# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .base_provider import DataProvider
# from .imagenet import ImagenetDataProvider
#from ofa.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler

import torch
import torchvision

__all__ = ["CIFARDataProvider"]

class CIFARDataProvider(DataProvider):
    DEFAULT_PATH = "/testing/cifar10"
    def __init__(
        self,
        save_path=None,
        train_batch_size=256,
        test_batch_size=512,
        valid_size=None,
        n_worker=32,
        resize_scale=0.08,
        distort_color=None,
        image_size=224,
        num_replicas=None,
        rank=None,
    ):


        warnings.filterwarnings("ignore")
        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = "None" if distort_color is None else distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
      
        self.active_img_size = self.image_size
        valid_transforms = self.build_valid_transform()
        train_loader_class = torch.utils.data.DataLoader

     
        
        self.CIFARpath =r"~/once-for-all/ofa/testing/cifar10/"
        train_dataset = self.train_dataset()

     
        if num_replicas is not None:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas, rank
            )
            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )
        self.valid = None

        test_dataset = self.test_dataset()
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas, rank
            )
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=n_worker,
                pin_memory=False,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=False,
            )

        if self.valid is None:
            self.valid = self.test



    @staticmethod
    def name():
        return "cifar10"
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size 
    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/once-for-all/ofa/testing/cifar10/")
        return self._save_path

    def train_dataset(self, _transforms=None):

        transform_train = transforms.Compose([
            torchvision.transforms.Resize((70, 70)),
            torchvision.transforms.RandomCrop((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        return torchvision.datasets.CIFAR10(self.CIFARpath, train=True, download=True,
                             transform=transform_train)

    def test_dataset(self, _transforms=None):

        transform_test = transforms.Compose([
            torchvision.transforms.Resize((70, 70)),
            torchvision.transforms.CenterCrop((64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        return torchvision.datasets.CIFAR10(self.CIFARpath, train=False, download=True,
                             transform=transform_test)