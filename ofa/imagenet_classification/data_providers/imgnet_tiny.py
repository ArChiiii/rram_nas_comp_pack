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
from .imagenet import ImagenetDataProvider
#from ofa.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler

import torch
import torchvision

__all__ = ["ImgnetTinyDataProvider"]

class ImgnetTinyDataProvider(ImagenetDataProvider):
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

     
        
        self.path =r"~/once-for-all/imgnet-tiny/tiny-imagenet-200/"
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
                shuffle=True,
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
                pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                num_workers=n_worker,
                pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test



    @staticmethod
    def name():
        return "imgnet-tiny"
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size 
    @property
    def n_classes(self):
        return 200

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/once-for-all/imgnet-tiny/tiny-imagenet-200/")
        return self._save_path

    def train_dataset(self, _transforms=None):

        transform = transforms.Compose([

            #  transforms.Resize(256), # Resize images to 256 x 256
            #     transforms.CenterCrop(224), # Center crop image
                # transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])

])




        return datasets.ImageFolder(self.path + 'train', transform=transform)

    def test_dataset(self, _transforms=None):

        transform = transforms.Compose([
            #  transforms.Resize(256), # Resize images to 256 x 256
            #     transforms.CenterCrop(224), # Center crop image
                # transforms.CenterCrop(56),
                transforms.ToTensor(),  # Converting cropped images to tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])

        return datasets.ImageFolder(self.path + 'val', transform=transform)
    

