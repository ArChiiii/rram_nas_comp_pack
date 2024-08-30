import os.path as osp
import numpy as np
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import copy
from ofa.utils import AverageMeter, accuracy


class AccuracyGenerator:
    def __init__(self, ofa_net, train_data_loader, data_loader, device="cuda:0"):
        self.device = device
        self.train_data_loader = train_data_loader
        self.data_loader = data_loader
        self.ofa_net = ofa_net.to(self.device)

    @torch.no_grad()
    def predict_accuracy(self, population, fine_tune=True):
        pred_acc =[]
        for sample in population:
            acc = self.predict_accuracy_by_sameple(sample, fine_tune)
            pred_acc.append(acc)
        return pred_acc

    def predict_accuracy_by_sameple(self, sample, fine_tune=True):
        
        ks_list = copy.deepcopy(sample["ks"])
        ex_list = copy.deepcopy(sample["e"])
        d_list = copy.deepcopy(sample["d"])

        self.ofa_net.set_active_subnet(
            ks=ks_list,
            e=ex_list,
            d=d_list
        )

        net = self.ofa_net.get_active_subnet()
        
        if fine_tune:
            self.fine_tune(net, self.train_data_loader, epoch=1)

        top1 = self.validate(net, self.data_loader)

        return top1/100
    
    def fine_tune(self, net, data_loader, epoch):
        # net = net.to(device)

        # cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

        net.train()
        # net = net.to(device)
        losses = AverageMeter()
        top1 = AverageMeter()

        for _ in range(epoch):
            # with tqdm(total=len(data_loader), desc="Fine Tune") as t:
            for i, (images, labels) in enumerate(data_loader):
                # print(i)
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                # losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # t.set_postfix({
                #     'loss': losses.avg,
                #     'top1': top1.avg,
                # })
                # t.update(1)
        return top1.avg

    @torch.no_grad()
    def validate(self, net, data_loader):
        # net = net.to(device)

        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss().to(self.device)

        net.eval()
        # net = net.to(device)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            # with tqdm(total=len(data_loader), desc="Validate") as t:
            for i, (images, labels) in enumerate(data_loader):
                # print(i)
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                # losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                # top5.update(acc5[0].item(), images.size(0))
                    # t.set_postfix(
                    #     {
                    #         "loss": losses.avg,
                    #         "top1": top1.avg,
                    #         "top5": top5.avg,
                    #         "img_size": images.size(2),
                    #     }
                    # )
                    # t.update(1)

        return top1.avg

