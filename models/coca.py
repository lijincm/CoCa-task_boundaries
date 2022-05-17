# Copyright 2021-present, Zhong Ji, Jin Li, Qiang Wang, Zhongfei Zhang.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import numpy as np
import torchvision.transforms as transforms
from utils.scloss import SupConLoss
import time
from datasets import get_dataset


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Calibration w task boundaries.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
    return parser
""" # for cifar-10
def rotate_img(img, s):
    transform = transforms.RandomResizedCrop(size=(64, 64), scale=(0.66, 0.67), ratio = (0.99,1.00))
    img = transform(img)
    return torch.rot90(img, s, [-1, -2])
"""
# for tiny ImageNet
def rotate_img(img, s):
    if s//4 == 0:
        transform = transforms.RandomResizedCrop(size=(64, 64), scale=(0.66, 0.67), ratio = (0.99,1.00))
    elif s // 4 == 1:
        transform = transforms.RandomResizedCrop(size=(64, 64), scale=(0.99, 1.00), ratio=(0.66, 0.67))
    elif s // 4 == 2:
        transform = transforms.RandomResizedCrop(size=(64, 64), scale=(0.99, 1.00), ratio=(1.32, 1.33))
    else:
        transform = lambda x: x
    img = transform(img)
    return torch.rot90(img, s%4, [-1, -2])

class Coca(ContinualModel):
    #Complementary Calibration w task boundaries
    NAME = 'coca'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Coca, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.criterion = SupConLoss()

        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.cls = get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def end_task(self, dataset):
        self.task += 1

    def observe(self, inputs, labels, not_aug_inputs):
        #start = time.time()
        self.opt.zero_grad()
        real_batch_size = inputs.shape[0]
        logits, _, bat_inv, _  = self.net(inputs, return_features=True)
        loss = self.loss(logits[: ,self.task * self.cls : (self.task+1) * self.cls ], labels- torch.tensor([self.task * self.cls], device = self.device))

        if not self.buffer.is_empty():
            """"""
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs , _, inv_feat,feat = self.net(buf_inputs, return_features = True)
            omega = 0.1
            diaN = torch.eye(buf_inputs.shape[0], device=self.device)
            A = F.softmax(torch.mm(inv_feat, inv_feat.transpose(0, 1))- diaN*0.1  , dim = 1)
            soft_targets = torch.mm((1 - omega) * torch.inverse(diaN - omega * A), buf_outputs)
            soft_targets = soft_targets.detach()
            gamma = 0.01
            loss += self.args.alpha * F.mse_loss(buf_outputs, (1 - gamma) * buf_logits + gamma * soft_targets)
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs, _, buf_inv, _  = self.net(buf_inputs, return_features = True)
            loss += self.args.beta * self.loss(buf_outputs[: , : (self.task+1) * self.cls], buf_labels)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            bat_inv = torch.cat((bat_inv, buf_inv))

        label_shot = torch.arange(16).repeat(inputs.shape[0])
        label_shot = label_shot.type(torch.LongTensor)
        choice = np.random.choice(a=inputs.shape[0] , size=inputs.shape[0], replace=False)
        rot_label = label_shot[choice].to(self.device)
        rot_inputs = inputs.cpu()
        for i in range(0, inputs.shape[0]):
            rot_inputs[i] = rotate_img(rot_inputs[i],rot_label[i])
        rot_inputs = rot_inputs.to(self.device)
        _ , rot_outputs, t_inv, _= self.net(rot_inputs, return_features = True)

        loss += 0.3 * self.criterion(bat_inv, t_inv, labels)
        loss += 0.3 * self.loss(rot_outputs, rot_label)


        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size],
                             logits=logits.data)

        return loss.item()
