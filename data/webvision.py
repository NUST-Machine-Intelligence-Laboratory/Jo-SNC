# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : webvision50.py
#   @Author      : Zeren Sun
#   @Created date: 2023/4/17 15:46
#   @Description :
#
# ================================================================
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os


class imagenet_dataset(Dataset):
    def __init__(self, root_dir, webvision_root, transform, num_class):
        self.root = os.path.join(root_dir, 'val')
        self.transform = transform
        self.val_data = []
        with open(os.path.join(webvision_root, 'info/synsets.txt')) as f:
            lines = f.readlines()
        synsets = [x.split()[0] for x in lines]
        for c in range(num_class):
            class_path = os.path.join(self.root, synsets[c])
            imgs = os.listdir(class_path)
            for img in imgs:
                self.val_data.append([c, os.path.join(class_path, img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return {'index': index, 'data': img, 'label': target}

    def __len__(self):
        return len(self.val_data)


class webvision_dataset(Dataset):
    def __init__(self, root_dir, transform, mode, num_class):
        self.root = root_dir
        self.transform = transform
        self.mode = mode

        if self.mode == 'test':
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
            self.samples = self.val_imgs
        else:
            with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
                lines = f.readlines()
            self.train_imgs = []
            self.train_labels = {}
            num_samples = 0
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.train_imgs.append(img)
                    self.train_labels[img] = target
                    num_samples += 1
            self.samples = self.train_imgs

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img = self.transform(image)
            return {'index': index, 'data': img, 'label': target}
        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(os.path.join(self.root, 'val_images_256/', img_path)).convert('RGB')
            img = self.transform(image)
            return {'index': index, 'data': img, 'label': target}
        else:
            raise AssertionError("mode is incorrect.")

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)
