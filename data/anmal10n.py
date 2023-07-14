# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : anmal10n.py
#   @Author      : Zeren Sun
#   @Created date: 2023/6/13 09:37
#   @Description :
#
# ================================================================
import os
from torch.utils.data.dataset import Dataset
from PIL import Image


class Animal10N(Dataset):
    def __init__(self, split='train', root_dir='animal10n', transform=None):
        super().__init__()
        self.root = os.path.join(root_dir, 'training' if split == 'train' else 'testing')
        self.image_files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        self.targets = [int(filename.split('_')[0]) for filename in self.image_files]
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.image_files[index])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[index]
        return {'index': index, 'data': image, 'label': target}

    def __len__(self):
        return len(self.image_files)

