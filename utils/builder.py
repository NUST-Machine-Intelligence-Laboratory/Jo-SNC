# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : builder.py.py
#   @Author      : Zeren Sun
#   @Created date: 2022/11/18 10:28
#   @Description :
#
# ================================================================

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from randaugment import CIFAR10Policy, ImageNetPolicy, Cutout, RandAugment
from data.noisy_cifar import NoisyCIFAR10, NoisyCIFAR100
from data.image_folder import IndexedImageFolder
from data.food101 import Food101
from data.food101n import Food101N
from data.clothing1m import Clothing1M
from data.webvision import webvision_dataset, imagenet_dataset
from data.anmal10n import Animal10N
from PIL import ImageFilter


# dataset --------------------------------------------------------------------------------------------------------------------------------------------

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_transform(rescale_size=512, crop_size=448, dataset='imagenet'):
    if dataset.startswith('cifar100n') or dataset.startswith('cifar80n'):
        normalization = torchvision.transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2009, 0.1984, 0.2023))
    elif dataset == 'food101n':
        normalization = torchvision.transforms.Normalize(mean=(0.5741, 0.4774, 0.3869), std=(0.2364, 0.2466, 0.2533))
    elif dataset == 'clothing1m':
        normalization = torchvision.transforms.Normalize(mean=(0.7215, 0.6846, 0.6678), std=(0.2503, 0.2628, 0.2622))
    elif dataset == 'web-aircraft':
        normalization = torchvision.transforms.Normalize(mean=(0.5108, 0.5413, 0.5649), std=(0.2062, 0.2038, 0.2209))
    elif dataset == 'web-bird':
        normalization = torchvision.transforms.Normalize(mean=(0.5313, 0.5339, 0.4821), std=(0.1836, 0.1828, 0.1930))
    elif dataset == 'web-car':
        normalization = torchvision.transforms.Normalize(mean=(0.4631, 0.4522, 0.4477), std=(0.2719, 0.2700, 0.2719))
    else:
        normalization = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    cifar_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=crop_size, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalization
    ])
    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalization
    ])
    cifar_train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=crop_size, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        torchvision.transforms.ToTensor(),
        normalization
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        normalization
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        normalization
    ])
    train_transform_strong_aug = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        RandAugment(),
        torchvision.transforms.ToTensor(),
        normalization
    ])

    train_transform_moco = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalization
    ])

    return {'train': train_transform, 'test': test_transform, 'train_strong_aug': train_transform_strong_aug,
            'cifar_train': cifar_train_transform, 'cifar_test': cifar_test_transform, 'cifar_train_strong_aug': cifar_train_transform_strong_aug,
            'train_moco': train_transform_moco}


def build_cifar10n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
    train_data = NoisyCIFAR10(root, train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=False)
    test_data = NoisyCIFAR10(root, train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=False)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data)}


def build_cifar100n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
    train_data = NoisyCIFAR100(root, train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=True)
    test_data = NoisyCIFAR100(root, train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=True)
    eval_train_data = NoisyCIFAR100(root, train=True, transform=test_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                                    openset_ratio=openset_ratio, verbose=False)
    train_indices_idn, train_indices_ood, train_indices_clean = train_data.get_sets()
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.data), 'n_test_samples': len(test_data.data),
            'train_indices_idn': train_indices_idn, 'train_indices_ood': train_indices_ood, 'train_indices_clean': train_indices_clean,
            'eval_train': eval_train_data}


def build_webfg_dataset(root, train_transform, test_transform):
    train_data = IndexedImageFolder(os.path.join(root, 'train'), transform=train_transform)
    eval_train_data = IndexedImageFolder(os.path.join(root, 'train'), transform=test_transform)
    test_data = IndexedImageFolder(os.path.join(root, 'val'), transform=test_transform)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples),
            'eval_train': eval_train_data}


def build_food101n_dataset(root, train_transform, test_transform):
    train_data = Food101N(root, transform=train_transform)
    test_data = Food101(os.path.join(root, 'food-101'), split='test', transform=test_transform)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}


def build_clothing1m_dataset(root, train_transform, test_transform):
    train_data = Clothing1M(root, split='train', transform=train_transform)
    valid_data = Clothing1M(root, split='val', transform=test_transform)
    test_data = Clothing1M(root, split='test', transform=test_transform)
    return {'train': train_data, 'test': test_data, 'val': valid_data,
            'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}


def build_mini_webvision_dataset(root, train_transform, test_transform, num_class=50):
    train_data = webvision_dataset(root, transform=train_transform, mode='train', num_class=num_class)
    test_data = webvision_dataset(root, transform=test_transform, mode='test', num_class=num_class)
    valid_data = imagenet_dataset(root.replace('mini-webvision' if num_class == 50 else 'webvision', 'imagenet'), webvision_root=root, transform=test_transform, num_class=num_class)
    return {'train': train_data, 'test': test_data, 'valid': valid_data,
            'n_train_samples': len(train_data.samples), 'n_test_samples': len(test_data.samples)}


def build_animal10n_dataset(root, train_transform, test_transform):
    train_data = Animal10N(split='train', root_dir=root, transform=train_transform)
    test_data = Animal10N(split='test', root_dir=root, transform=test_transform)
    return {'train': train_data, 'test': test_data, 'n_train_samples': len(train_data), 'n_test_samples': len(test_data)}
