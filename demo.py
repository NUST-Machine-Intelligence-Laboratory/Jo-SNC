import os
import pathlib
import time
import datetime
import argparse
import yaml
from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.eval import evaluate
from utils.builder import *
from utils.utils import *
from utils.model import DualHeadModel
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_dataset(cfg):
    transform = build_transform(cfg.rescale_size, cfg.crop_size, dataset=cfg.dataset)
    if cfg.dataset == 'animal10n':
        dataset = build_animal10n_dataset(os.path.join(cfg.data_root, cfg.dataset), transform['cifar_train'], transform['cifar_test'])
    elif cfg.dataset in ['web-aircraft', 'web-bird', 'web-car']:
        dataset = build_webfg_dataset(os.path.join(cfg.data_root, cfg.dataset), transform['train'], transform['test'])
    elif cfg.dataset == 'food101n':
        dataset = build_food101n_dataset(os.path.join(cfg.data_root, cfg.dataset), transform['train'], transform['test'])
    elif cfg.dataset in ['mini-webvision', 'webvision']:
        dataset = build_mini_webvision_dataset(os.path.join(cfg.data_root, cfg.dataset), transform['train'], transform['test'], num_class=cfg.n_classes)
    else:
        raise NotImplementedError(f'{cfg.dataset} is not supported.')
    return dataset


def build_model(cfg, arch=''):
    if cfg.dataset == 'animal10n':
        arch = 'vgg19_bn'
    elif cfg.dataset in ['web-aircraft', 'web-bird', 'web-car']:
        arch = 'resnet50'
    elif cfg.dataset == 'food101n':
        arch = 'resnet50'
    elif cfg.dataset == 'mini-webvision':
        arch = 'resnet50' if arch != 'InceptionResNetV2' else arch
    else:
        raise NotImplementedError(f'{cfg.dataset} is not supported.')
    model = DualHeadModel(arch=arch, num_classes=cfg.n_classes, mlp_hidden=2, feature_dim=512, pretrained=False).to(device)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--arch', type=str, default='')
    parser.add_argument('--gpu', type=str, default=0)
    args = parser.parse_args()
    dataset_cfg = edict(yaml.load(open(args.cfg, 'r'), Loader=yaml.FullLoader))

    set_seed(0)
    device = torch.device(f'cuda:{args.gpu}')

    dataset = build_dataset(dataset_cfg)
    test_loader = DataLoader(dataset['test'], batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    net = build_model(dataset_cfg).to(device)
    net.load_state_dict(torch.load(args.model_path))

    test_accuracy = evaluate(test_loader, net, device, progress_bar=True)
    print(f'Test accuracy: {test_accuracy:.2f}')


# python demo.py --cfg config/aircraft.yaml --model-path web_aircraft.pth --gpu 0