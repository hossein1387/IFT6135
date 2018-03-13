from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.sampler as sampler
import torchvision.datasets as datasets
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ipdb as pdb
#from models import *
from torch.autograd import Variable
import sys
import argparse
import ipdb as pdb
import config
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=True)
    parser.add_argument('-r', '--configtype', help='configurations to run', required=True)
    args = parser.parse_args()
    return vars(args)

def read_config_file(config_file):
    with open(config_file, 'r') as stream:
        try:
            configs = yaml.load(stream)
        except yaml.YAMLError as exc:
            print ("Error loading YAML file {0}".format(config_file))
            sys.exit()
    return configs


def load_dataset(config):
    image_size = config.image_size
    train_size = config.train_size
    valid_size = config.valid_size
    test_size = config.test_size
    batch_size_train = config.batch_size_train
    batch_size_valid = config.batch_size_valid
    batch_size_test = config.batch_size_test
    indices = list(range(train_size+valid_size+test_size))
    np.random.seed(123)
    np.random.shuffle(indices)

    train_idx, valid_idx, test_idx = indices[:train_size], indices[train_size:(train_size+valid_size)], indices[(train_size+valid_size):]

    train_sampler = sampler.SubsetRandomSampler(train_idx)
    valid_sampler = sampler.SubsetRandomSampler(valid_idx)
    test_sampler = sampler.SubsetRandomSampler(test_idx)

    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0], std=[1])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=config.data_set_path, transform=data_transform)

    train_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_train, sampler=train_sampler, num_workers=10)

    valid_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_valid,  sampler=valid_sampler, num_workers=10)

    test_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_test,  sampler=test_sampler, num_workers=10)

    print('Loaded images(train and valid), total',len(train_sampler)+len(valid_sampler))
    print('Loaded test images, total',len(test_sampler))
    print('Image size: ', dataset[0][0].size())
    return train_data_loader, valid_data_loader, test_data_loader
