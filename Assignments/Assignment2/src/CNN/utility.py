import os
import sys
import torch
import torchvision.transforms as transforms
import torch.utils.data.sampler as sampler
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ipdb as pdb
import argparse
import yaml
import config


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

def show_sample_image_loader(data_loader):
    img = iter(data_loader).next()
    img = img[0][0]
    img = img.numpy()
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    plt.imshow(img)
    plt.show()

def show_sample_image(img):
    img = img.numpy()
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)
    plt.imshow(img)
    plt.show()

def load_dataset(config):
    image_size = config.image_size
    train_size = config.train_size
    batch_size_train = config.batch_size_train
    valid_size = config.valid_size
    batch_size_valid = config.batch_size_valid
    test_size = config.test_size
    batch_size_test = config.batch_size_test
    indxes = list(range(train_size+valid_size+test_size))
    np.random.shuffle(indxes)

    train_idx, valid_idx, test_idx = indxes[:train_size], indxes[train_size:(train_size+valid_size)], indxes[(train_size+valid_size):]

    train_sampler = sampler.SubsetRandomSampler(train_idx)
    valid_sampler = sampler.SubsetRandomSampler(valid_idx)
    test_sampler = sampler.SubsetRandomSampler(test_idx)

    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        #                         0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=config.data_set_path, transform=data_transform)
    np.random.shuffle(dataset.imgs)
    train_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_train, sampler=train_sampler, num_workers=10)
    # pdb.set_trace()
    # show_sample_image(train_data_loader)
    test_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_test,  sampler=test_sampler, num_workers=10)

    valid_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_valid,  sampler=valid_sampler, num_workers=10)


    print('Total number of loaded images: {0}'.format(len(train_sampler)+len(test_sampler)+len(valid_sampler)))
    print('Image size: {0}'.format(dataset[0][0].size()))
    return train_data_loader, valid_data_loader, test_data_loader
