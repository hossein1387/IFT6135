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
    parser.add_argument('-t', '--configtype', help='configurations to run', required=True)
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
    batch_size = config['batch_size']
    for batch_num in range(config['num_batches']):
        seq_len = random.randint(config['seq_len_min'], config['seq_len_max'])
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, 8))
        seq = Variable(torch.from_numpy(seq))

        # The input includes an additional channel used for the delimiter
        inp = Variable(torch.zeros(seq_len + 1, batch_size, seq_width + 1))
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        seq2 = Variable(torch.zeros(seq_len, batch_size, seq_width)+0.5)
        act_inp = Variable(torch.zeros(seq_len, batch_size, seq_width + 1))
        act_inp[:seq_len, :, :seq_width] = seq2

        yield batch_num+1, inp.float(), outp.float(), act_inp.float()

    print('Total number of loaded images: {0}'.format(len(train_sampler)+len(test_sampler)+len(valid_sampler)))
    print('Image size: {0}'.format(dataset[0][0].size()))
    return train_data_loader, valid_data_loader, test_data_loader
