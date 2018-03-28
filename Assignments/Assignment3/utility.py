import os
import sys
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.autograd as autograd
import torch.distributions as distributions

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
    parser.add_argument('-l', '--data_set', help='configurations to run', required=False)
    args = parser.parse_args()
    return vars(args)

def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

def load_dataset(config_obj, min=None, max=None):
    config = config_obj.config_dict
    batch_size = config['batch_size']
    if min!=None and max!=None:
        seq_len = random.randint(min, max)
    else:
        seq_len = config_obj.seq_len
    print("seq_len = {0}\n".format(seq_len))
    if config['model_type'] == "LSTM":
        for batch_num in range(config['num_batches']):
            seq = np.random.binomial(1, 0.5, (seq_len, batch_size, 8))
            seq = Variable(torch.from_numpy(seq))

            # The input includes an additional channel used for the delimiter
            inp = Variable(torch.zeros(seq_len + 1, batch_size, config['data_width'] + 1))
            inp[:seq_len, :, :config['data_width']] = seq
            inp[seq_len, :, config['data_width']] = 1.0 # delimiter in our control channel
            outp = seq.clone()

            seq2 = Variable(torch.zeros(seq_len, batch_size, config['data_width'])+0.5)
            act_inp = Variable(torch.zeros(seq_len, batch_size, config['data_width'] + 1))
            act_inp[:seq_len, :, :config['data_width']] = seq2

            yield batch_num+1, inp.float(), outp.float(), act_inp.float()
    else:
        for batch_num in range(config['num_batches']):

            # All batches have the same sequence length
            seq = np.random.binomial(1, 0.5, (seq_len, batch_size, config['data_width']))
            seq = Variable(torch.from_numpy(seq))

            # The input includes an additional channel used for the delimiter
            inp = Variable(torch.zeros(seq_len + 1, batch_size, config['data_width'] + 1))
            inp[:seq_len, :, :config['data_width']] = seq
            inp[seq_len, :, config['data_width']] = 1.0 # delimiter in our control channel
            outp = seq.clone()
            yield batch_num+1, inp.float(), outp.float()

