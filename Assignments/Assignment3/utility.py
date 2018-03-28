import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.parallel
import yaml
from torch.autograd import Variable


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

def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def load_dataset(config, test=False, T=None):
    batch_size = config['batch_size']
    if test:
        num_batches = 20
        seq_len = T
    else:
        num_batches = config['num_batches']
    if config['model_type'] == "LSTM":
        for batch_num in range(num_batches):
            if not test:
                seq_len = random.randint(config['seq_len_min'], config['seq_len_max'])
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
            seq_len = random.randint(config['seq_len_min'], config['seq_len_max'])
            seq = np.random.binomial(1, 0.5, (seq_len, batch_size, config['data_width']))
            seq = Variable(torch.from_numpy(seq))

            # The input includes an additional channel used for the delimiter
            inp = Variable(torch.zeros(seq_len + 1, batch_size, config['data_width'] + 1))
            inp[:seq_len, :, :config['data_width']] = seq
            inp[seq_len, :, config['data_width']] = 1.0 # delimiter in our control channel
            outp = seq.clone()
            yield batch_num+1, inp.float(), outp.float()


def plot():
    sns.set()
    lstm_fname = 'LSTM_test_losses.np'
    lstm_fname = os.path.join('logs', lstm_fname)
    lstm_losses = np.loadtxt(lstm_fname)

    ntm_fname = 'NTM_LSTM_test_losses.np'
    ntm_fname = os.path.join('logs', ntm_fname)
    ntm_losses = np.loadtxt(ntm_fname)

    x = list(range(10, 110, 10))

    plt.plot(x, lstm_losses, label='lstm')
    plt.plot(x, ntm_losses, label='ntm')

    plt.title('Graph of losses with varying sequence lengths')
    plt.xlabel('sequence length')
    plt.ylabel('loss')
    plt.legend()

    plot_fname = 'test_losses.png'
    plot_fname = os.path.join('plots', plot_fname)
    plt.savefig(plot_fname, bbox_inches='tight')

    plt.clf()

plot()
