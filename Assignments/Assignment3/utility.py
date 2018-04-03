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
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=False)
    parser.add_argument('-t', '--configtype', help='configurations to run', required=False)
    parser.add_argument('-l', '--load_checkpoint', help='loads a checkpoint', required=False, action='store_true')
    parser.add_argument('-p', '--plot_all_average', help='plots all average cost', required=False, action='store_true')
    parser.add_argument('-v', '--visualize_read_write', help='visualization of read and write access', required=False, action='store_true')
    args = parser.parse_args()
    return vars(args)

def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

def load_dataset(config_obj, test=False, T=None, min=None, max=None):
    config = config_obj.config_dict
    if test:
        num_batches = 20
    else:
        num_batches = config['num_batches']
    batch_size = config['batch_size']
    if not test and (min!=None and max!=None):
        seq_len = random.randint(min, max)
    elif test:
        seq_len = T
    else:
        seq_len = config_obj.seq_len
#    if config['model_type'] == "LSTM":
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


def visualize_read_write(X,result,N) :
    T, batch_size, num_bits = X.size()
    T = T - 1
    num_bits = num_bits - 1
    
    plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3]) 
    
    ax = plt.subplot(gs[0,0])
    y_in = torch.cat((X[:,0,:].data,torch.zeros(T,num_bits+1)),dim=0)
    ax.imshow(torch.t(y_in), cmap='gray',aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('inputs')
    
    ax = plt.subplot(gs[0,1])
    y_out = torch.cat((torch.zeros(T+1,num_bits),result['y_out_binarized'][:,0,:]),dim=0)
    y_out = torch.cat((y_out,torch.zeros(2*T+1,1)),dim=1)
    ax.imshow(torch.t(y_out), cmap='gray',aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('outputs')
    
    states = result['states']
    read_state = torch.zeros(len(states),N)  # read weight
    write_state = torch.zeros(len(states),N) # write weight
    for i in range(0,len(states)) :
        reads, controller_state, heads_states = states[i]
        read_state[i,:] = heads_states[0][0].data
        write_state[i,:] = heads_states[1][0].data
        
        
    ax = plt.subplot(gs[1,0])
    ax.imshow(torch.t(write_state[:,90:]), cmap='gray',aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #ax.text(1,40,'Time', fontsize=11)
    ax.text(6,41,'Write Weightings',fontsize=12)
    #ax.arrow(6,60,60, fc="k", ec="k", head_width=0.5, head_length=1, color='w')
    #ax.annotate('Time', xy=(0.4, -0.1), xycoords='axes fraction', xytext=(0, -0.1),
    #            arrowprops=dict(arrowstyle="->", color='black'))
    #ax.annotate('Location', xy=(-0.2, 0.4), xycoords='axes fraction', xytext=(-0.26, 0), 
    #            arrowprops=dict(arrowstyle="->", color='black'))
    
    
    ax = plt.subplot(gs[1,1])
    ax.imshow(torch.t(read_state[:,90:]), cmap='gray',aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #ax.text(1,40,'Time', fontsize=11)
    ax.text(6,41,'Read Weightings',fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualization.pdf')
    plt.show()
