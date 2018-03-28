import os
import os.path
import time

import config
import models
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import utility
from torch.autograd import Variable

# Global Variables
# Records the model's performance
records = {"train": [[], [], "train records"], "test": [[], [], "test records"]}

def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000

def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


'''
def gen1seq():
    length=np.random.randint(2,SEQUENCE_MAX_LEN+1)
    # length=SEQ_SIZE+1
    seq=sample_binary.sample_n(9*length).view(length, 1, -1)
    seq[:,:,-1]=0.0
    seq[-1]=0.0
    seq[-1,-1,-1]=1.0
    return seq
'''

def gen1seq_act(length):
    seq=torch.zeros(9*length).view(length, 1, -1)+0.5
    seq[:,:,-1]=0.0
    seq[-1]=0.0
    seq[-1,-1,-1]=1.0
    return seq


def train_lstm_model(config, model, criterion, optimizer, seqs_loader):
    
    start_ms = get_ms()

    list_losses = []
    list_costs =[]
    list_seq_num=[]
    losses=0
    costs=0
    lengths = 0

    for batch_num, X, Y, act in seqs_loader:
        model.init_hidden(config['batch_size'])
        optimizer.zero_grad()
        model.forward(X)
        out_seq=model.forward(act)
        sigmoid_out=F.sigmoid(out_seq)
        loss = criterion(sigmoid_out, Y)
        loss.backward()
        optimizer.step()
        list_losses.append(loss.data[0])
        out_binarized = sigmoid_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))
        losses+=loss
        costs+=cost
        lengths += config['batch_size']
        if (batch_num) % config['interval']==0 :
            list_costs.append(costs/config['interval']/config['batch_size']) #per sequence
            list_losses.append(losses.data[0]/config['interval']/config['batch_size'])
            list_seq_num.append(lengths)  # per thousand
            mean_time = ((get_ms() - start_ms) / config['interval']) / config['batch_size']
            print ("Batch %d th, loss %f, cost %f, Time %.3f ms/sequence." % (batch_num, list_losses[-1], list_costs[-1], mean_time) )
            
            costs = 0
            losses = 0
            start_ms = get_ms()

    return list_losses,list_costs,list_seq_num

def train_ntm_model(config, model, criterion, optimizer, train_data_loader) : 
    list_seq_num = []
    list_loss = []
    list_cost = []
    lengthes = 0
    losses = 0
    costs = 0
    for batch_num, X, Y  in train_data_loader:
        # pdb.set_trace()
        inp_seq_len, _, _ = X.size()
        outp_seq_len, config['batch_size'], output_size = Y.size()
        model.init_sequence(config['batch_size'])
        optimizer.zero_grad()
        for i in range(inp_seq_len):
            model(X[i])
        y_out = Variable(torch.zeros(Y.size()))
        for i in range(outp_seq_len):
            y_out[i], _ =  model()
        length = config['batch_size']
        lengthes +=  length
        loss = criterion(y_out, Y)      # calculate loss
        losses += loss
        y_out_binarized = y_out.clone().data # binary output
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
        cost = torch.sum(torch.abs(y_out_binarized - Y.data))
        costs += cost
        loss.backward()
        clip_grads(model)
        optimizer.step()
        if batch_num % config['interval'] == 0  :
            list_loss.append(losses.data/config['interval']/config['batch_size'])
            list_seq_num.append(lengthes/1000) # per thousand
            list_cost.append(costs/config['interval']/config['batch_size'])
        if (batch_num % config['interval'] == 0 ): 
            print ("Epoch %d, loss %f, cost %f" % (batch_num, losses/config['interval']/config['batch_size'], costs/config['interval']/config['batch_size']) )
            costs = 0
            losses = 0
            
    return list_seq_num, list_loss, list_cost


def evaluate(model,criterion,optimizer, test_data_loader) : 
    costs = 0
    losses = 0
    lengths = 0
    for batch_num, X, Y, act in test_data_loader:
        model.init_hidden(config['batch_size'])
        optimizer.zero_grad()
        model.forward(X)
        out_seq=model.forward(act)
        sigmoid_out=F.sigmoid(out_seq)
        loss = criterion(sigmoid_out, Y)
        loss.backward()
        optimizer.step()
        lengths += 20
        losses += loss
        out_binarized = sigmoid_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))
        costs += cost
    print("T = %d, Average loss %f, average cost %f" % (
    Y.size(0), losses.data[0] / lengths, costs / lengths))  # TODO: Check loss averaging
    return losses.data / lengths, costs / lengths


def test_sequences(config, model, criterion, optimizer):
    seq_lengths = list(range(10, 110, 10))
    losses = []
    for seq_length in seq_lengths:
        test_seqs_loader = utility.load_dataset(config, test=True, T=seq_length)
        loss, _ = evaluate(model, criterion, optimizer, test_seqs_loader)
        losses.append(loss)

    losses = np.array(losses)
    fname = config['model_type'] + '_test_losses.np'
    fname = os.path.join('logs', fname)
    np.savetxt(fname, losses)

if __name__ == '__main__':
    #pdb.set_trace()
    args = utility.parse_args()
    config_type = args['configtype']
    config_file = args['configfile']
    config = config.Configuration(config_type, config_file).config
    model, criterion, optimizer = models.build_model(config)
    seqs_loader = utility.load_dataset(config)
    if config['model_type'] == "LSTM":
        train_lstm_model(config, model, criterion, optimizer, seqs_loader)
    else:
        train_ntm_model(config, model, criterion, optimizer, seqs_loader)
    test_sequences(config, model, criterion, optimizer)
