import random
import time
import numpy as np
import utility
import config
import matplotlib.pyplot as plt

import os
import os.path
import time
import sys

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


import ipdb as pdb

# Global Variables
# Records the model's performance
records = {"train": [[], [], "train records"], "test": [[], [], "test records"], "valid": [[],[], "valid records"]}


loss_function = nn.BCELoss()

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


sample_binary=distributions.Bernoulli(torch.Tensor([0.5]))


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm=nn.LSTM(BYTE_WIDTH+1, HIDDEN_NUM)
        self.mlp=nn.Linear(HIDDEN_NUM,BYTE_WIDTH)

    def init_hidden(self,batch_size):
        self.hidden= ( autograd.Variable(torch.randn(1, batch_size, HIDDEN_NUM)),\
          autograd.Variable(torch.randn((1, batch_size, HIDDEN_NUM))) )
        if cuda_available:
            self.hidden=self.hidden.cuda()

    def forward(self, sequence):
        # out, (self.hidden, self.cell) = self.lstm(inputs, None)
        out, self.hidden = self.lstm(sequence, self.hidden)
        out=self.mlp(out)
        return out

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000

def gen1seq():
    length=np.random.randint(2,SEQUENCE_MAX_LEN+1)
    # length=SEQ_SIZE+1
    seq=sample_binary.sample_n(9*length).view(length, 1, -1)
    seq[:,:,-1]=0.0
    seq[-1]=0.0
    seq[-1,-1,-1]=1.0
    return seq

def gen1seq_act(length):
    seq=torch.zeros(9*length).view(length, 1, -1)+0.5
    seq[:,:,-1]=0.0
    seq[-1]=0.0
    seq[-1,-1,-1]=1.0
    return seq

#total_batches: total of the batch 
#batch_size: in each batch, there are batch_size sequences together as same length of sequence.
def dataloader(total_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):
    """Generator of random sequences for the copy task.
    Creates random batches of "bits" sequences.
    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]
    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.
    return: batch_num,Xs(seq_len+1,Byte_size+1),Ys(seq_len,Byte_size),act_seqs(seq_len,Byte_size+1)
    act_seqs is for the action of copy!
    
    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(total_batches):

        # All batches have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
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

def train_model(model,criterion,optimizer, seqs_loader, interval=500):
    
    start_ms = get_ms()

    list_losses =[]
    list_costs =[]
    list_bits=[]
    list_seq_num=[]
    losses=0
    costs=0
    lengthes=0
    for batch_num, X, Y, act in seqs_loader:
        # start = time.time()
        if cuda_available:
            X, Y, act = X.cuda(), Y.cuda(), act.cuda()
        
        model.init_hidden(BATCH_SIZE)
        optimizer.zero_grad()
        # print(x.size(),y.size(),act.size())
        #input data
        model.forward(X)
        # do copy in useing the act_seq
        out_seq=model.forward(act)

        sigmoid_out=F.sigmoid(out_seq)
        loss = criterion(sigmoid_out, Y)
        loss.backward()
        optimizer.step()

        list_losses.append(loss.data[0])

        out_binarized = sigmoid_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        # The cost is the number of error bits per sequence
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))

        losses+=loss
        costs+=cost
        lengthes+=BATCH_SIZE
        # end = time.time()

        '''
        Because of LSTM has poor memory while sequence lengths >=20 However, 
        memory capacity is inversely proportional to the length of the sequence.
        But for longer sequences and shorter sequences, they cannot be compared 
        exactly in proportion to the number of remembered bits because long sequences
        can be more difficult. Therefore, the way we calculate the cost is to generate 
        random sequence lengths and sequence sizes of batch_size randomly according to 
        intervals of interval values. The sequence length averaged by the interval 
        number should be similar. At this time, calculate the cost after the average 
        nterval is used to test whether the cost convergence.
        TOTAL_BATCHES is a multiple of INTERVAL.
        '''
        if (batch_num) % INTERVAL==0 :
            list_costs.append(costs/INTERVAL/BATCH_SIZE) #per sequence
            list_losses.append(losses.data[0]/INTERVAL/BATCH_SIZE)
            list_seq_num.append(lengthes) # per thousand
            mean_time = ((get_ms() - start_ms) / INTERVAL) / BATCH_SIZE
            print ("Batch %d th, loss %f, cost %f, Time %.3f ms/sequence." % (batch_num, list_losses[-1], list_costs[-1], mean_time) )
            
            costs = 0
            losses = 0
            start_ms = get_ms()

    return list_losses,list_costs,list_seq_num

def evaluate(model,criterion,optimizer, test_data_loader) : 
    costs = 0
    losses = 0
    lengthes = 0
    # optimizer = optim.RMSprop(model.parameters(), lr=rmsprop_lr, momentum = MOMENTUM)

    for batch_num, X, Y, act in test_data_loader:
        # start = time.time()
        if cuda_available:
            X, Y, act = X.cuda(), Y.cuda(), act.cuda()
        
        model.init_hidden(BATCH_SIZE)
        optimizer.zero_grad()
        # print(x.size(),y.size(),act.size())
        #input data
        model.forward(X)
        # do copy in useing the act_seq
        out_seq=model.forward(act)

        sigmoid_out=F.sigmoid(out_seq)
        loss = criterion(sigmoid_out, Y)
        loss.backward()
        optimizer.step()

        lengthes+=BATCH_SIZE
    
        losses += loss
        
        out_binarized = sigmoid_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        # The cost is the number of error bits per sequence
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))

        costs += cost
        
    print ("T = %d, Average loss %f, average cost %f" % (Y.size(0), losses.data[0]/lengthes, costs/lengthes))
    return losses.data/lengthes, costs/lengthes

def saveCheckpoint(model,list_batch_num,list_loss, list_cost, path='lstm') :
    print('Saving..')
    state = {
        'model': model,
        'list_batch_num': list_batch_num,
        'list_loss' : list_loss,
        'list_cost' : list_cost
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+path)

def loadCheckpoint(path='lstm'):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+path)
    model = checkpoint['model']
    list_batch_num = checkpoint['list_batch_num']
    list_loss = checkpoint['list_loss']
    list_cost = checkpoint['list_cost']
    return model, list_batch_num, list_loss, list_cost
#%%

def build_model(config):
    model = LSTM()
    # model.apply(weights_init)
    criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(),  lr=config.lr0, momentum=config.momentum)
    optimizer = optim.Adam(model.parameters(),  lr=0.001)
    return model, criterion, optimizer

if __name__ == '__main__':
    pdb.set_trace()
    args = utility.parse_args()
    config_type = args['configtype']
    config_file = args['configfile']
    config = config.Configuration(config_type, config_file)
    model, criterion, optimizer = build_model(config)
    train_model(model, config, criterion, optimizer)

def not_used():
    model = LSTMcopy()
    loss_function = nn.BCELoss()
    # criterion = loss_function  #nn.CrossEntropyLoss() nn.MSELoss() nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=LR_0)
    # optimizer = optim.SGD(model.parameters(), lr=LR_0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # optimizer = optim.Adam(model.parameters(), lr=LR_0)
    optimizer = optim.RMSprop(model.parameters(), lr=rmsprop_lr, momentum = MOMENTUM)
        
    list_loss,list_cost,list_seq_num=train_model(model,loss_function,optimizer,train_loader)

    saveCheckpoint(model,list_seq_num,list_loss, list_cost, path='lstm1') 

    #%%

    model, list_seq_num, list_loss, list_cost = loadCheckpoint(path='lstm1')

    plt.figure()
    # plt.plot(range(0,TOTAL_BATCHES),list_cost,label='LSTM')
    plt.plot(list_seq_num,list_cost)
    # plt.plot(range(1,1000),valid_acc_list,label='Validation')
    plt.xlabel('Sequence number')
    plt.ylabel('Cost per sequence')
    plt.legend()
    plt.savefig('lstm-xx.pdf')
    #plt.show()

    #plot accuracy as a function of epoch
    # plt.figure()
    # # plt.plot(range(0,total_batches),loss_list,label='LSTM')
    # plt.plot(range(0,TOTAL_BATCHES),list_bits,label='Bits Num')
    # # plt.plot(range(1,1000),valid_acc_list,label='Validation')
    # plt.xlabel('Sequence number')
    # plt.ylabel('Loss per sequence')
    # plt.legend()
    # # plt.savefig('lstm1.pdf')
    # plt.show()

    #%%
    list_avg_loss = []
    list_avg_cost = []
    for T in range(10,110,10) : 
        test_data_loader = dataloader(TOTAL_BATCHES, BATCH_SIZE,
                        BYTE_WIDTH,min_len=T,max_len=T)
        avg_loss, avg_cost = evaluate(model,loss_function,optimizer,test_data_loader)
        list_avg_loss.append(avg_loss)
        list_avg_cost.append(avg_cost)

    #%%
        
    plt.plot(range(10,110,10),list_avg_cost)
    plt.xlabel('T')
    plt.ylabel('average cost')
    plt.savefig('lstm-cost-T.pdf')

    #%%
    test1=Variable(gen1seq())
    print(test1)
    model.init_hidden(1)
    model.forward(test1)
    actx=Variable(gen1seq_act(test1.size(0)))
    # model.init_hidden(1)
    print(torch.sigmoid(model.forward(actx)))
    # actx=Variable(gen1seq_act(test1.size(0)))
    # model.init_hidden(1)
    print(torch.sigmoid(model.forward(actx)))
