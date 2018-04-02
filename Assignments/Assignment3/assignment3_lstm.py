import matplotlib
matplotlib.use('Agg')
import os
import os.path
import time

import models
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import utility
from torch.autograd import Variable
import matplotlib.pyplot as plt


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

    list_loss = []
    list_cost =[]
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
        list_loss.append(loss.data[0])
        out_binarized = sigmoid_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))
        losses+=loss
        costs+=cost
        lengths += config['batch_size']
        if (batch_num) % config['interval']==0 :
            list_cost.append(costs/config['interval']/config['batch_size']) #per sequence
            list_loss.append(losses.data[0]/config['interval']/config['batch_size'])
            list_seq_num.append(lengths/1000)  # per thousand
            mean_time = ((get_ms() - start_ms) / config['interval']) / config['batch_size']
            print ("Batch %d th, loss %f, cost %f, Time %.3f ms/sequence." % (batch_num, list_loss[-1], list_cost[-1], mean_time) )
            
            costs = 0
            losses = 0
            start_ms = get_ms()

    saveCheckpoint(model,list_seq_num,list_loss, list_cost, path=config['filename']) 
    return list_seq_num, list_loss, list_cost

def evaluate_lstm(model,criterion,optimizer, test_data_loader, config) : 
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

def train_ntm_model(config, model, criterion, optimizer, train_data_loader) : 
    list_seq_num = []
    list_loss = []
    list_cost = []
    lengthes = 0
    losses = 0
    costs = 0
    for batch_num, X, Y, act  in train_data_loader:
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
        lengthes +=  config['batch_size']
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
            print ("Epoch %d, loss %f, cost %f" % (batch_num, losses/config['interval']/config['batch_size'], costs/config['interval']/config['batch_size']) )
            costs = 0
            losses = 0

    saveCheckpoint(model,list_seq_num,list_loss, list_cost, path=config['filename']) 
    return list_seq_num, list_loss, list_cost


def evaluate_ntm(model,criterion,optimizer, test_data_loader, config) :
    costs = 0
    losses = 0
    lengthes = 0
    for batch_num, X, Y, act in test_data_loader:
        model.init_sequence(config["batch_size"])
        optimizer.zero_grad()
        inp_seq_len = X.size(0)
        outp_seq_len, _, _ = Y.size()
        for i in range(inp_seq_len):
            model(X[i])
        y_out = Variable(torch.zeros(Y.size()))
        for i in range(outp_seq_len):
            y_out[i], _ = model()
        loss = criterion(y_out, Y)
        loss.backward()
        clip_grads(model)
        optimizer.step()
        lengthes+=config['batch_size']
        losses += loss
        out_binarized = y_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))
        costs += cost
    print ("T = %d, Average loss %f, average cost %f" % (Y.size(0), losses.data[0]/lengthes, costs/lengthes))
    return losses.data/lengthes, costs/lengthes

def evaluate_single_batch(net, criterion, X, Y):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = Variable(torch.zeros(Y.size()))
    for i in range(outp_seq_len):
        y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss.data[0],
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result

def report_result(model, criterion, optimizer, list_seq_num, list_loss, list_cost, config_obj, load_checkpoint):
    # pdb.set_trace()
    config = config_obj.config_dict
    model, list_seq_num, list_loss, list_cost = loadCheckpoint(path=config['filename'])
    plt.figure()
    plt.plot(list_seq_num,list_cost)
    plt.xlabel('Sequence number')
    plt.ylabel('Cost per sequence')
    plt.legend()
    plt.savefig('{0}_cost_per_seq.pdf'.format(config['filename']))

def report_average_cost(model, criterion, optimizer, list_seq_num, list_loss, list_cost, config_obj):
    list_avg_loss = []
    list_avg_cost = []
    list_T_num = []
    for T in range(10,110,10) : 
        print("Evaluating {0} model on sequence size {1}".format(config['model_type'], T))
        seqs_loader = utility.load_dataset(config_obj, max=T, min=T)
        if config['model_type'] == 'LSTM':
            avg_loss, avg_cost = evaluate_lstm(model, criterion, optimizer, seqs_loader, config)
        else:
            avg_loss, avg_cost = evaluate_ntm(model, criterion, optimizer, seqs_loader, config)
        list_avg_loss.append(avg_loss)
        list_avg_cost.append(avg_cost)
        list_T_num.append(T)
    saveCheckpoint(model,list_T_num,list_avg_loss, list_avg_cost, path="{0}_Ts".format(config['filename'])) 
    model, list_T_num, list_avg_loss, list_avg_cost = loadCheckpoint(path="{0}_Ts".format(config['filename']))
    plt.plot(list_T_num,list_avg_cost)
    plt.xlabel('T')
    plt.ylabel('average cost')
    plt.savefig('{0}_average_cost.pdf'.format(config['filename']))


def run():
    global config
    args = utility.parse_args()
    config_type = args['configtype']
    config_file = args['configfile']
    load_checkpoint = args['load_checkpoint']
    config_obj = config.Configuration(config_type, config_file)
    config = config_obj.config_dict
    model, criterion, optimizer = models.build_model(config)
    seqs_loader = utility.load_dataset(config_obj)
    if not load_checkpoint:
        print(config_obj.get_config_str())
    if config['model_type'] == "LSTM":
        if not load_checkpoint:
            list_seq_num, list_loss, list_cost = train_lstm_model(config, model, criterion, optimizer, seqs_loader)
            report_result(model, criterion, optimizer, list_seq_num, list_loss, list_cost, config_obj, load_checkpoint)
        else:
            print("Loading checkpoint from: {0}".format(config['filename']))
            report_average_cost(model, criterion, optimizer, 0, 0, 0, config_obj)
    else:
        if not load_checkpoint:
            list_seq_num, list_loss, list_cost = train_ntm_model(config, model, criterion, optimizer, seqs_loader)
            report_result(model, criterion, optimizer, list_seq_num, list_loss, list_cost, config_obj, load_checkpoint)
        else:
            print("Loading checkpoint from: {0}".format(config['filename']))
            report_average_cost(model, criterion, optimizer, 0, 0, 0, config_obj)


def hyperparam_search():
    global config
    model_types = ['MLP_NTM', 'LSTM', 'LSTM_NTM']
    learning_rates = [0.001, 0.0005, 0.00001]
    batch_sizes = [1, 2]
    config_obj = config.Configuration('LSTM', 'config.yaml')
    config = config_obj.config_dict

    for model_type in model_types:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:

                config['model_type'] = model_type
                config['learning_rate'] = learning_rate
                config['batch_size'] = batch_size
                config['filename'] = model_type + '_' + str(learning_rate) + '_' + str(batch_size)

                config_obj.config_dict = config

                model, criterion, optimizer = models.build_model(config)
                seqs_loader = utility.load_dataset(config_obj)

                if config['model_type'] == "LSTM":
                    list_seq_num, list_loss, list_cost = train_lstm_model(config, model, criterion, optimizer,
                                                                          seqs_loader)
                    report_result(model, criterion, optimizer, list_seq_num, list_loss, list_cost, config_obj, False)
                else:
                    list_seq_num, list_loss, list_cost = train_ntm_model(config, model, criterion, optimizer,
                                                                         seqs_loader)
                    report_result(model, criterion, optimizer, list_seq_num, list_loss, list_cost, config_obj, False)


if __name__ == '__main__':
    #pdb.set_trace()
    run()
    # hyperparam_search()
