# import matplotlib.pyplot as plt

import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from aio import EncapsulatedNTM


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config

        self.lstm=nn.LSTM(self.config['data_width']+1, self.config['num_hidden'])
        self.mlp=nn.Linear(self.config['num_hidden'], self.config['data_width'])

    def init_hidden(self, batch_size):
        num_hidden = self.config['num_hidden']
        self.hidden= ( autograd.Variable(torch.randn(1, batch_size, num_hidden)),\
                  autograd.Variable(torch.randn((1, batch_size, num_hidden))) )
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


def build_model(config):
    if config['model_type'] == 'LSTM':
        model = LSTM(config)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(),  lr=config['learning_rate'])
    elif config['model_type'] == 'LSTM_NTM':
        model = EncapsulatedNTM(num_inputs=config['data_width']+1, num_outputs=config['data_width']+1,\
                                controller_size=100, controller_layers=1, num_heads=1, N=128, M=20, \
                                controller_type ='lstm')
        criterion = nn.BCELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'], momentum = config['momentum'])
    elif config['model_type'] == 'MLP_NTM':
        model = EncapsulatedNTM(num_inputs=config['data_width']+1, num_outputs=config['data_width'],\
                                controller_size=100, controller_layers=1, num_heads=1, N=128, M=20, \
                                controller_type ='mlp')
        criterion = nn.BCELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'], momentum = config['momentum'])
    else:
        print ("Config type {0} is unknown".format(config['model_type']))
        sys.exit()
    return model, criterion, optimizer
