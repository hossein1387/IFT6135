# -*- coding: utf-8 -*-
import numpy            as np
from   torch.nn     import (Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, ReLU,
                            CrossEntropyLoss)
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from Quantize import *
from   layers       import *


#
# For the sake of ease of use, all models below inherit from this class, which
# exposes a constrain() method that allows re-applying the module's constraints
# to its parameters.
#
class ModelConstrained(torch.nn.Module):
    def constrain(self):
        def fn(module):
            if module is not self and hasattr(module, "constrain"):
                module.constrain()
        
        self.apply(fn)


class BNN(ModelConstrained):
    """
    https://arxiv.org/pdf/1602.02830.pdf
    """
    def __init__(self, config):
        super().__init__()
        self.dataset   = config['dataset']
        inChan         =     1 if self.dataset == "mnist"    else  3
        outChan        =   100 if self.dataset == "cifar100" else 10
        epsilon        = 1e-4   # Some epsilon
        alpha          = 1-0.9  # Exponential moving average factor for BN.
        
        self.conv1     = Conv2dBNN  (inChan, 128, (3,3), padding=1, H=1, W_LR_scale="Glorot")
        self.bn1       = BatchNorm2d( 128, epsilon, alpha)
        self.tanh1     = SignBNN    ()
        self.conv2     = Conv2dBNN  ( 128,  128, (3,3), padding=1, H=1, W_LR_scale="Glorot")
        self.maxpool2  = MaxPool2d  ((2,2), stride=(2,2))
        self.bn2       = BatchNorm2d( 128, epsilon, alpha)
        self.tanh2     = SignBNN    ()
        
        self.conv3     = Conv2dBNN  ( 128,  256, (3,3), padding=1, H=1, W_LR_scale="Glorot")
        self.bn3       = BatchNorm2d( 256, epsilon, alpha)
        self.tanh3     = SignBNN    ()
        self.conv4     = Conv2dBNN  ( 256,  256, (3,3), padding=1, H=1, W_LR_scale="Glorot")
        self.maxpool4  = MaxPool2d  ((2,2), stride=(2,2))
        self.bn4       = BatchNorm2d( 256, epsilon, alpha)
        self.tanh4     = SignBNN    ()
        
        self.conv5     = Conv2dBNN  ( 256,  512, (3,3), padding=1, H=1, W_LR_scale="Glorot")
        self.bn5       = BatchNorm2d( 512, epsilon, alpha)
        self.tanh5     = SignBNN    ()
        self.conv6     = Conv2dBNN  ( 512,  512, (3,3), padding=1, H=1, W_LR_scale="Glorot")
        self.maxpool6  = MaxPool2d  ((2,2), stride=(2,2))
        self.bn6       = BatchNorm2d( 512, epsilon, alpha)
        self.tanh6     = SignBNN    ()
        
        self.linear7   = LinearBNN  (25088, 1024, H=1, W_LR_scale="Glorot")
        self.tanh7     = SignBNN    ()
        self.linear8   = LinearBNN  (1024, 1024, H=1, W_LR_scale="Glorot")
        self.tanh8     = SignBNN    ()
        self.linear9   = LinearBNN  (1024,  outChan, H=1, W_LR_scale="Glorot")
    
    
    def forward(self, X):
        shape = (-1, 1, 28, 28) if self.dataset == "mnist" else (-1, 3, 32, 32)
        v = X.view(*shape)
        
        v = v*2-1
        
        v = self.conv1   (v)
        v = self.bn1     (v)
        v = self.tanh1   (v)
        v = self.conv2   (v)
        v = self.maxpool2(v)
        v = self.bn2     (v)
        v = self.tanh2   (v)
        

        # v = self.conv3   (v)
        # v = self.bn3     (v)
        # v = self.tanh3   (v)
        # v = self.conv4   (v)
        # v = self.maxpool4(v)
        # v = self.bn4     (v)
        # v = self.tanh4   (v)
        
        # v = self.conv5   (v)
        # v = self.bn5     (v)
        # v = self.tanh5   (v)
        # v = self.conv6   (v)
        # v = self.maxpool6(v)
        # v = self.bn6     (v)
        # v = self.tanh6   (v)
        
        v = v.view(v.size(0), -1)
        
        v = self.linear7 (v)
        v = self.tanh7   (v)
        v = self.linear8 (v)
        v = self.tanh8   (v)
        v = self.linear9 (v)
        
        return v
    
    def loss(self, Ypred, Y):
        onehotY   = torch.zeros_like(Ypred).scatter_(1, Y.unsqueeze(1), 1)*2 - 1
        hingeLoss = torch.mean(torch.clamp(1.0 - Ypred*onehotY, min=0)**2)
        return hingeLoss


class CNN(ModelConstrained):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc = nn.Linear(7*7*32, 10)
        self.batchnorm16 = nn.BatchNorm2d(16)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.maxpool2d = nn.MaxPool2d(2)

    def forward(self, x):
        # import ipdb as pdb; pdb.set_trace()
        layer1_out = self.maxpool2d(self._activation(self.batchnorm16(self.cnn1(x))))
        layer2_out = self.maxpool2d(self._activation(self.batchnorm32(self.cnn2(layer1_out))))
        out = layer2_out.view(layer2_out.size(0), -1)
        out = self.fc(out)
        return out
    def _activation(self, x):
        x = F.relu(x)
        # print(x)
        return x

# WAGE Model (2 conv layer)
class WAGE(ModelConstrained):
    def __init__(self, config):
        super().__init__()
        self.dataset   = config['dataset']
        inChan         =     1 if self.dataset == "mnist"    else  3
        outChan        =   100 if self.dataset == "cifar100" else 10
        epsilon        =     1e-4   # Some epsilon
        alpha          =     1-0.9  # Exponential moving average factor for BN.
        
        self.conv1     = Conv2dWAGE(inChan, 16, (5,5), config=config, padding=1, H=1, W_LR_scale="Glorot")
        self.conv2     = Conv2dWAGE(16, 32, (5,5), config=config, padding=1, H=1, W_LR_scale="Glorot")
        self.fc1       = LinearWAGE(18432, 1024, config=config)
        self.fc2       = nn.Linear(18432, 10)
    
    def forward(self, x):
        # import ipdb as pdb; pdb.set_trace()
        layer1_out = self.conv1(x)
        layer2_out = self.conv2(layer1_out)
        out = layer2_out.view(layer2_out.size(0), -1)
        #out = self.fc1(out)
        out = self.fc2(out)
        return out

    def _activation(self, x):
       x = F.relu(x)
       x = self._QE(x)
       x = self._QA(x)
       return x

    def loss(self, Ypred, Y):
        onehotY   = torch.zeros_like(Ypred).scatter_(1, Y.unsqueeze(1), 1)*2 - 1
        hingeLoss = torch.mean(torch.clamp(1.0 - Ypred*onehotY, min=0)**2)
        return hingeLoss
