# -*- coding: utf-8 -*-
import numpy            as np
import torch

from   torch.nn     import (Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, ReLU,
                            CrossEntropyLoss,)

from   functional   import *
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


class ModelBNN(ModelConstrained):
    """
    https://arxiv.org/pdf/1602.02830.pdf
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset   = dataset
        inChan         =     1 if dataset == "mnist"    else  3
        outChan        =   100 if dataset == "cifar100" else 10
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
