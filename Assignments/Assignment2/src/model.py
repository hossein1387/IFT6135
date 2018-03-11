
# coding: utf-8

# # IFT 6135 A2

# In[1]:

import csv
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
from os.path import dirname, abspath, join
import config
import numpy as np


# In[2]:

cuda_available = torch.cuda.is_available()
cuda_available


# In[3]:

parent_dir = dirname(dirname(abspath('__file__')))
yaml_file = join(parent_dir, 'config.yaml')
config = config.Configuration('Q1_1', yaml_file)

print(config)


# ## Load Data

# In[4]:

def load_dataset(config):
    mnist_train = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    
    test_sampler, valid_sampler = test_valid_split(mnist_test, config)
    
    trainloader = DataLoader(mnist_train, batch_size=config.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(mnist_test, batch_size=64, sampler=config.batch_size, num_workers=2)
    validloader = DataLoader(mnist_test, batch_size=64, sampler=config.batch_size, num_workers=2)
    return trainloader, testloader, validloader

def test_valid_split(test, config):
    num_test = len(test[0])
    indices = list(range(num_test))
    split = int(np.floor(num_test / 2))

    # split test set into validation and test set
    valid_idx, test_idx = indices[split:], indices[:split]

    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    return test_sampler, valid_sampler


# ## Define model

# In[5]:

class MLPb(nn.Module):
    def __init__(self):
        super(MLPb, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(1792, 600),
            nn.ReLU(),
            nn.Linear(600, 200),
            nn.Dropout(p=0.5), #p is a variable, switch off line
            nn.ReLU(),
            nn.Linear(200, 10), #compare with prediction with no dropout
            nn.Softmax())
        
    def forward(self, x):
        output = self.model(x)
        return output


# In[6]:

class MLPa(nn.Module):
    def __init__(self):
        super(MLPa, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1792, 600),
            nn.ReLU(),
            nn.Linear(600, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Softmax())

        
    def forward(self, x):
        output = self.model(x)
        return output


# In[7]:

class CNNa(nn.Module):
    def __init__(self):
        super(CNNa, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1792, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Logistic Regression
        self.clf = nn.Linear(128, 10)

    def forward(self, x):
        return self.clf(self.conv(x).squeeze())   


# In[8]:

class CNNb(nn.Module):
    def __init__(self):
        super(CNNb, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1792, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Logistic Regression
        self.clf = nn.Linear(128, 10)

    def forward(self, x):
        return self.clf(self.conv(x).squeeze())   


# In[9]:

def build_model():
    if config.model_type == 'MLPa':
        model = MLPa()
    elif config.model_type == 'MLPb':
        model = MLPb()
    elif config.model_type == 'CNN':
        if not config.batch_norm:
            model = CNNa()
        else:
            model = CNNb()

    if torch.cuda.is_available():
        model = model.cuda()
         
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr0, weight_decay = config.weight_decay/config.batch_size)
    return model, criterion, optimizer


# In[10]:

#torch.norm(input, p=2) #change input


# ## Train model

# In[11]:

# USES TENSORBOARDX http://tensorboard-pytorch.readthedocs.io/en/latest/tensorboard.html

def train_model(model, criterion, optimizer):
    losses = []
    writer = SummaryWriter('isaacsultan/IFT6135/Assignments/Assignment2/logs')
    # record the performance for this epoch
    trainloader, testloader, validloader = load_dataset(config)
    
    for epoch in range(config.num_epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            optimizer.zero_grad()
            # compute loss
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0])
            test_loss = find_testloss(model, test_loader)
            writer.add_scalars('learning_curve', {'train_loss':train_loss, 
                                             'test_loss':test_loss}, batch_idx)
        # print the results for this epoch
        print("Epoch {0} \n Train Loss : {1:.3f} \Test Loss : {2:.3f}".format(epoch, np.mean(losses), test_loss))
        


# In[12]:

def find_testloss(model,test_loader):
    model.eval() #import when using dropout
    test_loss_iter = []
    for inputs, targets in test_loader:
        if cuda_available:
             inputs, targets = inputs.torch.cuda(), targets.torch.cuda()
        inputs, targets = Variable(inputs, volatile=true), Variable(targets, volatile=true)
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        test_loss_iter.append(test_loss)
        iteration_test_loss = np.mean(test_loss_iter)
    return iteration_test_loss


# In[13]:

model, criterion, optimizer = build_model()
train_model(model, criterion, optimizer)


# In[ ]:




# In[ ]:



