
# coding: utf-8

# # IFT 6135 A2

# In[94]:

import csv
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.distributions import * 
from os.path import dirname, abspath, join
from torch import norm
import config
import numpy as np
import pickle as pk


# In[95]:

cuda_available = torch.cuda.is_available()
print(cuda_available)


# In[96]:

parent_dir = dirname(dirname(abspath('__file__')))
yaml_file = join(parent_dir, 'config.yaml')
config = config.Configuration('Q1_1', yaml_file)

print(config)


# ## Load Data

# In[97]:

def load_dataset(config):
    mnist_train = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    mnist_test = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    
    test_sampler, valid_sampler = test_valid_split(mnist_test, config)
    
    trainloader = DataLoader(mnist_train, batch_size=config.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(mnist_test, batch_size=64, sampler=test_sampler, num_workers=2)
    validloader = DataLoader(mnist_test, batch_size=64, sampler=valid_sampler, num_workers=2)
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

# In[98]:

class MLPa(nn.Module):
    def __init__(self):
        super(MLPa, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(784, 600),
            nn.ReLU(),
            nn.Linear(600, 200),
            nn.ReLU(),
            nn.Linear(200, 10), 
            nn.Softmax(dim=1))
        
    def forward(self, x):
        output = self.model(x)
        return output


# In[99]:

class MLPb(nn.Module):
    def __init__(self):
        super(MLPb, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(784, 600)
        self.fc2 = nn.Linear(600, 200)
        self.dropout = nn.Dropout(p=0.5) #last layer dropout
        self.fc3 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.dropout(x))
        x = F.softmax(self.fc3(x))
        return x
    


# In[100]:

class CNNa(nn.Module):
    def __init__(self):
        super(CNNa, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
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


# In[101]:

class CNNb(nn.Module):
    def __init__(self):
        super(CNNb, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
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


# In[102]:

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
    optimizer = optim.SGD(model.parameters(), lr=config.lr0, weight_decay = (config.weight_decay/9912422)*config.batch_size)
    return model, criterion, optimizer


# ## Train model

# In[ ]:


def train_model(config, model, criterion, optimizer):
    losses = []
    parameter_norms = []
    train_losses = []
    test_losses = []
    # record the performance for this epoch
    trainloader, testloader, validloader = load_dataset(config)
    
    for epoch in range(config.num_epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            parameter_norm = []
            optimizer.zero_grad()
             
            if config.model_type == 'MLPa' or config.model_type == 'MLPb':
                inputs = Variable(inputs).view(-1,784)
                targets = Variable(targets).view(-1)
            elif config.model_type == 'CNN':
                inputs = Variable(inputs).view(-1,1,28,28)
                targets = Variable(targets).view(-1)
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            # compute loss
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0])
            if config.config_type == 'Q1_1' or config.config_type == 'Q1_2':
                for param in model.parameters():
                    parameter_norm.append(norm(param).data)
            parameter_norms.append(parameter_norm)
        # print the results for this epoch
        test_loss = find_testloss(model, testloader) #change
        test_losses.append(test_loss)
        loss_at_epoch = np.mean(losses)
        train_losses.append(loss_at_epoch)
        print("Epoch {0} \n Train Loss : {1:.3f} \Test Loss : {2:.3f}".format(epoch, loss_at_epoch, test_loss))
    
    test_loss_filename = 'test_loss_' + config.config_type + '.pk'
    train_loss_filename = 'train_loss_' + config.config_type + '.pk'
    pk.dump(test_losses, open(join(parent_dir, test_loss_filename), 'wb'))
    pk.dump(train_losses, open(join(parent_dir, train_loss_filename),'wb'))

    if config.config_type == 'Q1_1' or config.config_type == 'Q1_2':
        params_filename = 'parameters_' + config.config_type + '.pk'
        pk.dump(parameter_norms, open(join(parent_dir, params_filename), 'wb'))
    
    print("Finished training Model {}".format(config.config_type))
    


# In[104]:

def evaluate(model, dataset_loader):
    LOSSES = 0
    COUNTER = 0
    for batch in dataset_loader:
        optimizer.zero_grad()

        x, y = batch
      
        x = Variable(x, volatile=True).view(-1,1,28,28)
        y = Variable(y).view(-1)

        loss = criterion(model(x), y)
        n = y.size(0)
        LOSSES += loss.sum().data.cpu().numpy() * n
        COUNTER += n
    
    return LOSSES / float(COUNTER)


# In[105]:

def find_testloss(model,test_loader):
    model.eval() 
    test_loss_iter = []
    optimizer.zero_grad()
    for data in test_loader:
        inputs, targets = data

        if config.model_type == 'MLPa' or config.model_type == 'MLPb':
            inputs = Variable(inputs, volatile=True).view(-1,784)
            targets = Variable(targets, volatile=True).view(-1)
        elif config.model_type == 'CNN':
            inputs = Variable(inputs, volatile=True).view(-1,1,28,28)
            targets = Variable(targets).view(-1)
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        test_loss_iter.append(test_loss)
    iteration_test_loss = np.mean(test_loss_iter)
    return iteration_test_loss.data[0]


# In[125]:

def dropout_validate(model, N, validloader):
    model.eval()
    mask_probs = np.ones(200)*0.5
    bern = Bernoulli(torch.from_numpy(mask_probs))
    
    correct = 0
    total = 0
    for x, y in validloader:
        
        x = Variable(x, volatile=True).view(-1,784)
        y = Variable(y, volatile=True).view(-1)
        
        x = F.relu(model.fc1(x))
        x = F.relu(model.fc2(x))
        sum_out = 0
        for i in range(N):
            dropout_mask = bern.sample()
            sum_out += F.relu(x*Variable(dropout_mask))
        x = sum_out / N        
        outputs = F.Softmax(model.fc3(x)) 
        
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum()
    
    return correct/total


# In[113]:

model, criterion, optimizer = build_model()
train_model(config, model, criterion, optimizer)


# In[116]:

#torch.save(model.state_dict(), join(parent_dir, 'MLP'))



# In[126]:

trainloader, testloader, validloader = load_dataset(config)

if config.config_type == 'Q1_4' or config.config_type == 'Q1_5':
    dropout_validates = []
    ns = [i*10 for i in range(1,11)]
    for n in ns:
        dv = dropout_validate(model, n, validloader)
        dropout_validates.append(dv)
    dropout_validate_filename = 'dropoutvalidation_' + config.config_type + '.pk'
    pk.dump(dropout_validates, open(join(parent_dir, dropout_validate_filename), 'wb'))


# In[ ]:

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=True)
    parser.add_argument('-r', '--configtype', help='configurations to run', required=True)
    args = parser.parse_args()
    return vars(args)

