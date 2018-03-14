import torchvision
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import ipdb as pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utility
import sys
import config

# Global Variables
# Records the model's performance
records = {"train": [[], [], "train records"], "test": [[], [], "test records"], "valid": [[],[], "valid records"]}

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("Linear") != -1: 
        # pdb.set_trace()
        # f_in  = np.shape(m.weight)[1]
        # f_out = np.shape(m.weight)[0]
        # glorot_init = np.sqrt(6.0/(f_out+f_in))
        # m.weight.data.uniform_(-glorot_init, glorot_init)
        m.bias.data.fill_(0)
        nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def build_model(config):
    model = CNN()
    # model.apply(weights_init)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model.parameters(),  lr=config.lr0, momentum=config.momentum)
    optimizer = optim.Adam(model.parameters(),  lr=0.001)
    return model, criterion, optimizer

def evaluate(dataset_loader, model, criterion):
    for i in range(dataset_loader[0].shape[0]):
        data   = torch.from_numpy(dataset_loader[0][i])
        labels = torch.from_numpy(dataset_loader[1][i])
        if not isinstance(data, Variable):
           data = Variable(data).view(-1,784)
           labels = Variable(labels).view(-1)
        output = model(data)
        loss = criterion(model(data), labels)
        prediction = torch.max(output.data, 1)[1]
        accuracy = (prediction.eq(labels.data).sum() / float(labels.size(0))) * 100
    return loss.data[0], accuracy

def record_performance(data, model, criterion, type="none"):
    # Record train accuracy
    if type is "train":
        train_loss, train_acc = evaluate(data, model, criterion)
        records["train"][0].append(train_loss)
        records["train"][1].append(train_acc)
        return train_loss, train_acc
    elif type is "test":
        # Record test accuracy
        test_loss, test_acc = evaluate(data, model, criterion)
        records["test"][0].append(test_loss)
        records["test"][1].append(test_acc)
        return test_loss, test_acc
    elif type is "valid":
        # Record valid accuracy
        valid_loss, valid_acc = evaluate(data, model, criterion)
        records["valid"][0].append(valid_loss)
        records["valid"][1].append(valid_acc)
        return valid_loss, valid_acc
    else:
        raise ValueError("unknown data type was passed for performance recording")

def train_model(model, config, criterion, optimizer):
    train_data, valid_data, test_data = utility.load_dataset(config)
    # iterate over batches
    for epoch in range(config.num_epochs): 
        running_loss = 0.0
        data_ = iter(train_data).next()
        for i, data in enumerate(train_data, 0):
            # get the inputs
            # data = data_
            x, labels = data
            # wrap them in Variable
            x, labels = Variable(x), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y = model(x)
            # pdb.set_trace()
            # compute loss
            loss = criterion(y.squeeze(), labels.float())
            # compute gradients and update parameters
            loss.backward()
            optimizer.step()
            print loss.data.cpu().numpy()
            # print statistics
            # running_loss += loss.data[0]
            # if i % 2 == 0:
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

if __name__ == '__main__':
    args = utility.parse_args()
    config_type = args['configtype']
    config_file = args['configfile']
    config = config.Configuration(config_type, config_file)
    model, criterion, optimizer = build_model(config)
    train_model(model, config, criterion, optimizer)

