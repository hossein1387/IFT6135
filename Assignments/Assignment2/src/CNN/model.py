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
        m.bias.data.fill_(0)
        nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn32  = nn.BatchNorm2d(32)
        self.bn64  = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        pdb.set_trace()
        x = self.pool(F.relu(self.bn64(self.conv1(x))))
        x = self.pool(F.relu(self.bn64(self.conv2(x))))
        x = self.pool(F.relu(self.bn64(self.conv3(x))))
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
    for i, data in enumerate(dataset_loader, 0):
        # pdb.set_trace()
        x, labels = data
        x, labels = Variable(x), Variable(labels)
        y = model(x)
        loss = criterion(y.squeeze(), labels.float())
        prediction = torch.max(y.data, 1)[1]
        accuracy = (prediction.eq(labels.data).sum() / float(labels.size(0))) * 100
    return loss.data[0], accuracy

def record_performance(data, model, criterion, type="none", print_res=False, epoch=0):
    # Record train accuracy
    if type is "train":
        train_loss, train_acc = evaluate(data, model, criterion)
        records["train"][0].append(train_loss)
        records["train"][1].append(train_acc)
        if print_res:
            print("[{0}] train accuracy={1:.3f} loss={2:.3f}".format(epoch, train_acc, train_loss))
        return train_loss, train_acc
    elif type is "test":
        # Record test accuracy
        test_loss, test_acc = evaluate(data, model, criterion)
        records["test"][0].append(test_loss)
        records["test"][1].append(test_acc)
        if print_res:
            print("[{0}] test accuracy={1:.3f} loss={2:.3f}".format(epoch, test_acc, test_loss))
        return test_loss, test_acc
    elif type is "valid":
        # Record valid accuracy
        valid_loss, valid_acc = evaluate(data, model, criterion)
        records["valid"][0].append(valid_loss)
        records["valid"][1].append(valid_acc)
        if print_res:
            print("[{0}] valid accuracy={1:.3f} loss={2:.3f}".format(epoch, valid_acc, valid_loss))
        return valid_loss, valid_acc
    else:
        raise ValueError("unknown data type was passed for performance recording")

def train_model(model, config, criterion, optimizer):
    train_data, valid_data, test_data = utility.load_dataset(config)
    # iterate over batches
    train_loss, train_acc = record_performance(train_data, model, criterion, "train", True, 0)
    for epoch in range(config.num_epochs): 
        running_loss = 0.0
        iter_cnt = 0
        # pdb.set_trace()
        # valid_loss, valid_acc = record_performance(valid_data, model, criterion, "valid", True, epoch)
        # test_loss, test_acc = record_performance(test_data, model, criterion, "test", True, epoch)
        for i, data in enumerate(train_data, 0):
            # get the inputs
            # data = data_
            x, labels = data
            # wrap them in Variable
            x, labels = Variable(x), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # compute model output
            y = model(x)
            # pdb.set_trace()
            # compute loss
            loss = criterion(y.squeeze(), labels.float())
            # compute gradients and update parameters
            loss.backward()
            # take one optimization step toward minimum
            optimizer.step()

        train_loss, train_acc = record_performance(train_data, model, criterion, "train", True, epoch)
        # valid_loss, valid_acc = record_performance(valid_data, model, criterion, "valid", True, epoch)
        # test_loss, test_acc = record_performance(test_data, model, criterion, "test", True, epoch)


if __name__ == '__main__':
    args = utility.parse_args()
    config_type = args['configtype']
    config_file = args['configfile']
    config = config.Configuration(config_type, config_file)
    model, criterion, optimizer = build_model(config)
    train_model(model, config, criterion, optimizer)

