import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import ipdb as pdb
from Assignments.Assignment1.src.utility2 import load_dataset
import numpy as np

# Global Variables
num_epochs = 10
lr0 = 0.005
model_type = 'MLP'
store_every = 1000
# Records the model's performance
train_record = [[],[]]
valid_record = [[],[]]
test_record  = [[],[]]


# building model
class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 10))

    def forward(self, x):
        output = self.model(x)
        return output

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr0)


def evaluate(dataset_loader):
    for batch in dataset_loader:
        data, labels = batch
        if not isinstance(data, Variable):
           data = Variable(data).view(-1,784)
           labels = Variable(labels).view(-1)
        output = model(data)
        loss = criterion(model(data), labels)
        prediction = torch.max(output.data, 1)[1]
        accuracy = (prediction.eq(labels.data).sum() / float(labels.size(0))) * 100
    return loss.data[0], accuracy

def record_performance(dataloader, type="none"):
    # Record train accuracy
    if type is "train":
        train_loss, train_acc = evaluate(dataloader)
        train_record[0].append(train_loss)
        train_record[1].append(train_acc)
        return train_loss, train_acc
    elif type is "test":
        # Record test accuracy
        test_loss, test_acc = evaluate(dataloader)
        test_record[0].append(test_loss)
        test_record[1].append(test_acc)
        return test_loss, test_acc
    elif type is "valid":
        # Record valid accuracy
        valid_loss, valid_acc = evaluate(dataloader)
        valid_record[0].append(valid_loss)
        valid_record[1].append(valid_acc)
        return valid_loss, valid_acc
    else:
        raise ValueError("unknown data type was passed for performance recording")

def train_model():
    losses = 0
    iter = 0
    # record the performance for this epoch
    train_loader, valid_loader, test_loader = load_dataset()
    record_performance(test_loader, "test")
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            x = Variable(x).view(-1,784)
            y = Variable(y).view(-1)
            optimizer.zero_grad()
            # compute loss
            loss = criterion(model(x), y)
            # compute gradients and update parameters
            loss.backward()
            # take one SGD step
            optimizer.step()
        
        # record the performance for this epoch
        train_loss, train_acc = record_performance(test_loader, "test")
        # print the results for this epoch
        print("Epoch {0} \nLoss : {1:.3f} \nAcc : {2:.3f}".format(epoch, train_loss, train_acc))

#            pdb.set_trace()


if __name__ == '__main__':
    train_model()


