import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import ipdb as pdb
import utils

# Global Variables
num_epochs = 10
lr0 = 0.02
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
            nn.Linear(784, 600),
            nn.ReLU(),
            nn.Linear(600, 100),
            nn.ReLU(),
            nn.Linear(100, 10))

    def forward(self, x):
        output = self.model(x)
        return output

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr0)


def evaluate(data, labels):
    if not isinstance(data, Variable):
       data = Variable(data)
       labels = Variable(labels)
    output = model(data)
    loss = criterion(model(x), y)
    prediction = torch.max(output.data, 1)[1]
    accuracy = (prediction.eq(labels.data).sum() / labels.size(0)) * 100
    return loss.data[0], accuracy

def record_performance():

    # Record train accuracy
    train_loss, train_acc = evaluate(x_train, y_train)
    train_record[0].append(train_loss)
    train_record[1].append(train_acc)

    # Record valid accuracy
    valid_loss, valid_acc = evaluate(x_valid, y_valid)
    valid_record[0].append(valid_loss)
    valid_record[1].append(valid_acc)

    # Record test accuracy
    test_loss, test_acc = evaluate(x_test, y_test)
    test_record[0].append(test_loss)
    test_record[1].append(test_acc)

def train_model():
    losses = 0
    iter = 0
    # record the performance for this epoch
    record_performance()
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            x, y = batch
            x = Variable(x).view(-1,784)
            y = Variable(y).view(-1)

            # compute loss
            loss = criterion(model(x), y)
            # compute gradients and update parameters
            loss.backward()
            # take one SGD step
            optimizer.step()
            
        # record the performance for this epoch
        record_performance()
        # print the results for this epoch
        print("Epoch {0} \nLoss : {1:.3f} \nAcc : {2:.3f}".format(epoch, valid_loss, valid_acc))

#            pdb.set_trace()


if __name__ == '__main__':
    train_model()


