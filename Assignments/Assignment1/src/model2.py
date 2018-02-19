import argparse
import csv

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Assignments.Assignment1.src.utility2 import *

# Global Variables
num_epochs = 20
momentum = 0.9
is_cuda = True


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(111140, 100),
            nn.ReLU(),
            nn.Linear(100, 20))

    def forward(self, x):
        output = self.model(x)
        return output

model = MLP()
if is_cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()


def train_model(train, test, procedure, lr0=0.05, batch_size=100):
    train_loader, valid_loader, test_loader = load_dataset(train, test, batch_size)

    optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=momentum)

    train_accuracys = []
    test_accuracys = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            if is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_prediction = torch.max(outputs.data, 1)[1]
            correct += (train_prediction.eq(labels.data).sum())
            total += labels.size(0)
        train_accuracy = correct / float(total) * 100
        test_accuracy = evaluate(test_loader)

        train_accuracys.append(train_accuracy)
        test_accuracys.append(test_accuracys)

        print("epoch:   {}, training accuracy:    {}, test accuracy   {}".format(epoch, train_accuracy, test_accuracy))

    log_name = 'logfile_' + str(procedure) + '_' + str(lr0) + '_' + str(batch_size) + '.csv'
    with open(log_name, 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(train_accuracys)
        wr.writerow(test_accuracys)

    print('Finished Training')


def evaluate(test_loader):
    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data

        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs), labels
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description='Model 2')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    is_cuda = args.cuda

    read_20()

    raw_train = pk.load(open('data/raw_train', 'rb'))
    raw_test = pk.load(open('data/raw_test', 'rb'))

    for procedure in range(1,3):
        print("pre-process procedure {}: ".format(procedure))
        train, test = preprocess_dataset(procedure, raw_train, raw_test)
        for lr in [0.1, 0.05, 0.01]:
            print("results with learning rate: {}:".format(lr))
            train_model(train, test, procedure, lr, 100, )

    print("results with batch size 1: ")
    train, test = preprocess_dataset(2, raw_train, raw_test)
    train_model(train, test, 1, 0.1, 1)


if __name__== '__main__':
    main()

