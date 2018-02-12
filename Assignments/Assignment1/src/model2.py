from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from Assignments.Assignment1.src.utility2 import *


# Global Variables
num_epochs = 20
momentum = 0.9


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(100, 20),
        nn.ReLU())

    def forward(self, x):
        output = self.model(x)
        return output

model = MLP()
criterion = nn.CrossEntropyLoss()


def train_model(x_train, x_test, lr0 = 0.05):

    train_loader, valid_loader, test_loader = load_dataset(x_train, x_test)

    optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=momentum)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

def main():
    raw_train = datasets.load_files('data/20news-bydate-train')
    raw_test = datasets.load_files('data/20news-bydate-test')

    for procedure in range(1,3):
        print("pre-process procedure {}: ".format(procedure))
        x_train, x_test = preprocess_dataset(procedure, raw_train, raw_test)
        for lr in [0.1, 0.01, 0.001]:
            print("results with learning rate: {}".format(lr))
            train_model(x_train, x_test, lr)

if __name__== '__main__':
    main()

