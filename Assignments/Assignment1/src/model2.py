import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from Assignments.Assignment1.src.utility2 import *

# Global Variables
num_epochs = 20
momentum = 0.9


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(130088, 100),
            nn.ReLU(),
            nn.Linear(100, 20))

    def forward(self, x):
        output = self.model(x)
        return output

model = MLP()
criterion = nn.CrossEntropyLoss()


def train_model(train, test, lr0=0.05, batch_size=100):
    train_loader, valid_loader, test_loader = load_dataset(train, test, batch_size)

    optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=momentum)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        correct = 0
        total = 0
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

            train_prediction = torch.max(outputs.data, 1)[1]
            correct += (train_prediction.eq(labels.data).sum())
            total += labels.size(0)
        train_accuracy = correct / float(total) * 100

        test_accuracy = evaluate(test_loader)

        print("epoch:   {}, training accuracy:    {}, validation accuracy   {}".format(epoch, train_accuracy,
                                                                                       test_accuracy))

    print('Finished Training')


def evaluate(test_loader):
    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs), labels
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100 * correct / total

def main():
    # read_20()

    raw_train = pk.load(open('data/raw_train', 'rb'))
    raw_test = pk.load(open('data/raw_test', 'rb'))

    # for procedure in range(1, 3):
    #     train, test = preprocess_dataset(procedure, raw_train, raw_test)
    #     name_train = 'train_' + str(procedure)
    #     name_test = 'test_' + str(procedure)
    #     with open(name_train, 'wb') as f:
    #         pk.dump(train, f)
    #     with open(name_test, 'wb') as f:
    #         pk.dump(test, f)

    batch_size = 100
    for procedure in range(1,3):
        print("pre-process procedure {}: ".format(procedure))
        # name_train = 'train_' + str(procedure)
        # name_test = 'test_' + str(procedure)
        # train = pk.load(open(name_train, 'rb'))
        # test = pk.load(open(name_test, 'rb'))
        train, test = preprocess_dataset(procedure, raw_train, raw_test)
        for lr in [0.1, 0.05, 0.01]:
            print("results with learning rate: {}:".format(lr))
            train_model(train, test, lr, batch_size)

    print("results with batch size 1: ")
    train, test = preprocess_dataset(2, raw_train, raw_test)
    train_model(train, test, 0.1, 1)


if __name__== '__main__':
    main()

