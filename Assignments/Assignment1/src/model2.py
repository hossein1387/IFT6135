import csv

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import utility2 

# Global Variables
num_epochs = 1
momentum = 0.9
is_cuda = False


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

        for i, data in enumerate(train_loader, 0):

            if i % 250 == 0:
                correct = 0
                total = 0

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

            if (i % 250 == 0):
                train_accuracy = correct / float(total) * 100
                train_accuracys.append(train_accuracy)

                print("iteration:   {}, training accuracy:    {}".format(i, train_accuracy))

    log_name = 'logfile_' + str(procedure) + '_' + str(lr0) + '_' + str(batch_size) + '.csv'
    with open(log_name, 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(train_accuracys)

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
    # read_20()

    raw_train = utility2.pk.load(open('data/raw_train', 'rb'))
    raw_test = utility2.pk.load(open('data/raw_test', 'rb'))

    # for procedure in range(1,3):
    #     print("pre-process procedure {}: ".format(procedure))
    #     train, test = preprocess_dataset(procedure, raw_train, raw_test)
    #     for lr in [0.1, 0.05, 0.01]:
    #         print("results with learning rate: {}:".format(lr))
    #         train_model(train, test, procedure, lr, 100, )

    print("results with batch size 1: ")
    train, test = preprocess_dataset(2, raw_train, raw_test)
    train_model(train, test, 1, 0.1, 1)
    train_model(train, test, 1, 0.1, 100)


if __name__== '__main__':
    main()

