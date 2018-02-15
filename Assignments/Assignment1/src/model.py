import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import ipdb as pdb
import utility
import numpy as np
import sys
import config

# Global Variables
# Records the model's performance
records = {"train": [[], [], "train records"], "test": [[], [], "test records"], "valid": [[],[], "valid records"]}

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if config.init_type== "zero":
            m.weight.data.fill_(0)
            m.bias.data.fill_(0)
        elif config.init_type == "normal":
            m.weight.data.normal_(0, 1)
            m.bias.data.fill_(0)
        elif config.init_type == "glorot":
            f_in  = np.shape(m.weight)[1]
            f_out = np.shape(m.weight)[0]
            glorot_init = np.sqrt(6.0/(f_out+f_in))
            m.weight.data.uniform_(-glorot_init, glorot_init)
            m.bias.data.fill_(0)
        else:
            print ("Unsupported config type".format(config.init_type))
            sys.exit()

# building model
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(784, config.num_neorons_l1),
            nn.ReLU(),
            nn.Linear(config.num_neorons_l1, config.num_neorons_l2),
            nn.ReLU(),
            nn.Linear(config.num_neorons_l2, 10))

    def forward(self, x):
        output = self.model(x)
        return output

def build_model(config):
    model = MLP(config)
    model.apply(weights_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr0)
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

#def init_params(model):
#    conv1Params = list(net.conv1.parameters())

def train_model(model, config, criterion, optimizer):
    losses = 0
    iter = 0
    # record the performance for this epoch
    train_data, valid_data, test_data = utility.load_dataset("mnist.pkl", config)
    record_performance(train_data, model, criterion, "train")
    record_performance(valid_data, model, criterion, "valid")
    for epoch in range(config.num_epochs):
        # iterate over batches
        # the shape of train_data[0] must be 500 x 100 x 784
        # the shape of train_data[1] must be 500 x 100
        for i in range(train_data[0].shape[0]):
            optimizer.zero_grad()
            x = Variable(torch.from_numpy(train_data[0][i]))
            y = Variable(torch.from_numpy(train_data[1][i]))
            optimizer.zero_grad()
            # compute loss
            loss = criterion(model(x), y)
            # compute gradients and update parameters
            loss.backward()
            # take one SGD step
            optimizer.step()

        # record the performance for this epoch
        train_loss, train_acc = record_performance(train_data, model, criterion, "train")
        valid_loss, valid_acc = record_performance(valid_data, model, criterion, "valid")
        # print the results for this epoch
        print("Epoch {0} \nLoss : {1:.3f} \nAcc : {2:.3f}".format(epoch, train_loss, train_acc))
    # print the results for this epoch
    print("Validation Results:\nLoss : {0:.3f} Acc : {1:.3f}".format(valid_loss, valid_acc))
    data_to_plot = (records["train"], records["valid"])
    utility.plot_sample_data(data_to_plot, config, "train vs Valid accuracy", True)


if __name__ == '__main__':
    args = utility.parse_args()
    config_type = args['configtype']
    config_file = args['configfile']
#    pdb.set_trace()
    config = config.Configuration(config_type, config_file)
    model, criterion, optimizer = build_model(config)
    train_model(model, config, criterion, optimizer)


