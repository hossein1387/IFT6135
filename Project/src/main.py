import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import ipdb as pdb
import utility
import config
import models 

def build_model(config):
    model_type = config['model_type']
    if model_type =='cnn':
        model = models.CNN(config)
    elif model_type =='bnn':
        model = models.BNN(config)
    elif model_type =='wage':
        model = models.WAGE(config)
    else:
        print("model_type={0} is not supported yet!".fortmat(model_type))
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    return model, criterion, optimizer

def test_model(test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


def train_model(model, criterion, optimizer, train_loader, train_dataset, test_loader, config):
    # Train the Model
    num_epochs = config['num_epochs']
    batch_size = config['batchsize']
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # if (i+1) % 100 == 0:
            #     print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
            #            %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
        test_model(test_loader)


if __name__ == '__main__':
    args = utility.parse_args()
    model_type = args['modelype']
    config_file = args['configfile']
    config = config.Configuration(model_type, config_file)
    print(config.get_config_str())
    config = config.config_dict
    model, criterion, optimizer = build_model(config)
    train_loader, test_loader, train_dataset, test_dataset = utility.load_dataset(config)
    train_model(model, criterion, optimizer, train_loader, train_dataset, test_loader, config)
    # test_model(test_loader)
    # Save the Trained Model
    torch.save(model.state_dict(), 'trained_model.pkl')
