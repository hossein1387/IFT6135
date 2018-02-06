import torch
import torchvision
import torchvision.transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import ipdb as pdb

batch_size = 64
valid_size = 0.1 # 10 percent of train data is used for validation
def load_dataset():
    # Data Preprocessing: define a transformer to be used by data loader

    mnist_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()])

    # Use defined transformer to load MNIST data for test and train
    mnist_train = torchvision.datasets.MNIST(
            root='./data', train=True, 
            transform=mnist_transforms, download=True)
    mnist_test = torchvision.datasets.MNIST(
            root='./data', train=False, 
            transform=mnist_transforms, download=True)

    mnist_valid = torchvision.datasets.MNIST(
            root='./data', train=True, 
            transform=mnist_transforms, download=True)

    num_train = len(mnist_train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # split train set into train and validation set
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(mnist_train, 
                    batch_size=batch_size, sampler=train_sampler, 
                    num_workers=1)

    valid_loader = torch.utils.data.DataLoader(mnist_valid, 
                    batch_size=batch_size, sampler=valid_sampler, 
                    num_workers=1)

    # Load to local variable
#    train_loader = torch.utils.data.DataLoader(
#            mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
            mnist_test, batch_size=batch_size, shuffle=True, num_workers=1)

    sample_loader = torch.utils.data.DataLoader(mnist_train, 
                                                    batch_size=9, 
                                                    shuffle=True, 
                                                    num_workers=1)
    data_iter = iter(sample_loader)
    images, labels = data_iter.next()
    X = images.numpy()
    X = np.transpose(X, [0, 2, 3, 1])
    plot_images(X, labels)

    return train_loader, test_loader

def plot_images(images, cls_true, cls_pred=None):

    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i, :, :, :], interpolation='spline16')
        # get its equivalent class name
        cls_true_name = label_names[cls_true[i]]
            
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
            
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

train_data, valid_data = load_dataset()
print train_data.size()
print valid_data.size()
