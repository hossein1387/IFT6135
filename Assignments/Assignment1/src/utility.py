import torch
import torchvision
import torchvision.transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import ipdb as pdb

batch_size = 100
valid_size = 0.1 # 10 percent of train data is used for validation
def load_dataset():
    print "Loading dataset..."
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

    # Load to local variable
    train_loader = torch.utils.data.DataLoader(mnist_train, 
                    batch_size=batch_size, sampler=train_sampler, 
                    num_workers=1)
    valid_loader = torch.utils.data.DataLoader(mnist_valid, 
                    batch_size=batch_size, sampler=valid_sampler, 
                    num_workers=1)
    test_loader = torch.utils.data.DataLoader(
            mnist_test, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, valid_loader, test_loader

def plot_sample_data(dataset, batch_size=9, plot_name="Title"):
    print "Plotting sample data"
    sample_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=9, 
                                                shuffle=True, 
                                                num_workers=1)
    data_iter = iter(sample_loader)
    images, labels = data_iter.next()
    X = images.numpy()
    X = np.transpose(X, [0, 2, 3, 1])
    plot_images(X, labels, plot_name)


# plot only a batch of 9 images in a 3 by 3 plot
def plot_images(images, labels, plot_name="Title"):

    assert len(images) == len(labels) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)
    if plot_name is not "Title":
        plt.suptitle(plot_name)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i, :, :, 0], cmap='gray')
        xlabel = "Pred: {0}".format(labels[i])
        
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
