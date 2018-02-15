import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipdb as pdb
import pickle
import os
import sys
import config
import argparse

def load_dataset(file, config):
    # check if the file exist on disk
    if os.path.exists(file):
        # make sure dataset is a pickled dataset 
        if file.split(".")[-1] != "pkl":
            print ("Can read only a pickled dataset")
            sys.exit()
        with open(file, 'rb') as f:
                mnist = pickle.load(f)
                batch_size = config.batch_size
                print("Loading {0} data set...\n".format(file))
                train_data  = (mnist[0][0].reshape((50000/batch_size, batch_size, 784)),
                              mnist[0][1].reshape((50000/batch_size, batch_size)))
                valid_data  = (mnist[1][0].reshape((10000/batch_size, batch_size, 784)),
                              mnist[1][1].reshape((10000/batch_size, batch_size)))
                test_data   = (mnist[2][0].reshape((10000/batch_size, batch_size, 784)),
                              mnist[2][1].reshape((10000/batch_size, batch_size)))
    else:
        print ("File {0} not found".format(file))
        sys.exit()
    return train_data, valid_data, test_data

def plot_sample_image(dataset, batch_size=9, plot_name="Title"):
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

def plot_sample_data(data, config, plot_name=None, save_image=False):
    print("Plotting sample data {0}".format(plot_name))
    num_records = np.size(np.shape(data))
    fig, ax = plt.subplots()
    for i in range(0, num_records):
        accuracy = data[i][1]
        loss = data[i][0]
        label = data[i][2]
        x = range(0, len(accuracy))
        plt.plot(x, accuracy, label=label)
    plt.legend(loc=4)
    if not save_image:
        if plot_name is not None:
            plt.title(plot_name)
        plt.show()
    else:
        lr0 = config.lr0
        init_type = config.init_type
        batch_size = config.batch_size
        num_epochs = config.num_epochs
        config_str = "lr0={0} initialization={1} batch_size={2} num_epochs={3}".format(lr0, init_type, batch_size, num_epochs)
        pdb.set_trace()
        plt.rc('figure', titlesize=2)
        ax.grid(linestyle='-', linewidth=1)
        ax.grid(True)
        plt.title(config_str)
        plt.savefig('{0}.png'.format(config.filename))   # save the figure to file

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=True)
    parser.add_argument('-t', '--configtype', help='type of test to be done', required=True)
    args = parser.parse_args()
    return vars(args)

