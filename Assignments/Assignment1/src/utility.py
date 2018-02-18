import torch
import numpy as np
import pylab 
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
                subsample_ratio = int(config.subsample_ratio * 50000)
                data_indx = np.random.randint(0, 50000, (subsample_ratio,))
                if config.subsample_ratio != 1:
                    mnist_data = mnist[0][0][data_indx]
                    mnist_label= mnist[0][1][data_indx]
                else:
                    mnist_data = mnist[0][0]
                    mnist_label= mnist[0][1]

                print("Loading {0} data set...\n".format(file))
                train_data  = (mnist_data.reshape((subsample_ratio/batch_size, batch_size, 784)),
                              mnist_label.reshape((subsample_ratio/batch_size, batch_size)))
                valid_data  = (mnist[1][0].reshape((10000/batch_size, batch_size, 784)),
                              mnist[1][1].reshape((10000/batch_size, batch_size)))
                test_data   = (mnist[2][0].reshape((10000/batch_size, batch_size, 784)),
                              mnist[2][1].reshape((10000/batch_size, batch_size)))
    else:
        print ("File {0} not found".format(file))
        sys.exit()
    return train_data, valid_data, test_data

def plot_sample_data(data, config, plot_name=None, save_image=False, plot_loss=False, add_config_str=False):
    print("Plotting sample data {0}".format(plot_name))
    filename = config.filename
    num_records = np.size(np.shape(data))
    pdb.set_trace()

    for i in range(0, num_records):
        if num_records > 1:
            loss      = data[i][0]
            accuracy  = data[i][1]
            plot_func = data[i][2]
        else:
            loss      = data[0]
            accuracy  = data[1]
            plot_func = data[2]
        if plot_loss:
            x = range(0, len(loss))
            pylab.plot(x, loss, label='loss of{0}'.format(plot_func))
        else:
            x = range(0, len(accuracy))
            pylab.plot(x, accuracy, label="accuracy of {0}".format(plot_func))
    pylab.legend(loc='lower right')
    if not save_image:
        if plot_name is not None:
            pylab.title(plot_name)
        pylab.show()
    else:
        lr0 = config.lr0
        init_type = config.init_type
        batch_size = config.batch_size
        num_epochs = config.num_epochs
        config_str = "lr0={0} initialization={1} batch_size={2} num_epochs={3}".format(lr0, init_type, batch_size, num_epochs)
        pylab.grid(True, which="both", ls="-")
        pylab.xlabel("Epoch")
        if add_config_str:
            pylab.title(config_str)
        if plot_loss:
            filename = "{0}_loss".format(filename)
            pylab.ylabel("Training Loss")
            pylab.title(plot_name)
        else:
            filename = "{0}_accuracy".format(filename)
            pylab.ylabel("Training Loss")
            pylab.title(plot_name)
        pylab.savefig('{0}.png'.format(filename))   # save the figure to file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=True)
    parser.add_argument('-t', '--configtype', help='type of test to be done', required=True)
    args = parser.parse_args()
    return vars(args)

