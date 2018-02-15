import pickle as pk

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# train_data = np.loadtxt('data/20news-bydate/matlab/train.data', dtype = int, delimiter= ' ')
# train_label = np.loadtxt('data/20news-bydate/matlab/train.label', dtype = int, delimiter= ' ')
# map = np.loadtxt('data/20news-bydate/matlab/train.map', dtype = int, delimiter= ' ')
#
# test_data = np.loadtxt('data/20news-bydate/matlab/train.data', dtype = int, delimiter= ' ')
# test_label = np.loadtxt('data/20news-bydate/matlab/test.label', dtype = int, delimiter= ' ')


def read_20():
    raw_train = datasets.load_files('data/20news-bydate-train')
    raw_test = datasets.load_files('data/20news-bydate-test')

    pk.dump(raw_train, open('data/raw_train', 'wb'))
    pk.dump(raw_test, open('data/raw_test', 'wb'))


def preprocess_dataset(procedure, train, test):
    train_y = train.target
    test_y = test.target

    if procedure == 2:
        tfidf_vectorizer = TfidfVectorizer(decode_error='replace')
        tfidf_x_train = tfidf_vectorizer.fit_transform(train.data)
        tfidf_x_test = tfidf_vectorizer.fit_transform(test.data)
        return to_tensor(tfidf_x_train), to_tensor(tfidf_x_test)

    if procedure == 1 or procedure ==3:
        count_vectorizer = CountVectorizer(decode_error='replace')
        count_x_train = count_vectorizer.fit_transform(train.data)
        count_x_test = count_vectorizer.fit_transform(test.data)

        if procedure == 1:
            return to_tensor(count_x_train, train_y), to_tensor(count_x_test, test_y)
        else:
            normalized_x_train = scale(count_x_train)
            normalized_x_test = scale(count_x_test)
            return to_tensor(normalized_x_train), to_tensor(normalized_x_test)  # implement epsilon = 1e-5


def to_tensor(sparse_matrix, labels):
    x = sparse_matrix.toarray()

    dataset = data_utils.TensorDataset(torch.FloatTensor(x), torch.LongTensor(labels))
    return dataset


def load_dataset(x_train, x_test, batch_size):

    num_test = len(x_test)
    indices = list(range(num_test))
    split = int(np.floor(num_test / 2))


    # split test set into validation and test set
    valid_idx, test_idx = indices[split:], indices[:split]

    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(x_train,
                              batch_size=batch_size, shuffle=True,
                              num_workers=1)
    valid_loader = DataLoader(x_test,
                              batch_size=batch_size, sampler=valid_sampler,
                              num_workers=1)
    test_loader = DataLoader(
        x_train, batch_size=batch_size, num_workers=1, sampler=test_sampler)

    return train_loader, valid_loader, test_loader
