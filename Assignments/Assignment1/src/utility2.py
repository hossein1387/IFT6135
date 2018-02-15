import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 100


# train_data = np.loadtxt('data/20news-bydate/matlab/train.data', dtype = int, delimiter= ' ')
# train_label = np.loadtxt('data/20news-bydate/matlab/train.label', dtype = int, delimiter= ' ')
# map = np.loadtxt('data/20news-bydate/matlab/train.map', dtype = int, delimiter= ' ')
#
# test_data = np.loadtxt('data/20news-bydate/matlab/train.data', dtype = int, delimiter= ' ')
# test_label = np.loadtxt('data/20news-bydate/matlab/test.label', dtype = int, delimiter= ' ')

def preprocess_dataset(procedure, train_x, test_x):

    if procedure == 2:
        tfidf_vectorizer = TfidfVectorizer(decode_error='replace')
        tfidf_x_train = tfidf_vectorizer.fit_transform(train_x.data)
        tfidf_x_test = tfidf_vectorizer.fit_transform(test_x.data)
        return to_tensor(tfidf_x_train), to_tensor(tfidf_x_test)

    if procedure == 1 or procedure ==3:
        count_vectorizer = CountVectorizer(decode_error='replace')
        count_x_train = count_vectorizer.fit_transform(train_x.data)
        count_x_test = count_vectorizer.fit_transform(test_x.data)

        if procedure == 1:
            return to_tensor(count_x_train), to_tensor(count_x_test)
        else:
            normalized_x_train = scale(count_x_train)
            normalized_x_test = scale(count_x_test)
            return to_tensor(normalized_x_train), to_tensor(normalized_x_test)  # implement epsilon = 1e-5


def to_tensor(sparse_matrix):
    x = sparse_matrix.toarray()
    return torch.FloatTensor(x)


def load_dataset(x_train, x_test):
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
