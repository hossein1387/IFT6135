from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import scale
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch import from_numpy
from torch.utils.data import DataLoader

batch_size = 100


def preprocess_dataset(procedure, train_x, test_x):

    if procedure == 2:
        tfidf_vectorizer = TfidfVectorizer(decode_error='replace')
        tfidf_x_train = tfidf_vectorizer.fit_transform(train_x.data)
        tfidf_x_test = tfidf_vectorizer.fit_transform(test_x.data)
        return tfidf_x_train, tfidf_x_test

    if procedure == 1 or procedure ==3:
        count_vectorizer = CountVectorizer(decode_error='replace')
        count_x_train = from_numpy(count_vectorizer.fit_transform(train_x.data).toarray())
        count_x_test = from_numpy(count_vectorizer.fit_transform(test_x.data).toarray())

        if procedure == 1:
            return count_x_train, count_x_test
        else:
            normalized_x_train = scale(count_x_train)
            normalized_x_test = scale(count_x_test)
            return normalized_x_train, normalized_x_test


def load_dataset(x_train, x_test):
    num_test = len(x_test)
    indices = list(range(num_test))
    split = np.floor(num_test / 2)

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
        x_train, batch_size=batch_size, shuffle=True, num_workers=1, sampler=test_sampler)

    return train_loader, valid_loader, test_loader
