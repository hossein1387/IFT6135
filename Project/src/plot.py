import os

import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_fig4(df):
    df = df.astype(float)

    mean = df.mean()
    mean.index = np.arange(1, len(mean) + 1)
    _, ax = plt.subplots()
    mean.plot(ax=ax, label='mean error', marker='o', mfc='none')
    bp = df.plot(kind='box', showfliers=False, ax=ax, color={'medians': 'red'})

    plt.legend()
    plt.xlabel(r'$k_{e}$')
    plt.ylabel('Test error rate')
    plt.title(r'Accuracies with different $k_{e}$')

    plt.show()

    plt.show()


def parseTxt(filename):
    test_errors = []
    with open(os.path.join('logs', filename), 'r') as f:
        for line in f:
            cols = line.split(' ')
            try:
                test_error = cols[6]
                test_errors.append(test_error)
            except IndexError:
                pass
    return test_errors


if __name__ == '__main__':
    files = os.listdir('logs')
    test_errors_dict = dict()
    for filename in filter(lambda fname: fname.endswith('.txt'), files):
        test_errors = parseTxt(filename)[:100]
        test_errors_dict[filename.split('.')[0]] = test_errors
    df = pd.DataFrame(data=test_errors_dict)
    df = df[['2', '4', '8', '10', '12']]
    plot_fig4(df)
