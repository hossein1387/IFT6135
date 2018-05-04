import os

import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_fig4(df):
    df = df.astype(float)

    _, ax = plt.subplots()
    df.mean().plot(ax=ax)
    df.boxplot(showfliers=False, ax=ax)

    plt.xlabel(r'$k_{e}$')
    plt.ylabel('Test error rate')
    plt.title(r'Accuracies with different $k_{e}$')

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
    df = df[['6', '8', '15']]
    plot_fig4(df)
