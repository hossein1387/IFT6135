import csv
import glob

import matplotlib.pyplot as plt

axis = [i for i in range(1, 21)]
print(axis)
for file in glob.glob('/Users/isaacsultan/IFT6135_A1/Assignments/Assignment1/src/*.csv'):
    data = []
    data.append(axis)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
        train_accuracy = data[0]
        test_accuracy = data[1]
        plt.plot(  # FIX)
            plt.show()
