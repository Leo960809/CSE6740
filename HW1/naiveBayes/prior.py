import scipy.io as sio
import numpy as np


# Read ecoli.mat.
def read_mat(mat_file):
    mat = sio.loadmat(mat_file)
    y_train = mat['yTrain']

    return y_train


def prior(yTrain):
    # Count the number of classes.
    list = np.ndarray.tolist(yTrain)
    total = []
    total.append(list[0])
    c = 1
    for i in range(1, len(list)):
        if list[i] not in total:
            total.append(list[i])
            c += 1
        else:
            pass

    # Print the prior probability of each class.
    p = np.zeros((c, 1), dtype = float)
    for i in range(c):
        p[i, 0] = float(list.count(total[i])) / len(yTrain)
    print p
    return p


def main():
    y_train = read_mat('ecoli.mat')
    prior(y_train)

if __name__ == "__main__":
    main()