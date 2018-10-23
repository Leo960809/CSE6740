import scipy.io as sio
import numpy as np


# Read ecoli.mat.
def read_mat(mat_file):
    mat = sio.loadmat(mat_file)
    x_train = mat['xTrain']
    y_train = mat['yTrain']

    return x_train, y_train


# Get the number of classes and the prior probability of each class.
def prior(yTrain):
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

    # Count the amount of samples of each class.
    count_list = [0, 0, 0, 0, 0]
    for i in range(c):
        for j in range(len(yTrain)):
            if yTrain[j] == total[i]:
                count_list[i] += 1
            else:
                pass

    p = np.zeros((c, 1), dtype = float)
    for i in range(c):
        p[i, 0] = float(list.count(total[i])) / len(yTrain)

    return c, total, count_list, p


# Calculate the conditional probability of feature i given class j.
def likelihood (xTrain, yTrain):
    # Get the number of features (m).
    m = len(xTrain[0])
    c, total, count_list, p = prior(yTrain)
    # M is the mean and V is the variance.
    M, V = [], []

    for i in range(c):
        k = 0
        X_label = np.zeros((count_list[i], m))
        for j in range(len(xTrain)):
            if yTrain[j][0] == total[i]:
                X_label[k] = xTrain[j, :]
                k += 1
        M.append(np.mean(X_label, axis = 0).tolist())
        V.append(np.square(np.std(X_label, ddof = 1, axis = 0).tolist()))

    M = np.array(M).reshape(m, c)
    V = np.array(V).reshape(m, c)

    print "M:\n", M, "\nV:\n", V
    return M, V


def main():
    x_train, y_train = read_mat('ecoli.mat')
    likelihood(x_train, y_train)

if __name__ == "__main__":
    main()