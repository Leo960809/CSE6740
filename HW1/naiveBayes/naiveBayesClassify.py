import scipy.io as sio
import numpy as np


# Read ecoli.mat
def read_mat(mat_file):
    mat = sio.loadmat(mat_file)
    x_train = mat['xTrain']
    y_train = mat['yTrain']
    x_test = mat['xTest']
    y_test = mat['yTest']

    return x_train, y_train, x_test, y_test



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

    return M, V


# Naive Bayes Classification.
def naiveBayesClassify(xTest, M, V, p):
    data = np.repeat(np.expand_dims(xTest, axis = 2), p.shape[0], axis = 2)
    prob = np.log(1 / np.power(2 * np.pi * V.T, 0.5) * np.exp(-np.power(data - M.T, 2) / (2 * V.T)))

    classify = np.log(np.transpose(p)) + np.sum(prob, axis = 1)
    nb = np.expand_dims(np.argmax(classify, axis = 1) + 1, axis = 1)
    print nb
    return nb


# Evaluate the result.
def evaluation(yTest, nb, total, test_file):
    match = 0 # Number of match.
    match_1 = 0 # Number of match as class 1.
    match_5 = 0 # Number of match as class 5
    np_1 = 0 # Number of data predicted as class 1.
    np_5 = 0 # Number of data predicted as class 5.
    num_1 = 0 # Number of true class 1.
    num_5 = 0 # Number of true class 5.

    for i in range(len(nb)):
        if nb[i] == yTest[i]:
            match += 1
            if nb[i] == total[0]:
                match_1 += 1
            elif nb[i] == total[4]:
                match_5 += 1
            else:
                pass
        else:
            pass

    for i in range(len(nb)):
        if nb[i] == total[0]:
            np_1 += 1
        elif nb[i] == total[4]:
            np_5 += 1
        else:
            pass

    for i in range(len(yTest)):
        if yTest[i] == total[0]:
            num_1 += 1
        elif yTest[i] == total[4]:
            num_5 += 1
        else:
            pass


    with open(test_file, 'w') as f:
        f.write("%f\n" % (float(match) / len(yTest)))
        f.write("%f\n" % (float(match_1) / np_1))
        f.write("%f\n" % (float(match_1) / num_1))
        f.write("%f\n" % (float(match_5) / np_5))
        f.write("%f" % (float(match_5) / num_5))



def main():
    x_train, y_train, x_test, y_test = read_mat('ecoli.mat')
    _, total, _, p = prior(y_train)
    M, V = likelihood(x_train, y_train)
    nb = naiveBayesClassify(x_test, M, V, p)
    evaluation(y_test, nb, total, test_file)

if __name__ == "__main__":
    test_file = 'evaluation.txt'
    main()