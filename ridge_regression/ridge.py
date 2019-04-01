import numpy as np


def read_data(data_file):
    # Read the data X and Y from train-matrix.txt.
    f = open(data_file)
    lines = f.readlines()
    global r_number
    r_number = int(lines[0])
    global c_number
    c_number = int(lines[1])


    # Generate a Null matrix for the train matrix.
    N = np.zeros((r_number * 2 + 2, c_number), dtype = float)
    N_row = 0
    for line in lines:
        list = line.strip('\n').split(' ')
        N[N_row:] = list[0:c_number]
        N_row += 1

    # Generate the design matrix.
    X = np.zeros((r_number, c_number), dtype = float)
    for i in range(r_number):
        for j in range(c_number):
            X[i][j] = N[i + 2][j]
    X = np.mat(X)

    # Generate the response vector.
    y = np.zeros((r_number, 1), dtype = float)
    for i in range(r_number):
        y[i][0] = N[2 + r_number + i][0]
    y = np.mat(y)

    index = int(len(X) / 10)

    return X, y, index



# Ridge regression.
def ridge(X, y, Lambda):
    beta = np.dot(np.linalg.inv(np.transpose(X) * X + 2 * r_number * Lambda * np.eye(X.shape[1])), np.transpose(X) * y)
    return beta

def get_train(X, i, index):
    if i == 0:
        V = [index, r_number - 1]
        X_train = X[V, :]
    if i == 9:
        V = [0, index * 9 -1]
        X_train = X[V, :]
    else:
        V = [0, index * i - 1]
        X_1 = X[V, :]
        V = [index * (i + 1), r_number - 1]
        X_2 = X[V, :]
        X_train = np.vstack((X_1, X_2))
    return X_train


def get_test(X, i, index):
    V = [index * i, index * (i + 1) - 1]
    X_test = X[V, :]
    return X_test

# Ten-fold cross-validation.
def ten_fold(X, y, Lambdas):
    index = int(len(X) / 10)

    error = [0, 0, 0, 0, 0]
    best = 0
    min_error = error[0]

    for i in range(len(Lambdas)):
        test_lambda = Lambdas[i]
        for j in range(10):
            temp_beta = ridge(get_train(X, j, index), get_train(y, j, index), test_lambda)
            error[i] += np.square(np.linalg.norm((get_test(y, j, index) - get_test(X, j, index) * temp_beta),
                                                 ord = 2)) / r_number
    for k in range(1, len(Lambdas)):
        if error[k] < min_error:
            min_error = error[k]
            best = k
        else:
            pass

    Lambda = Lambdas[best]
    beta = ridge(X, y, Lambda)
    print beta, "\n", Lambda
    return beta, Lambda

def main():
    X, y, index = read_data("train-matrix.txt")
    opt_lambda = [0.0125, 0.025, 0.05, 0.1, 0.2]
    ten_fold(X, y, opt_lambda)

if __name__ == "__main__":
    main()