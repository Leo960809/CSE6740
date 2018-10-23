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


def read_test_data(test_file):
    # Read the data X and Y from test-matrix.txt.
    f1 = open(test_file)
    lines = f1.readlines()

    global test_row
    test_row = int(lines[0])
    global test_col
    test_col = int(lines[1])


    # Generate a Null matrix for the test matrix.
    N_test = np.zeros((test_row * 2 + 2, test_col), dtype = float)
    N_test_row = 0
    for line in lines:
        list = line.strip('\n').split(' ')
        N_test[N_test_row:] = list[0:test_row]
        N_test_row += 1

    # Generate the design matrix.
    X_test_mat = np.zeros((test_row, test_col), dtype = float)
    for i in range(test_row):
        for j in range(test_col):
            X_test_mat[i][j] = N_test[i + 2][j]
    X_test_mat = np.mat(X_test_mat)

    # Generate the response vector.
    y_test_mat = np.zeros((test_row, 1), dtype = float)
    for i in range(test_row):
        y_test_mat[i][0] = N_test[2 + test_row + i][0]
    y_test_mat = np.mat(y_test_mat)

    return X_test_mat, y_test_mat, test_row, test_col


def read_true_beta(true_beta_file):
    # Read the data from true-beta.txt.
    f2 = open(true_beta_file)
    b_lines = f2.readlines()
    beta_number = int(b_lines[0])
    true_beta = np.zeros((beta_number, 1), dtype = float)
    b_row = -1
    for line in b_lines:
        list = line.strip('\n').split(' ')
        true_beta[b_row:] = list[0:1]
        b_row += 1
    true_beta = np.mat(true_beta)

    return true_beta



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
    return beta, Lambda


# Calculate the prediction error.
def cal_error(X, y, beta, number):
    pred_error = np.square(np.linalg.norm(y - X * beta)) / number
    print pred_error
    return pred_error

# Calculate the bias between the predicted beta and true beta.
def cal_bias(test_beta, true_beta):
    bias = np.square(np.linalg.norm((test_beta - true_beta)))
    print bias
    return bias


def main():
    X, y, index = read_data('train-matrix.txt')
    X_test_mat, y_test_mat, test_row, test_col = read_test_data('test-matrix.txt')
    true_beta = read_true_beta('true-beta.txt')
    opt_lambda = [0.0125, 0.025, 0.05, 0.1, 0.2]
    test_beta, Lambda = ten_fold(X, y, opt_lambda)
    cal_error(X_test_mat, y_test_mat, true_beta, test_row)
    cal_bias(test_beta, true_beta)


if __name__ == "__main__":
    main()