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

    return X, y



def greedy(X, y, K):
    # Initialize A and beta.
    A = []
    beta = np.zeros((c_number, 1), dtype = float)

    for k in range(K):
        # Get i(k) and store it into A.
        max = abs(np.transpose(X[:, 0]) * (X * beta - y))
        i_number = 0
        for i in range(1, c_number):
            check = abs(np.transpose(X[:, i]) * (X * beta -y))
            if check > max:
                i_number = i
                max = check
            else:
                continue
        A.append(i_number + 1)

        # Calculate beta(k).
        beta = np.linalg.inv(np. transpose(X) * X) * np.transpose(X) * y
        # Set beta(j) to 0 if j does not belong to A(k).
        for j in range(c_number):
            if (j + 1) not in A:
                beta[j] = 0
            else:
                pass

    print "A(k):\n", A
    print "beta(k):\n", beta
    return A, beta


def main():
    X, y = read_data('train-matrix.txt')
    greedy(X, y, 6)

if __name__ == "__main__":
    main()
