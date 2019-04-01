import numpy as np
import random

def myGMM(X, K, maxIter):

    '''
    This is a function that performs GMM clustering
    The input is X: N*d, the input data
                 K: integer, number of clusters
                 maxIter: integer, number of iterations
             
    The output is C: K*d the center of clusters
                  I: N*1 the label of data
                  Loss: [maxIter] likelihood in each
                  step
    '''
    
    # number of vectors in X
    [N, d] = np.shape(X)
    
    # construct indicator matrix (each entry corresponds to the cluster of each point in X)
    I = np.zeros((N, 1))
    
    # construct centers matrix
    C = np.zeros((K, d))
    
    # the list to record error
    Loss = []

    #####################################################################
    # TODO: Implement the EM method for Gaussian mixture model          #
    #####################################################################
    # Randomly generate the starting centers matrix
    for k in range(K):
        C[k, :] = np.mean(X[k * int(N / K):(k + 1) * int(N / K), :], 0)

    cov = np.zeros((K, d, d))
    for k in range(K):
        cov[k, :, :] = np.identity(d)

    # Initialize the weight of each Gaussian distribution
    w = [1.0 / K] * K

    # Initialize the responsibility matrix
    r = np.zeros((N, K))

    # Iteration
    for i in range(maxIter):
        # E-step
        for n in range(N):
            for k in range(K):
                r[n, k] = w[k] * 1 / (2 * np.pi) ** (d / 2) / (np.linalg.det(cov[k, :, :])) ** 0.5 * np.exp(-1 / 2 * \
                            np.dot(np.dot((X[n, :] - C[k, :]), np.linalg.inv(cov[k, :, :])), np.transpose([X[n, :] - C[k, :]])))
            r[n, :] /= np.sum(r[n, :])
            N_k = np.sum(r, axis=0)

        # M-step
        for k in range(K):
            C[k, :] = 1.0 / N_k[k] * np.sum(r[:, k] * X.T, axis=1).T
            cov[k, :, :] = np.array(1 / N_k[k] * np.dot(np.multiply((np.matrix(X - C[k, :])).T, r[:, k]), np.matrix(X - C[k, :])))
            w[k] = 1.0 / N * N_k[k]

        Loss.append(np.sum(np.log(np.sum(r, axis=1))))
    I = np.argmax(r, axis=1)
    #####################################################################
    #                      END OF YOUR CODE                             #
    #####################################################################
    return C, I, Loss