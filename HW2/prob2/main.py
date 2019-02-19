import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib 
fontsize = 20
matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize) 

from myGMM import *

def plot(X, C, K):   
    #####################################################################
    # TODO: Implement the plot function                                 #
    #####################################################################
    plt.scatter(X[:, 0], X[:, 1], c=C)
    plt.title('Visualization of K = ' + str(K), fontsize=fontsize)
    plt.savefig('Scatter_of_' + str(K)+ '.png')
    #you may want to use
    #plt.scatter(X[:,0], X[:,1], c=C)
    #plt.title('Visualization of K = '+str(K), fontsize=fontsize)
    #plt.save(...)
    #####################################################################
    #                      END OF YOUR CODE                             #
    #####################################################################
    
    
def plot_losses(Losses, max_iter):
    #####################################################################
    # TODO: Implement the plot function                                 #
    #####################################################################
    plt.figure()
    iter = [i for i in range(max_iter)]
    iter = np.array(iter)
    plt.plot(iter, Losses[0], 'r', label='K = 2')
    plt.plot(iter, Losses[1], 'b', label='K = 3')
    plt.plot(iter, Losses[2], 'g', label='K = 4')
    plt.title('Plot of losses', fontsize=fontsize)
    plt.savefig('Losses.png')
    #you may want to use
    #plt.title('Plot of losses', fontsize=fontsize)
    #plt.save(...)
    #####################################################################
    #                      END OF YOUR CODE                             #
    #####################################################################

if __name__ == "__main__":
    data = scipy.io.loadmat('data/Q2.mat')['X']
    
    #Set parameters
    max_iter = 1000 # choose one that is suitable
    
    Losses = []
    for K in [2, 3, 4]: #You should do K=2, 3, 4
        #Do clustering
        print(K)
        C, I, Loss = myGMM(data, K, max_iter)
        
    #   Plot your result
        plot(data, I, K)
        Losses.append(Loss)
        
    #plot the losses together
    print("Losses[0]", Losses[0])
    print("Losses[1]", Losses[1])
    print("Losses[2]", Losses[2])
    plot_losses(Losses, max_iter)
    plt.show()
