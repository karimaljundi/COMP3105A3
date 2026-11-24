import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

def minMulDev(X, Y):
    n,d = X.shape
    k = Y.shape[1]
    def loss(w):
        #minimize works with flat vectors only so we turn it into d * k
        W = w.reshape(d,k)

        scores = X @ W

        #first part of the loss: log(sum(exp(scores))) for each row
        first_term = logsumexp

        #second part: y_i is one hot so this just grabs the score of the correct class
        second_term = np.sum(Y * scores, axis=1)

        #average over all n samples
        total = np.sum(first_term - second_term)
        return total / n
    #start with all zeros
    w0 = np.zeros(d * k)
    result = minimize(loss, w0)

    W = result.x.reshape(d,k)
    return W
        