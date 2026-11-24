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
def classify(Xtest, W):
    #scores for every test point, row i: W^T * x_i
    scores = Xtest @ W

    predicted = np.argmax(scores, axis=1)

    m = Xtest.shape[0]
    k = W.shape[1]
    Yhat = np.zeros((m, k))
    Yhat[np.arange(m), predicted] = 1
    return Yhat
def calculateAcc(Yhat, Y):

    #which class each row predicted and which class it actually is
    predicted = np.argmax(Yhat, axis=1)
    actual = np.argmax(Y, axis=1)

    #num of rows that match
    correct = np.sum(predicted == actual)


    #calculate accuracy
    m = Y.shape[0]
    acc = correct / m
    return acc
        