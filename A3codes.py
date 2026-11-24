import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import eig

def minMulDev(X, Y):
    n,d = X.shape
    k = Y.shape[1]
    def loss(w):
        #minimize works with flat vectors only so we turn it into d * k
        W = w.reshape(d,k)

        scores = X @ W

        #first part of the loss: log(sum(exp(scores))) for each row
        first_term = logsumexp(scores, axis=1)

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
def PCA(X, k):
    n,d = X.shape

    #mean of each feature, kept as a d x 1 column
    mu = np.mean(X, axis=0)
    mu = mu.reshape(d,1)

    #subtract mean from every row
    Xc = X - mu.T

    #the d x d matrix we want the eigenvectors of
    C = Xc.T @ Xc

    #eig sorts eigenvalues smallest to biggest, so the top k are the last k
    #subset_by_index means we only those ones
    vals, vecs = eig(C, subset_by_index=[d-k, d-1])

    #eigenvectors come back as columns in ascending order
    #we flip so the biggest is first
    vecs = vecs[:, ::-1]
    U = vecs.T
    return U, mu
def projPCA(Xtest, mu, U):

    #mu is d x 1, so mu.T is a 1 x d row
    #subtract that row from every row of Xtest
    Xc = Xtest - mu.T

    #project each centered point onto the k directions in U

    Xproj = Xc @ U.T
    return Xproj
def kernelPCA(X, k, kernel_func):
    #make sure kernel values dont overflow
    X = X.astype(float)
    n = X.shape[0]

    K = kernel_func(X, X)

    # (1/n) 1 K mean of each column, same for every row
    # (1/n) K 1 mean of each row, same for every column
    # (1/n^2) 1 K 1 mean of the whole matrix
    col_means = np.mean(K, axis=0, keepdims=True)
    row_means = np.mean(K, axis=1, keepdims=True)
    total_mean = np.mean(K)
    K_centered = K - col_means - row_means + total_mean

    #top k eigenvectors
    vals, vecs = eig(K_centered, subset_by_index=[n-k, n-1])


    #make the biggest eigenvalue first
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    # eigh gives unit length vectors, divide each column by sqrt of its eigenvalue
    alphas = vecs / np.sqrt(vals)


    A = alphas.T

    return A

        