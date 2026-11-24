import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.linalg import eigh
from A3helpers import generateData, augmentX

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
    vals, vecs = eigh(C, subset_by_index=[d-k, d-1])

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
    vals, vecs = eigh(K_centered, subset_by_index=[n-k, n-1])


    #make the biggest eigenvalue first
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    # eigh gives unit length vectors, divide each column by sqrt of its eigenvalue
    alphas = vecs / np.sqrt(vals)


    A = alphas.T

    return A
def projKernelPCA(Xtest, Xtrain, kernel_func, A):
    #float so the kernel values dont overflow

    Xtest = Xtest.astype(float)
    Xtrain = Xtrain.astype(float)

    K_te_tr = kernel_func(Xtrain, Xtrain) # m x n
    K_tr_tr = kernel_func(Xtrain, Xtest) # n x n

    # (1/n) 1 K_tr_tr column means of the TRAIN kernel, 1 x n
    # (1/n) K_te_tr 1 row means of the test-train kernel, m x 1
    # (1/n^2) 1 K_tr_tr 1 mean of the whole TRAIN kernel
    train_col_means = np.mean(K_tr_tr, axis=0, keepdims=True)
    test_row_means = np.mean(K_te_tr, axis=1, keepdims=True)
    train_total_mean = np.mean(K_tr_tr)
    K_centered = K_te_tr - train_col_means - test_row_means + train_total_mean

    #project into the direction of A
    Xproj = K_centered @ A.T
    return Xproj
def synClsExperimentsPCA():
    n_runs = 100
    n_train = 128
    n_test = 1000
    dim_list = [1, 2]
    gen_model_list = [1, 2]
    train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    np.random.seed(56)
    for r in range(n_runs):
        for i, k in enumerate(dim_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = generateData(n=n_test, gen_model=gen_model)
    
                U, mu = PCA(Xtrain, k)
                Xtrain_proj = projPCA(Xtrain, mu, U)
                Xtest_proj = projPCA(Xtest, mu, U)

                Xtrain_proj = augmentX(Xtrain_proj) # add augmentation
                Xtest_proj = augmentX(Xtest_proj)

                W = minMulDev(Xtrain_proj, Ytrain) # from Q1
                Yhat = classify(Xtrain_proj, W) # from Q1
                train_acc[i, j, r] = calculateAcc(Yhat, Ytrain) # from Q1

                Yhat = classify(Xtest_proj, W)
                test_acc[i, j, r] = calculateAcc(Yhat, Ytest)
    train_avg = np.mean(train_acc, axis=2)
    test_avg = np.mean(test_acc, axis=2)

    return train_avg, test_avg
    

        