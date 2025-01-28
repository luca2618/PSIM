import numpy as np
from scipy.linalg import solve


def NLARS(XtX, Xty, inc_path=False, maxK=np.inf):
    """
    Usage:
        beta, path, lambda = NLARS(XtX, Xty, inc_path, maxK)
    
        solves the problem argmin_beta 1/2||y-X*beta||_2^2 s.t. beta>=0
        The problem is solved using a modification of the Least Angle Regression and Selection
        algorithm. As such the entire regularization path for the LASSO problem
        min 1/2||y-X*beta||_2^2 + lambda|beta|_1    s.t. beta>=0
        for all values of lambda is given in path.
    
        Please make reference to the following article describing the algorithm:
    
            Mørup, M and Madsen, K.H. and Hansen, L.K. "Approximate L0
            constrained Non-negative Matrix and Tensor Factorization"   IEEE International Symposium on Circuits and Systems, ISCAS 2008.
            pp. 1328 - 1331, 2008
            
        This code was translated from matlab to python by chatGPT
    
    Input:
        XtX         matrix of size D x D           
        Xty         vector of size D x 1
        inc_path    include the path in the output (default: False)
        maxK        maximal number of active elements in beta (default: inf)
    Output:
        beta    D x 1 solution vector 
        path    the entire solution path
        lambda  regularization value at beginning of step corresponds to
                value of negative gradient
    """

    p = len(Xty)
    k = 0
    if inc_path:
        path = np.zeros((p, 4 * p))
        lambda_ = np.zeros(4 * p)
    else:
        path = None
        lambda_ = None

    # Initialization
    I = list(range(p))  # inactive set
    A = []  # active set
    beta = np.zeros_like(Xty)

    XtXbeta = XtX @ beta
    c = (Xty - XtXbeta)
    C = np.max(c[I])
    j = np.argmax(c[I])
    A.append(I[j])
    I.pop(j)

    # Insert initial zero solution in path.
    k += 1
    if inc_path:
        path[:, k - 1] = beta
        lambda_[k - 1] = C if C.size > 0 else 0

    # NLARS main loop
    while np.sum(c[A]) / len(A) >= 1e-9 and np.count_nonzero(beta) < maxK:
        s = np.ones(len(A))
        w = solve(XtX[np.ix_(A, A)], s)
        XtXw = XtX[:, A] @ w
        
        # Gradient in inactive set of same value as of active set
        gamma1 = (C - c[I]) / (XtXw[A[0]] - XtXw[I])
        
        # Check step length towards zero for active elements
        gamma2 = -beta[A] / w
        
        # Check step to zero gradient for active elements
        gamma3 = np.array([c[A[0]] / XtXw[A[0]]])
        
        # Concatenate the three step types
        # gamma = np.concatenate([gamma1, gamma2, gamma3])
        # concatenate the three step types, they have different shapes so they need to be concatenated differently
        gamma = np.zeros(len(gamma1) + len(gamma2) + len(gamma3))
        gamma[:len(gamma1)] = gamma1
        gamma[len(gamma1):len(gamma1) + len(gamma2)] = gamma2
        gamma[len(gamma1) + len(gamma2):] = gamma3
        
        # Make negative and machine impression steps irrelevant
        gamma[gamma <= 1e-9] = np.inf
        
        # Find smallest step
        mu = np.min(gamma)
        t = np.argmin(gamma)
        
        # Update beta                       
        beta[A] = beta[A] + mu * w
        
        #enforce non-negativity
        beta[beta < 0] = 0

        # Check whether an active element has reached zero
        if len(gamma1) <= t < len(gamma1) + len(gamma2):
            lassocond = 1
            j = t - len(gamma1)
            I.append(A[j])
            A.pop(j)
        else:
            lassocond = 0
        
        # Recalculate gradient
        # print('recalc')
        XtXbeta = XtX @ beta
        c = (Xty - XtXbeta)
        
        if len(I) == 0:
            break
        
        C = np.max(c[I])
        j = np.argmax(c[I])
        
        # Update path
        k += 1
        if inc_path:
            path[:, k - 1] = beta
            lambda_[k - 1] = C if C.size > 0 else 0
        
        if not lassocond:
            A.append(I[j])
            I.pop(j)
    
    # Trim path
    if inc_path:
        path = path[:, :k]
        lambda_ = lambda_[:k]

    return beta, path, lambda_

def calc_scoring(X, H, inc_path=False, maxK=np.inf):
    paths, lambdas = [], []
    W = np.zeros((X.shape[0], H.shape[0]))
    #print(W.shape)
    for i in range(X.shape[0]):
        # print(i)
        XtX = H @ H.T
        Xty = H @ X[i,:].T
        # Xty = Xty.reshape(len(Xty))
        
        # print(XtX, Xty)
        
        W[i,:], path, lambda_ = NLARS(XtX, Xty, inc_path=inc_path, maxK=maxK)
        paths.append(path)
        lambdas.append(lambda_)
    
    return W, paths, lambdas

def get_optimal_W(X, H_hyb, threshold):
    Ws = {}
    Loss = {}
    optimal_W = None
    W_optim, path, lambdas = calc_scoring(X, H_hyb, inc_path=True)
    X = W_optim @ H_hyb
    
    for l in range(H_hyb.shape[0]):
        Ws[l] = np.zeros((X.shape[0], H_hyb.shape[0]))
        for i in range(X.shape[0]):
            # Ws[l][i,:], _, _ = NLARS(H_hyb @ H_hyb.T, H_hyb @ X[i,:].T, maxK=l+1)
            Ws[l][i,:], _, _ = NLARS(H_hyb @ H_hyb.T, H_hyb @ X[i,:].T, inc_path=False, maxK=l+1)
        #calculate the frobenius loss
        Loss[l] = np.linalg.norm(X - Ws[l] @ H_hyb, 'fro')/np.linalg.norm(X, 'fro')
    
    #get the path
    
    #set optimal W to the last W
    optimal_W = Ws[l]
    #find W with the highest loss still below the threshold
    for l in range(H_hyb.shape[0]):
        if Loss[l] <= threshold:
            optimal_W = Ws[l]
            break
    return optimal_W, path, lambdas