import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Lasso, LinearRegression


def nnls(A, b, alpha=0, x0=None):
    """Non-negative least squares with L1 regularization using Lasso.

    Parameters
    ----------
    A : ndarray
        The input data matrix.
    b : ndarray
        The target values.
    alpha : float
        The regularization parameter.
    x0 : ndarray, optional
        The initial guess for the solution. If None, a zero vector is used.

    Returns
    -------
    x : ndarray
        The solution to the problem.
    """
    # Use a zero vector as the initial guess if not provided
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    
    # Create Lasso model with non-negative constraints and fit to the data
    
    #if alpha is 0, fit with normal equation
    if alpha == 0:
        lasso = LinearRegression(fit_intercept=False, positive=True)
    else:
        lasso = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=100000, tol=1e-7)
    lasso.fit(A, b)
    
    # Extract the solution
    x = lasso.coef_
    
    return x