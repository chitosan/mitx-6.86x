"""Mixture model using EM"""
import math
from typing import Tuple
import numpy as np
from common import GaussianMixture
import common



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    '''print(X.shape)
    i, d = X.shape
    print("X  shape i,d: ", i, d)
    K, d =mixture.mu.shape
    print("mu shape K,d: ", K, d)
    N = np.zeros([i,K])
    numerator = np.zeros([i, K])
    for j in range(K):
        for i_ in range(i):
            ePower = -(np.linalg.norm(X[i_]-mixture.mu[j])**2)/(2*mixture.var[j])
            eCoef = (2*math.pi*mixture.var[j])**(d/2)
            N[i_][j] = eCoef*math.exp(ePower)
            numerator[i_][j] = mixture.p[j] * N[i_][j]
    denominator = np.dot(N,mixture.p).reshape(-1,1)
    post = numerator/denominator

    log_lh = d'''
    #### --------------------------------------------------------
    n, d = X.shape
    mu, var, pi = mixture  # Unpack mixture tuple
    K = mu.shape[0]
    # Compute normal dist. matrix: (N, K)
    pre_exp = (2 * np.pi * var) ** (d / 2)
    # Calc exponent term: norm matrix/(2*variance)
    post = np.linalg.norm(X[:, None] - mu, ord=2, axis=2) ** 2  # Vector version
    post = np.exp(-post / (2 * var))
    post = post / pre_exp  # Final Normal matrix: will be (n, K)
    numerator = post * pi
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)  # This is the vector p(x;theta)
    post = numerator / denominator  # This is the matrix of posterior probs p(j|i)
    log_lh = np.sum(np.log(denominator), axis=0).item()  # Log-likelihood

    return post, log_lh


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    nj_hat = np.sum(post, axis=0)  # shape is (K, )
    pij_hat = nj_hat / n  # Cluster probs; shape is (K, )
    mu_hat = (post.T @ X) / nj_hat.reshape(-1, 1)  # Revised means; shape is (K,d)
    norms = np.linalg.norm(X[:, None] - mu_hat, ord=2, axis=2) ** 2  # Vectorized version
    var_hat = np.sum(post * norms, axis=0) / (nj_hat * d)  # Revised variance; shape is (K, )
    return GaussianMixture(mu_hat, var_hat, pij_hat)

def run(X: np.ndarray,
        mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_lh = None
    new_log_lh = None
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):
        old_log_lh = new_log_lh
        # E-step
        post, new_log_lh = estep(X, mixture)
        # M-step
        mixture = mstep(X, post)
    if new_log_lh!=float:
        return mixture, post, old_log_lh
    else:
        return mixture, post, new_log_lh





X = np.array([[0.85794562,0.84725174],
             [0.6235637 ,0.38438171],
             [0.29753461, 0.05671298],
             [0.27265629, 0.47766512],
             [0.81216873, 0.47997717],
             [0.3927848 , 0.83607876],
             [0.33739616, 0.64817187],
             [0.36824154, 0.95715516],
             [0.14035078, 0.87008726],
             [0.47360805, 0.80091075],
             [0.52047748, 0.67887953],
             [0.72063265, 0.58201979],
             [0.53737323, 0.75861562],
             [0.10590761, 0.47360042],
             [0.18633234, 0.73691818]])
Mu = np.array([[0.6235637, 0.38438171],
                 [0.3927848, 0.83607876],
                 [0.81216873, 0.47997717],
                 [0.14035078, 0.87008726],
                 [0.36824154, 0.95715516],
                 [0.10590761, 0.47360042]])
Var = np.array([0.10038354, 0.07227467, 0.13240693, 0.12411825, 0.10497521, 0.12220856])
pi_ = np.array([0.1680912, 0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])

theta = GaussianMixture(Mu,Var,pi_)
print(theta)
#post = np.zeros([len(X),len(Mu)])
post, new_log_lh = estep(X, theta)
theta, post, log_lh = run(X,theta,post)
print(theta)
print("BIC")
print(common.bic(X,theta,log_lh))