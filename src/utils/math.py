import numpy as np

EPS = 1e-16


def log_stable(arr):

    return np.log(arr + EPS)


def compute_exp_normalizing(omega, k):
    """
    Quick function for computing the big exponential /softmax term at the end of the update
    """
    omega_C = 1.0 - omega
    exp_k_omega_special = np.exp(k * (1.0 - 2 * omega))
    numer = omega + omega_C * exp_k_omega_special
    denom = 1.0 + exp_k_omega_special
    return 2.0 * (numer / denom)
