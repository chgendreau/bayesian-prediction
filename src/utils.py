from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
from typing import Tuple


def wasserstein_distance(x, y):
    """Compute the Wasserstein distance between two distributions."""
    return stats.wasserstein_distance(x, y)


def ecdf(data):
    """Compute the empirical cumulative distribution function (ECDF)
    of a dataset."""
    return ECDF(data)


def ecdf_inv(data, y):
    """Generalized inverse of the ECDF"""
    return np.quantile(data, y)


# Helper functions for var1_2d_pr
def var1_s_statistics_2d(x_obs: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute the S:
    S_{n-1}^{(1)} = sum(y_i^{(1)}^2), S_{n-1}^{(2)} = sum(y_i^{(2)}^2), 
    S_{n-1}^{(1,2)} = sum(y_i^{(1)}*y_i^{(2)})
    """
    # Exclude the last observation for these statistics
    x = x_obs[:-1]
    s11 = np.sum(x[:, 0]**2)
    s22 = np.sum(x[:, 1]**2)
    s12 = np.sum(x[:, 0] * x[:, 1])

    return s11, s22, s12


def var1_h_statistics_2d(
        x_obs: np.ndarray
        ) -> Tuple[float, float, float, float]:
    """
    Compute the h:
    h_n^{(1)} = sum(y_i^{(1)}*y_{i+1}^{(1)}), h_n^{(2)} = sum(y_i^{(2)}*y_{i+1}^{(2)}),
    h_n^{(1,2)} = sum(y_i^{(1)}*y_{i+1}^{(2)}), h_n^{(2,1)} = sum(y_i^{(2)}*y_{i+1}^{(1)})
    """
    # Use all observations except the last for the first term
    x_current = x_obs[:-1]
    # Use all observations except the first for the second term
    x_next = x_obs[1:]
    h11 = np.sum(x_current[:, 0] * x_next[:, 0])
    h22 = np.sum(x_current[:, 1] * x_next[:, 1])
    h12 = np.sum(x_current[:, 0] * x_next[:, 1])
    h21 = np.sum(x_current[:, 1] * x_next[:, 0])
    
    return h11, h22, h12, h21


def var1_estimate_A_2d(x_obs: np.ndarray) -> np.ndarray:
    """
    Estimate the VAR(1) coefficient matrix A using the method of moments.
    Only works for 2D data.
    """
    # Compute S and h statistics
    s11, s22, s12 = var1_s_statistics_2d(x_obs)
    h11, h22, h12, h21 = var1_h_statistics_2d(x_obs)
    # Compute the determinant
    D = s11 * s22 - s12**2
    # Compute A using the formula derived earlier
    a11 = (h11 * s22 - h21 * s12) / D
    a12 = (-h11 * s12 + h21 * s11) / D
    a21 = (h12 * s22 - h22 * s12) / D
    a22 = (-h12 * s12 + h22 * s11) / D

    A_hat = np.array([[a11, a12], [a21, a22]])

    return A_hat


def var1_estimate_sigma_eps(
        x_obs: np.ndarray, A_hat: np.ndarray = None
        ) -> np.ndarray:
    """
    Estimate the covariance matrix $\Sigma_\varepsilon$ of the residuals.
    Any dimension is allowed.
    """
    n = len(x_obs)
    if A_hat is None:
        # Estimate A_hat using the method of moments
        A_hat = var1_estimate_A_2d(x_obs)
    # Calculate residuals: ε_t = Y_t - A_hat Y_{t-1}
    residuals = x_obs[1:] - np.array([A_hat @ x for x in x_obs[:-1]])
    # Estimate the covariance matrix
    Sigma_hat = (residuals.T @ residuals) / (n - 1)

    return Sigma_hat
