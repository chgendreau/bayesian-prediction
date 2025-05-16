"""File with all PR methods for resampling the data."""
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
from src.utils import var1_estimate_A_2d, var1_estimate_sigma_eps


def empirical_res(X_obs: np.ndarray, N: int) -> np.ndarray:
    """Resample the data using the empirical cumulative distribution function (ECDF).

    Args:
        X_obs (np.ndarray): Observed data (1 dim only).
        N (int): Number of samples to generate.

    Returns:
        np.ndarray: Resampled data (including the observed one).
    """
    if len(X_obs.shape) != 1:
        raise ValueError("X_obs must be a 1D array.")
    n_obs = X_obs.shape[0]
    n_res = N - n_obs
    X_resampled = X_obs.copy()
    if n_res <= 0:
        return X_resampled[:N]
    
    # Generating new samples
    for n in tqdm(range(n_obs, N), desc="Resampling"):
        X_new = np.random.choice(X_resampled, size=1, replace=True)
        X_resampled = np.append(X_resampled, X_new)
    
    return X_resampled


def empirical_normal_res(X_obs: np.ndarray, N: int) -> np.ndarray:
    """Resample the data using Empirical Normal Distribution (Garelli).

    Args:
        X_obs (np.ndarray): Observed data (1 dim only).
        N (int): Number of samples to generate.

    Returns:
        np.ndarray: Resampled data (including the observed one).
    """
    if len(X_obs.shape) != 1:
        raise ValueError("X_obs must be a 1D array.")
    n_obs = X_obs.shape[0]
    n_res = N - n_obs
    X_resampled = X_obs.copy()
    if n_res <= 0:
        return X_resampled[:N]
    mean = np.mean(X_resampled)
    variance = np.var(X_resampled, ddof=0)  # default biased estimator 

    # Resampling data
    for n in tqdm(range(n_obs, N), desc="Resampling"):
        # Create new sample and add it to the data
        X_new = np.random.normal(loc=mean, scale=np.sqrt(variance), size=1)
        X_resampled = np.append(X_resampled, X_new)
        # Updating mean and variance recursively (faster than using np.mean and np.var)
        variance = (n*variance + (n*(X_new - mean)**2)/(n+1))/(n+1)  # Use mean_n
        mean = (n*mean + X_new) / (n + 1)  # Or mean = np.mean(X_resampled)

    return X_resampled


def empirical_t_res(X_obs: np.ndarray, N: int, df: int | float = None) -> np.ndarray:
    """Resample the data using Empirical Student-T Distribution.

    Args:
        X_obs (np.ndarray): Observed data (1 dim only).
        N (int): Number of samples to generate.
        df (int or float): Degrees of freedom. If None estimated using Kurtosis
            (default is None).
    Returns:
        np.ndarray: Resampled data (including the observed one).
    """
    if len(X_obs.shape) != 1:
        raise ValueError("X_obs must be a 1D array.")
    n_obs = X_obs.shape[0]
    n_res = N - n_obs
    X_resampled = X_obs.copy()
    if n_res <= 0:
        return X_resampled[:N]
    mean = np.mean(X_resampled)
    variance = np.var(X_resampled, ddof=0)  # default biased estimator 
    # Estimate degrees of freedom
    if df is None:
        df, _, _ = stats.t.fit(X_resampled, floc=mean)

    # Resampling data
    for n in tqdm(range(n_obs, N), desc="Resampling"):
        # Create new sample and add it to the data
        X_new = stats.t.rvs(df, loc=mean, scale=np.sqrt(variance*(df-2)/df), size=1)
        X_resampled = np.append(X_resampled, X_new)
        # Updating mean and variance recursively (faster than using np.mean and np.var)
        variance = (n*variance + (n*(X_new - mean)**2)/(n+1))/(n+1)  # Use mean_n
        mean = (n*mean + X_new) / (n + 1)  # Or mean = np.mean(X_resampled)
    return X_resampled


############################
# VAR(1) 2D
############################
# TODO: Write the sequential upgrade of A_hat and Sigma_hat
def var1_2d_res(X_obs: np.ndarray, N: int) -> np.ndarray:
    """Resample the data using VAR(1) model.
    Args:
        X_obs (np.ndarray): Observed data (2 dim only).
        N (int): Number of samples to generate.
    Returns:
        np.ndarray: Resampled data (including the observed one).
    """
    # Initial parameter estimates using observed data
    n_obs = X_obs.shape[0]
    A_hat = var1_estimate_A_2d(X_obs)
    Sigma_hat = var1_estimate_sigma_eps(X_obs, A_hat)

    # Array to store all the data
    X_resampled = X_obs.copy()
    if n_obs >= N:
        return X_resampled[:N]

    for n in tqdm(range(n_obs, N), desc="Resampling"):
        # Getting last observation
        X_prev = X_resampled[-1]
        # Forecast mean
        x_mean = A_hat @ X_prev
        # Generate random error
        eps = np.random.multivariate_normal(mean=np.zeros(2), cov=Sigma_hat)
        # Generate new sample
        X_new = x_mean + eps
        # Append new sample to the data
        X_resampled = np.append(X_resampled, X_new.reshape(1, 2), axis=0)
        # Update the A_hat and Sigma_hat using the new sample
        A_hat = var1_estimate_A_2d(X_resampled)
        Sigma_hat = var1_estimate_sigma_eps(X_resampled, A_hat)

    # Return the resampled data
    return X_resampled
