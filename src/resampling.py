# flake8: noqa
"""File with all PR methods for resampling the data."""
import warnings
import scipy.stats as stats
import numpy as np
from src.utils import (
    var1_estimate_2d_sequential,
    var1_s_statistics_2d,
)
from src.parameters import (
    var1_estimate_A_2d,
    var1_estimate_sigma_eps,
)


def empirical_res(X_obs: np.ndarray, N: int, random_seed: int = None) -> np.ndarray:
    """Resample the data using the empirical cumulative distribution function (ECDF).

    Args:
        X_obs (np.ndarray): Observed data (1 dim only).
        N (int): Number of samples to generate.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Resampled data (including the observed one).
    """
    # Create a local RNG
    rng = np.random.default_rng(random_seed)

    if len(X_obs.shape) != 1:
        raise ValueError("X_obs must be a 1D array.")
    n_obs = X_obs.shape[0]
    n_res = N - n_obs
    X_resampled = X_obs.copy()
    if n_res <= 0:
        return X_resampled[:N]
    
    # Generating new samples
    for n in range(n_obs, N):
        X_new = rng.choice(X_resampled, size=1, replace=True)
        X_resampled = np.append(X_resampled, X_new)

    return X_resampled


def empirical_normal_res(X_obs: np.ndarray, N: int, random_seed: int = None) -> np.ndarray:
    """Resample the data using Empirical Normal Distribution (Garelli).

    Args:
        X_obs (np.ndarray): Observed data (1 dim only).
        N (int): Number of samples to generate.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Resampled data (including the observed one).
    """
    # Create a local RNG
    rng = np.random.default_rng(random_seed)
    
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
    for n in range(n_obs, N):
        # Create new sample and add it to the data
        X_new = rng.normal(loc=mean, scale=np.sqrt(variance), size=1)
        X_resampled = np.append(X_resampled, X_new)
        # Updating mean and variance recursively (faster than using np.mean and np.var)
        variance = (n*variance + (n*(X_new - mean)**2)/(n+1))/(n+1)  # Use mean_n
        mean = (n*mean + X_new) / (n + 1)  # Or mean = np.mean(X_resampled)

    return X_resampled


def empirical_t_res(X_obs: np.ndarray, N: int, df: int | float = None, random_seed = None) -> np.ndarray:
    """Resample the data using Empirical Student-T Distribution.

    Args:
        X_obs (np.ndarray): Observed data (1 dim only).
        N (int): Number of samples to generate.
        df (int or float): Degrees of freedom. If None estimated using Kurtosis
            (default is None).
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
    Returns:
        np.ndarray: Resampled data (including the observed one).
    """
    # Create a local RNG
    rng = np.random.default_rng(random_seed)

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
        warnings.warn(
            "#######\nDegrees of freedom not provided. Estimating using `scipy.stats.t.fit()`.\n#######",
            UserWarning
        )
        df, _, _ = stats.t.fit(X_resampled)

    # Resampling data
    for n in range(n_obs, N):
        # Create new sample and add it to the data
        X_new = stats.t.rvs(df, loc=mean, scale=np.sqrt(variance*(df-2)/df), size=1, random_state=rng)
        X_resampled = np.append(X_resampled, X_new)
        # Updating mean and variance recursively (faster than using np.mean and np.var)
        variance = (n*variance + (n*(X_new - mean)**2)/(n+1))/(n+1)  # Use mean_n
        mean = (n*mean + X_new) / (n + 1)  # Or mean = np.mean(X_resampled)
    return X_resampled


############################
# VAR(1) 2D
############################
def var1_2d_res(X_obs: np.ndarray, N: int, get_statistics: bool = False, random_seed = None) -> np.ndarray:
    """Resample the data using VAR(1) model.
    Args:
        X_obs (np.ndarray): Observed data (2 dim only).
        N (int): Number of samples to generate.
        get_statistics (bool): If True, return the statistics used to
            generate the data.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
    Returns:
        np.ndarray: Resampled data (including the observed one).
        OR if get_statistics is True:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Resampled data (including
            the observed one), A_hat and Sigma_hat.
    """
    # Create a local RNG
    rng = np.random.default_rng(random_seed)

    # Initial parameter estimates using observed data
    n_obs, m = X_obs.shape
    if n_obs < m:
        raise ValueError(f"n_obs={n_obs} is smaller than dim m={m}.")

    # Getting initial estimates and statistics
    A_hat = var1_estimate_A_2d(X_obs)
    Sigma_hat = var1_estimate_sigma_eps(X_obs, A_hat)
    E_statistic = Sigma_hat.copy() * n_obs

    s11, s22, s12 = var1_s_statistics_2d(X_obs)
    S_statistic = np.array([[s11, s12],
                            [s12, s22]])

    # Array to store all the data
    X_resampled = X_obs.copy()
    if n_obs >= N:
        raise ValueError("N must be greater than n_obs.")
        return X_resampled[:N]

    for n in range(n_obs+1, N+1):
        # Getting last observation
        X_prev = X_resampled[-1]  # Last observation (m x 1)
        # Forecast 
        x_forecast = A_hat @ X_prev  # Forecast (m x 1)
        x_forecast = x_forecast.reshape(m,)  # Convert to 1D array to keep consistent size
        # Generate random error
        eps = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma_hat)
        # Generate new sample
        X_new = x_forecast + eps

        # Append new sample to the data
        X_resampled = np.append(X_resampled, X_new.reshape(1, -1), axis=0)

        # Update the statistics using new sample
        A_hat, S_statistic, E_statistic = var1_estimate_2d_sequential(
            X_new, X_prev, A_hat, S_statistic, E_statistic
        )
        Sigma_hat = E_statistic / n

    if get_statistics:
        # Return the resampled data and the statistics
        Sigma_hat = E_statistic / N
        return X_resampled, A_hat, Sigma_hat
    else:
        # Return the resampled data
        return X_resampled
