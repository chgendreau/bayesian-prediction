# Functions to simulate data given specific distributions.  # noqa
import numpy as np
import pymc as pm


def generate_exchangeable_data(
        dist_name: str,
        dist_params: dict,
        n_samples: int,
) -> np.ndarray:
    """
    Generate n_samples i.i.d. realizations from a specified PyMC distribution.

    Args:
        dist_name: Name of the PyMC distribution (e.g., 'Normal', 'Gamma', 
        'Beta')
        dist_params: Dictionary of parameters for the distribution
        n_samples: Number of samples to generate

    Returns:
        NumPy array of samples with shape (n_samples,) or (n_samples, d) for 
        multivariate distributions
    """

    # Get the distribution class from PyMC
    if not hasattr(pm, dist_name):
        raise ValueError(f"Distribution '{dist_name}' not found in PyMC")

    dist_class = getattr(pm, dist_name)

    # Create a PyMC model and sample from it
    with pm.Model():
        # Create the distribution
        rand_var = dist_class('rand_var', **dist_params)  # noqa
        # Sample from the prior (no conditioning)
        samples = pm.sample_prior_predictive(samples=n_samples)
        # Extract the samples
        data = samples.prior['rand_var'].values

    return data


def generate_var1_data(
    A: np.ndarray,  # Coefficient matrix (m x m)
    Sigma_eps: np.ndarray,  # Error covariance matrix (m x m)
    n_samples: int,  # Number of samples to generate
    X0: np.ndarray = None,  # Initial value (default: zeros)
) -> np.ndarray:
    """
    Generate data from a Vector Autoregressive model of order 1 (VAR(1)).

    The VAR(1) model is defined as:
        X_t = A * X_{t-1} + ε_t
    where ε_t ~ N(0, Sigma_eps)
    
    Parameters
    ----------
    A : numpy.ndarray
        Coefficient matrix of size (m x m)
    Sigma_eps : numpy.ndarray
        Covariance matrix of the Gaussian white noise, size (m x m)
    n_samples : int
        Number of time steps to generate
    X0 : numpy.ndarray, optional
        Initial value of size m. If None, zeros will be used.
        
    Returns
    -------
    numpy.ndarray
        Generated time series of shape (n_samples, m)
    """
    A = np.array(A)
    Sigma_eps = np.array(Sigma_eps)
    # Check dimensions
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    m = A.shape[0]
    if Sigma_eps.shape != (m, m):
        raise ValueError(f"Sigma_eps must be a square matrix of size {m}x{m}")
    # Check if Sigma_eps is positive definite
    try:
        np.linalg.cholesky(Sigma_eps)
    except np.linalg.LinAlgError:
        raise ValueError("Sigma_eps must be positive definite")
    # Initialize X0 if not provided
    if X0 is None:
        X0 = np.zeros(m)
    # Ensure X0 is the right shape (m,)
    X0 = np.asarray(X0).flatten()
    if X0.shape[0] != m:
        raise ValueError(f"X0 must have length {m}")

    # Initialize the output array
    X = np.zeros((n_samples, m))
    X[0, :] = X0

    # Generate multivariate normal errors
    # We generate n_samples-1 errors since we already have X0
    errors = np.random.multivariate_normal(
        mean=np.zeros(m),
        cov=Sigma_eps,
        size=n_samples-1
    )  # (n_samples -1, m), transpose to get shape (m, n_samples-1)

    # Generate the time series
    for t in range(1, n_samples):
        X[t, :] = A @ X[t-1, :] + errors[t-1]

    return X
