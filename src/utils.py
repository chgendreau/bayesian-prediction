from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import json
import os
import numpy as np
from typing import Dict, Any, Tuple


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


def frobenius_error_normalized(A_hat, A):
    """
    Compute the normalized Frobenius error between two matrices.
    The error is normalized by the Frobenius norm of the true matrix A.
    """
    if A.size == 0:
        return 0.0
    if (A.ndim != 2) or (A.ndim != 2):
        return 0.0

    error = np.linalg.norm(A_hat - A, 'fro')
    norm_A = np.linalg.norm(A, 'fro')
    return error / norm_A if norm_A != 0 else error


def load_inference_results(experiment_name: str) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    """Load inference results, config from JSON files and X_obs.npy file."""
    # Load inference results
    file_path = os.path.join("inference_results", experiment_name, "inferences.json")  
    with open(file_path, 'r') as f:
        results = json.load(f)

    # Load configuration file
    config_path = os.path.join("inference_results", experiment_name, "experiment_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load X_obs.npy file
    x_obs_path = os.path.join("inference_results", experiment_name, "X_obs.npy")
    X_obs = np.load(x_obs_path)
    
    # Convert list-of-lists to numpy arrays where appropriate
    for theta_name in results:
        for method_name in results[theta_name]:
            samples = results[theta_name][method_name]

            # randomly filter samples if 'B' is specified in config
            if 'B' in config and isinstance(samples, list):
                B = config['B']
                if isinstance(samples, list) and len(samples) > B:
                    # Create a local RNG
                    rng = np.random.default_rng(111)
                    samples = rng.choice(samples, size=B, replace=False).tolist()
                else:
                    samples = samples[:B]
      
            # Check if samples are a list of lists (matrix) or list of list of lists (3D)
            if isinstance(samples, list):
                if all(isinstance(x, list) for x in samples):
                    if all(isinstance(y, list) for x in samples for y in x):
                        # 3D array: list of matrices
                        results[theta_name][method_name] = np.array(samples)
                    else:
                        # 2D array: matrix
                        results[theta_name][method_name] = np.array(samples)
                else:
                    # 1D array: vector
                    results[theta_name][method_name] = np.array(samples)
    
    return results, config, X_obs


def get_theta_samples_errors(theta_samples, true_theta):
    """Get normalized error of theta_samples using frobenius norm
    Args:
        theta_samples (dict): Dictionary of theta samples
        true_theta (np.ndarray): True theta value
    Returns:
        dict: Dictionary of normalized errors for each method
    """
    errors = {}
    for method, samples in theta_samples.items():
        if samples.ndim == 1:  # theta is scalar
            errors[method] = np.abs(samples - true_theta) / np.abs(true_theta) if true_theta != 0 else np.abs(samples - true_theta)
        elif samples.ndim == 2:  # theta is vector
            errors[method] = np.linalg.norm(samples - true_theta, ord='fro', axis=1) / np.linalg.norm(true_theta, ord='fro')
        elif samples.ndim == 3:  # theta is matrix
            errors[method] = np.linalg.norm(samples - true_theta, ord='fro', axis=(1, 2)) / np.linalg.norm(true_theta, ord='fro', axis=(0,1))
        else:
            raise ValueError(f"Unsupported dimension for method {method}: {samples.ndim}")
    return errors


# Helper functions for var1_2d_pr
def var1_s_statistics_2d(x_obs: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute the matrix S (S = S^T):
    S_{n-1}^{(1)} = sum(y_i^{(1)}^2), S_{n-1}^{(2)} = sum(y_i^{(2)}^2), 
    S_{n-1}^{(1,2)} = sum(y_i^{(1)}*y_i^{(2)})
    """
    # Exclude the last observation for these statistics
    x = x_obs[:-1]
    s11 = np.sum(x[:, 0]**2)     # Sum of squares for first variable
    s22 = np.sum(x[:, 1]**2)     # Sum of squares for second variable
    s12 = np.sum(x[:, 0] * x[:, 1])  # Sum of products between variables

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


# Recursive update for VAR(1) model
def var1_estimate_2d_sequential(
        x_n: np.ndarray,
        x_n_1: np.ndarray,
        A_n_1: np.ndarray,
        S_n_1: np.ndarray,
        E_n_1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one step of the 2D VAR(1) recursive update to estimate \hat A and \hat \Sigma_\epsilon.

    Parameters
    ----------
    x_n : array-like, shape (2,)
        The new observation x_n.
    x_n_1 : array-like, shape (2,)
        The previous observation x_{n-1}.
    A_n_1 : array-like, shape (2,2), optional
        The previous estimator \hat A_{n-1}. If provided together with
        S_prev, the full recursive update is performed.
    S_n_1 : array-like, shape (2,2), optional
        The previous scatter matrix S_{n-1} = \sum_{i=0}^{n-1} x_i x_i^T.
    E_n_1 : array-like, shape (2,2), optional
        The previous sum of squared residuals matrix E_{n-1} = \sum_{i=0}^{n-1} (x_i - A_n_1 x_{i-1})(x_i - A_n_1 x_{i-1})^T.
        = \sum_{i=0}^{n-1} r_i r_i^T.

    Returns
    -------
    Tuple
        A_new : array-like, shape (2,2)
            The updated estimator \hat A_n.
        S_new : array-like, shape (2,2)
            The updated scatter matrix S_n = \sum_{i=0}^{n} x_i x_i^T.
        E_new : array-like, shape (2,2)
            The updated sum of squared residuals matrix E_n = \sum_{i=0}^{n} (x_i - A_n x_{i-1})(x_i - A_n x_{i-1})^T.
            = \sum_{i=0}^{n} r_i r_i^T.
    """
    x_n = np.asarray(x_n).reshape(2,)
    x_n_1 = np.asarray(x_n_1).reshape(2,)

    # update scatter matrix
    S_new = (S_n_1 if S_n_1 is not None else np.zeros((2, 2))) \
        + np.outer(x_n_1, x_n_1)

    A_n_1 = np.asarray(A_n_1).reshape(2, 2)
    # innovation
    r = x_n - A_n_1.dot(x_n_1)
    # determinant
    delta = np.linalg.det(S_new)
    # adjugate of a 2×2
    adj = np.array([[S_new[1, 1], -S_new[0, 1]],
                    [-S_new[1, 0], S_new[0, 0]]])
    # inverse
    S_inv = adj / delta
    # matrix increment
    Delta = np.outer(r, x_n_1).dot(S_inv)
    # updated A
    A_new = A_n_1 + Delta
    E_new = E_n_1 + np.outer(r, r)

    return A_new, S_new, E_new


def rename_keys(obj, mapping):
    """
    Recursively rename dict keys according to mapping.
    
    - If obj is a dict: build a new dict with keys replaced via mapping.
    - If obj is a list or tuple: process each element.
    - Otherwise: return obj unchanged.
    """
    if isinstance(obj, dict):
        new_d = {}
        for k, v in obj.items():
            # rename key if in mapping, otherwise keep original
            new_key = mapping.get(k, k)
            new_d[new_key] = rename_keys(v, mapping)
        return new_d

    elif isinstance(obj, list):
        return [rename_keys(item, mapping) for item in obj]

    elif isinstance(obj, tuple):
        return tuple(rename_keys(item, mapping) for item in obj)

    else:
        return obj


