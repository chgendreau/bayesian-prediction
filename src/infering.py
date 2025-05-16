from src.resampling import (
    empirical_res,
    empirical_normal_res,
    empirical_t_res,
    var1_2d_res,
)
from typing import Literal, List, Callable
from tqdm import tqdm

RESAMPLING_MAP = {
    "empirical": empirical_res,
    "empirical_normal": empirical_normal_res,
    "empirical_t": empirical_t_res,
    "var1_2d": var1_2d_res,
}


def predictive_resampling(
    X_obs: List[float],
    N: int,
    method: Literal["empirical", "empirical_normal", "empirical_t", "var1_2d"],
    theta_hat_func_dict: Callable | dict,
    B: int = 100,
    df: int | float = None,
) -> List[float]:
    """Resample the data using the specified method.

    Args:
        X_obs (List[float]): Observed data (1 dim only).
        N (int): Number of samples to generate.
        method (str): Resampling method to use.
        theta_hat_func_dict (dict): Dictionary containing multiple theta_hat
            functions. The keys should be the names of the parameters and the
            values should be the functions to estimate them.
        B (int): Number of bootstrap samples (default is 100).
        df (int or float): Degrees of freedom. Only used for method="empirical_t".

    Returns:
        List[float]: Resampled data (including the observed one).
    """
    if method not in RESAMPLING_MAP:
        raise ValueError(f"Method {method} is not supported.")
    
    resampling_function = RESAMPLING_MAP[method]
    theta_hat_vals_dict = {key: [] for key in theta_hat_func_dict.keys()}
    # Resampling parameters
    resamp_params = {"X_obs": X_obs, "N": N}
    # appending new param for empirical_t
    if resampling_function == empirical_t_res:
        resamp_params["df"] = df
    
    # Iterate over B bootstrap samples
    for _ in tqdm(range(B), desc="Bootstrapping"):
        # Resample the data
        X_resampled = resampling_function(**resamp_params)
        # Estimate the parameter
        for key, theta_hat_func in theta_hat_func_dict.items():
            theta_hat_vals_dict[key].append(theta_hat_func(X_resampled))

    return theta_hat_vals_dict
