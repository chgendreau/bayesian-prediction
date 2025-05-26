"""
Pipeline to choose N and observe convergence of predictive resampling.
This Pipeline computes the distance of improvement as N increases
to see if convergence is achieved.
"""
from locale import normalize
import numpy as np
from src.parameters import THETA_HAT_FUNC_DICT, compute_real_theta_from_config
from src.generating_data import generate_exchangeable_data, generate_var1_data


def choose_N_1D(
    experiment_config: dict,
    N_max: int = 10000,
    N_min: int = 100,
    N_step: int = 10,
    iterations: int = 10,
):
    # Create dict to store errors for all iterations
    all_errors_dict = {k: [] for k in THETA_HAT_FUNC_DICT.keys() if "VAR1" not in k}
    # Compute real values of theta
    real_theta_dict = compute_real_theta_from_config(experiment_config)

    # iterate to average the error
    for i in range(iterations):
        # generate data
        X = generate_exchangeable_data(
            dist_name=experiment_config['dist_name'],
            dist_params=experiment_config['dist_params'],
            n_samples=N_max,
        )[0]  # Get the first element (1D data)

        N_range = np.arange(N_min, N_max + 1, N_step)

        # Dict to store the error for each N
        theta_hat_error_dict = {
            k: [] for k in all_errors_dict.keys()
        }
                
        for theta_name in theta_hat_error_dict.keys():
            real_theta = real_theta_dict[theta_name]
            for N in N_range:
                # Computing estimate of theta with N samples
                theta_hat_N = THETA_HAT_FUNC_DICT[theta_name](X[:N])
                # Compute the error with respect to the real theta
                normalizer = np.abs(real_theta) if np.abs(real_theta) > 1e-6 else 1.0
                theta_hat_error_dict[theta_name].append(
                    np.abs(theta_hat_N - real_theta)/normalizer
                )
            all_errors_dict[theta_name].append(
                np.array(theta_hat_error_dict[theta_name])
            )
        
    # average the errors over all iterations
    avg_errors_dict = {}
    for theta_name in all_errors_dict.keys():
        avg_errors_dict[theta_name] = np.mean(
            np.array(all_errors_dict[theta_name]), axis=0
        )
    return avg_errors_dict




def choose_N(
    experiment_config: dict,
    VAR1: bool = False,  # If True, use VAR1 methods
    N_max: int = 10000,
    N_min: int = 100,
    N_step: int = 10,
    iterations: int = 10,
):
    # Create dict to store errors for all iterations
    if VAR1:
        all_errors_dict = {k: [] for k in THETA_HAT_FUNC_DICT.keys() if "VAR1" in k}
    else:
        all_errors_dict = {k: [] for k in THETA_HAT_FUNC_DICT.keys() if "VAR1" not in k}
    
    # Compute real values of theta
    real_theta_dict = compute_real_theta_from_config(experiment_config)
    

    # iterate to average the error
    for i in range(iterations):
        # generate data
        if VAR1:
            X = generate_var1_data(
                A=experiment_config['A'],
                Sigma_eps=experiment_config['Sigma_eps'],
                n_samples=N_max,
            )
        else:
            # Generate exchangeable data (1D)
            X = generate_exchangeable_data(
                dist_name=experiment_config['dist_name'],
                dist_params=experiment_config['dist_params'],
                n_samples=N_max,
            )[0]  # Get the first element (1D data)

        N_range = np.arange(N_min, N_max + 1, N_step)

        # Dict to store the error for each N
        theta_hat_error_dict = {
            k: [] for k in all_errors_dict.keys()
        }
                
        for theta_name in theta_hat_error_dict.keys():
            real_theta = real_theta_dict[theta_name]
            for N in N_range:
                # Computing estimate of theta with N samples
                theta_hat_N = THETA_HAT_FUNC_DICT[theta_name](X[:N])
                # Compute the error with respect to the real theta
                if theta_hat_N.ndim > 1:
                    metric = lambda x: np.linalg.norm(x, ord='fro')
                elif theta_hat_N.ndim == 1:
                    metric = lambda x: np.abs(x) if np.abs(x) > 1e-6 else 1.0

                normalizer = metric(real_theta) if metric(real_theta) > 1e-6 else 1.0
                theta_hat_error_dict[theta_name].append(
                    metric(theta_hat_N - real_theta) / normalizer
                )
            all_errors_dict[theta_name].append(
                np.array(theta_hat_error_dict[theta_name])
            )
        
    # average the errors over all iterations
    avg_errors_dict = {}
    for theta_name in all_errors_dict.keys():
        avg_errors_dict[theta_name] = np.mean(
            np.array(all_errors_dict[theta_name]), axis=0
        )
    return avg_errors_dict
