# main pipeline for inferences
import json
import argparse
import random
import numpy as np
from pathlib import Path

from src.config import CONFIG_VAR1
from src.parameters import THETA_HAT_FUNC_DICT
from src.generating_data import generate_var1_data
from src.infering import (
    predictive_resampling_posterior,
    bvar_inference,
    bvar_analytical_posterior,
)


def inference_pipeline_VAR1(experiment_name: str, experiment_config: dict):
    """
    Pipeline for 1D inferences.
    """
    random_seed = experiment_config.get('random_seed', None)
    # Generates data using src.generating_data.generate_exchangeable_data
    X_all = generate_var1_data(
        A=experiment_config['A'],
        Sigma_eps=experiment_config['Sigma_eps'],
        n_samples=experiment_config.get('N', 2000),
        X0=experiment_config.get('X0', None),
        random_seed=random_seed,
    )
    n_obs = experiment_config['n_obs']
    X_obs = X_all[:n_obs]

    # Filter theta_hat_func_dict to only include the specified thetas
    theta_funcs = {k: THETA_HAT_FUNC_DICT[k] for k in experiment_config['thetas'] if k in THETA_HAT_FUNC_DICT}

    theta_samples_dict = {}  # first layer the theta_name, second the method name, third the samples

    ###############################################################
    # Inference
    ###############################################################
    # Process resampling methods
    for method_name in experiment_config.get('resampling_methods', []):
        print(f"Running resampling inference with {method_name}...")

        # Infer data using predictive_resampling_posterior
        theta_samples_method = predictive_resampling_posterior(
            X_obs=X_obs,
            N=experiment_config.get('N', n_obs),
            method=method_name,
            theta_hat_func_dict=theta_funcs,
            B=experiment_config.get('B', 1000),
            random_seed=random_seed,
        )

        # Update the theta_samples_dict
        for theta_name, samples in theta_samples_method.items():
            if theta_name not in theta_samples_dict:
                theta_samples_dict[theta_name] = {}

            method_full_name = f"resampling_{method_name}"
            theta_samples_dict[theta_name][method_full_name] = np.array(samples)

    # Process classic Bayesian methods
    for method_name, method_config in experiment_config.get('classic_bayesian', {}).items():
        print(f"Running Bayesian inference with {method_name}...")

        if "analytical" in method_name:
            # Infer data using bvar_analytical_posterior
            theta_samples_method = bvar_analytical_posterior(
                X=X_obs,
                p=experiment_config['p'],
                prior_mean=method_config.get('prior_mean', None),
                prior_precision=method_config.get('prior_precision', None),
                prior_scale=method_config.get('prior_scale', None),
                prior_df=method_config.get('prior_df', None),
                n_draws=experiment_config.get('B', 1000),
                random_seed=experiment_config.get('random_seed', None),
            )
        else:
            # Infer data using bvar_inference
            theta_samples_method = bvar_inference(
                X=X_obs,
                p=experiment_config['p'],
                bvar_prior_config=method_config,
                n_draws=experiment_config.get('B', 1000),
                random_seed=experiment_config.get('random_seed', None),
            )
        # Renaming for technical consistency
        theta_samples_method['VAR1_A_hat'] = theta_samples_method.pop('A1', None)
        theta_samples_method['VAR1_sigma_eps_hat'] = theta_samples_method.pop('covariance', None)
        theta_samples_method['VAR1_X0_hat'] = theta_samples_method.pop('constant', None)
        # Update the theta_samples_dict
        for theta_name, samples in theta_samples_method.items():
            if theta_name not in theta_samples_dict:
                theta_samples_dict[theta_name] = {}

            method_full_name = f"bayesian_{method_name}"
            theta_samples_dict[theta_name][method_full_name] = np.array(samples)

    ##############################################################
    # Save results
    ##############################################################
    # Create output directory
    output_dir = Path(f"inference_results/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "img"
    img_dir.mkdir(exist_ok=True)

    # Save results - convert numpy arrays to lists for JSON serialization
    serializable_dict = {}
    for theta_name, methods in theta_samples_dict.items():
        serializable_dict[theta_name] = {method: samples.tolist() for method, samples in methods.items()}

    with open(output_dir / "inferences.json", "w") as f:
        json.dump(serializable_dict, f, indent=2)

    # Saving config file used
    with open(output_dir / "experiment_config.json", "w") as f:
        # Convert any numpy values to Python types for JSON serialization
        config_serializable = json.loads(
            json.dumps(experiment_config, default=lambda x: float(x) if isinstance(x, np.number) else x)
        )
        json.dump(config_serializable, f, indent=2)

    # Save the generated data
    np.save(output_dir / "X_obs.npy", X_obs)

    # Compute true parameter values for comparison (Using X_all)
    true_theta_values = {}
    for theta_name, theta_func in theta_funcs.items():
        true_theta_values[theta_name] = theta_func(X_all)


def main():
    """
    Main function to run the inference pipeline for all experiments in the config.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run inference pipeline for experiments")
    parser.add_argument("--experiments", nargs="+", help="Specify experiment names to run (optional)")
    args = parser.parse_args()

    # Determine which experiments to run
    experiments_to_run = args.experiments if args.experiments else CONFIG_VAR1.keys()

    random_seeds =[111, 222, 333]  # 444, 555, 321, 234, 432, 795] [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # [111, 222, 333, 444, 555, 321, 234, 432, 795]
    for random_seed in random_seeds:
        for experiment_name in experiments_to_run:
            if experiment_name not in CONFIG_VAR1:
                print(f"Warning: Experiment '{experiment_name}' not found in config, skipping.")
                continue
            exp_config = CONFIG_VAR1[experiment_name].copy()
            exp_config['random_seed'] = random_seed

            experiment_name = f"{experiment_name}, seed={random_seed}"
            print(f"Running inference pipeline for {experiment_name}...")
            inference_pipeline_VAR1(experiment_name, exp_config)
            print(f"Inference pipeline for {experiment_name} completed.")
    print()


if __name__ == "__main__":
    main()
