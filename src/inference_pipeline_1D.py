# main pipeline for inferences
import json
import argparse
import numpy as np
from pathlib import Path
from src.config import CONFIG
from src.parameters import THETA_HAT_FUNC_DICT
from src.generating_data import generate_exchangeable_data
from src.infering import (
    predictive_resampling_posterior,
    likelihood_prior_posterior,
    RESAMPLING_MAP,
    empirical_t_res,
)


def inference_pipeline1D(experiment_name: str, experiment_config: dict):
    """
    Pipeline for 1D inferences.
    """
    # Setting random seed for reproducibility
    random_seed = experiment_config.get('random_seed', None)
    print(f"Running Pileline for {experiment_name} with random seed: '{random_seed}'")

    # Generates data using src.generating_data.generate_exchangeable_data
    X_all = generate_exchangeable_data(
        dist_name=experiment_config['dist_name'],
        dist_params=experiment_config['dist_params'],
        n_samples=experiment_config['N'],
        random_seed=random_seed,
    )
    X_all = X_all.flatten()  # Ensure X_all is 1D
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
            B=experiment_config['B'],
            df=experiment_config['dist_params'].get('nu', None) if RESAMPLING_MAP[method_name] == empirical_t_res else None,
            random_seed=random_seed,
        )

        # Update the theta_samples_dict
        for theta_name, samples in theta_samples_method.items():
            if theta_name not in theta_samples_dict:
                theta_samples_dict[theta_name] = {}

            method_full_name = f"PR\n{method_name}"
            theta_samples_dict[theta_name][method_full_name] = np.array(samples)

    # Process classic Bayesian methods
    for method_name, method_config in experiment_config.get('classic_bayesian', {}).items():
        print(f"Running Bayesian inference with {method_name}...")

        # Infer data using likelihood_prior_posterior
        theta_samples_method = likelihood_prior_posterior(
            X=X_obs,
            likelihood_prior_config=method_config,
            theta_hat_func_dict=theta_funcs,
            N=experiment_config['N'],
            n_samples=method_config.get('n_samples', 1000),
            n_tune=method_config.get('n_tune', 1000),
            random_seed=random_seed,
        )

        # Update the theta_samples_dict
        for theta_name, samples in theta_samples_method.items():
            if theta_name not in theta_samples_dict:
                theta_samples_dict[theta_name] = {}

            method_full_name = f"LP\n{method_name}"
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

    with open(output_dir / "experiment_config.json", "w") as f:
        # Convert any numpy values to Python types for JSON serialization
        config_serializable = json.loads(
            json.dumps(experiment_config, default=lambda x: float(x) if isinstance(x, np.number) else x)
        )
        json.dump(config_serializable, f, indent=2)

    np.save(output_dir / "X_obs.npy", X_obs)


def main():
    """
    Main function to run the inference pipeline for all experiments in the config.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run inference pipeline for experiments")
    parser.add_argument("--experiments", nargs="+", help="Specify experiment names to run (optional)")
    args = parser.parse_args()

    # Determine which experiments to run
    experiments_to_run = args.experiments if args.experiments else CONFIG.keys()

    random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # [111, 222, 333, 444, 555, 321, 234, 432, 795]
    for random_seed in random_seeds:
        for experiment_name in experiments_to_run:
            if experiment_name not in CONFIG:
                print(f"Warning: Experiment '{experiment_name}' not found in config, skipping.")
                continue
            # Getting config for the current experiment
            exp_config = CONFIG[experiment_name].copy()
            exp_config['random_seed'] = random_seed

            experiment_name = f"{experiment_name}, seed={random_seed}"

            print(f"###############\nRunning inference pipeline for {experiment_name}...\n")
            inference_pipeline1D(experiment_name, exp_config)
            print(f"###############\nInference pipeline for {experiment_name} completed.\n")


if __name__ == "__main__":
    main()
