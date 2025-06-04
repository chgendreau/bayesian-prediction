# main pipeline for inferences
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import CONFIG
from src.parameters import THETA_HAT_FUNC_DICT
from src.generating_data import generate_exchangeable_data
from src.infering import (
    predictive_resampling_posterior,
    likelihood_prior_posterior,
)
from src.plotting import plot_posterior_distributions


def inference_pipeline1D(experiment_name: str, experiment_config: dict):
    """
    Pipeline for 1D inferences.
    """
    # Generates data using src.generating_data.generate_exchangeable_data
    X_all = generate_exchangeable_data(
        dist_name=experiment_config['dist_name'],
        dist_params=experiment_config['dist_params'],
        n_samples=experiment_config.get('N', 2000)
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
            B=experiment_config.get('B', 1000),
            df=experiment_config['dist_params'].get('df', None) if method_name == "empirical_t" else None
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

        # Infer data using likelihood_prior_posterior
        theta_samples_method = likelihood_prior_posterior(
            X=X_obs,
            likelihood_prior_config=method_config,
            theta_hat_func_dict=theta_funcs,
            N=experiment_config.get('N', 5000),
            n_samples=method_config.get('n_samples', 1000),
            n_tune=method_config.get('n_tune', 1000),
            random_seed=method_config.get('random_seed', 42)
        )

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

    with open(output_dir / "experiment_config.json", "w") as f:
        # Convert any numpy values to Python types for JSON serialization
        config_serializable = json.loads(
            json.dumps(experiment_config, default=lambda x: float(x) if isinstance(x, np.number) else x)
        )
        json.dump(config_serializable, f, indent=2)

    np.save(output_dir / "X_obs.npy", X_obs)

    # Compute true parameter values for comparison
    # true_theta_values = {}
    # for theta_name, theta_func in theta_funcs.items():
    #     true_theta_values[theta_name] = theta_func(X_all)

    # ############################################################
    # # Plotting
    # ############################################################
    # # Plot the results for each theta
    # for theta_name, method_samples in theta_samples_dict.items():
    #     # Create a clean dictionary for plotting - exactly matching the expected format
    #     plot_data = {method_name: samples for method_name, samples in method_samples.items()}
        
    #     # Create the plot
    #     fig = plot_posterior_distributions(
    #         plot_data,  # This is now a dict with {method_name: samples_array}
    #         true_theta=true_theta_values.get(theta_name),
    #         plot_type='both',
    #         bins=100,
    #         figsize=(10, 6),
    #         title=f"Posterior Distribution of {theta_name} - {experiment_name}",
    #         xlabel=f"{theta_name} estimates",
    #         ylabel="Density",
    #         show_legend=True,
    #         alpha=0.3
    #     )
        
    #     # Save the plot
    #     fig.savefig(img_dir / f"{theta_name}_posterior.png", dpi=300, bbox_inches='tight')
    #     plt.close(fig)


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

    for experiment_name in experiments_to_run:
        if experiment_name not in CONFIG:
            print(f"Warning: Experiment '{experiment_name}' not found in config, skipping.")
            continue
   
        print(f"Running inference pipeline for {experiment_name}...")
        inference_pipeline1D(experiment_name, CONFIG[experiment_name])
        print(f"Inference pipeline for {experiment_name} completed.")


if __name__ == "__main__":
    main()
