"""Pipeline to make predictions on a parameter theta"""
import json
import numpy as np
from src.predictions import *


def mean_function(data: np.ndarray):
    return np.mean(data, axis=0)


def std_function(data: np.ndarray):
    return np.std(data, axis=0)


def skewness_function(data: np.ndarray):
    return np.mean((data - np.mean(data, axis=0)) ** 3, axis=0) / (np.std(data, axis=0) ** 3)


def kurtosis_function(data: np.ndarray):
    return np.mean((data - np.mean(data, axis=0)) ** 4, axis=0) / (np.std(data, axis=0) ** 4) - 3


THETA_FUNCS = {
    "mean": mean_function,
    "std": std_function,
    "skewness": skewness_function,
    # "kurtosis": kurtosis_function,
}

DATA_PATHS = ["data/april08_exchangeable_data.npy", "data/april08_tmixture_data.npy"]
N_OBS = [100, 1000, 3500]
N_RESAMPLES = [100, 500, 1000, 3500]
CONFIG_NAME = "april08"

def predictions_pipeline():
    # Loop through each theta function
    for theta_name, theta_func in THETA_FUNCS.items():
        for data_path in DATA_PATHS:
            # Load the data
            data = np.load(data_path)
            # Extract function
            real_theta = theta_func(data)

            ##############################
            # Predictive resampling
            ##############################
            pr_normal_predictor = PredictiveResamplingNormal(theta_func)
            predictions_dict = {}
            for n_obs in N_OBS:
                print(f"Working on: theta: {theta_name}, data: {data_path.split('/')[-1].split('.')[0]}, n_obs: {n_obs}")
                predictions_dict[n_obs] = dict()
                # Predict theta
                theta_dist_dict = pr_normal_predictor.predict_theta(data[: n_obs], n_resamples=N_RESAMPLES)
                # Save the results converting to serializable elements
                theta_dist_dict = {key: value.tolist() for key, value in theta_dist_dict.items()}
                predictions_dict[n_obs] = theta_dist_dict
            
            predictions_dict["real_theta"] = real_theta
            # Save the results as json
            save_path = f"predictions/{theta_name}/{CONFIG_NAME}_{data_path.split('/')[-1].split('.')[0]}_pr.json"
            json.dump(predictions_dict, open(save_path, "w"))
    return None


if __name__ == "__main__":
    predictions_pipeline()
                
            #Hello
