from collections import defaultdict
from functools import partial
import scipy.stats as stats
from tqdm import tqdm
import numpy as np
import pymc as pm
from scipy import integrate
from typing import Callable, Optional, List

from src.utils import ecdf_inv


def confidence_interval(sample_values, alpha: float = 0.05) -> tuple:
        """
        Calculate the confidence interval and expected value for a parameter theta.

        Parameters:
        - alpha: float, the significance level for the confidence interval (e.g., 0.05 for 95% confidence).

        Returns:
        - lower_bound: float, the lower bound of the confidence interval.
        - upper_bound: float, the upper bound of the confidence interval.
        - expected_value: float, the expected value of theta.
        """
        if sample_values is None:
            raise ValueError("Theta distribution is not set.")

        sorted_theta = np.sort(sample_values)
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower_bound = np.percentile(sorted_theta, lower_percentile)
        upper_bound = np.percentile(sorted_theta, upper_percentile)
        median = np.median(sample_values)

        return lower_bound, median, upper_bound


class Predictor:
    """
    Base class for making predictions on the distribution of a parameter theta = f(X_1, X_2, ...)
    """
    def __init__(self):
        self.theta_dist = None  # array of realizations of theta. can be multidimensional

    def predict_theta(self):
        """
        Placeholder method for predicting theta. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class LikelihoodPriorPredictor(Predictor):
    """
    Base class for defining a prior distribution for a parameter theta = f(X_1, X_2, ...)
    """
    def __init__(self,
                 theta_support: np.ndarray,
                 prior_dist: Optional[np.ndarray] = None,
                 prior_func: Optional[Callable] = None,
                 model_pdf: Callable = None,
                 likelihood: Optional[np.ndarray] = None,
                 evidence: Optional[float] = None):
        super().__init__()
        self.theta_support = theta_support

        # Initialize prior
        if prior_dist is not None:
            if len(prior_dist) != len(theta_support):
                raise ValueError("Length of prior_dist must match length of theta_support.")
            self.prior = prior_dist / np.sum(prior_dist)  # Normalize the prior
        elif prior_func is not None:
            self.prior = prior_func(self.theta_support)
        else:
            # Default to a uniform prior if neither is provided
            self.prior = np.ones_like(theta_support) / len(theta_support)

        # Initialize model PDF
        self.model_pdf = model_pdf if model_pdf else partial(stats.norm.pdf, loc=self.theta_support, scale=1)

        # Initialize likelihood
        if likelihood is not None:
            if len(likelihood) != len(theta_support):
                raise ValueError("Length of likelihood must match length of theta_support.")
            self.likelihood = likelihood
        else:
            self.likelihood = np.ones(self.theta_support.shape[0])  # Default to ones if not provided

        # Initialize evidence
        self.evidence = evidence  # Can be None if not provided

        self.unnormalized_posterior = None

    def compute_likelihood(self, data_obs: np.ndarray) -> np.ndarray:
        if self.likelihood is None or np.all(self.likelihood == 1):
            self.likelihood = np.ones(self.theta_support.shape[0])
            for datum in data_obs:
                likelihood_out = self.model_pdf(x=datum)
                likelihood_out = likelihood_out / np.sum(likelihood_out)
                self.likelihood *= likelihood_out
        return self.likelihood

    def predict_theta(self, data_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the predicted distribution of theta based on the prior and likelihood.
        :param data_obs: Observed data (optional)
        :return: posterior distribution
        """
        if data_obs is not None:
            self.likelihood = self.compute_likelihood(data_obs)

        self.unnormalized_posterior = self.prior * self.likelihood

        if self.evidence is None:
            self.evidence = integrate.trapezoid(self.unnormalized_posterior, self.theta_support)

        self.theta_dist = self.unnormalized_posterior / self.evidence
        return self.theta_dist


class PredictiveResamplingNormal(Predictor):
    """
    Class for predictive resampling based on the normal distribution.
    """
    def __init__(self, theta_hat_func: Callable):
        super().__init__()
        self.B = 100  # number of samples of theta to estimate its distribution
        self.theta_hat_func = theta_hat_func  # function to estimate theta f(X_1, X_2, ...)
    
    def resample(self, data_obs: np.ndarray, n_resamples: int) -> np.ndarray:
        """
        Resample the data to create a new sample of size n_resamples: (x_1, ..., x_n_obs, x_{n_obs+1}, ..., x_{n_obs + n_resamples})
        Uses the strategy given by Garelli: alpha_i = N(mean of X_i, covariance of X_i)
        :param data: Original data
        :param n_resamples: Size of the resampled data
        :return: Resampled data
        """
        # initialize the strategy
        mu = np.mean(data_obs)
        sigma = np.std(data_obs)
        alpha = pm.Normal.dist(mu=mu, sigma=sigma)
        resampled_data = data_obs.copy()
        for j in range(n_resamples):
            # adding new sample
            x_new = pm.draw(alpha, draws=1)
            resampled_data = np.append(resampled_data, x_new)
            # updating the strategy
            mu = np.mean(resampled_data)
            sigma = np.std(resampled_data)
            alpha = pm.Normal.dist(mu=mu, sigma=sigma)
        return resampled_data

    def predict_theta(self, data_obs: np.ndarray, n_resamples: int | List[int]) -> np.ndarray:
        if isinstance(n_resamples, int):
            n_resamples_list = [n_resamples]
        else:
            n_resamples_list = n_resamples
        n_resamples_max = max(n_resamples_list)  # resampling until the maximum number of resamples

        n_obs = data_obs.shape[0]
        
        theta_dist_dict = defaultdict(list)
        for i in tqdm(range(self.B)):
            # Get resampled data
            resampled_data = self.resample(data_obs, n_resamples_max)
            for n_res in n_resamples_list:
                # Compute theta for the resampled data
                theta_i = self.theta_hat_func(resampled_data[:n_obs + n_res])
                # Store the result
                theta_dist_dict[n_res].append(theta_i)
        # Convert the dictionary lists to an array
        self.theta_dist = {k: np.array(v) for k, v in theta_dist_dict.items()}
        return self.theta_dist
    

class PredictiveResamplingAR1(Predictor):
    """
    Class for predictive resampling based sor an AR1 stationary model
    """
    def __init__(self, theta_hat_func: Callable, B: int = 500):
        super().__init__()
        self.B = B  # number of samples of theta to estimate its distribution
        self.theta_hat_func = theta_hat_func  # function to estimate theta f(X_1, X_2, ...)
    
    def resample(self, data_obs: np.ndarray, n_resamples: int, error_var = None) -> np.ndarray:
        """
        Resample the data to create a new sample of size n_resamples: (x_1, ..., x_n_obs, x_{n_obs+1}, ..., x_{n_obs + n_resamples})
        Uses X_{n+1} = \hat{c}*X_n + \epsilon_{n+1}, error_var_hat = 1/(n-1) * \sum_{i=1}^{n-1} (X_i - \hat{c}*X_{i-1})^2
        :param data: Original data
        :param n_resamples: Size of the resampled data
        :return: Resampled data
        """
        # initialize the strategy by estimating the parameters
        y_t_1 = data_obs[:-1]
        y_t = data_obs[1:]
        c_hat = np.sum((y_t_1-np.mean(y_t_1))*(y_t-np.mean(y_t))) / np.sum((y_t_1-np.mean(y_t_1))**2)
        a_hat = np.mean(y_t) - c_hat*np.mean(y_t_1)

        # fixing error var if given
        fixed_error_var = True
        if error_var is None:
            fixed_error_var = False
            error_var = np.var(data_obs[1:] - c_hat * data_obs[:-1])
        
        # Resampling
        resampled_data = data_obs.copy()
        for j in range(n_resamples):
            # adding new sample
            resampled_data = np.append(resampled_data, a_hat + c_hat * resampled_data[-1] + np.random.normal(0, error_var))
            # updating the strategy
            c_hat = self.theta_hat_func(resampled_data)
            if not fixed_error_var:
                error_var = np.var(resampled_data[1:] - a_hat - c_hat * resampled_data[:-1])

        return resampled_data

    def predict_theta(self, data_obs: np.ndarray, n_resamples: int | List[int]) -> np.ndarray:
        if isinstance(n_resamples, int):
            n_resamples_list = [n_resamples]
        else:
            n_resamples_list = n_resamples
        n_resamples_max = max(n_resamples_list)  # resampling until the maximum number of resamples
        n_obs = data_obs.shape[0]
        
        theta_dist_dict = defaultdict(list)
        for i in tqdm(range(self.B)):
            # Get resampled data
            resampled_data = self.resample(data_obs, n_resamples_max)
            for n_res in n_resamples_list:
                # Compute theta for the resampled data
                theta_i = self.theta_hat_func(resampled_data[:n_obs + n_res])
                # Store the result
                theta_dist_dict[n_res].append(theta_i)
        # Convert the dictionary lists to an array
        self.theta_dist = {k: np.array(v) for k, v in theta_dist_dict.items()}
        return self.theta_dist
    

class PredictiveResamplingECDF(Predictor):
    """
    Class for predictive resampling based on the ECDFs.
    """
    def __init__(self, theta_hat_func: Callable):
        super().__init__()
        self.B = 100  # number of samples of theta to estimate its distribution
        self.theta_hat_func = theta_hat_func  # function to estimate theta f(X_1, X_2, ...)
    
    def resample(self, data_obs: np.ndarray, n_resamples: int) -> np.ndarray:
        """
        Resample the data to create a new sample of size n_resamples: (x_1, ..., x_n_obs, x_{n_obs+1}, ..., x_{n_obs + n_resamples})
        Uses the strategy of empirical cdfs
        :param data_obs: Original data
        :param n_resamples: Size of the resampled data
        :return: Resampled data
        """
        resampled_data = data_obs.copy()
        for j in range(n_resamples):
            # adding new sample
            u = np.random.uniform(0, 1)
            x_new = ecdf_inv(resampled_data, u)
            resampled_data = np.append(resampled_data, x_new)
        return resampled_data

    def predict_theta(self, data_obs: np.ndarray, n_resamples: int | List[int]) -> np.ndarray:
        if isinstance(n_resamples, int):
            n_resamples_list = [n_resamples]
        else:
            n_resamples_list = n_resamples
        n_resamples_max = max(n_resamples_list)  # resampling until the maximum number of resamples

        n_obs = data_obs.shape[0]
        
        theta_dist_dict = defaultdict(list)
        for i in tqdm(range(self.B)):
            # Get resampled data
            resampled_data = self.resample(data_obs, n_resamples_max)
            for n_res in n_resamples_list:
                # Compute theta for the resampled data
                theta_i = self.theta_hat_func(resampled_data[:n_obs + n_res])
                # Store the result
                theta_dist_dict[n_res].append(theta_i)
        # Convert the dictionary lists to an array
        self.theta_dist = {k: np.array(v) for k, v in theta_dist_dict.items()}
        return self.theta_dist


 

# # Example subclass for a specific Bayesian method
# class BayesianPredictor(PriorPredictor):
#     def __init__(self, theta_support: np.ndarray, prior_dist: Optional[np.ndarray] = None, prior_func: Optional[Callable] = None, model_pdf: Callable = None):
#         super().__init__(theta_support, prior_dist, prior_func, model_pdf)
#         # Additional initialization for Bayesian-specific logic

#     # Override or extend methods as needed for Bayesian-specific behavior

# from functools import partial
# import pymc as pm
# import scipy.stats as stats
# import numpy as np
# from scipy import integrate

# class Predictor:
#     """
#     Base class for making predictions on the distribbution of a parameter theta = f(X_1, X_2, ...)
#     """
#     def __init__(self):
#         self.theta_dist = None  # array of realisations of theta. can be multidimensional

#     def confidence_interval(self, alpha=0.05):
#         """
#         Calculate the confidence interval and expected value for a parameter theta.

#         Parameters:
#         - alpha: float, the significance level for the confidence interval (e.g., 0.05 for 95% confidence).

#         Returns:
#         - lower_bound: float, the lower bound of the confidence interval.
#         - upper_bound: float, the upper bound of the confidence interval.
#         - expected_value: float, the expected value of theta.
#         """
#         # Sort the distribution
#         sorted_theta = np.sort(self.theta_dist)

#         # Calculate the lower and upper percentiles
#         lower_percentile = alpha / 2 * 100
#         upper_percentile = (1 - alpha / 2) * 100

#         # Get the confidence interval bounds
#         lower_bound = np.percentile(sorted_theta, lower_percentile)
#         upper_bound = np.percentile(sorted_theta, upper_percentile)

#         # Calculate the expected value
#         expected_value = np.mean(self.theta_dist)

#         return lower_bound, upper_bound, expected_value

# class PriorPredictor:
#     """
#     Base class for defining a prior distribution for a parameter theta = f(X_1, X_2, ...)
#     """
#     def __init__(self, theta_support, model_pdf = None):
#         self.data_obs = None
#         self.theta_support = theta_support  # The support of the prior distribution
#         # prior
#         self.prior = stats.uniform.pdf(self.theta_support) + 1
#         self.prior = self.prior / np.sum(self.prior)  # P(theta) for each theta default: uniform prior
#         # Model pdf
#         self.model_pdf = partial(stats.norm.pdf, loc=self.theta_support, scale=0.1)  #None  # P(Data|theta)
#         # likelihood
#         self.likelihood = None  # P(Data|theta)
#         # unnormalized_posterior
#         self.unnormalized_posterior = None
#         # evidence
#         self.evidence = None

#     def get_likelihood(self, data_obs):
#         # Returning the likelihood for every theta in the support.
#         # WE return an array of size the number of thetas in the support
#         # and the likelihood for each theta (the product of the model pdf at each data point)
#         self.likelihood = np.ones(self.theta_support.shape[0])  # various pdfs for each theta

#         # Loop over all the data points
#         # and multiply the likelihood for each theta
#         # with the likelihood for the data point
#         for datum in data_obs:
#             likelihood_out = self.model_pdf(x=datum)
#             likelihood_out = likelihood_out / np.sum(likelihood_out)
#             self.likelihood *= likelihood_out
#         return self.likelihood

#     def get_posterior(self, data_obs: np.array = None):
#         """
#         Update the posterior distribution based on the observed data
#         :return: posterior distribution
#         """
#         if data_obs is not None:
#             self.data_obs = data_obs
#             # Update the likelihood based on the observed data
#             self.likelihood = self.get_likelihood(self.data_obs)
#         # Compute the unnormalized posterior
#         self.unnormalized_posterior = self.prior * self.likelihood
#         # Compute the evidence
#         self.evidence = integrate.trapezoid(self.unnormalized_posterior, self.theta_support)
#         # Compute the posterior by normalizing the unnormalized posterior
#         self.posterior = self.unnormalized_posterior / self.evidence

#         return self.posterior
