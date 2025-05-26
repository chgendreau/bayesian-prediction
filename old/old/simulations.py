import numpy as np
from typing import Iterable, Optional, Union, Dict, Any
import pymc as pm
from pymc.pytensorf import TensorVariable


############################################################
# Global class to simulate different types of data
############################################################
class Simulator:
    """
    Base class for simulating data. This class is not meant to be
    instantiated directly. Instead, use one of the subclasses
    (e.g., ExchangeableSimulator, TMixtureSimulator) to simulate
    specific types of data.
    """
    def __repr__(self):
        return f"Simulator(k={self.k}, n={self.n}, seed={self.seed})"

    def __init__(self, k: int = 1, n: int = 1000, seed: int = None):
        self.k = k
        self.n = n
        self.seed = seed
        self.data = None
    
    def save_data(self, file_name: str):
        """
        Save the simulated data to a file. The file format is
        determined by the file extension.
        """
        if self.data is None:
            raise ValueError("No data to save. Please run simulate() first.")
        # Save the data to a file
        if file_name.endswith(".csv"):
            np.savetxt(file_name, self.data, delimiter=",")
        elif file_name.endswith(".npy"):
            np.save(file_name, self.data)
        else:
            file_name = file_name + ".npy"
            np.save(file_name, self.data)
        return None


############################################################
# Sub-classes for specific types of simulations
############################################################
class ExchangeableSimulator(Simulator):
    """
    Simulate exchangeable data using custom distributions and parameters.
    Parameters can be either fixed values or random variables specified using PyMC distributions.
    Example usage:
    ```
        normal_random_mean = ExchangeableSimulator(
        model=pm.Normal,
        params={
            'mu': pm.Uniform.dist(lower=0, upper=1),
            'sigma': 1.0
        },
        n=1000,
        seed=42
    )
    data = normal_random_mean.simulate()
    ```
    """
    def __repr__(self):
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"ExchangeableSimulator(distribution={self.distribution.__name__}, {params_str})"

    def __init__(self, *,
                 model: pm.Distributions = pm.Normal,
                 params: Dict[str, Union[float, TensorVariable]] = {"mu": pm.Uniform.dist(0, 1), "sigma": 10},
                 k: int = 1,
                 n: int = 1000,
                 seed: int = None):
        """
        Parameters:
        -----------
        distribution : callable
            The PyMC distribution to use for simulation
        params : Dict[str, Union[float, pm.Distribution]]
            Dictionary of parameters for the distribution.
            Each parameter can be either a scalar value or a PyMC distribution
        k : int
            Number of independent sequences
        n : int
            Number of samples per sequence
        seed : int or None
            Random seed for reproducibility
        """
        super().__init__(k, n, seed)
        self.distribution = model
        self.params = params
        self.data = self.simulate()

    def _sample_param(self, param: Union[float, pm.Distribution]) -> float:
        """
        Sample a value from a parameter if it's a distribution,
        otherwise return the parameter value
        """
        if isinstance(param, pm.Distribution):
            return pm.draw(param, draws=1)
        return param

    def simulate(self, n: Optional[int] = None, save_name: str = "") -> np.array:
        """
        Simulate exchangeable data using the specified distribution and parameters
        
        Parameters:
        -----------
        n : Optional[int]
            Number of samples (if different from self.n)
        save_name : str
            If provided, save the data to this file
            
        Returns:
        --------
        np.array
            Simulated data
        """
        if n is not None:
            self.n = n

        # Set random seed
        np.random.seed(self.seed)

        # Sample values for random parameters
        sampled_params = {
            name: self._sample_param(param)
            for name, param in self.params.items()
        }
        distribution = self.distribution.dist(**sampled_params)

        data = pm.draw(distribution, draws=self.n, random_seed=self.seed)

        if save_name:
            self.save_data(save_name)
            
        return data

    def get_params_info(self) -> Dict[str, Any]:
        """
        Get information about the parameters used in the simulation
        """
        return {
            name: {
                'type': type(param).__name__,
                'value': self._sample_param(param) if isinstance(param, pm.Distribution) else param
            }
            for name, param in self.params.items()
        }


class TMixtureSimulator(Simulator):
    """
    Simulate data from a mixture of t-distributions.
    The mixture is defined by the following parameters:
    - weights: list of weights for each component
    - mus: list of means for each component
    - nus: list of degrees of freedom for each component
    """
    def __repr__(self):
        return f"TMixtureSimulator(weights={self.weights}, mus={self.mus}, \
              nus={self.nus}, n={self.n}, seed={self.seed})"
    
    def __init__(
            self,
            weights: Iterable,
            mus: Iterable,
            nus: Iterable, *,
            k: int = 1, n: int = 1000, seed: int = None,
    ):
        super().__init__(k, n, seed)
        if len(weights) != len(mus) or len(weights) != len(nus):
            raise ValueError("weights, mus and nus must have the same length")
        if not np.isclose(sum(weights), 1):
            raise ValueError("weights must sum to 1")
        self.weights = weights
        self.mus = mus
        self.nus = nus
        self.data = None

    def simulate(self, n: Optional[int] = None, save_name: str = "") -> np.array:
        if n is not None:
            self.n = n
        # Create the mixture model
        comp_dists = [
            pm.StudentT.dist(nu=self.nus[i], mu=self.mus[i])
            for i in range(len(self.weights))
        ]
        model_dist = pm.Mixture.dist(w=self.weights, comp_dists=comp_dists)

        # Simulating data
        self.data = pm.draw(model_dist, draws=self.n, random_seed=self.seed)

        if save_name:
            self.save_data(save_name)
        return self.data


class StationaryMarkovSimulator(Simulator):
    """
    Simulate Stationary Markov (AR(1)) data
    y[t] = mu + phi * (y[t-1] - mu) + epsilon[t]
    where epsilon[t] ~ N(0, sigma^2)
    """
    def __repr__(self):
        return f"StationaryMarkovSimulator(phi={self.phi}, mu={self.mu}, sigma={self.sigma})"

    def __init__(self, *, n: int = 1000, seed: int = None, phi: float = 0.7, 
                 mu: float = 0, sigma: float = 1.0):
        super().__init__(n, seed)
        if abs(phi) >= 1:
            raise ValueError("phi must be between -1 and 1 for stationarity")
        self.phi = phi
        self.mu = mu
        self.sigma = sigma
        self.data = None

    def simulate(self, n: Optional[int] = None, save_name: str = "") -> np.array:
        """
        Simulate stationary AR(1) Markov process
        
        Parameters:
        -----------
        n : Optional[int]
            Number of samples to generate (if different from self.n)
        save_name : str
            If provided, save the data to this file
            
        Returns:
        --------
        np.array
            Simulated AR(1) process
        """
        if n is not None:
            self.n = n
            
        np.random.seed(self.seed)
        data = np.zeros(self.n)
        # Generate white noise
        epsilon = np.random.normal(0, self.sigma, self.n)
        # initalize the first value
        data[0] = np.random.normal(self.mu, self.sigma / np.sqrt(1 - self.phi**2))
        # Generate AR(1) process
        for t in range(1, self.n):
            data[t] = self.mu + self.phi * (data[t-1] - self.mu) + epsilon[t]

        self.data = data
        if save_name:
            self.save_data(save_name)
        return self.data


# def simulate_exchangeable_data(n: int = 1000, seed: int = None) -> np.array:
#     """
#     Simulate exchangeable data using a normal distribution with
#     mean mu ~ Unif(0,1)
#     and variance sigma^2 = 1
#     """
#     np.random.seed(seed)
#     mu = np.random.uniform(0, 1)
#     sigma2 = 1
#     data = np.random.normal(mu, np.sqrt(sigma2), n)
#     return data


# def simulate_non_exchangeable_data(weights: List[Callable], n: int = 1000, \
#  seed: int = None) -> np.array:
#     """
#     Simulate exchangeable data using a normal distribution with
#     mean mu ~ Unif(0,1)
#     and variance sigma^2 = 1
#     """
#     np.random.seed(seed)
#     data = simulate_exchangeable_data()
#     return data, weights
