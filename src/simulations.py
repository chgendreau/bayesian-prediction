import numpy as np
from typing import Iterable, Optional
import pymc as pm

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
    Simulate exchangeable data using a normal distribution with
    mean mu ~ Unif(0,1)
    and variance sigma^2 = 1
    """
    def __repr__(self):
        return f"ExchangeableSimulator(k={self.k}, n={self.n}, \
              seed={self.seed})"

    def __init__(self, *, k: int = 1, n: int = 1000, seed: int = None):
        super().__init__(k, n, seed)
        self.data = self.simulate()

    def simulate(self, n: Optional[int] = None, save_name: str = "") -> np.array:
        """
        Simulate exchangeable data using a normal distribution with
        mean mu ~ Unif(0,1)
        and variance sigma^2 = 1
        """
        if n is not None:
            self.n = n
        np.random.seed(self.seed)
        mu = np.random.uniform(0, 1)
        sigma2 = 10
        data = np.random.normal(mu, np.sqrt(sigma2), self.n)

        if save_name:
            self.save_data(save_name)
        return data


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
