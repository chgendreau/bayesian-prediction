from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np


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
