# parameters computing functions
import numpy as np
from scipy import stats
from src.utils import var1_estimate_A_2d, var1_estimate_sigma_eps

THETA_HAT_FUNC_DICT = {
    'mean': lambda x: np.mean(x),
    'median': lambda x: np.median(x),
    'std': lambda x: np.std(x),
    'variance': lambda x: np.var(x, ddof=0),  # default biased estimator
    'skewness': lambda x: stats.skew(x),
    'kurtosis': lambda x: stats.kurtosis(x),  # excess kurtosis
    'expected_shortfall_95': lambda x: np.mean(x[x < np.percentile(x, 5)]),
    'expected_shortfall_99': lambda x: np.mean(x[x < np.percentile(x, 1)]),
    'VAR1_A_hat': lambda x: var1_estimate_A_2d(x),
    'VAR1_sigma_eps_hat': lambda x: var1_estimate_sigma_eps(x),
}
