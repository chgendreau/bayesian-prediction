# parameters computing functions
import numpy as np
from scipy import stats
from src.utils import var1_s_statistics_2d, var1_h_statistics_2d
from tqdm import tqdm
from scipy.integrate import quad

THETA_HAT_FUNC_DICT = {
    'mean': lambda x: np.mean(x),
    'median': lambda x: np.median(x),
    'std': lambda x: np.std(x),
    'variance': lambda x: np.var(x, ddof=0),  # default biased estimator
    'skewness': lambda x: stats.skew(x),
    'kurtosis': lambda x: stats.kurtosis(x),  # excess kurtosis
    'VaR_95': lambda x: np.percentile(x, 5),
    'VaR_99': lambda x: np.percentile(x, 1),
    'CVaR_95': lambda x: np.mean(x[x < np.percentile(x, 5)]),
    'CVaR_99': lambda x: np.mean(x[x < np.percentile(x, 1)]),
    'VAR1_A_hat': lambda x: var1_estimate_A_2d(x),
    'A': lambda x: var1_estimate_A_2d(x),  # duplicate for compatibility
    'VAR1_sigma_eps_hat': lambda x: var1_estimate_sigma_eps(x),
    '$\\Sigma$': lambda x: var1_estimate_sigma_eps(x),  # duplicate for compatibility
}

TRUE_THETA_FUNC_NORMAL_DICT = {
    'mean': lambda mu, sigma: mu,
    'median': lambda mu, sigma: mu,
    'std': lambda mu, sigma: sigma,
    'variance': lambda mu, sigma: sigma**2,  # default biased estimator
    'skewness': lambda mu, sigma: 0.0,
    'kurtosis': lambda mu, sigma: 0.0,  # excess kurtosis
    'VaR_95': lambda mu, sigma: mu + stats.norm.ppf(0.05) * sigma,
    'VaR_99': lambda mu, sigma: mu + stats.norm.ppf(0.01) * sigma,
    'CVaR_95': lambda mu, sigma: mu - sigma * stats.norm.pdf(stats.norm.ppf(0.95)) / 0.05,
    'CVaR_99': lambda mu, sigma: mu - sigma * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01,
    # see: https://blog.smaga.ch/expected-shortfall-closed-form-for-normal-distribution/ for VaR and CVaR
}

TRUE_THETA_FUNC_T_DICT = {
    'mean': lambda nu, mu, sigma: mu,
    'median': lambda nu, mu, sigma: mu,
    'std': lambda nu, mu, sigma: sigma * np.sqrt(nu / (nu - 2)) if nu > 2 else np.nan,
    'variance': lambda nu, mu, sigma: sigma**2 * nu / (nu - 2) if nu > 2 else np.nan,  # default biased estimator
    'skewness': lambda nu, mu, sigma: 0.0 if nu > 3 else np.nan,
    'kurtosis': lambda nu, mu, sigma: 6 / (nu - 4) if nu > 4 else np.nan,  # excess kurtosis
    'VaR_95': lambda nu, mu, sigma: mu + stats.t.ppf(0.05, nu) * sigma,
    'VaR_99': lambda nu, mu, sigma: mu + stats.t.ppf(0.01, nu) * sigma,
    'CVaR_95': lambda nu, mu, sigma: _CVaR_t(0.05, nu, mu, sigma) if nu > 1 else np.nan,
    'CVaR_99': lambda nu, mu, sigma: _CVaR_t(0.01, nu, mu, sigma) if nu > 1 else np.nan,
}

TRUE_THETA_FUNC_SKEWNORMAL_DICT = {
    'mean': lambda alpha, mu, sigma: mu + np.sqrt(2 / np.pi) * (sigma * alpha / np.sqrt(1 + alpha**2)),
    'median': lambda alpha, mu, sigma: mu + np.sqrt(2 / np.pi) * (sigma * alpha / np.sqrt(1 + alpha**2)),  # same as mean
    'std': lambda alpha, mu, sigma: sigma * np.sqrt(1 - (2*alpha**2 / ((1 + alpha**2)*np.pi))),
    'variance': lambda alpha, mu, sigma: sigma**2 * (1 - (2*alpha**2 / ((1 + alpha**2)*np.pi))),
    'skewness': lambda alpha, mu, sigma: ((4 - np.pi) * (alpha/np.sqrt(1 + alpha**2) * np.sqrt(2/np.pi))**3) / (2 * (1 - 2*(alpha/np.sqrt(1 + alpha**2))**2/np.pi)**(3/2)),
    'kurtosis': lambda alpha, mu, sigma: (2 * (np.pi - 3) * (alpha/np.sqrt(1 + alpha**2) * np.sqrt(2/np.pi))**4) / ((1 - 2*(alpha/np.sqrt(1 + alpha**2))**2/np.pi)**2),  # excess kurtosis
    'VaR_95': lambda alpha, mu, sigma: stats.skewnorm.ppf(1 - 0.95, alpha, loc=mu, scale=sigma),
    'VaR_99': lambda alpha, mu, sigma: stats.skewnorm.ppf(1 - 0.99, alpha, loc=mu, scale=sigma),
    'CVaR_95': lambda alpha, mu, sigma: quad(lambda x: x * stats.skewnorm.pdf(x, alpha, loc=mu, scale=sigma), -np.inf, stats.skewnorm.ppf(0.05, alpha, loc=mu, scale=sigma))[0] / (1 - 0.95),  # noqa
    'CVaR_99': lambda alpha, mu, sigma: quad(lambda x: x * stats.skewnorm.pdf(x, alpha, loc=mu, scale=sigma), -np.inf, stats.skewnorm.ppf(0.01, alpha, loc=mu, scale=sigma))[0] / (1 - 0.99)  # noqa
}

TRUE_THETA_FUNC_DICT = {
    'Normal': TRUE_THETA_FUNC_NORMAL_DICT,
    'StudentT': TRUE_THETA_FUNC_T_DICT,
    'SkewNormal': TRUE_THETA_FUNC_SKEWNORMAL_DICT,
}


def _CVaR_t(u, nu, mu, sigma):
    """
    Compute the Conditional Value at Risk (CVaR) for the Student's t-distribution.
    See https://arxiv.org/pdf/1102.5665 section 3.3
    Args:
        u: Significance level (e.g., 0.05 for 95% CVaR).
        nu: Degrees of freedom of the t-distribution.
        mu: Mean of the distribution.
        sigma: Scale parameter (standard deviation).
    Returns:
        CVaR value.
    """
    def helper_func_k(t, nu):
        from scipy.special import gamma
        numerator = nu**(nu/2) * gamma((nu-1)/2) * (nu + t**2)**(1/2 - nu/2)  # noqa
        denominator = 2 * np.sqrt(np.pi) * gamma(nu/2)  # noqa
        return numerator / denominator
    
    cvar_u = mu - sigma * helper_func_k(stats.t.ppf(u, nu), nu) / u if nu > 1 else np.nan  # noqa
    return cvar_u


def compute_theta_hat_evolution(
        X: np.ndarray,
        theta_hat_func_dict: dict,
        lag: int = 1
) -> dict:
    """
    This function computes the estimator of theta at each sample size
    and returns a dictionary with the name of the estimator as key
    and the value as the estimator of theta.
    Args:
        X: observed data
        theta_hat_func_dict: Dictionary of functions to compute theta_hat
        lag: Lag for the distance of improvement before considering convergence
    Returns:
        dict: Dictionary with keys the name of theta and values an array with the distance of
        improvement
    """
    # Getting number of samples
    n_samples = X.shape[-1]
    # Converting to 1d data if necessary
    if X.ndim == 2 and X.shape[0] == 1:
        X = X.flatten()

    theta_hat_dict = {
        theta_name: [] for theta_name in theta_hat_func_dict.keys()
    }
    # Iterating over the samples
    # and computing the estimator of theta for each sample size
    for i in tqdm(range(1, n_samples // lag + 1), desc="Computing theta hat"):
        # computing for each parameter theta
        for theta_name, theta_func in theta_hat_func_dict.items():
            # computing with only fraction of data
            theta_hat_dict[theta_name].append(theta_func(X[: i * lag]))
    return theta_hat_dict


def var1_estimate_A_2d(X: np.ndarray) -> np.ndarray:
    """
    Estimate the VAR(1) coefficient matrix A using the method of moments.
    Only works for 2D data.
    """
    # if 1D data is passed, return nan
    if X.ndim == 1:
        return np.nan
    
    # Compute S and h statistics
    s11, s22, s12 = var1_s_statistics_2d(X)
    h11, h22, h12, h21 = var1_h_statistics_2d(X)
    # Compute the determinant
    D = s11 * s22 - s12**2
    # Compute A using the formula derived earlier
    a11 = (h11 * s22 - h21 * s12) / D
    a12 = (-h11 * s12 + h21 * s11) / D
    a21 = (h12 * s22 - h22 * s12) / D
    a22 = (-h12 * s12 + h22 * s11) / D

    A_hat = np.array([[a11, a12], [a21, a22]])

    return A_hat


def var1_estimate_sigma_eps(
        X: np.ndarray, A_hat: np.ndarray = None
) -> np.ndarray:
    """
    Estimate the covariance matrix $\Sigma_\varepsilon$ of the residuals.
    Any dimension is allowed.
    """
    # if 1D data is passed, return None
    if X.ndim == 1:
        return np.nan
    n, m = X.shape
    if m > n:
        raise ValueError(f"Dimension m={m} is higher than observations n={n}!")

    if A_hat is None:
        # Estimate A_hat using the method of moments
        A_hat = var1_estimate_A_2d(X)
    # Calculate residuals: ε_t = Y_t - A_hat Y_{t-1}
    # Correct approach using direct matrix multiplication
    # residuals = X[:, 1:] - A_hat @ X[:, :-1]  # if X is (m,n)
    residuals = X[1:] - np.array([A_hat @ x for x in X[:-1]])

    # Estimate the covariance matrix
    Sigma_hat = (residuals.T @ residuals) / n

    return Sigma_hat


def compute_real_theta_from_config(experiment_config: dict) -> dict:
    """Computes real value of theta from the experiment config (for exchangeable data).
    Return a dictionary with keys the name of theta and values the real value of theta."""
    if 'dist_name' not in experiment_config:
        return get_real_theta_from_config_var1(experiment_config)
    # applying the correct function based on the distribution name
    if experiment_config['dist_name'] == 'Normal':
        dist_params = experiment_config.get('dist_params', {})
        mu = dist_params.get('mu', 0)
        sigma = dist_params.get('sigma', 1)
        return {theta_name: func(mu, sigma) for theta_name, func in TRUE_THETA_FUNC_DICT['Normal'].items()}
    elif experiment_config['dist_name'] == 'StudentT':
        dist_params = experiment_config.get('dist_params', {})
        nu = dist_params.get('nu', None)
        mu = dist_params.get('mu', 0)
        sigma = dist_params.get('sigma', 1)
        if nu is None:
            raise ValueError("The 'nu' parameter must be provided for StudentT distribution.")
        return {theta_name: func(nu, mu, sigma) for theta_name, func in TRUE_THETA_FUNC_DICT['StudentT'].items()}
    elif experiment_config['dist_name'] == 'SkewNormal':
        dist_params = experiment_config.get('dist_params', {})
        alpha = dist_params.get('alpha', None)
        mu = dist_params.get('mu', None)
        sigma = dist_params.get('sigma', None)
        return {theta_name: func(alpha, mu, sigma) for theta_name, func in TRUE_THETA_FUNC_DICT['SkewNormal'].items()}
    else:
        print(f"WARNING: Distribution {experiment_config['dist_name']} is not supported for real theta computation.")
        return {}


def get_real_theta_from_config_var1(
        experiment_config: dict,
) -> dict:
    A = experiment_config.get('A', None)
    Sigma_eps = experiment_config.get('Sigma_eps', None)
    return {
        'A': A,
        '$\\Sigma$': Sigma_eps,
    }
