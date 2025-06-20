# flake8: noqa
from logging import warning
from pickle import TRUE
from re import U
import warnings
from src.resampling import (
    empirical_res,
    empirical_normal_res,
    empirical_t_res,
    var1_2d_res,
)
from typing import Literal, List, Callable
from tqdm import tqdm
import numpy as np
import pymc as pm
from typing import Dict, Optional, Callable, List, Any
from scipy.stats import skew, kurtosis
from src.parameters import TRUE_THETA_FUNC_DICT

RESAMPLING_MAP = {
    "Empirical": empirical_res,
    "Normal": empirical_normal_res,
    "Student-T": empirical_t_res,
    "var1_2d": var1_2d_res,
}


def predictive_resampling_posterior(
    X_obs: np.ndarray,
    N: int,
    method: Literal["Empirical", "Normal", "Student-T", "var1_2d"],
    theta_hat_func_dict: dict,
    B: int = 1000,
    df: int | float = None,
    random_seed: int | None = None,
) -> dict:
    """Resample the data using the specified method to compute posterior of theta.

    Args:
        X_obs (np.ndarray): Observed data (1 dim only).
        N (int): Number of samples to generate.
        method (str): Resampling method to use.
        theta_hat_func_dict (dict): Dictionary containing multiple theta_hat
            functions. The keys should be the names of the parameters and the
            values should be the functions to estimate them.
        B (int): Number of bootstrap samples (default is 100).
        df (int or float): Degrees of freedom. Only used for method="empirical_t".
        random_seed (int or None): Random seed for reproducibility.

    Returns:
        dict: Dictionnary with key quantities computed and values array with sample distribution.
    """
    # Get resampling function based on method
    if method not in RESAMPLING_MAP:
        raise ValueError(f"Method {method} is not supported.")
    resampling_function = RESAMPLING_MAP[method]

    # Initialize
    theta_hat_vals_dict = {key: [] for key in theta_hat_func_dict.keys()}
    resamp_params = {"X_obs": X_obs, "N": N}

    # appending new param for empirical_t
    if resampling_function == empirical_t_res:
        resamp_params["df"] = df

    # Iterate over B bootstrap samples
    for l in tqdm(range(B), desc="Bootstrapping"):
        if random_seed is not None:
            # update resamp_params
            resamp_params["random_seed"] = random_seed + l  # Ensure different seed for each iteration
        try:
            # Resample the data
            X_resampled = resampling_function(**resamp_params)  
            # Estimate the parameter
            for key, theta_hat_func in theta_hat_func_dict.items():
                theta_hat_vals_dict[key].append(theta_hat_func(X_resampled))
        except Exception as e:  # Exception might happen because of rare events
            print("Warning: resampling failed for iteration", l)
            print(f"Error: {e}")
            continue

    return theta_hat_vals_dict


#######################################
# Functions for Bayesian Inference with PyMC
#######################################
def likelihood_prior_posterior(
    X: np.ndarray,
    likelihood_prior_config: dict, 
    theta_hat_func_dict: dict,
    N: int = 5000,
    n_samples: int = 2000,
    n_tune: int = 1000,
    random_seed: int = None,
) -> Dict[str, np.ndarray]:
    """
    Perform Bayesian inference using specified priors.
    
    Args:
        X: Observed data
        likelihood_prior_config: Configuration for likelihood and priors. Examples:
            # Normal likelihood with flat prior on mean, fixed variance
            {'likelihood_model': 'Normal', 
             'parameters': {'mu': 'Flat', 'sigma': 1.5}}
            
            # Normal likelihood with informative prior on mean, Jeffreys prior on sigma
            {'likelihood_model': 'Normal', 
             'parameters': {
                 'mu': {'dist': 'Normal', 'args': {'mu': 0, 'sigma': 10}},
                 'sigma': 'HalfFlat'
             }}
             
            # Student's t-distribution with df=5, uninformative priors
            {'likelihood_model': 'StudentT', 
             'parameters': {
                 'nu': 5,  # Fixed df parameter
                 'loc': 'Flat',  # Prior on location
                 'scale': 'HalfFlat'  # Prior on scale
             }}
             
            # Poisson likelihood with gamma prior on rate
            {'likelihood_model': 'Poisson', 
             'parameters': {
                 'mu': {'dist': 'Gamma', 'args': {'alpha': 2, 'beta': 0.5}}
             }}
             
            # Binomial likelihood with beta prior on probability
            {'likelihood_model': 'Binomial', 
             'parameters': {
                 'p': {'dist': 'Beta', 'args': {'alpha': 1, 'beta': 1}},
                 'n': 10  # Fixed trials parameter
             }}
             
        theta_hat_func_dict: Dictionary of functions to compute statistics, e.g.:
            {'mean': np.mean, 'std': np.std, 'skewness': scipy.stats.skew}
        N: Number of posterior predictive samples to generate to compute statistics
        n_samples: Number of posterior samples per chain
        n_tune: Number of tuning samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary of parameter samples and derived statistics
    """
    # Create the Bayesian model
    with pm.Model() as model:
        # Setting priors:
        params_with_prior = {}
        params_fixed = {}
        
        for param_name, val in likelihood_prior_config['parameters'].items():
            # Check if this is a simple named prior (string)
            if isinstance(val, str):
                if hasattr(pm, val):
                    # This is a named prior without parameters
                    params_with_prior[param_name] = getattr(pm, val)(param_name)
                else:
                    raise ValueError(f"Unsupported distribution: {val}")
                    
            # Check if this is a parameterized prior (dict)
            elif isinstance(val, dict) and 'dist' in val:
                if hasattr(pm, val['dist']):
                    # Get distribution class
                    dist_cls = getattr(pm, val['dist'])
                    # Extract args if present, otherwise empty dict
                    args = val.get('args', {})
                    # Create the prior
                    params_with_prior[param_name] = dist_cls(param_name, **args)
                else:
                    raise ValueError(f"Unsupported distribution: {val['dist']}")
                    
            # Check if this is a fixed value
            elif isinstance(val, (float, int)):
                # This is a fixed value
                params_fixed[param_name] = pm.Deterministic(param_name, pm.math.constant(val))
                
            else:
                raise ValueError(f"Unsupported prior specification: {val}")
        
        # Get likelihood model
        likelihood_model = likelihood_prior_config['likelihood_model']
        if hasattr(pm, likelihood_model):
            # Create dictionary of likelihood arguments
            likelihood_args = {}
            
            # Add fixed parameters
            likelihood_args.update(params_fixed)
            
            # Add parameters with priors
            likelihood_args.update(params_with_prior)
            
            # Create the likelihood (variable unused but needed in PyMC)
            likelihood = getattr(pm, likelihood_model)(
                'likelihood',
                observed=X,
                **likelihood_args
            )
            # Create the X_pred variable for posterior predictive sampling
            X_pred = getattr(pm, likelihood_model)(
                'X_pred',
                observed=X,
                shape=N,
                **likelihood_args
            )
        else:
            raise ValueError(f"Unsupported likelihood model: {likelihood_model}")

        # Sampling
        trace = pm.sample(
            n_samples,
            chains=4,
            tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=False
        )
        # Generate posterior predictive samples
        # Update
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=['X_pred'],  # same as likelihood but with different sizes
            random_seed=random_seed,
        )

    # Extract inferred parameters samples
    theta_hat_vals_dict = {}
    for param_name in params_with_prior.keys():
        theta_hat_vals_dict[param_name] = trace.get_values(param_name)
    
    # # Compute statistics samples using model parameters (if real function defined)
    # ATTN: If we want a fully Bayesian predictive approach, we should compute the statistics!!!


    # # cases depending on the exact distribution
    # if likelihood_prior_config['likelihood_model'] == 'Normal':
    #     for key in theta_hat_func_dict.keys():
    #         if key not in theta_hat_vals_dict:  # check if not already here
    #             # Apply TRUE_THETA_FUNC_DICT to get the real value with all samples
    #             theta_func = TRUE_THETA_FUNC_DICT['Normal'][key]
    #             theta_hat_vals_dict[key] = [theta_func(mu, sigma) for mu, sigma in zip(trace.get_values('mu'), trace.get_values('sigma'))]
    # elif likelihood_prior_config['likelihood_model'] == 'StudentT':
    #     for key in theta_hat_func_dict.keys():
    #         if key not in theta_hat_vals_dict:
    #             # Apply TRUE_THETA_FUNC_DICT to get the real value with all samples
    #             theta_func = TRUE_THETA_FUNC_DICT['StudentT'][key]
    #             theta_hat_vals_dict[key] = [theta_func(nu, mu, sigma) for nu, mu, sigma in zip(
    #                 trace.get_values('nu'), trace.get_values('mu'), trace.get_values('sigma')
    #             )]
    # elif likelihood_prior_config['likelihood_model'] == 'SkewNormal':
    #     for key in theta_hat_func_dict.keys():
    #         if key not in theta_hat_vals_dict:
    #             # Apply TRUE_THETA_FUNC_DICT to get the real value with all samples
    #             theta_func = TRUE_THETA_FUNC_DICT['SkewNormal'][key]
    #             theta_hat_vals_dict[key] = [theta_func(alpha, mu, sigma) for alpha, mu, sigma in zip(
    #                 trace.get_values('alpha'), trace.get_values('mu'), trace.get_values('sigma')
    #             )]
    else:
        # Else, Samples N from P(X_{n+1} | X_{1:n}) = \int P(X_{n+1} | X_{1:n}, \theta) P(\theta | X_{1:n}) d\theta and compute empirical estimate
        # raise warning
        # warnings.warn(f"{'#'*10}\nLikelihood model {likelihood_prior_config['likelihood_model']} is not supported for theta_hat computation. \
        #               Using {N} posterior predictive samples to estimate statistics.\n{'#'*10}", UserWarning)
        # Get predictive samples
        pred_samples = ppc.posterior_predictive['X_pred'].values
        
        # Reshape if needed (from 3D to 2D)
        if len(pred_samples.shape) == 3:
            pred_samples = pred_samples.reshape(-1, pred_samples.shape[-1])

        # Compute derived statistics from predictive samples
        for key, theta_hat_func in theta_hat_func_dict.items():
            if key not in theta_hat_vals_dict:
                # Apply the function to each predictive dataset
                theta_hat_vals_dict[key] = np.apply_along_axis(
                    theta_hat_func,
                    axis=1,  # Apply along the observations dimension
                    arr=pred_samples
                )
                
    return theta_hat_vals_dict


#########################################
# Functions for Bayesian VAR Inference
#########################################

# Define the fixed functions
def _create_var_design_matrix(X: np.ndarray, p: int) -> np.ndarray:
    """
    Create design matrix Z for VAR model with p lags.
    
    Args:
        X: Time series data with shape (m, n) where m is the dimension 
           and n is the number of observations
        p: Number of lags
        
    Returns:
        Design matrix Z with shape (n-p, m*p + 1) where n-p is the number of 
        usable observations and m*p + 1 includes all lagged variables plus constant
    """
    n, m = X.shape
    # We still want Z to have n-p rows (one per usable observation)
    Z = np.ones((n-p, m*p + 1))  # +1 for constant term
    
    # Fill in lagged values
    for i in range(p):
        # Note the transposition to get the right orientation in Z
        # We want each row of Z to contain a flattened set of lagged observations
        Z[:, i*m+1:(i+1)*m+1] = X[p-i-1:n-i-1, :]
        
    return Z


def _create_minnesota_prior_variance(m: int, p: int, lambda1: float, lambda2: float, lambda3: float) -> np.ndarray:
    """
    Create variance matrix for Minnesota prior.
    
    Args:
        m: Dimension of the process (number of variables)
        p: Number of lags
        lambda1: Overall tightness
        lambda2: Cross-variable tightness
        lambda3: Lag decay
        
    Returns:
        Variance matrix for Minnesota prior
    """
    # k = m*p + 1 (coefficients per equation: m vars * p lags + constant)
    k = m*p + 1
    prior_var = np.zeros((k, k))
    
    # Set large variance for constant term
    prior_var[0, 0] = 100
    
    # Set variance for lag coefficients
    for i in range(1, k):
        # Determine which lag this coefficient belongs to
        lag = ((i-1) // m) + 1
        # Determine if it's own lag or cross-variable lag
        var_idx = (i-1) % m
        eq_idx = (i-1) % m
        
        if var_idx == eq_idx:  # Own lag
            prior_var[i, i] = lambda1**2 / lag**(2*lambda3)
        else:  # Cross-variable lag
            prior_var[i, i] = (lambda1*lambda2)**2 / lag**(2*lambda3)
    
    return prior_var


# Fixed BVAR inference function
def bvar_inference(
    X: np.ndarray,
    p: int,
    bvar_prior_config: dict,
    n_draws: int = 250,
    n_tune: int = 1000,
    random_seed: int = None,
) -> Dict[str, np.ndarray]:
    """
    Perform Bayesian inference on Vector Autoregression (VAR) model parameters.
    
    Args:
        X: Observed multivariate time series data with shape (n, m)
            where n is number of observations and m is dimension of the process
        p: Order of the VAR model (number of lags)
        bvar_prior_config: Configuration for VAR priors. Examples:
            # Uninformative/flat priors on all parameters
            {'prior_type': 'Uninformative',
             'parameters': {
                 'coefficients': 'Flat',  # Flat prior on VAR coefficients
                 'covariance': 'InverseWishart',  # Jeffreys prior on covariance
             }}
             
            # Minnesota prior (informative)
            {'prior_type': 'Minnesota',
             'parameters': {
                 'coefficients': {
                     'dist': 'NormalMinnesota', 
                     'args': {
                         'lambda1': 0.1,  # Overall tightness
                         'lambda2': 0.5,  # Cross-variable tightness
                         'lambda3': 1.0,  # Lag decay
                     }
                 },
                 'covariance': {'dist': 'InverseWishart', 'args': {'df': 5, 'scale': None}}
             }}
             
            # Normal-Inverse Wishart prior (informative)
            {'prior_type': 'NormalInverseWishart',
             'parameters': {
                 'coefficients': {
                     'dist': 'Normal',
                     'args': {
                         'mean': None,  # Prior mean matrix (k x m)
                         'precision': None,  # Prior precision matrix
                     }
                 },
                 'covariance': {
                     'dist': 'InverseWishart',
                     'args': {'df': 5, 'scale': None}  # Prior scale matrix (m x m)
                 }
             }}
        n_draws: Number of posterior samples per chain
        n_tune: Number of tuning samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary of parameter posterior samples with keys:
        - 'coefficients': Raw coefficient samples (shape: samples x k x m)
        - 'covariance': Error covariance matrix samples (shape: samples x m x m)
        - 'constant': Constant term samples (shape: samples x m)
        - 'A1', 'A2', ..., 'Ap': Coefficient matrices for each lag
    """
    # # Transpose X to match paper notation (n x m)
    # X = X.T 
    # n, m = X.shape  # n observation, m dimension
    # if m > n:
    #     raise ValueError("Number of dimensions (m) cannot exceed number of observations (n).")
    n, m = X.shape  # m dimension, n observations
    # Create lag matrix Z
    Z = _create_var_design_matrix(X, p)
    
    # Adjust X to match Z dimensions (remove first p rows)
    X_adj = X[p:]
    
    # VAR coefficients dimensions: k x m where k = m*p + 1 (including constant)
    k = m * p + 1
    
    # Create the Bayesian model using PyMC
    with pm.Model() as model:
        # Setting priors based on configuration
        prior_type = bvar_prior_config['prior_type']
        parameters = bvar_prior_config['parameters']
        
        # Initialize parameter containers
        coef_prior = None
        cov_prior = None
        
        # Set up coefficient priors for each equation separately
        coefficients = []
        
        if prior_type == 'Uninformative':
            # Flat prior on coefficients - need to handle each equation separately
            if parameters['coefficients'] == 'Flat':
                for i in range(m):
                    # Create a multivariate normal for each equation
                    coef_i = pm.MvNormal(
                        f'coefficients_{i}',
                        mu=np.zeros(k),
                        cov=np.eye(k) * 1e6,  # Very large variance for uninformative prior
                        shape=k
                    )
                    coefficients.append(coef_i)
            
        elif prior_type == 'Minnesota':
            # Minnesota prior - handle each equation separately
            lambda1 = parameters['coefficients']['args'].get('lambda1', 0.1)
            lambda2 = parameters['coefficients']['args'].get('lambda2', 0.5)
            lambda3 = parameters['coefficients']['args'].get('lambda3', 1.0)
            
            # Construct prior variance matrix
            prior_var = _create_minnesota_prior_variance(m, p, lambda1, lambda2, lambda3)
            
            for i in range(m):
                # Set up Minnesota prior mean (random walk for own lags, zeros elsewhere)
                prior_mean = np.zeros(k)
                # Set the first own lag coefficient to 1
                prior_mean[i*p + 1] = 1.0
                
                coef_i = pm.MvNormal(
                    f'coefficients_{i}',
                    mu=prior_mean,
                    cov=prior_var,
                    shape=k
                )
                coefficients.append(coef_i)
            
        elif prior_type == 'NormalInverseWishart':
            # Normal prior - handle each equation separately
            for i in range(m):
                if 'mean' in parameters['coefficients']['args'] and parameters['coefficients']['args']['mean'] is not None:
                    prior_mean = parameters['coefficients']['args']['mean'][:, i]
                else:
                    prior_mean = np.zeros(k)
                    
                if 'precision' in parameters['coefficients']['args'] and parameters['coefficients']['args']['precision'] is not None:
                    prior_precision = parameters['coefficients']['args']['precision']
                    prior_cov = np.linalg.inv(prior_precision)
                else:
                    prior_cov = np.eye(k) * 10  # Default moderate precision
                
                coef_i = pm.MvNormal(
                    f'coefficients_{i}',
                    mu=prior_mean,
                    cov=prior_cov,
                    shape=k
                )
                coefficients.append(coef_i)
            
        else:
            raise ValueError(f"Unsupported prior type: {prior_type}")
        
        # Stack the coefficients for each equation into a matrix
        coefficients_matrix = pm.Deterministic('coefficients', pm.math.stack(coefficients, axis=1))
  
        # Set up covariance prior
        if parameters['covariance'] == 'LKJ' or \
        (isinstance(parameters['covariance'], dict) and parameters['covariance']['dist'] == 'LKJ'):
            
            # Get parameters for LKJ prior
            if isinstance(parameters['covariance'], dict) and 'args' in parameters['covariance']:
                eta = parameters['covariance']['args'].get('eta', 2.0)
                sd_dist = parameters['covariance']['args'].get('sd_dist', 'HalfNormal')
                sd_args = parameters['covariance']['args'].get('sd_args', {'sigma': 1.0})
            else:
                # Default parameters
                eta = 2.0  # Controls correlation: higher = closer to identity
                sd_dist = 'HalfNormal'
                sd_args = {'sigma': 1.0}
            
            # Create standard deviation prior
            if sd_dist == 'HalfNormal':
                sd = pm.HalfNormal.dist(sigma=sd_args.get('sigma', 1.0), shape=m)
            elif sd_dist == 'HalfCauchy':
                sd = pm.HalfCauchy.dist(beta=sd_args.get('beta', 1.0), shape=m)
            else:
                raise ValueError(f"Unsupported standard deviation distribution: {sd_dist}")
            
            # Create LKJ prior for covariance matrix
            chol, corr, sigmas = pm.LKJCholeskyCov(
                'cholesky', 
                n=m, 
                eta=eta,
                sd_dist=sd, 
                compute_corr=True
            )
            
            # Compute covariance matrix from cholesky factor
            cov_prior = pm.Deterministic('covariance', pm.math.dot(chol, chol.T)) #* pm.math.diag(sigmas)
        else:
            raise ValueError(f"Unsupported covariance prior: {parameters['covariance']}")
        
        # Define likelihood function
        pm.MvNormal(
            'likelihood',
            mu=pm.math.dot(Z, coefficients_matrix),
            cov=cov_prior,
            observed=X_adj
        )
        
        # Sample from posterior
        trace = pm.sample(
            n_draws,
            tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=False
        )
    
    # Extract parameter samples
    posterior_samples = {}
    posterior_samples['coefficients'] = trace.get_values('coefficients')
    posterior_samples['covariance'] = trace.get_values('covariance')
    
    # Reshape coefficient matrices for easier interpretation
    # From (samples, k, m) to dictionary of coefficient matrices
    coef_samples = posterior_samples['coefficients']
    
    # Extract constant term
    posterior_samples['constant'] = coef_samples[:, 0, :]
    
    # Extract lag coefficients
    for lag in range(1, p+1):
        posterior_samples[f'A{lag}'] = coef_samples[:, (lag-1)*m+1:lag*m+1, :].transpose(0, 2, 1)  # Transpose for technical reasons
                
    return posterior_samples


from scipy import stats
def bvar_analytical_posterior(
    X: np.ndarray,
    p: int,
    prior_mean: Optional[np.ndarray] = None,
    prior_precision: Optional[np.ndarray] = None, 
    prior_scale: Optional[np.ndarray] = None,
    prior_df: Optional[float] = None,
    n_draws: int = 1000,
    random_seed: int = None,
) -> Dict[str, np.ndarray]:
    """
    Compute the exact analytical posterior distribution for a VAR(p) model
    with Normal-Inverse Wishart prior.
    
    This implements the posterior derivation from pages 10-11 of the LSE paper:
    https://www.lse.ac.uk/CFM/assets/pdf/CFM-Discussion-Papers-2018/CFMDP2018-08-Paper.pdf
    
    Args:
        X: Observed multivariate time series with shape (n, m)
           where n is number of observations and m is dimension
        p: Order of the VAR model
        prior_mean: Prior mean for VAR coefficients, shape (k, m)
                   Default: zero matrix
        prior_precision: Prior precision matrix for VAR coefficients, shape (k, k)
                        Default: 0.001 * identity matrix (diffuse prior)
        prior_scale: Prior scale matrix for error covariance, shape (m, m)
                    Default: identity matrix
        prior_df: Prior degrees of freedom for inverse Wishart
                 Default: m + 2 (minimum for proper prior)
        n_draws: Number of draws from the posterior
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary of parameter posterior samples
    """
    # Set RNG for reproducibility
    rng = np.random.default_rng(random_seed)

    n, m = X.shape
    if m > n:
        raise ValueError("Number of dimensions (m) cannot exceed number of observations (n).")
    
    # Create design matrix Z (called X in the paper)
    Z = _create_var_design_matrix(X, p)
    n_obs = Z.shape[0]  # effective sample size after accounting for lags
    
    # Number of coefficients per equation
    k = m * p + 1  # includes constant
    
    # Adjust X to match Z dimensions (remove first p rows)
    Y = X[p:]  # called Y in the paper
    
    # Default prior parameters if not provided
    if prior_mean is None:
        prior_mean = np.zeros((k, m))
    
    if prior_precision is None:
        prior_precision = 0.001 * np.eye(k)  # diffuse prior
    
    if prior_scale is None:
        prior_scale = np.eye(m)
    
    if prior_df is None:
        prior_df = m + 2  # minimum for proper prior
    
    # Calculate posterior parameters
    # 1. OLS estimates (needed for posterior computation)
    Z_T_Z = Z.T @ Z
    Z_T_Z_inv = np.linalg.inv(Z_T_Z)
    A_OLS = Z_T_Z_inv @ Z.T @ Y
    
    # 2. Calculate posterior precision matrix (V_bar in the paper)
    posterior_precision = prior_precision + Z.T @ Z
    posterior_precision_inv = np.linalg.inv(posterior_precision)
    
    # 3. Calculate posterior mean (A_bar in the paper)
    posterior_mean = posterior_precision_inv @ (prior_precision @ prior_mean + Z.T @ Z @ A_OLS)
    
    # 4. Calculate residuals from OLS
    residuals = Y - Z @ A_OLS
    S_OLS = residuals.T @ residuals
    
    # 5. Calculate posterior scale matrix (S_bar in the paper)
    posterior_scale = prior_scale + S_OLS + \
                     (A_OLS - prior_mean).T @ \
                     np.linalg.inv(np.linalg.inv(prior_precision) + Z_T_Z_inv) @ \
                     (A_OLS - prior_mean)
    
    # 6. Calculate posterior degrees of freedom (v_bar in the paper)
    posterior_df = prior_df + n_obs
    
    # Draw samples from the posterior
    posterior_samples = {}
    
    # Storage for coefficient draws
    coef_samples = np.zeros((n_draws, k, m))
    cov_samples = np.zeros((n_draws, m, m))
    
    # Draw from the posterior
    for i in range(n_draws):
        # First draw from inverse Wishart
        cov_i = stats.invwishart.rvs(df=posterior_df, scale=posterior_scale, random_state=rng)
        cov_samples[i] = cov_i
        
        # Then draw from matrix normal, which we implement as multivariate normal for each column
        for j in range(m):
            # For each column of A, draw from multivariate normal
            mean_j = posterior_mean[:, j]
            cov_j = posterior_precision_inv * cov_i[j, j]  # Scale by the jth diagonal element of Sigma
            coef_samples[i, :, j] = rng.multivariate_normal(mean_j, cov_j)
    
    # Store results
    # posterior_samples['coefficients'] = coef_samples
    posterior_samples['covariance'] = cov_samples
    
    # Extract constant term
    posterior_samples['constant'] = coef_samples[:, 0, :]
    
    # Extract lag coefficients
    for lag in range(1, p+1):
        posterior_samples[f'A{lag}'] = coef_samples[:, (lag-1)*m+1:lag*m+1, :].transpose(0, 2, 1)  # Transpose for technical reasons
    
    # Also store the analytical posterior parameters
    posterior_samples['posterior_mean'] = posterior_mean
    posterior_samples['posterior_precision'] = posterior_precision
    posterior_samples['posterior_scale'] = posterior_scale
    posterior_samples['posterior_df'] = posterior_df
    
    return posterior_samples
