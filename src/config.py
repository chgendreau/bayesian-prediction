# Configuration file with all the experiments
# FORMAT:
# {
#     "experiment_name": {
#         "dist_name": "Distribution name",
#         "dist_params": {
#             "param1": value1,
#             "param2": value2,
#             ...
#         },
#         "n_obs": number of observations,
#         "N": number of samples,
#         "B": number of bootstrap samples,
#         "resampling_methods": [
#             "resampling_method1",
#             "resampling_method2",
#             ...
#         ],
#         "classic_bayesian": {  # Optional if Bayesian method available
#             "name_classic_bayesian_inference1": {
#                 'likelihood_model': 'Likelihood model name',
#                 'parameters': {
#                     'param1': value1,
#                     'param2': value2,
#                     ...
#                 }
#             },
#             "name_classic_bayesian_inference2": {
#                 'likelihood_model': 'Likelihood model name',
#                 'parameters': {
#                     'param1': value1,
#                     'param2': value2,
#                     ...
#                 }
#             }
#         },
#         "thetas": [  # Name of the parameters to compute
#             'theta1',
#             'theta2',
#             ...
#         ],
# ...

CONFIG = {
    "Normal (n=1000)": {
        "dist_name": "Normal",
        "dist_params": {
            "mu": 0,
            "sigma": 2,
        },
        "n_obs": 1000,
        "N": 3000,
        "B": 1000,
        "resampling_methods": [
            "empirical",
            "empirical_normal",
            "empirical_t",
        ],
        "classic_bayesian": {
            "correct_likelihood_uninformative": {
                'likelihood_model': 'Normal',
                'parameters': {
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 3.0}  # Uniform prior for sigma
                    }
                }
            },
            "incorrect_likelihood": {
                'likelihood_model': 'StudentT',
                'parameters': {
                    'nu': 3,
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 3.0}  # Uniform prior for sigma
                    }
                }
            },
        },
        "thetas": [
            'mean',
            'median',
            'std',
            'variance',
            'skewness',
            'kurtosis',  # excess kurtosis
            'VaR_95',
            'CVaR_95',
        ],
    },

    "Student-T (nu=5, n=1000)": {
        "dist_name": "StudentT",
        "dist_params": {
            "nu": 5,
            "mu": 0,
            "sigma": 1,
        },
        "n_obs": 1000,
        "N": 3000,
        "B": 1000,
        "resampling_methods": [
            "empirical",
            "empirical_normal",
            "empirical_t",
        ],
        "classic_bayesian": {
            "incorrect_likelihood": {
                'likelihood_model': 'Normal',
                'parameters': {
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 3.0}  # Uniform prior for sigma
                    }
                }
            },
            "correct_likelihood": {
                'likelihood_model': 'StudentT',
                'parameters': {
                    'nu': 5,
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 3.0}  # Uniform prior for sigma
                    }
                }
            },
        },
        "thetas": [
            'mean',
            'median',
            'std',
            'variance',
            'skewness',
            'kurtosis',  # excess kurtosis
            'VaR_95',
            'CVaR_95',
        ],
    },
}


CONFIG_VAR1 = {
    "VAR1 (n=1000)": {
        "A": [[0.7, 0.3], [0.1, 0.4]],  # Coefficient matrix (m x m)
        'p': 1,
        "Sigma_eps": [[1, 0.2], [0.2, 2]],  # Error covariance matrix (m x m)
        "n_obs": 1000,
        "N": 3000,
        "B": 1000,
        "resampling_methods": ["var1_2d"],
        "classic_bayesian": {
            'bvar_analytical': {},
            'bvar_uninformative': {
                'prior_type': 'Uninformative',
                'parameters': {
                    'coefficients': 'Flat',
                    'covariance': {
                        'dist': 'LKJ',
                        'args': {
                            'eta': 2.0,             # Controls correlation strength (higher = closer to identity)
                            'sd_dist': 'HalfNormal',  # Prior on standard deviations
                            'sd_args': {'sigma': 1.0}  # Parameters for the sd prior
                        }
                    }
                }
            },
        },
        "thetas": [
            'VAR1_A_hat',
            'VAR1_sigma_eps_hat',
        ],
    },
}
