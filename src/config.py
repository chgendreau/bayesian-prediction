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

    # "Normal, n=1000": {
    #     "dist_name": "Normal",
    #     "dist_params": {
    #         "mu": 0,
    #         "sigma": 1,
    #     },
    #     "n_obs": 1000,
    #     "N": 5000,
    #     "B": 1000,
    #     "resampling_methods": [
    #         "Empirical",
    #         "Normal",
    #     ],
    #     "classic_bayesian": {
    #         "Uninf. prior": {
    #             'likelihood_model': 'Normal',
    #             'parameters': {
    #                 'mu': 'Flat',  # Uninformative prior for mu
    #                 'sigma': {
    #                     'dist': 'HalfNormal',
    #                     'args': {'sigma': 100}  # Uniform prior for sigma
    #                 }
    #             },
    #             "n_samples": 500,  # 2 chains, so B=1000 samples in total
    #             "n_tune": 1000,
    #         },
    #         "Inform. prior": {
    #             'likelihood_model': 'Normal',
    #             'parameters': {
    #                 'mu': 'Normal',  # informative prior for mu
    #                 'sigma': {
    #                     'dist': 'HalfNormal',
    #                     'args': {'sigma': 3.0}  # Uniform prior for sigma
    #                 }
    #             },
    #             "n_samples": 500,  # 2 chains, so B=1000 samples in total
    #             "n_tune": 1000,
    #         },
    #         "Wrong Student-T L": {
    #             'likelihood_model': 'StudentT',
    #             'parameters': {
    #                 'nu': {
    #                     'dist': 'Gamma',
    #                     'args': {'alpha': 2, 'beta': 0.1}  # Weakly informative prior for nu
    #                 },
    #                 'mu': 'Flat',  # Uninformative prior for mu
    #                 'sigma': {
    #                     'dist': 'HalfNormal',
    #                     'args': {'sigma': 100}  # Uniform prior for sigma
    #                 }
    #             },
    #             "n_samples": 500,
    #             "n_tune": 1000,
    #         },
    #     },
    #     "thetas": [
    #         'mean',
    #         'median',
    #         'std',
    #         'variance',
    #         'skewness',
    #         'kurtosis',  # excess kurtosis
    #         'VaR_95',
    #         'CVaR_95',
    #     ],
    # },

    "Normal, n=100": {
        "dist_name": "Normal",
        "dist_params": {
            "mu": 0,
            "sigma": 1,
        },
        "n_obs": 100,
        "N": 5000,
        "B": 1000,
        "resampling_methods": [
            "Empirical",
            "Normal",
        ],
        "classic_bayesian": {
            "Uninf. prior": {
                'likelihood_model': 'Normal',
                'parameters': {
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 100}  # Uniform prior for sigma
                    }
                },
                "n_samples": 500,  # 2 chains, so B=1000 samples in total
                "n_tune": 1000,
            },
            "Inform. prior": {
                'likelihood_model': 'Normal',
                'parameters': {
                    'mu': 'Normal',  # informative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 3.0}  # Uniform prior for sigma
                    }
                },
                "n_samples": 500,  # 2 chains, so B=1000 samples in total
                "n_tune": 1000,
            },
            "Wrong Student-T L": {
                'likelihood_model': 'StudentT',
                'parameters': {
                    'nu': {
                        'dist': 'Gamma',
                        'args': {'alpha': 2, 'beta': 0.1}  # Weakly informative prior for nu
                    },
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 100}  # Uniform prior for sigma
                    }
                },
                "n_samples": 500,
                "n_tune": 1000,
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

    # "Student-T, nu=40, n=1000": {
    #     "dist_name": "StudentT",
    #     "dist_params": {
    #         "nu": 40,
    #         "mu": 0,
    #         "sigma": 1,
    #     },
    #     "n_obs": 1000,
    #     "N": 10000,
    #     "B": 1000,
    #     "resampling_methods": [
    #         "Empirical",
    #         "Normal",
    #         "Student-T"
    #     ],
    #     "classic_bayesian": {
    #         "Uninf. prior": {
    #             'likelihood_model': 'StudentT',
    #             'parameters': {
    #                 'nu': 40,
    #                 'mu': 'Flat',  # Uninformative prior for mu
    #                 'sigma': {
    #                     'dist': 'HalfNormal',
    #                     'args': {'sigma': 100}  # Uniform prior for sigma
    #                 }
    #             },
    #             "n_samples": 1000,  # 2 chains, so B=1000 samples in total
    #             "n_tune": 1000,
    #         },
    #         "Wrong Normal L": {
    #             'likelihood_model': 'Normal',
    #             'parameters': {
    #                 'mu': 'Flat',  # Uninformative prior for mu
    #                 'sigma': {
    #                     'dist': 'HalfNormal',
    #                     'args': {'sigma': 100}  # Uniform prior for sigma
    #                 }
    #             },
    #             "n_samples": 500,  # 2 chains, so B=1000 samples in total
    #             "n_tune": 1000,
    #         },
    #     },
    #     "thetas": [
    #         'mean',
    #         'median',
    #         'std',
    #         'variance',
    #         'skewness',
    #         'kurtosis',  # excess kurtosis
    #         'VaR_95',
    #         'CVaR_95',
    #     ],
    # },

    # "Student-T, nu=10, n=1000": {
    #     "dist_name": "StudentT",
    #     "dist_params": {
    #         "nu": 10,
    #         "mu": 0,
    #         "sigma": 1,
    #     },
    #     "n_obs": 1000,
    #     "N": 5000,
    #     "B": 1000,
    #     "resampling_methods": [
    #         "Empirical",
    #         "Normal",
    #         "Student-T"
    #     ],
    #     "classic_bayesian": {
    #         "Uninf. prior": {
    #             'likelihood_model': 'StudentT',
    #             'parameters': {
    #                 'nu': 10,
    #                 'mu': 'Flat',  # Uninformative prior for mu
    #                 'sigma': {
    #                     'dist': 'HalfNormal',
    #                     'args': {'sigma': 100}  # Uniform prior for sigma
    #                 }
    #             },
    #             "n_samples": 1000,
    #             "n_tune": 1000,
    #         },
    #         "Wrong Normal L": {
    #             'likelihood_model': 'Normal',
    #             'parameters': {
    #                 'mu': 'Flat',  # Uninformative prior for mu
    #                 'sigma': {
    #                     'dist': 'HalfNormal',
    #                     'args': {'sigma': 100}  # Uniform prior for sigma
    #                 }
    #             },
    #             "n_samples": 500,  # 2 chains, so B=1000 samples in total
    #             "n_tune": 1000,
    #         },
    #     },
    #     "thetas": [
    #         'mean',
    #         'median',
    #         'std',
    #         'variance',
    #         'skewness',
    #         'kurtosis',  # excess kurtosis
    #         'VaR_95',
    #         'CVaR_95',
    #     ],
    # },

    "Student-T, nu=10, n=100": {
        "dist_name": "StudentT",
        "dist_params": {
            "nu": 10,
            "mu": 0,
            "sigma": 1,
        },
        "n_obs": 100,
        "N": 5000,
        "B": 1000,
        "resampling_methods": [
            "Empirical",
            "Normal",
            "Student-T"
        ],
        "classic_bayesian": {
            "Uninf. prior": {
                'likelihood_model': 'StudentT',
                'parameters': {
                    'nu': 10,
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 100}  # Uniform prior for sigma
                    }
                },
                "n_samples": 1000,
                "n_tune": 1000,
            },
            "Wrong Normal L": {
                'likelihood_model': 'Normal',
                'parameters': {
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 100}  # Uniform prior for sigma
                    }
                },
                "n_samples": 500,  # 2 chains, so B=1000 samples in total
                "n_tune": 1000,
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

    "SkewNormal, alpha=-2, n=100": {
        "dist_name": "SkewNormal",
        "dist_params": {
            "alpha": -2,
            "mu": 0,
            "sigma": 1,
        },
        "n_obs": 1000,
        "N": 10000,
        "B": 1000,
        "resampling_methods": [
            "Empirical",
            "Normal",
            # "Student-T"
        ],
        "classic_bayesian": {
            "Uninf. prior": {
                "likelihood_model": "SkewNormal",
                "parameters": {
                    "alpha": "Flat",  # Uninformative prior for alpha
                    "mu": "Flat",  # Uninformative prior for mu
                    "sigma": {
                        "dist": "HalfNormal",
                        "args": {"sigma": 100}  # Uniform prior for sigma
                    }
                }
            },
            "Wrong Normal L": {
                'likelihood_model': 'Normal',
                'parameters': {
                    'mu': 'Flat',  # Uninformative prior for mu
                    'sigma': {
                        'dist': 'HalfNormal',
                        'args': {'sigma': 100}  # Uniform prior for sigma
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
    "testVAR1 (n=100)": {
        "A": [[0.7, 0.3], [0.1, 0.4]],  # Coefficient matrix (m x m)
        'p': 1,
        "Sigma_eps": [[0.4, 0.1], [0.1, 0.5]],  # Error covariance matrix (m x m)
        "n_obs": 100,
        "N": 4000,
        "B": 1000,
        "resampling_methods": ["var1_2d"],
        "classic_bayesian": {
            'bvar_analytical': {},
            # 'bvar_uninformative': {
            #     'prior_type': 'Uninformative',
            #     'parameters': {
            #         'coefficients': 'Flat',
            #         'covariance': {
            #             'dist': 'LKJ',
            #             'args': {
            #                 'eta': 2.0,             # Controls correlation strength (higher = closer to identity)
            #                 'sd_dist': 'HalfNormal',  # Prior on standard deviations
            #                 'sd_args': {'sigma': 1.0}  # Parameters for the sd prior
            #             }
            #         }
            #     }
            # },
            'bvar_minnesota_wrong': {
                'prior_type': 'Minnesota',
                'parameters': {
                    'coefficients': {
                        'dist': 'NormalMinnesota',
                        'args': {
                            'lambda1': 1.0,  # Overall tightness
                            'lambda2': 1.0,  # Cross-variable tightness
                            'lambda3': 1.0,  # Lag decay
                        }
                    },
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
            'bvar_minnesota_chat_gpt': {
                'prior_type': 'Minnesota',
                'parameters': {
                    'coefficients': {
                        'dist': 'NormalMinnesota',
                        'args': {
                            'lambda1': 0.2,  # Overall tightness  Suggested by chat GPT
                            'lambda2': 0.5,  # Cross-variable tightness
                            'lambda3': 100,  # Lag decay
                        }
                    },
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
    # "VAR1 (n=1000)": {
    #     "A": [[0.7, 0.3], [0.1, 0.4]],  # Coefficient matrix (m x m)
    #     'p': 1,
    #     "Sigma_eps": [[0.4, 0.1], [0.1, 0.5]],  # Error covariance matrix (m x m)
    #     "n_obs": 1000,
    #     "N": 4000,
    #     "B": 1000,
    #     "resampling_methods": ["var1_2d"],
    #     "classic_bayesian": {
    #         'bvar_analytical': {},
    #         # 'bvar_uninformative': {
    #         #     'prior_type': 'Uninformative',
    #         #     'parameters': {
    #         #         'coefficients': 'Flat',
    #         #         'covariance': {
    #         #             'dist': 'LKJ',
    #         #             'args': {
    #         #                 'eta': 2.0,             # Controls correlation strength (higher = closer to identity)
    #         #                 'sd_dist': 'HalfNormal',  # Prior on standard deviations
    #         #                 'sd_args': {'sigma': 1.0}  # Parameters for the sd prior
    #         #             }
    #         #         }
    #         #     }
    #         # },
    #         'bvar_minnesota_wrong': {
    #             'prior_type': 'Minnesota',
    #             'parameters': {
    #                 'coefficients': {
    #                     'dist': 'NormalMinnesota',
    #                     'args': {
    #                         'lambda1': 1.0,  # Overall tightness
    #                         'lambda2': 1.0,  # Cross-variable tightness
    #                         'lambda3': 1.0,  # Lag decay
    #                     }
    #                 },
    #                 'covariance': {
    #                     'dist': 'LKJ',
    #                     'args': {
    #                         'eta': 2.0,             # Controls correlation strength (higher = closer to identity)
    #                         'sd_dist': 'HalfNormal',  # Prior on standard deviations
    #                         'sd_args': {'sigma': 1.0}  # Parameters for the sd prior
    #                     }
    #                 }
    #             }
    #         },
    #         'bvar_minnesota_chat_gpt': {
    #             'prior_type': 'Minnesota',
    #             'parameters': {
    #                 'coefficients': {
    #                     'dist': 'NormalMinnesota',
    #                     'args': {
    #                         'lambda1': 0.2,  # Overall tightness  Suggested by chat GPT
    #                         'lambda2': 0.5,  # Cross-variable tightness
    #                         'lambda3': 100,  # Lag decay
    #                     }
    #                 },
    #                 'covariance': {
    #                     'dist': 'LKJ',
    #                     'args': {
    #                         'eta': 2.0,             # Controls correlation strength (higher = closer to identity)
    #                         'sd_dist': 'HalfNormal',  # Prior on standard deviations
    #                         'sd_args': {'sigma': 1.0}  # Parameters for the sd prior
    #                     }
    #                 }
    #             }
    #         },
    #     },
    #     "thetas": [
    #         'VAR1_A_hat',
    #         'VAR1_sigma_eps_hat',
    #     ],
    # },
    # "VAR1 (n=100) minnesota": {
    #     "A": [[0.7, 0.3], [0.1, 0.4]],  # Coefficient matrix (m x m)
    #     'p': 1,
    #     "Sigma_eps": [[0.4, 0.1], [0.1, 0.5]],  # Error covariance matrix (m x m)
    #     "n_obs": 100,
    #     "N": 4000,
    #     "B": 1000,
    #     "classic_bayesian": {
    #         'bvar_minnesota_wrong': {
    #             'prior_type': 'Minnesota',
    #             'parameters': {
    #                 'coefficients': {
    #                     'dist': 'NormalMinnesota',
    #                     'args': {
    #                         'lambda1': 0.1,  # Overall tightness
    #                         'lambda2': 0.5,  # Cross-variable tightness
    #                         'lambda3': 1.0,  # Lag decay
    #                     }
    #                 },
    #                 'covariance': {
    #                     'dist': 'LKJ',
    #                     'args': {
    #                         'eta': 2.0,             # Controls correlation strength (higher = closer to identity)
    #                         'sd_dist': 'HalfNormal',  # Prior on standard deviations
    #                         'sd_args': {'sigma': 1.0}  # Parameters for the sd prior
    #                     }
    #                 }
    #             }
    #         },
    #         'bvar_minnesota_chat_gpt': {
    #             'prior_type': 'Minnesota',
    #             'parameters': {
    #                 'coefficients': {
    #                     'dist': 'NormalMinnesota',
    #                     'args': {
    #                         'lambda1': 0.2,  # Overall tightness  Suggested by chat GPT
    #                         'lambda2': 0.5,  # Cross-variable tightness
    #                         'lambda3': 100,  # Lag decay
    #                     }
    #                 },
    #                 'covariance': {
    #                     'dist': 'LKJ',
    #                     'args': {
    #                         'eta': 2.0,             # Controls correlation strength (higher = closer to identity)
    #                         'sd_dist': 'HalfNormal',  # Prior on standard deviations
    #                         'sd_args': {'sigma': 1.0}  # Parameters for the sd prior
    #                     }
    #                 }
    #             }
    #         },
    #     },
    #     "thetas": [
    #         'VAR1_A_hat',
    #         'VAR1_sigma_eps_hat',
    #     ],
    # },
}
