import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union, List, Tuple


def plot_posterior_distributions(
    theta_samples: Dict[str, np.ndarray],
    true_theta: Union[float, None] = None,
    plot_type: str = 'both',
    bins: Union[int, str] = 'auto',
    figsize: tuple = (10, 6),
    colors: Union[Dict[str, str], None] = None,
    title: str = 'Posterior Distributions',
    xlabel: str = 'θ',
    ylabel: str = 'Density',
    show_legend: bool = True,
    alpha: float = 0.5,
    x_range: Union[Tuple[float, float], None] = None
) -> None:
    """
    Plot histogram and/or density plots for multiple sets of theta samples.
    
    Parameters:
    -----------
    theta_samples : Dict[str, np.ndarray]
        Dictionary with inference method names as keys and theta samples as values
    true_theta : float or None
        True value of theta to be plotted as vertical line
    plot_type : str
        Type of plot to create ('hist', 'density', or 'both')
    bins : int or str
        Number of bins for histogram or method to calculate bins
    figsize : tuple
        Figure size (width, height)
    colors : Dict[str, str] or None
        Dictionary mapping inference methods to colors
    title : str
        Plot title
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    show_legend : bool
        Whether to show the legend
    alpha : float
        Transparency level for the plots
    x_range : tuple or None
        Optional (min, max) tuple to specify the x-axis range for zooming
    """
    
    plt.figure(figsize=figsize)
    
    # Set default colors if none provided
    if colors is None:
        colors = {method: color for method, color 
                 in zip(theta_samples.keys(), 
                       sns.color_palette("husl", len(theta_samples)))}
    
    # Find common x-axis limits if x_range not specified
    if x_range is None:
        all_samples = np.concatenate(list(theta_samples.values()))
        x_min, x_max = np.min(all_samples), np.max(all_samples)
        x_range = x_max - x_min
        x_min -= x_range * 0.1  # Add 10% padding
        x_max += x_range * 0.1
    else:
        x_min, x_max = x_range
    
    for method, samples in theta_samples.items():
        if plot_type in ['hist', 'both']:
            plt.hist(samples, bins=bins, density=True, alpha=alpha,
                    label=f'{method} (hist)', color=colors[method],
                    range=(x_min, x_max))  # Limit histogram range
            
        if plot_type in ['density', 'both']:
            sns.kdeplot(data=samples, label=f'{method} (density)',
                       color=colors[method], linewidth=2,
                       clip=(x_min, x_max))  # Limit density range
    
    # Plot true theta if provided
    if true_theta is not None:
        plt.axvline(true_theta, color='black', linestyle='--', linewidth=2,
                    label='True θ')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    
    if show_legend:
        plt.legend(fontsize=10)
        
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    
    # Make the plot look nice
    sns.set_style("whitegrid")
    plt.tight_layout()
    
    return plt.gcf()



def plot_var1_posteriors(
    results: Dict,
    true_A: Union[np.ndarray, None] = None,
    true_Sigma: Union[np.ndarray, None] = None,
    bins: Union[int, str] = 'auto',
    alpha: float = 0.5,
    method_names: Union[List[str], None] = None,
    colors: Union[Dict[str, str], None] = None
) -> None:
    """
    Plot posterior distributions for VAR(1) model parameters.
    
    Parameters:
    -----------
    results : Dict or List[Dict]
        Dictionary containing bootstrap samples or other posterior samples.
        Expected to contain 'A_samples' and 'Sigma_samples' keys.
        Can also be a list of such dictionaries for comparing methods.
    true_A : np.ndarray or None
        True A matrix for comparison (optional)
    true_Sigma : np.ndarray or None
        True Sigma matrix for comparison (optional)
    bins : int or str
        Number of bins for histogram or method to calculate bins
    alpha : float
        Transparency for plots
    method_names : List[str] or None
        Names of methods if multiple methods are being compared
    colors : Dict[str, str] or None
        Dictionary mapping methods to colors
    """
    # Extract samples (handling both single method and multiple methods)
    if isinstance(results, list):
        # Multiple methods case
        A_samples_list = [res['A_samples'] for res in results]
        Sigma_samples_list = [res['Sigma_samples'] for res in results]
        multi_method = True
    else:
        # Single method case
        A_samples_list = [results['A_samples']]
        Sigma_samples_list = [results['Sigma_samples']]
        multi_method = False
    
    # If no method names provided, create default names
    if method_names is None:
        if multi_method:
            method_names = [f'Method {i+1}' for i in range(len(A_samples_list))]
        else:
            method_names = ['Bootstrap']
    
    # Matrix element labels
    A_labels = [['a_11', 'a_12'], ['a_21', 'a_22']]
    Sigma_labels = [['σ_11', 'σ_12'], ['σ_21', 'σ_22']]

    
    # Process each element of the A matrix
    for i in range(2):
        for j in range(2):
            # Create samples dictionary for this parameter
            theta_samples = {}
            for k, method in enumerate(method_names):
                theta_samples[method] = A_samples_list[k][:, i, j]
            
            # Plot the posterior without creating a new figure
            true_value = true_A[i, j] if true_A is not None else None
            plot_posterior_distributions(
                theta_samples=theta_samples,
                true_theta=true_value,
                plot_type='both',
                bins=bins,
                colors=colors,
                title=f'Posterior for {A_labels[i][j]}',
                xlabel=A_labels[i][j],
                ylabel='Density',
                show_legend=(i==0 and j==0),  # Only show legend in first subplot
                alpha=alpha
            )
    
    # Process each element of the Sigma matrix
    for i in range(2):
        for j in range(2):
            # Create samples dictionary for this parameter
            theta_samples = {}
            for k, method in enumerate(method_names):
                theta_samples[method] = Sigma_samples_list[k][:, i, j]
            
            # Plot the posterior without creating a new figure
            true_value = true_Sigma[i, j] if true_Sigma is not None else None
            plot_posterior_distributions(
                theta_samples=theta_samples,
                true_theta=true_value,
                plot_type='both',
                bins=bins,
                colors=colors,
                title=f'Posterior for {Sigma_labels[i][j]}',
                xlabel=Sigma_labels[i][j],
                ylabel='Density',
                show_legend=(i==0 and j==0),  # Only show legend in first subplot
                alpha=alpha
            )
            # Show the plots
            plt.show()


# Frequentist view of VAR(1) Posterior Plotting Function
def plot_predictive_resampling_results(y_obs: np.ndarray, results: Dict, true_A: np.ndarray = None, 
                                      true_Sigma: np.ndarray = None, alpha: float = 0.1):
    """
    Plot the results of predictive resampling
    
    Parameters:
    -----------
    y_obs : np.ndarray, shape (m, 2)
        Observed time series
    results : Dict
        Results dictionary from predictive_resampling function
    true_A : np.ndarray, optional
        True A matrix if known (for comparison)
    true_Sigma : np.ndarray, optional
        True Sigma matrix if known (for comparison)
    alpha : float
        Transparency of forecast paths
    """
    m = len(y_obs)
    n_future = results['forecasts'].shape[1]
    n_bootstrap = results['forecasts'].shape[0]
    total_time = m + n_future
    
    # Create time indices
    time_obs = np.arange(m)
    time_future = np.arange(m, total_time)
    
    # 1. Plot time series and forecasts
    plt.figure(figsize=(15, 10))
    
    # Plot for first variable
    plt.subplot(2, 2, 1)
    plt.plot(time_obs, y_obs[:, 0], 'b-', label='Observed')
    
    for b in range(n_bootstrap):
        if b == 0:
            plt.plot(time_future, results['forecasts'][b, :, 0], 'r-', alpha=alpha, label='Forecasts')
        else:
            plt.plot(time_future, results['forecasts'][b, :, 0], 'r-', alpha=alpha)
    
    plt.axvline(x=m-1, color='k', linestyle='--')
    plt.title('Variable 1: Observed and Forecasted Values')
    plt.legend()
    
    # Plot for second variable
    plt.subplot(2, 2, 2)
    plt.plot(time_obs, y_obs[:, 1], 'b-', label='Observed')
    
    for b in range(n_bootstrap):
        if b == 0:
            plt.plot(time_future, results['forecasts'][b, :, 1], 'r-', alpha=alpha, label='Forecasts')
        else:
            plt.plot(time_future, results['forecasts'][b, :, 1], 'r-', alpha=alpha)
    
    plt.axvline(x=m-1, color='k', linestyle='--')
    plt.title('Variable 2: Observed and Forecasted Values')
    plt.legend()
    
    # 2. Plot distribution of A parameters
    plt.subplot(2, 2, 3)
    A_params = [results['A_samples'][:, 0, 0], results['A_samples'][:, 0, 1],
                results['A_samples'][:, 1, 0], results['A_samples'][:, 1, 1]]
    labels = ['a11', 'a12', 'a21', 'a22']
    
    plt.boxplot(A_params, labels=labels)
    
    if true_A is not None:
        true_values = [true_A[0, 0], true_A[0, 1], true_A[1, 0], true_A[1, 1]]
        plt.plot(range(1, 5), true_values, 'ro', label='True Values')
        plt.legend()
    
    plt.title('Distribution of A Parameters')
    
    # 3. Plot distribution of Sigma parameters
    plt.subplot(2, 2, 4)
    Sigma_params = [results['Sigma_samples'][:, 0, 0], results['Sigma_samples'][:, 0, 1],
                    results['Sigma_samples'][:, 1, 0], results['Sigma_samples'][:, 1, 1]]
    labels = ['σ11', 'σ12', 'σ21', 'σ22']
    
    plt.boxplot(Sigma_params, labels=labels)
    
    if true_Sigma is not None:
        true_values = [true_Sigma[0, 0], true_Sigma[0, 1], true_Sigma[1, 0], true_Sigma[1, 1]]
        plt.plot(range(1, 5), true_values, 'ro', label='True Values')
        plt.legend()
    
    plt.title('Distribution of Sigma Parameters')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("A Matrix Summary:")
    print("Mean:", results['A_mean'])
    print("95% Confidence Intervals:")
    for i in range(2):
        for j in range(2):
            print(f"A[{i},{j}]: ({results['A_quantiles'][i,j,0]:.4f}, {results['A_quantiles'][i,j,1]:.4f})")
    
    print("\nSigma Matrix Summary:")
    print("Mean:", results['Sigma_mean'])
    print("95% Confidence Intervals:")
    for i in range(2):
        for j in range(2):
            print(f"Sigma[{i},{j}]: ({results['Sigma_quantiles'][i,j,0]:.4f}, {results['Sigma_quantiles'][i,j,1]:.4f})")
    return
