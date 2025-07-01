from matplotlib.pylab import eig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union, List, Tuple, Optional
import os


def plot_posterior_boxplots(
    theta_samples: Dict[str, np.ndarray], 
    param_name: str = "",
    true_theta: Optional[Union[float, np.ndarray]] = None,
    theta_hat_obs: Optional[Union[float, np.ndarray]] = None,
    title: str = "",
    xlabel: str = "Value",
    figsize_base: tuple = (8, 4),
    show_method_names: bool = True,
    method_name_pos: str = "left",  # 'left', 'above', or 'below'
    whisker_width: float = 1.5,
    showfliers: bool = True
) -> plt.Figure:
    """
    Plot horizontal boxplots for parameter samples from different inference methods.
    
    Args:
        theta_samples: Dictionary with method names as keys and sample arrays as values
        param_name: Name of the parameter being plotted
        true_theta: True parameter value(s) to mark with vertical line(s)
        theta_hat_obs: Observed estimated value of the parameter (optional)
        title: Plot title (if empty, will use param_name)
        xlabel: Label for the x-axis
        figsize_base: Base figure size (width, height) - will be adjusted for number of methods
        colors: Optional list of colors for the boxplots
        show_method_names: Whether to show method names
        method_name_pos: Position of method names ('left', 'above', 'below')
        whisker_width: Width of the boxplot whiskers (in IQR units)
        showfliers: Whether to show outliers
        
    Returns:
        The matplotlib figure object for further customization
    """
    # Determine number of methods and adjust figure size
    n_methods = len(theta_samples)
    figsize = (figsize_base[0], figsize_base[1] + 0.5 * n_methods)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for boxplot removing nan
    method_names = list(theta_samples.keys())
    samples_list = [theta_samples[method][~np.isnan(theta_samples[method])] for method in method_names]

    positions = np.arange(n_methods, 0, -1)  # Positions for boxplots
    
    # Create boxplots
    boxplots = ax.boxplot(
        samples_list, 
        vert=False, 
        patch_artist=True,
        positions=positions,
        widths=0.6,
        whis=whisker_width,
        showfliers=showfliers
    )
    
    # Set colors for boxplots
    for i, box in enumerate(boxplots['boxes']):
        # box.set(facecolor=box_color, alpha=0.6)
        box.set(alpha=0.6)
        
        # Also set the color for whiskers, caps, and median
        # for whisker in boxplots['whiskers'][i*2:i*2+2]:
        #     whisker.set(color=box_color, linewidth=1.5)
        # for cap in boxplots['caps'][i*2:i*2+2]:
        #     # cap.set(color=box_color, linewidth=1.5)
        #     cap.set(linewidth=1.5)
        # boxplots['medians'][i].set(color='black', linewidth=2)
    
    # Add method names 
    if show_method_names:
        ax.set_yticks(positions)
        ax.set_yticklabels(method_names)
    else:
        # Remove y-ticks if not showing method names
        ax.set_yticks([])
    
    # Add true value if provided
    if true_theta is not None:
        if isinstance(true_theta, (int, float, np.number)):
            # Single scalar value
            ax.axvline(x=true_theta, color='black', linestyle='--', linewidth=2, 
                      label='True $\\theta$')
            # ax.legend(loc='upper right')
        elif isinstance(true_theta, np.ndarray) and true_theta.ndim == 1:
            # Vector of true values - add all as vertical lines
            for i, val in enumerate(true_theta):
                ax.axvline(x=val, color='black', linestyle='--', linewidth=2, 
                          label='True $\\theta$' if i == 0 else "")
            # ax.legend(loc='upper right')

    # Add observed estimate if provided
    if theta_hat_obs is not None:
        if isinstance(theta_hat_obs, (int, float, np.number)):
            # Single scalar value
            ax.axvline(x=theta_hat_obs, color='red', linestyle=':', linewidth=1, 
                      label='$\\hat{\\theta}_{obs}$')
            # ax.legend(loc='upper left')
        elif isinstance(theta_hat_obs, np.ndarray) and theta_hat_obs.ndim == 1:
            # Vector of observed estimates - add all as vertical lines
            for i, val in enumerate(theta_hat_obs):
                ax.axvline(x=val, color='red', linestyle=':', linewidth=1, 
                          label='$\\hat{\\theta}_{obs}$' if i == 0 else "")
            # ax.legend(loc='upper left')
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    elif param_name:
        ax.set_title(f"Posterior Distribution of {param_name}", fontsize=14, pad=10)
        
    ax.set_xlabel(xlabel, fontsize=11)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig, ax


def plot_posterior_eigvals_boxplots(
    theta_samples: Dict[str, np.ndarray], 
    true_theta: Optional[Union[float, np.ndarray]] = None,
    theta_hat_obs: Optional[Union[float, np.ndarray]] = None,
    figsize_base: tuple = (12, 4),
    whisker_width: float = 1.5,
    showfliers: bool = True
) -> plt.Figure:
    """
    Plot horizontal boxplots for parameter samples from different inference methods.
    
    Args:
        theta_samples: Dictionary with method names as keys and sample arrays of matrices as values
        param_name: Name of the parameter being plotted
        true_theta: True parameter matrix-value (optional)
        theta_hat_obs: Observed estimated value of the parameter (optional)
        figsize_base: Base figure size (width, height) - will be adjusted for number of methods
        colors: Optional list of colors for the boxplots
        show_method_names: Whether to show method names
        method_name_pos: Position of method names ('left', 'above', 'below')
        whisker_width: Width of the boxplot whiskers (in IQR units)
        showfliers: Whether to show outliers
        
    Returns:
        The matplotlib figure object for further customization
    """
    # Determine number of methods and adjust figure size
    n_methods = len(theta_samples)
    figsize = (figsize_base[0], figsize_base[1] + 0.5 * n_methods)
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for idx, ax in enumerate(axes):
        # Prepare data for boxplot
        method_names = list(theta_samples.keys())
        eigval_i_samples = {
            method: np.real(np.linalg.eigvals(theta_samples[method])[:, idx])
            for method in theta_samples.keys()
        }
        samples_list = [eigval_i_samples[method] for method in method_names]
        positions = np.arange(n_methods, 0, -1)  # Positions for boxplots

        # Create boxplots
        boxplots = ax.boxplot(
            samples_list, 
            vert=False, 
            patch_artist=True,
            positions=positions,
            widths=0.6,
            whis=whisker_width,
            showfliers=showfliers,
        )

        # remove ticks to put custom ones
        ax.set_yticks([])
    
        # Set add transparency
        for box in boxplots['boxes']:
            box.set(alpha=0.6)
        
        # Add true value if provided
        if true_theta is not None:
            true_eigval_i = np.linalg.eigvals(true_theta)[idx]
            ax.axvline(x=true_eigval_i, color='black', linestyle='--', linewidth=2, 
                label='True value')

        # Add observed estimate if provided
        if theta_hat_obs is not None:
            obs_eigval_i = np.linalg.eigvals(theta_hat_obs)[idx]
            ax.axvline(x=obs_eigval_i, color='red', linestyle=':', linewidth=1.5, 
                label='Obs. value')
    
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # add ticks with method names
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(method_names)
    
    return fig, axes




##########################################################
# Functions for KDE or histogram plots of posterior distributions
###########################################################
def plot_posterior_distributions(
    theta_samples: Dict[str, np.ndarray],
    true_theta: Union[float, None] = None,
    theta_hat_obs: Union[float, None] = None,
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
):
    """
    Plot histogram and/or density plots for multiple sets of theta samples.
    
    Parameters:
    -----------
    theta_samples : Dict[str, np.ndarray]
        Dictionary with inference method names as keys and theta samples as values
    true_theta : float or None
        True value of theta to be plotted as vertical line
    theta_hat_obs : float or None
        Observed estimated value of theta to be plotted as vertical line
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
        x_min, x_max = np.nanmin(all_samples), np.nanmax(all_samples)
        x_range = x_max - x_min
        x_min -= x_range * 0.1  # Add 10% padding
        x_max += x_range * 0.1
    else:
        x_min, x_max = x_range
    
    for method, samples in theta_samples.items():
        if plot_type in ['hist', 'both']:
            plt.hist(samples, bins=bins, density=True, alpha=alpha, color=colors[method],
                    range=(x_min, x_max))  # Limit histogram range
            
        if plot_type in ['density', 'both']:
            sns.kdeplot(data=samples, label=f'{method}',
                       color=colors[method], linewidth=2,
                       clip=(x_min, x_max))  # Limit density range
    
    # Plot true theta if provided
    if true_theta is not None:
        plt.axvline(true_theta, color='black', linestyle='--', linewidth=2,
                    label='True θ')
    # Plot observed estimate if provided
    if theta_hat_obs is not None:
        plt.axvline(theta_hat_obs, color='red', linestyle=':', linewidth=1,
                    label='Observed θ')
    
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


# # Frequentist view of VAR(1) Posterior Plotting Function
# def plot_predictive_resampling_results(y_obs: np.ndarray, results: Dict, true_A: np.ndarray = None, 
#                                       true_Sigma: np.ndarray = None, alpha: float = 0.1):
#     """
#     Plot the results of predictive resampling
    
#     Parameters:
#     -----------
#     y_obs : np.ndarray, shape (m, 2)
#         Observed time series
#     results : Dict
#         Results dictionary from predictive_resampling function
#     true_A : np.ndarray, optional
#         True A matrix if known (for comparison)
#     true_Sigma : np.ndarray, optional
#         True Sigma matrix if known (for comparison)
#     alpha : float
#         Transparency of forecast paths
#     """
#     m = len(y_obs)
#     n_future = results['forecasts'].shape[1]
#     n_bootstrap = results['forecasts'].shape[0]
#     total_time = m + n_future
    
#     # Create time indices
#     time_obs = np.arange(m)
#     time_future = np.arange(m, total_time)
    
#     # 1. Plot time series and forecasts
#     plt.figure(figsize=(15, 10))
    
#     # Plot for first variable
#     plt.subplot(2, 2, 1)
#     plt.plot(time_obs, y_obs[:, 0], 'b-', label='Observed')
    
#     for b in range(n_bootstrap):
#         if b == 0:
#             plt.plot(time_future, results['forecasts'][b, :, 0], 'r-', alpha=alpha, label='Forecasts')
#         else:
#             plt.plot(time_future, results['forecasts'][b, :, 0], 'r-', alpha=alpha)
    
#     plt.axvline(x=m-1, color='k', linestyle='--')
#     plt.title('Variable 1: Observed and Forecasted Values')
#     plt.legend()
    
#     # Plot for second variable
#     plt.subplot(2, 2, 2)
#     plt.plot(time_obs, y_obs[:, 1], 'b-', label='Observed')
    
#     for b in range(n_bootstrap):
#         if b == 0:
#             plt.plot(time_future, results['forecasts'][b, :, 1], 'r-', alpha=alpha, label='Forecasts')
#         else:
#             plt.plot(time_future, results['forecasts'][b, :, 1], 'r-', alpha=alpha)
    
#     plt.axvline(x=m-1, color='k', linestyle='--')
#     plt.title('Variable 2: Observed and Forecasted Values')
#     plt.legend()
    
#     # 2. Plot distribution of A parameters
#     plt.subplot(2, 2, 3)
#     A_params = [results['A_samples'][:, 0, 0], results['A_samples'][:, 0, 1],
#                 results['A_samples'][:, 1, 0], results['A_samples'][:, 1, 1]]
#     labels = ['a11', 'a12', 'a21', 'a22']
    
#     plt.boxplot(A_params, labels=labels)
    
#     if true_A is not None:
#         true_values = [true_A[0, 0], true_A[0, 1], true_A[1, 0], true_A[1, 1]]
#         plt.plot(range(1, 5), true_values, 'ro', label='True Values')
#         plt.legend()
    
#     plt.title('Distribution of A Parameters')
    
#     # 3. Plot distribution of Sigma parameters
#     plt.subplot(2, 2, 4)
#     Sigma_params = [results['Sigma_samples'][:, 0, 0], results['Sigma_samples'][:, 0, 1],
#                     results['Sigma_samples'][:, 1, 0], results['Sigma_samples'][:, 1, 1]]
#     labels = ['σ11', 'σ12', 'σ21', 'σ22']
    
#     plt.boxplot(Sigma_params, labels=labels)
    
#     if true_Sigma is not None:
#         true_values = [true_Sigma[0, 0], true_Sigma[0, 1], true_Sigma[1, 0], true_Sigma[1, 1]]
#         plt.plot(range(1, 5), true_values, 'ro', label='True Values')
#         plt.legend()
    
#     plt.title('Distribution of Sigma Parameters')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print summary statistics
#     print("A Matrix Summary:")
#     print("Mean:", results['A_mean'])
#     print("95% Confidence Intervals:")
#     for i in range(2):
#         for j in range(2):
#             print(f"A[{i},{j}]: ({results['A_quantiles'][i,j,0]:.4f}, {results['A_quantiles'][i,j,1]:.4f})")
    
#     print("\nSigma Matrix Summary:")
#     print("Mean:", results['Sigma_mean'])
#     print("95% Confidence Intervals:")
#     for i in range(2):
#         for j in range(2):
#             print(f"Sigma[{i},{j}]: ({results['Sigma_quantiles'][i,j,0]:.4f}, {results['Sigma_quantiles'][i,j,1]:.4f})")
#     return
