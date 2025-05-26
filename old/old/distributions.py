import numpy as np
from typing import Callable


def inverse_transform_sampling_numerical(
        cdf: Callable,
        x_min: float,
        x_max: float,
        size: int = 1000,
        num_points=10000,
):
    """Generate samples using numerical approximation of inverse CDF.
 
    Args:
        cdf: The cumulative distribution function
        x_min: Minimum value of the support
        x_max: Maximum value of the support
        size: Number of samples to generate
        num_points: Number of points for numerical approximation
    """
    # Create grid of x values
    x = np.linspace(x_min, x_max, num_points)

    # Compute CDF values
    cdf_values = cdf(x)
    cdf_values /= cdf_values[-1]  # Normalize to 1
    
    # Generate uniform random numbers
    u = np.random.uniform(size=size)
    
    # Inverse sampling by interpolation
    return np.interp(u, cdf_values, x)
