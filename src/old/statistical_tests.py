# Tests for various properties of data
import pandas as pd
import numpy as np
from scipy.stats import shapiro


# Normality tests
def normality_test(data):
    """
    Shapiro-Wilk test for normality
    :param data: data to be tested
    :return: p-value
    """
    return shapiro(data)[1]


# Exchangeability tests:
def exchangeability_test(data, k: int = 10**5, seed: int = None):
    """
    Test for exchangeability of data using Runs up and down test
    :param data: data to be tested
    :param k: number of permutations
    :param seed: random seed
    :return: p-value
    """
    # Define a function to calculate the runs for an input vector
    def runs(x):
        n = len(x)
        S = np.zeros(n-1, dtype=bool)
        for i in range(n-1):
            S[i] = x[i+1] >= x[i]
        return 1 + np.sum(S[:-1] != S[1:])

    # Simulate the runs statistic for k permutations
    np.random.seed(seed)
    RR = np.zeros(k, dtype=int)
    for i in range(k):
        x_perm = np.random.permutation(data)
        RR[i] = runs(x_perm)

    # Generate the frequency table for the simulated runs
    FREQS = pd.DataFrame({'RR': RR}).value_counts().reset_index(name='Freq')
    # Calculate the p-value of the runs test
    R = runs(data)
    R_FREQ = FREQS.loc[FREQS['RR'] == R, 'Freq'].values[0]
    p = np.sum(FREQS['Freq'] * (FREQS['Freq'] <= R_FREQ)) / k

    return p
