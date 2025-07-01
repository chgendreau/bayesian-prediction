import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Constants for nicer names in plots
NICE_NAMES = {
    "VAR1_A_hat": "A",
    "VAR1_sigma_eps_hat": "$\\Sigma$",
    "resampling_var1_2d": "PR",
    "bayesian_bvar_analytical": "BVAR\nJeffreys",
    "bayesian_bvar_minnesota_wrong": "BVAR\nMinnesota\n(uninf.)",
    "bayesian_bvar_minnesota_chat_gpt": "BVAR\nMinnesota\n(ChatGPT)",
    "LP\nUninf. prior": "LP\nJeffreys",
    "PR\nStudent-T": "PR\nStudent-t",
    "LP\nWrong Normal L": "LP\n(wrong) Normal\n Likelihood",
}


def correct_names(df):
    """
    Replace method names in the DataFrame with more readable versions.
    If not found in NICE_NAMES, it will keep the original name.
    """
    df['method'] = df['method'].replace(NICE_NAMES)
    df['theta'] = df['theta'].replace(NICE_NAMES)
    return df


# LOADERS 
def load_1d_experiment(base_dir, experiment_name):
    """
    Load posterior samples for one experiment (all seeds) into a DataFrame and .

    Parameters
    ----------
    base_dir : str
        Path containing subfolders like "experiment_name, seed=0", "experiment_name, seed=1", …
    experiment_name : str
        The name of the experiment to gather.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        A tuple containing:
        - DataFrame with columns ['experiment', 'seed', 'theta', 'method', 'sample']
        - config_data: dictionary with the last config.json found in the experiment folder.
    """
    records = []
    # match folders exactly like "<experiment_name>, seed=<number>"
    pattern = re.compile(rf"^{re.escape(experiment_name)}, seed=(\d+)$")
    config_path = ""
    for folder in os.listdir(base_dir):
        m = pattern.match(folder)
        if not m:
            continue
        seed = int(m.group(1))
        json_path = os.path.join(base_dir, folder, "inferences.json")
        config_path = os.path.join(base_dir, folder, "experiment_config.json")  # updating config path so we get the last one
        if not os.path.isfile(json_path):
            continue

        with open(json_path, "r") as f:
            inf = json.load(f)

        for theta, methods in inf.items():
            for method, samples in methods.items():
                for idx, s in enumerate(samples):
                    records.append({
                        "experiment": experiment_name,
                        "seed": seed,
                        "theta": theta,
                        "method": method,
                        "sample_idx": idx,
                        "sample": s
                    })
    
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # improving names + dataframe structure
    df = correct_names(pd.DataFrame.from_records(records))
    
    # ensure same number of posterior samples for each method. Take only the first B samples if more are present.
    if 'B' in config_data:
        df = df[df['sample_idx'] < config_data['B']]

    return df, config_data


def load_2d_experiment(base_dir, experiment_name):
    """
    Load 2D posterior samples for one experiment (all seeds) into a DataFrame,
    storing each draw as a list of lists, plus the last experiment config.

    Returns
    -------
    df : pd.DataFrame
        Columns: ['experiment','seed','theta','method','sample_idx','sample']
        where 'sample' is always a list of lists (a matrix).
    config_data : dict
        JSON from the last experiment_config.json found.
    """
    records = []
    pattern = re.compile(rf"^{re.escape(experiment_name)}, seed=(\d+)$")
    config_path = None

    for folder in os.listdir(base_dir):
        m = pattern.match(folder)
        if not m:
            continue
        seed = int(m.group(1))
        inf_path = os.path.join(base_dir, folder, "inferences.json")
        cfg_path = os.path.join(base_dir, folder, "experiment_config.json")
        if os.path.isfile(cfg_path):
            config_path = cfg_path
        if not os.path.isfile(inf_path):
            continue

        with open(inf_path, "r") as f:
            inf = json.load(f)

        for theta_name, methods in inf.items():
            for method_name, samples in methods.items():
                # ensure we have a list to iterate
                if not isinstance(samples, list):
                    samples = [samples]

                for idx, sam in enumerate(samples):
                    # if this draw isn't already a matrix, wrap into 1×1
                    if not (isinstance(sam, list) and all(isinstance(row, list) for row in sam)):
                        sam = [[sam]]
                    records.append({
                        "experiment":  experiment_name,
                        "seed":        seed,
                        "theta":       theta_name,
                        "method":      method_name,
                        "sample_idx":  idx,
                        "sample":      sam
                    })

    # load last config, if any
    config_data = {}
    if config_path:
        with open(config_path, "r") as f:
            config_data = json.load(f)

    # improving names + dataframe structure
    df = correct_names(pd.DataFrame.from_records(records))

    # ensure same number of posterior samples for each method. Take only the first B samples if more are present.
    if 'B' in config_data:
        df = df[df['sample_idx'] < config_data['B']]

    return df, config_data


# BOXPLOTS
def plot_boxplots_by_seed(df, theta_name):
    dfθ = df[df['theta'] == theta_name]
    seeds   = sorted(dfθ['seed'].unique())
    methods = dfθ['method'].unique()

    # Collect data in order: for each seed, for each method
    data, labels = [], []
    for seed in seeds:
        for method in methods:
            samp = dfθ[(dfθ['seed']==seed) & (dfθ['method']==method)]['sample'].values
            data.append(samp)
            labels.append(f"{method}\n(seed={seed})")

    plt.figure(figsize=(max(10, len(labels)*0.4), 6))
    plt.boxplot(data, labels=labels, notch=True)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Comparison of {theta_name} posteriors across seeds & methods")
    plt.xlabel("Method (seed)")
    plt.ylabel("Sample value")
    plt.tight_layout()
    plt.show()


def get_y_range(df_theta, quantile=0.95):
    """
    Compute y-limits for boxplots of a given parameter, ignoring outliers.

    Parameters
    ----------
    df_theta : pd.DataFrame
        Subset of your full DataFrame with df_theta['theta'] == parameter_name.
        Must contain columns ['method','seed','sample'].
    quantile : float, optional
        Upper quantile to include (default 0.95). Lower quantile is 1-quantile.
    """
    lower_q = 1 - quantile
    grouped = df_theta.groupby(['method', 'seed'])['sample']
    lows = grouped.quantile(lower_q)
    highs = grouped.quantile(quantile)
    y_min = lows.min()
    y_max = highs.max()
    return y_min, y_max


def plot_box_overlap(df, theta_name, alpha=0.1, width_per_method=1.2,
                     height=4, true_theta=None):
    """
    For a given parameter, draw:
      - one semi-transparent box per seed (grey, alpha)
      - one bold box aggregating all seeds (black)
      - optional dashed line at the true parameter value
    Figure width scales with number of methods, and y-limits ignore outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['theta','method','seed','sample']
    theta_name : str
        The parameter to plot.
    alpha : float, optional
        Transparency for seed-wise boxes (default 0.1).
    width_per_method : float, optional
        Inches of figure width per method (default 1.2).
    height : float, optional
        Figure height in inches (default 4).
    true_theta : float, optional
        If provided, draw a dashed black line at this value.
    """
    df_param = df[df['theta'] == theta_name]
    methods = df_param['method'].unique()
    n_methods = len(methods)

    fig_width = max(6, n_methods * width_per_method)
    fig, ax = plt.subplots(figsize=(fig_width, height))

    positions = np.arange(n_methods)

    # seed-wise boxes
    all_seeds = df_param['seed'].unique()
    eps = 0.3 * width_per_method / len(all_seeds)  # small width offset to separate boxes for different seeds
    for i, method in enumerate(methods):
        df_m = df_param[df_param['method'] == method]
        for j, seed in enumerate(df_m['seed'].unique()):
            samp = df_m[df_m['seed'] == seed]['sample']
            ax.boxplot(
                samp,
                positions=[i],
                widths=width_per_method * 0.3 + (j + 1) * eps,
                boxprops={'color': 'red', 'alpha': alpha},
                whiskerprops={'color': 'grey', 'alpha': 0},
                capprops={'color': 'blue', 'alpha': alpha},
                medianprops={'color': 'orange', 'alpha': 0},
                flierprops={'marker': '', 'alpha': alpha},
                showfliers=False,
            )

    # aggregate boxes with thicker lines
    agg_color = 'black'
    agg_alpha = 0.5
    agg_linewidth = 1.5
    for i, method in enumerate(methods):
        df_m = df_param[df_param['method'] == method]
        all_samp = df_m['sample']
        ax.boxplot(
            all_samp,
            positions=[i],
            widths=width_per_method * 0.3,
            boxprops={'color': agg_color, 'linewidth': agg_linewidth, 'alpha': agg_alpha},
            whiskerprops={'color': agg_color, 'linewidth': agg_linewidth, 'alpha': agg_alpha},
            capprops={'color': agg_color, 'linewidth': agg_linewidth, 'alpha': agg_alpha},
            medianprops={'color': agg_color, 'linewidth': agg_linewidth, 'alpha': agg_alpha},
            flierprops={'marker': '', 'alpha': 0},
            showfliers=False
        )

    # true parameter line
    if true_theta is not None:
        ax.axhline(true_theta, color='black', linestyle='--', linewidth=1.5)

    # labels
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel(f"Posterior {theta_name}", fontsize=11)

    # # apply y-range ignoring outliers
    # y_min, y_max = get_y_range(df_param, quantile=quantile)
    # pad = (y_max - y_min) * margin_ratio
    # ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(True, linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    return fig, ax
