# Predictive Approach to Bayesian Inference

This repository contains the code and notebooks used for the numerical experiments of the Master Thesis project at EPFL during the Spring semester of 2025. It includes all pipelines, configuration files, and Jupyter notebooks necessary to reproduce the results, as well as scripts to generate the final report plots. The results are stored here

**Author:**
* **Charles Gendreau**

**Supervisors:**

* **Prof. Anthony C. Davison**, École Polytechnique Fédérale de Lausanne (EPFL)
* **Dr. Riccardo Passeggeri**, Imperial College London

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/chgendreau/bayesian-prediction.git
   cd bayesian-prediction
   ```

2. Install dependencies using Poetry (recommended):

   ```bash
   poetry install
   ```

3. (Optional) Activate the virtual environment:

   ```bash
   poetry shell
   ```

## Configuration

All experiment settings (hyperparameters, data paths, etc.) are defined in `src/config.py`. 

## Running Pipelines

Execute any of the three pipelines to run the corresponding experiments:

```bash
# 1D experiments
poetry run python -m src.inference_pipeline_1D.py

# VAR experiments
poetry run python -m src.inference_pipeline_var1.py

# Choose N experiments
Call chooseN_pipeline from a Notebook.
```

Results (e.g., summary tables, figures) will be saved in the `inference_results/` directory by default.

## Machine Learning Experiments

Open and run the `machine_learning.ipynb` notebook to perform the computations on the Machine Learning section. 

## Generating Report Plots

After running the pipelines, use `report_plots.ipynb` to recreate all plots found in the final thesis report. This notebook reads results from the `inference_results/` folder and generates the figures.


