# Imbalanced Fraud Detection

This repository contains the code for my thesis "Empirical Evaluation of Machine Learning Models and Ensemble Methods for Imbalanced Learning and Anomaly Detection" on the topic of imbalanced fraud detection. Despite the thesis is written in German, the code and repository are in English.

# Table of Contents

- [Repository Structure](#repository-structure)
- [Installation & Requirements](#installation--requirements)
- [Datasets](#datasets)
- [Code](#code)

# Repository Structure

```
├── data
│   ├── creditcard
│   └── kddcup
├── model_utils
├── models
├── requirements.txt
```

# Installation & Requirements

A `requirements.txt` file is provided in the root directory of the repository. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

This will also install the local package `model_utils` which is used for the unified evaluation of all models. 

NOTE: The `model_utils` package is not yet available on PyPi. A virtual environment is also recommended for the usage of this repository.

# Datasets

Only the Credit Card Fraud 2013 dataset is included in the repository. However, the original dataset is in the `Rdata` format and must be converted to a `CSV` file first using the provided R-script `Rdata2CSV.ipynb`. Preprocessing can be done using the provided Jupyter Notebook `CC_preprocessing.ipynb`. 

For the KDDCUP99-dataset, preprocessing can be done using the provided Jupyter Notebook `KDDCUP_preprocessing.ipynb`. The dataset `corrected.gz` can be downloaded from the [UCI KDD Archive](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html). Prepare the dataset by running the provided Jupyter Notebook `KDD_preprocessing.ipynb`.

# Code

The `models` directory contains the code for all 18 models, sampling, boosting and imputation methods. The entire code is provided in Jupyter Notebooks.
