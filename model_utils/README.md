# model_utils

`model_utils` is a python package that provides utilities for machine learning models. Especially it provides an easy way to evaluate models with the following metrics:

- AUCPRC
- F1
- G-Mean
- Precision
- Recall
- Accuracy
- TP, FP, TN, FN


# Installation

You can install the package with `pip` or `conda`.

### install with pip
```bash
pip install -e .
```

### uninstall with pip

```bash
pip uninstall model_utils
```

### install with conda

```bash
conda develop .
``` 

### uninstall with conda

```bash
conda develop -u .
```

# Basic Usage

```python
from model_utils.evaluation import evaluate_model

evaluate_model([my_keras_model, my_xgb_classifier], X_test, y_test, threshold=0.5, as_table=True)
```

If `as_table` is set to `True`, the function will display a table in Jupyter Notebook. If it is set to `False`, the function will return a dictionary with the metrics. A threshold can be set to calculate the metrics and the operator direction can be set with the `op` parameter. The default is `>`.

# Run tests

Navigate to the root directory of the package and run the following command:

```bash
pytest
```