import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    auc,
    classification_report,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score
import pandas as pd
from imbens.ensemble import SelfPacedEnsembleClassifier
from scikeras.wrappers import KerasClassifier
from xgboost import XGBClassifier
from keras import Sequential, Model
from IPython.display import display

SEED = 123
np.random.seed(SEED)

# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def assert_dicts_almost_equal(dict1, dict2, precision=0.3):
    assert dict1.keys() == dict2.keys(), "Keys do not match."
    for key in dict1:
        if isinstance(dict1[key], float):
            assert (
                abs(dict1[key] - dict2[key]) < precision
            ), f"Mismatch at {key}: {dict1[key]} vs {dict2[key]}"
        else:
            assert (
                dict1[key] == dict2[key]
            ), f"Mismatch at {key}: {dict1[key]} vs {dict2[key]}"


def table(metrics, sort_by="AUCPRC", asc=False):

    columns = [
        "Model Name",
        "AUCPRC",
        "F1",
        "G-Mean",
        "MCC",
        "Precision",
        "Recall",
        "ROCAUC",
        "ACCURACY",
        "TP",
        "FP",
        "TN",
        "FN",
    ]

    # Mapping your dictionary keys to the names used in the 'columns' list
    column_mapping = {
        "model_name": "Model Name",
        "AUCPRC": "AUCPRC",
        "ROCAUC": "ROCAUC",
        "F1": "F1",
        "GMEAN": "G-Mean",
        "MCC": "MCC",
        "ACC": "ACCURACY",
        "tp": "TP",
        "fp": "FP",
        "tn": "TN",
        "fn": "FN",
        "precision": "Precision",
        "recall": "Recall",
    }

    results_df = pd.DataFrame.from_dict(metrics)

    # Rename the columns using the mapping
    results_df.rename(columns=column_mapping, inplace=True)

    # Reorder the columns as specified in the 'columns' list
    results_df = results_df.loc[:, columns]

    results_df.sort_values(by=sort_by, ascending=asc, inplace=True)

    # Display results in a table without index
    display(results_df)


def get_metrics(y_true, y_scores, threshold=0.5, op=">"):
    # y_scores must be 1d-array
    if y_scores.ndim != 1:
        print(
            "y_scores must be a 1D array (flat). use [:, 1] on sklearn predict scores or .flatten() on keras scores"
        )

    if op == ">":
        y_preds = (y_scores > threshold).astype(int)
    elif op == "<":
        y_preds = (y_scores < threshold).astype(int)
    else:
        raise ValueError("Please use either < or > as operator.")

    # create dictionary
    metrics = {}

    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel().astype("float32")

    # add to dic
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    metrics["precision"] = round(precision, 4)
    metrics["recall"] = round(recall, 4)

    precision_array, recall_array, _ = precision_recall_curve(y_true, y_scores)
    metrics["AUCPRC"] = round(auc(recall_array, precision_array), 4)

    metrics["F1"] = round(fbeta_score(y_true, y_preds, beta=1), 4)

    metrics["ROCAUC"] = round(roc_auc_score(y_true, y_scores), 4)

    metrics["MCC"] = round(matthews_corrcoef(y_true, y_preds), 4)

    metrics["ACC"] = round(accuracy_score(y_true, y_preds), 4)

    metrics["GMEAN"] = round(geometric_mean_score(y_true, y_preds), 4)

    return metrics


def evaluate_model(model, X, y_true, threshold=0.5, op=">", names=None, as_table=False):
    # If model is not a list, convert it to a list
    if not isinstance(model, list):
        model = [model]

    # Check if names are provided and match the number of models
    if names is not None and len(names) != len(model):
        raise ValueError("The length of 'names' must match the number of models.")

    results = []

    for idx, m in enumerate(model):

        if isinstance(m, XGBClassifier) or isinstance(m, Pipeline):
            m_scores = m.predict_proba(X)[:, 1]  # Assuming binary classification
            metrics = get_metrics(y_true, m_scores, threshold=threshold, op=op)

        elif isinstance(m, KerasClassifier):
            m_scores = m.predict_proba(X, verbose=0)[:, 1]  # Assuming binary classification
            metrics = get_metrics(y_true, m_scores, threshold=threshold, op=op)

        elif isinstance(m, SelfPacedEnsembleClassifier):
            m_scores = m.predict_proba(X)[:, 1]  # Assuming binary classification
            metrics = get_metrics(y_true, m_scores, threshold=threshold, op=op)

        elif isinstance(m, Sequential) or isinstance(m, Model):
            m_scores = m.predict(X, verbose=0).flatten()
            metrics = get_metrics(y_true, m_scores, threshold=threshold, op=op)

        else:
            # Model not supported
            raise TypeError("Model of type ", type(m), " not supported")

        metrics["model_name"] = names[idx] if names else m.__class__.__name__

        results.append(metrics)

    if as_table:
        return table(results, sort_by="AUCPRC", asc=False)

    return results
