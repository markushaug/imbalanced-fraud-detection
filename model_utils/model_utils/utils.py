# load libraries
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# set seed
SEED=123
np.random.seed(SEED)

# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def imb_ratio(value_counts):
    """
    Calculates the imbalance ratio (IR) between two values based on a dataframe value_counts.
    The majority class is automatically identified and used as the numerator.
    
    Parameters:
    value_counts (tuple): A tuple containing the counts of the two classes.
    
    Returns:
    float: The imbalance ratio between two classes.
    """
    a, b = value_counts
    return round(max(a,b) / min(a,b), 2)

def get_device():
    """
    Returns the device (cuda or cpu) that will be used throughout the notebook.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_cm(cm):
    """
    Plot a confusion matrix using seaborn's heatmap.

    Parameters:
    cm (array): A confusion matrix.
    """
    plt.figure(figsize=(5, 2))  # You can adjust the size to fit your needs
    sns.set(font_scale=1)  # Adjust to make labels larger or smaller
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thres = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label = 'AUC') 
    plt.plot([0,1], [0,1], ':', label = 'Random') 
    plt.legend() 
    plt.grid() 
    plt.ylabel("TPR") 
    plt.xlabel("FPR") 
    plt.title('ROC') 
    plt.show()
    
def plot_auc_prc_curve(y_true, y_pred):
    precision_array, recall_array, _ = precision_recall_curve(y_true, y_pred)
    auc_prc = auc(recall_array, precision_array)
    
    # Plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall_array, precision_array, label=f'AUC-PRC = {auc_prc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

