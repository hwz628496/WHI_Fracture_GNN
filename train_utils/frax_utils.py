import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def FRAX_maximize_youden_j(y_true: pd.Series, y_prob: pd.Series) -> float:
    """
    Finds the optimal threshold that maximizes Youden’s J Statistic (TPR - FPR).

    :param y_true: Pandas Series of true binary labels (0 or 1).
    :param y_prob: Pandas Series of predicted probabilities.
    :return: The optimal threshold for classification.
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Youden’s J statistic
    j_scores = tpr - fpr

    #optimal threshold (maximum J score)
    best_threshold = thresholds[np.argmax(j_scores)]
    print(f"Optimal Threshold (Max Youden's J): {best_threshold:.4f}")

    return best_threshold

