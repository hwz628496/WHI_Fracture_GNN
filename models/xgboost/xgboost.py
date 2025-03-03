from xgboost import XGBClassifier
import numpy as np

def get_xgboost():
    cv_avail = True

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method = "hist", device = "cuda:0")
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': np.logspace(-3, 0, 10),  # From 0.001 to 1.0
        'subsample': np.linspace(0.6, 1.0, 5),  # Fraction of samples per tree
        'colsample_bytree': np.linspace(0.6, 1.0, 5),  # Fraction of features per tree
        'gamma': np.logspace(-3, 1, 5),  # Minimum loss reduction for a split
        'reg_alpha': np.logspace(-4, 1, 10),  # L1 regularization (like Lasso)
        'reg_lambda': np.logspace(-4, 1, 10)  # L2 regularization (like Ridge)
    }
    log_path = "models/xgboost/xgboost"
    scoring = "roc_auc"

    return model, cv_avail, param_dist, log_path, scoring
