from sklearn.linear_model import LogisticRegression
import numpy as np


def get_elasticnet():
    cv_avail = True

    model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000)
    param_dist = {
        'C': np.logspace(-4, 1, 50),  
        'l1_ratio': np.linspace(0, 1, 10) 
    }
    log_path = "models/elasticnet/elasticnet"
    scoring = "roc_auc"

    return model, cv_avail, param_dist, log_path, scoring
