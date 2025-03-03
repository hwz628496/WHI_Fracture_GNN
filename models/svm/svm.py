import numpy as np
from sklearn.svm import SVC


def get_svm():
    cv_avail = True

    model = SVC(probability=True, random_state=42)
    param_dist = {
        'C': np.logspace(-3, 2, 10),  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 5)),  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        'degree': [2, 3, 4]  # Degree of polynomial kernel (only used for 'poly' kernel)
    }
    log_path = "models/svm/svm"
    scoring = "roc_auc"

    return model, cv_avail, param_dist, log_path, scoring
