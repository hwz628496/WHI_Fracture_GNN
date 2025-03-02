from sklearn.linear_model import LogisticRegression
import numpy as np



def get_lasso():
    #return only features that relate to random forest
    cv_avail = True

    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000) 
    param_dist = {
        'C': np.logspace(-4, 1, 50)  
    }
    log_path = "models/lasso/lasso"
    scoring = "roc_auc"

    return model, cv_avail, param_dist, log_path, scoring
