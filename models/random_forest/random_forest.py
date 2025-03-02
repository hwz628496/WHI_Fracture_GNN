import scipy
from sklearn.ensemble import RandomForestClassifier


def get_random_forest(random_state = 42):

    #return only features that relate to random forest
    cv_avail = True

    param_dist = {
    'n_estimators': scipy.stats.randint(100, 500),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': scipy.stats.randint(2, 20),
    'min_samples_leaf': scipy.stats.randint(1, 10),
    'max_features': ['sqrt', 'log2']
    }

    model = RandomForestClassifier(random_state = random_state)
    log_path = "models/random_forest/random_forest"
    scoring = "accuracy"

    return model, cv_avail, param_dist, log_path, scoring