from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pandas as pd

from train_utils.weighted_downsample import weighted_downsample_LABELS
from train_utils.eval import eval_frax, eval_run

n_splits = 5
random_state = 45
#downsample
target_ratio = 0.5

def train(model = None, 
          random_cv_avail: bool = False, 
          scoring = None, 
          param_dist = None,
          dataset: pd.DataFrame = None, 
          labels: pd.Series = None, 
          n_splits = 5, 
          random_state = 45, 
          target_ratio = 0.5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        print(f"Fold {fold+1}/{n_splits}--------------------------")

        # Split train and test sets
        X_train, X_test = dataset.iloc[train_idx], dataset.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        
        # Perform weighted downsampling on training data
        X_train_balanced, y_train_balanced = weighted_downsample_LABELS(X_train, y_train, target_ratio = 0.3)
        print("Training Fold Label Distr:", dict(pd.Series(y_train_balanced).value_counts()))

        #get frax after downsample
        frax_score_train, frax_score_test = X_train_balanced["FRAX_SCORE"], X_test["FRAX_SCORE"]
        X_train_balanced, X_test = X_train_balanced.drop(columns = ["FRAX_SCORE"]), X_test.drop(columns = ["FRAX_SCORE"])

        if random_cv_avail == True:
            random_search = RandomizedSearchCV(model, param_distributions = param_dist, n_iter=20, cv=5, scoring = scoring, n_jobs=-1, random_state=42)
            random_search.fit(X_train_balanced, y_train_balanced)
            print("Best Parameters:", random_search.best_params_)
            model = random_search.best_estimator_

        eval_run(model, X_train_balanced, y_train_balanced, descr = "MODEL Train:")
        FRAX_threshold = eval_frax(frax_score_train, y_train_balanced, descr = "FRAX Train:")
        
        print("Test Fold Label Distr:", dict(pd.Series(y_test).value_counts()))

        # Evaluate on the original test set (without downsampling)
        eval_run(model, X_test, y_test, descr = "MODEL Test:")
        eval_frax(frax_score_test, y_test, descr = "FRAX Test:", threshold = FRAX_threshold)

        print("\n")
