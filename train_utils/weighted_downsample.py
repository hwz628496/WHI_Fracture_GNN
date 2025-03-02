import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

import pandas as pd
import numpy as np

def weighted_downsample(X: pd.DataFrame, 
                        y: pd.Series, 
                        feature_list: list = ["ETHNICNIH", "RACENIH"],
                        weights_dict: dict = None, 
                        frac: float = 0.5, 
                        random_state: int = 42):

    valid_features = [f for f in feature_list if f in X.columns]
    if not valid_features:
        raise ValueError("None of the specified features exist in X.")

    #no weights_dict is provided
    if weights_dict is None:
        weights_dict = {feature: 1 / len(valid_features) for feature in valid_features}

    total_weight = sum(weights_dict.values())
    normalized_weights = {k: v / total_weight for k, v in weights_dict.items() if k in valid_features}
    sampling_weights = X[valid_features].mul(normalized_weights).sum(axis=1)
    sampling_weights = np.maximum(sampling_weights, 1e-10)  # Avoid zero probability
    sampling_weights /= sampling_weights.sum()
    downsampled_indices = (
        pd.concat([X, y], axis=1) 
        .groupby(y.name, group_keys=False)  
        .apply(lambda group: group.sample(frac=frac, weights=sampling_weights.loc[group.index], random_state=random_state))
        .index
    )

    X_downsampled = X.loc[downsampled_indices]
    y_downsampled = y.loc[downsampled_indices]

    return X_downsampled, y_downsampled


def weighted_downsample_LABELS(df, labels, target_ratio=0.5, random_state=42):
    """
    Performs weighted downsampling to balance classes in the dataset.

    :param df: DataFrame with features.
    :param labels: Series with target labels.
    :param target_ratio: Desired ratio of the minority class.
    :param random_state: Random seed for reproducibility.
    :return: Downsampled features (X) and labels (y).
    """
    # Combine features and labels into one DataFrame
    df['label'] = labels
    
    # Identify majority and minority classes
    class_counts = df['label'].value_counts()
    min_class = class_counts.idxmin()
    maj_class = class_counts.idxmax()

    # Compute the target number of samples for the majority class
    n_min = class_counts[min_class]
    n_maj = int(n_min / target_ratio - n_min)

    # Downsample the majority class
    df_minority = df[df['label'] == min_class]
    df_majority_downsampled = resample(df[df['label'] == maj_class], 
                                       replace=False, 
                                       n_samples=n_maj, 
                                       random_state=random_state)

    # Combine the balanced dataset
    df_balanced = pd.concat([df_minority, df_majority_downsampled])

    # Separate features and labels again
    y_balanced = df_balanced['label']
    X_balanced = df_balanced.drop(columns=['label'])

    return X_balanced, y_balanced
