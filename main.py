#stl
import os
import warnings

#data handling
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import missingno as mso

#stats
import scipy
import sklearn

#network
import networkx as nx

#vis
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
import time
sns.set(font_scale = 1)
sns.set_style("whitegrid")

#os
import importlib.metadata
import json
import logging
import os
import re
import tempfile
import time
import ast
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

import argparse

from models.random_forest.random_forest import get_random_forest
from models.lasso.lasso import get_lasso
from models.elasticnet.elasticnet import get_elasticnet
from models.svm.svm import get_svm
from models.mlp.mlp_3_layer import get_3_layer_mlp
from models.mlp.mlp_5_layer import get_5_layer_mlp

from train_utils.train import train
from train_utils.logging import logging

def main(extract_model, model_type, cohort = None):
    df = pd.read_csv("dataset/dataset.csv", index_col = [0])

    #extract cohort, labels, patid
    if cohort == 1 or cohort == 0:
        df = df[df["CTFLAG"] == cohort]
    else:
        cohort = "all"

    #do not drop frax score yet
    labels = df["ANYFX"]
    dataset = df.drop(columns = ["ANYFX", "CTFLAG", "ID"])
    
    cv_avail = False
    model, cv_avail, param_dist, log_path, scoring = extract_model()
    print(model)

    logger = logging(log_file = log_path + "_cohort%s.log" % cohort)
    print("Dataset Shape:", dataset.shape)

    train(model = model, 
        random_cv_avail = cv_avail, 
        param_dist = param_dist,
        scoring = scoring,
        dataset = dataset, 
        labels = labels, 
        n_splits = 5, 
        random_state = 45, 
        target_ratio = 0.3)
        
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MLP Model Selection and Cohort Specification")
    parser.add_argument("--model", type=str, required=True, help="Specify the MLP model (e.g., 'mlp_3layer', 'mlp_5layer', or custom model)")
    parser.add_argument("--cohort", type=str, required=True, help="Specify the cohort name (e.g., '0', '1', 'all')")
    args = parser.parse_args()
 
    model_type = args.model
    cohort = args.cohort

    if "model_type" == "xgboost":
        from models.xgboost.xgboost import get_xgboost
    else:
        get_xgboost = None

    model_list = {"random_forest": get_random_forest, 
                  "elasticnet": get_elasticnet, 
                  "lasso": get_lasso, 
                  "svm": get_svm, 
                  "xgboost": get_xgboost,
                  "mlp_3layer": get_3_layer_mlp,
                  "mlp_5layer": get_5_layer_mlp}

    assert model_type in model_list.keys()
    main(extract_model = model_list[model_type], model_type = model_type, cohort = cohort)