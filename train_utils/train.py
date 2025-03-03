from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pandas as pd
import torch

from tqdm import tqdm

from train_utils.weighted_downsample import weighted_downsample_LABELS
from train_utils.eval import eval_frax, eval_run



n_splits = 5
random_state = 45
#downsample
target_ratio = 0.5

#wrapper for 5 fold CV to make it 5 fold nested
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Implements Focal Loss for binary classification.
        Args:
            alpha (float): Balancing factor for class weights.
            gamma (float): Focusing parameter for hard examples.
            reduction (str): 'mean' (default) or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")  # Keep raw loss for focal scaling

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (before sigmoid).
            targets: Ground truth labels (binary, 0 or 1).
        Returns:
            Focal Loss value.
        """
        bce_loss = self.bce_loss(logits, targets)  # Compute standard BCE loss
        probs = torch.sigmoid(logits)  # Convert logits to probabilities
        targets = targets.float()

        # Compute focal loss scaling factor
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_factor = (1 - p_t) ** self.gamma

        # Apply alpha weighting (for class balance)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Compute final focal loss
        loss = alpha_factor * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # No reduction (per-sample loss)


def train_5foldnest_DL(model_init=None, 
                       labels=None, 
                       dataset=None, 
                       n_splits=5, 
                       random_state=42,
                       epoch=200,
                       param_dist=None,
                       batch_size = 750,
                       device = "cuda:0",
                       best_score = 0):  # Added batch_size argument
    
    skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=random_state)
    scores = []
    
    # Ensure param_dist has enough values
    assert all(len(param_dist[key]) == 5 for key in param_dist), "Each hyperparameter must have 5 values"

    print(model_init)
    model_list = [
        model_init(
            hidden_dim1=param_dist["hidden_dim1"][i],
            hidden_dim2=param_dist["hidden_dim2"][i],
            hidden_dim3=param_dist["hidden_dim3"][i],
            dropout = param_dist["dropout"][i]
        ).to(device)
        for i in range(5)
    ]

    dataset_tensor = torch.tensor(dataset.to_numpy(), dtype = torch.float32).to(device)
    labels_tensor = torch.tensor(labels.to_numpy(), dtype = torch.float32).unsqueeze(1).to(device)  # Make labels (N,1)
    

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        print(f"Nested {fold+1}/{n_splits}")

        model = model_list[fold] 
        print("Model Params:")
        for name, param in model.named_parameters():
            print(f"\t{name}: {param.shape}")

        model.train()  

        X_train, X_test = dataset_tensor[train_idx], dataset_tensor[test_idx]
        y_train, y_test = labels_tensor[train_idx], labels_tensor[test_idx]

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

        # Loss, Optimizer, Scheduler
        criterion = FocalLoss()  
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay = 1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  

        # Training Loop
        for epoch_num in tqdm(range(epoch), desc="Param Tuning"):
            total_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

        model.eval()
        metrics = eval_run(model, x = X_test, y = y_test, model_type = "deep learning", descr = "[NESTED_FOLD]:")

        if metrics["AUC"] > best_score:
            best_score = metrics["AUC"]
            best_model = model  # Save the best model

    return best_model

def train(model = None, 
          random_cv_avail: bool = False, 
          scoring = None, 
          param_dist = None,
          dataset: pd.DataFrame = None, 
          labels: pd.Series = None, 
          n_splits = 5, 
          random_state = 45, 
          target_ratio = 0.3,
          device = "cuda:0"):  
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    model_init = model
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        print(f"Fold {fold+1}/{n_splits}--------------------------")

        # Split train and test sets
        X_train, X_test = dataset.iloc[train_idx], dataset.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        
        # Perform weighted downsampling on training data
        X_train_balanced, y_train_balanced = weighted_downsample_LABELS(X_train, y_train, target_ratio = target_ratio)
        print("Training Fold Label Distr:", dict(pd.Series(y_train_balanced).value_counts()))

        #get frax after downsample
        frax_score_train, frax_score_test = X_train_balanced["FRAX_SCORE"], X_test["FRAX_SCORE"]
        X_train_balanced, X_test = X_train_balanced.drop(columns = ["FRAX_SCORE"]), X_test.drop(columns = ["FRAX_SCORE"])


        #if sklearn wrapper
        if random_cv_avail == True:
            random_search = RandomizedSearchCV(model, param_distributions = param_dist, n_iter=20, cv=5, scoring = scoring, n_jobs=-1, random_state=42)
            random_search.fit(X_train_balanced, y_train_balanced)
            print("Best Parameters:", random_search.best_params_)
            model = random_search.best_estimator_

            eval_run(model, X_train_balanced, y_train_balanced, descr = "MODEL Train:")
            FRAX_threshold = eval_frax(frax_score_train, y_train_balanced, descr = "FRAX Train:")
            
            print("Test Fold Label Distr:", dict(pd.Series(y_test).value_counts()))
            eval_frax(frax_score_test, y_test, descr = "FRAX Test:", threshold = FRAX_threshold)

            # Evaluate on the original test set (without downsampling)
            eval_run(model, X_test, y_test, descr = "MODEL Test:")

        #do nested 5 fold CV for myself for DL
        else:

            model = train_5foldnest_DL(model_init = model_init,
                                       labels =  y_train_balanced, 
                                       param_dist = param_dist,
                                       dataset = X_train_balanced,
                                       random_state = 42)
            
            FRAX_threshold = eval_frax(frax_score_train, y_train_balanced, descr = "FRAX Train:")

            eval_frax(frax_score_test, y_test, descr = "FRAX Test:", threshold = FRAX_threshold)
            X_test, y_test = torch.tensor(X_test.to_numpy(), dtype = torch.float32).to(device), torch.tensor(y_test.to_numpy(), dtype = torch.float32).to(device)
            eval_run(model, x = X_test, y = y_test, model_type = "deep learning", descr = "[FINAL TEST]:")
        print("\n")
