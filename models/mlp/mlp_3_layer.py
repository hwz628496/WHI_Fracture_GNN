import torch
import numpy as np

class mlp_3_layer(torch.nn.Module):
    def __init__(self, input_size = 63, hidden_dim1 = 75, hidden_dim2 = 125, hidden_dim3 = 64, output_size = 1, dropout=0.2):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(input_size, hidden_dim1)  # First hidden layer
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)  # Second hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim2, output_size)  # Output layer
        self.fc_out = torch.nn.Linear(hidden_dim3, output_size)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)  # Regularization
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation here for raw logits
        return x

def get_3_layer_mlp():
    cv_avail = False

    #input shape (sample, 63) is 
    model = mlp_3_layer
    param_dist = {
        "hidden_dim1": [128, 256, 512, 256, 128],  
        "hidden_dim2": [64, 128, 256, 128, 64],    
        "hidden_dim3": [32, 64, 128, 64, 32],      
        "dropout": [0.2, 0.3, 0.4, 0.3, 0.2]        
    }
    log_path = "models/mlp/mlp_3_layer"
    scoring = None

    return model, cv_avail, param_dist, log_path, scoring
