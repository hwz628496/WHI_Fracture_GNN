
import torch
import numpy as np

class mlp_5_layer(torch.nn.Module):
    def __init__(self, input_size=63, hidden_dim1=75, hidden_dim2=125, hidden_dim3=100, hidden_dim4=80, hidden_dim5=64, output_size=1, dropout=0.2):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(input_size, hidden_dim1)  # First hidden layer
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2)  # Second hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim2, hidden_dim3)  # Third hidden layer
        self.fc4 = torch.nn.Linear(hidden_dim3, hidden_dim4)  # Fourth hidden layer
        self.fc5 = torch.nn.Linear(hidden_dim4, hidden_dim5)  # Fifth hidden layer
        self.fc_out = torch.nn.Linear(hidden_dim5, output_size)  # Output layer

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)  # Regularization
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc_out(x)  # No activation here for raw logits
        return x

def get_5_layer_mlp():
    cv_avail = False

    # Input shape (sample, 63)
    model = mlp_5_layer
    param_dist = {
        "hidden_dim1": [128, 256, 512, 256, 128],  
        "hidden_dim2": [64, 128, 256, 128, 64],    
        "hidden_dim3": [128, 256, 512, 256, 128],  
        "hidden_dim4": [64, 128, 256, 128, 64],    
        "hidden_dim5": [32, 64, 128, 64, 32],      
        "dropout": [0.2, 0.3, 0.4, 0.3, 0.2]        
    }
    log_path = "models/mlp/mlp_5_layer"
    scoring = None

    return model, cv_avail, param_dist, log_path, scoring
