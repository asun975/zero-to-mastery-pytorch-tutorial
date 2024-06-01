import torch
from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(
            torch.randn(1, dtype=torch.float), 
            requires_grad=True  # <- can we update this value with gradient descent?)
        ) 

        self.bias = nn.Parameter(
            torch.randn(1, dtype=torch.float), 
            requires_grad=True  # <- can we update this value with gradient descent?))
        ) 

    # Forward defines the computation in the model
    # "x" is the input data (e.g. training/testing features)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.weights * x + self.bias # (y = m*x + b)

# Linear Regression model using nn.Linear() for creating model parameters
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)
    
    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)