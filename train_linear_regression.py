import argparse
from torch import nn
from torch import optim


parser=argparse.ArgumentParser()
parser.add_argument(
    "--model", 
    default="linear_regression",
    choices=["linear_regression", 'Maths', 'Biology'])
args=parser.parse_args()
print ("My subject is ", args.sub)

# Linear Regression model V1
# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))