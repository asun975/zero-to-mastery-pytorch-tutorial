# %%
# Import PyTorch and matplotlib
import torch
from torch import nn # building blocks for neural networks
import matplotlib.pyplot as plt

from models.model_cls import LinearRegressionModel


# Check pytorch version
torch.__version__

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from helper_functions import plot_predictions
from prepare_load_data import linear_regression_data
# 1. Data (preparing and loading)
# Known parameters
weight = 0.7
bias = 0.3

# Train/test split 80/20
# X: features, y: labels
X, y = linear_regression_data(weight, bias)
print(f"{X[:20]}\n{y[:20]}")
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
#print(f"Train: {X_train}, {y_train}\nTest: {X_test}, {y_test}")

plot_predictions(
  train_data=X_train,
  train_labels=y_train,
  test_data=X_test,
  test_labels=y_test,
)
# 2. Build model
# Check contents of pytorch model
# Set manual seed since nn.Parameter are randomly initialzied
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel().to(device)

# Check the nn.Parameter(s) within the nn.Module subclass we created
list(model_0.parameters())

# Set model to GPU if it's availalble, otherwise it'll default to CPU
model_0.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
next(model_0.parameters()).device

#################### untested code below!
# %% Make predictions with torch.inference() context manager
# Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)

# Note: in older PyTorch code you might also see torch.no_grad()
# with torch.no_grad():
#   y_preds = model_0(X_test)

# %%
# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

# 3. Train model

# %% Linear Regression model V1
# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01) # learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))

# %%
def training_loop(model,
                  train_data,
                  train_labels):
  ### Training

  # Put model in training mode (this is the default state of a model)
  model.train()

  # 1. Forward pass on train data using the forward() method inside 
  y_pred = model_0(train_data)
  # print(y_pred)

  # 2. Calculate the loss (how different are our models predictions to the ground truth)
  loss = loss_fn(y_pred, train_labels)

  # 3. Zero grad of the optimizer
  optimizer.zero_grad()

  # 4. Loss backwards
  loss.backward()

  # 5. Progress the optimizer
  optimizer.step()

  return loss

# %%
def testing_loop(model,
                 test_data,
                 test_labels):
  ### Testing
  # Put the model in evaluation mode
  model.eval()

  with torch.inference_mode():
    # 1. Forward pass on test data
    test_pred = model(test_data)

    # 2. Caculate loss on test data
    test_loss = loss_fn(test_pred, test_labels.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type
  return test_loss
# %%
torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 200

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    loss = training_loop(model=model_0,
                         train_data=X_train,
                         train_labels=y_train)
    """### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()
    """

    test_loss = testing_loop(model=model_0,
                             test_data=X_test,
                             test_labels=y_test)
    """### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
      # 1. Forward pass on test data
      test_pred = model_0(X_test)

      # 2. Caculate loss on test data
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type"""

      # Print out what's happening
    if epoch % 10 == 0:
          epoch_count.append(epoch)
          train_loss_values.append(loss.detach().numpy())
          test_loss_values.append(test_loss.detach().numpy())
          print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
# %%
# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
# %%
# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# 4. Making predictions with a trained pytorch model (inference)
# %%
# Only use what is needed for inference - faster computation and less cross-device errors
# 1. Set the model in evaluation mode
model_0.eval()

# 2. Setup the inference mode context manager
with torch.inference_mode():
  # 3. Make sure the calculations are done with the model and data on the same device
  # in our case, we haven't setup device-agnostic code yet so our data and model are
  # on the CPU by default.
  # model_0.to(device)
  # X_test = X_test.to(device)
  y_preds = model_0(X_test)
y_preds
# %%
plot_predictions(predictions=y_preds)

# 5. Save and load pytorch model
"""
Note: As stated in Python's pickle documentation, the pickle module is not secure. 
That means you should only ever unpickle (load) data you trust. 
That goes for loading PyTorch models as well. 
Only ever use saved PyTorch models from sources you trust.
"""
# %%
# Recommended way is to save with pytorch model's state_dict()
from pathlib import Path

# 1. Create models directory 
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model 
# Common convention to save with .pt or .pth
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 
# %%
# Load saved model state_dict()
# dictionary of learned parameters not entire model for flexibility
# Instantiate a new instance of our model (this will be instantiated with random weights)
loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
# %%
# Make inferences on loaded model
# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model
# %%
# Compare previous model predictions with loaded model predictions (these should be the same)
y_preds == loaded_model_preds