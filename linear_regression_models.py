from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt

from torch import cuda
from torch import float32
from torch import inference_mode
from torch import load
from torch import optim
from torch import nn
from torch import __version__

from helper_functions import plot_predictions, data_split, save_model, set_seeds
from models.model_cls import LinearRegressionModel, LinearRegressionModelV2
from prepare_load_data import linear_regression_data


print(f"Running PyTorch version {__version__}")

# Set up device
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using device {device}")

# 1. Prepare and load data
# known parameters
weight = 0.7
bias = 0.3

# Train/test data split 80/20
# X: features, y: labels
X, y = linear_regression_data(weight=weight, bias=bias)
print(f"{X[:10]}\n{y[:10]}")
X_train, y_train, X_test, y_test = data_split(
    train_percent=0.8,
    X_features=X,
    y_labels=y
)
plot_predictions(X_train, y_train, X_test, y_test)

# Put data on the available device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# 2. Build your model
# Set manual seed, nn.Parameter are randomly initialized
set_seeds()

model_0 = LinearRegressionModel()
model_0.to(device)  # set GPU if available
print(f"Model_0 parameters: {list(model_0.parameters())}")
print(f"{next(model_0.parameters()).device}")   # ?

# 2.1. Make predictions with context manager
with inference_mode():  # use no_grad() for older pytorch
    y_preds = model_0(X_test)

print(f"Number of testing samples: {len(X_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

# 3. Train model
loss_fn = nn.L1Loss()
optimizer = optim.SGD(params=model_0.parameters(), lr=0.01)

set_seeds()
epochs = 200

# Track loss values 
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    # Training
    # Training
    model_0.train()  # default mode
    y_pred = model_0(X_train)  # forward pass
    loss = loss_fn(y_pred, y_train)  # loss: model preds vs to ground truth
    optimizer.zero_grad()  # zero gradients to start over with each epoch
    loss.backward()  # backpropagation on loss fn
    optimizer.step()  # progress the optimizer

    # Testing
    model_0.eval()  # evaluation mode

    with inference_mode():
        test_pred = model_0(X_test)  # forward pass
        test_loss = loss_fn(test_pred, y_test.type(float32))
    # predictions made in torch.float (float32)

    # Print train and test loss over epochs
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")

# Plot the loss curve
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Learned parameters
print("The model learned the following values for weights and bias:")
pprint(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# 4. Predictions with a trained model
model_0.eval()
with inference_mode():
    y_preds = model_0(X_test)

plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_preds
)

# Only load/use saved PyTorch models from trusted sources!
# save the state_dict of the model (learned parameters)
MODEL_PATH = Path("models")
MODEL_NAME = "pytorch01_workflow_video.pth"
MODEL_SAVE_PATH = save_model(model_0, MODEL_PATH, MODEL_NAME)

# Instantiate a new model (will have random weights)
MODEL_SAVE_PATH = Path("models\pytorch01_workflow_video.pth")
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(load(f=MODEL_SAVE_PATH))
loaded_model_0.to(device)  # put model to target device

print(f"Loaded model:\n{loaded_model_0}")
print(f"Model on device:\n{next(loaded_model_0.parameters()).device}")

# Make inferences on loaded model
loaded_model_0.eval()
with inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

# Compare previous model preds with loaded model preds (should be same)
print("Are the predictions from previous and loaded model the same? ")
print(f"{y_preds == loaded_model_preds}")
