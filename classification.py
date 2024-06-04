from os import path
from pathlib import Path
import sys
from traceback import print_exception

from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from torch import nn

from sklearn.model_selection import train_test_split

from helper_functions import accuracy_fn, plot_decision_boundary, save_model, set_device, set_seeds
from prepare_load_data import sklearn_circle_data
from models.model_cls import CircleModelV0, CircleModelV1, CircleModelV2


PROG_NAME = "Train/test NN Classification model"
PROG_DESC = "Section 02 PyTorch Neural Networks: Classification models"
# TODO: implement loading different models to compare
CHOICES = ('v0', 'v1', 'v2')
DEFAULT = CHOICES[0]

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    parser.add_argument('--model', type=str, default=DEFAULT, choices=CHOICES, 
                        help='model name')
    parser.add_argument('--epochs', type=int, default=100, help='epochs to run')
    parser.add_argument('--epoch_step', type=int, default=10, 
                        help='print train/test loss every n epochs')
    parser.add_argument('--load_model', type=str, default=None, 
                        help='file name of saved model (must be same type as model arg)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save current model')
    
    return parser.parse_args()

def is_valid_model(model_name: str) -> bool:
    # Look for .pth file in models\
    path_to_model = path.join("models", f"{model_name}.pth")
    if path.exists(path_to_model):
        return True
    return False

def get_model(model_name: str, messages: list) ->nn.Module:
    if model_name == "v1":
        model = CircleModelV1()
        print(messages[1])
    elif model_name == "v2":
        model = CircleModelV2()
        print(messages[2])
    else:  # default 
        model = CircleModelV0()
        print(messages[0])
    
    return model

def main(args) -> None:
    """ARGS"""
    MODEL = args.model
    EPOCHS = args.epochs
    EPOCH_STEP = args.epoch_step
    LOAD_MODEL = args.load_model
    SAVE = args.save

    try:
        print(f"Running PyTorch version {torch.__version__}")
        device = set_device()  # set gpu or cpu

        # Make classification data
        n_samples = 1000
        X, y = sklearn_circle_data(n_samples, noise=0.03, random_state=42)
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y, 
            test_size=0.2,  # 20% test, 80% train
            random_state=42 # make the random split reproducible
        )
        print(f"Train data {len(X_train)}, test data {len(X_test)}")
        print(f"Train labels {len(y_train)}, test labels {len(y_test)}")

        # Build model based on args
        get_model_msgs = (
            "Baseline model",
            "Model with 1 extra linear layer",
            "Model with 1 ReLU layer",
            "BlobModel",
        )
        model = get_model(MODEL, get_model_msgs)
        model.to(device)  # set to target device

        # Predictions
        untrained_preds = model(X_test.to(device))
        print(f"Length of preds: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
        print(f"\nFirst 10 preds:\n{untrained_preds[:10]}")
        print(f"\nFirst 10 labels:\n{y_test[:10]}")

        """# Prediction labels -for info only (causes TypeError in train loop!)
        # First 5 outputs of forward pass on test data
        y_logits = model(X_test.to(device))[:5]
        print(f"Logits are raw outputs from the model: {y_logits}") 
        y_pred_probs = torch.sigmoid(y_logits)
        print(f"Use sigmoid activation function on logits to use them like truth labels: {y_pred_probs}")
        y_preds = torch.round(y_pred_probs)
        print(f"Round down to get prediction labels: {y_preds}")
        y_pred_labels = torch.round(torch.sigmoid(model(X_test).to(device))[:5])  # one step
        print(f"Check for equality: {torch.eq(y_preds.squeeze(), y_pred_labels.squeeze())}")
        y_preds.squeeze()
        print(f"Model preds are in the same form as our truth labels: {y_preds}")"""

        # Train model
        # Loss function with built-in sigmoid
        loss_fn = nn.BCEWithLogitsLoss()

        # Optimizer SGD
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

        set_seeds()
        epochs = EPOCHS
        # Put data to target device
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        test_pred = 0 # variable to track test predictions
        # Training and evaluating loop
        for epoch in range(epochs):
            # Training
            model.train()  # default
            # forward pass - model outputs logits
            y_logits = model(X_train).squeeze()
            # Logits -> decision boundary -> pred labels
            y_pred = torch.round(torch.sigmoid(y_logits))

            # Calculate loss/accuracy
            loss = loss_fn(y_logits, y_train)
            acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

            optimizer.zero_grad()  # Zero grad every epoch
            loss.backward()  # Loss backwards back propagation
            optimizer.step()

            # Testing
            model.eval()
            with torch.inference_mode():
                # Forward pass
                test_logits = model(X_test).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))
                # Loss/Accuracy
                test_loss = loss_fn(test_logits, y_test)
                test_acc = accuracy_fn(
                    y_true=y_test, 
                    y_pred=test_pred
                )
            
            # Print every 10 epochs
            if epoch % EPOCH_STEP == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

        # Plot decision boundary for training and test sets
        plt.figure(figsize=(12,6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model, X_train, y_train)
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model, X_test, y_test)
        plt.show()

        if SAVE:
            # Only load/use saved PyTorch models from trusted sources!
            # save the state_dict of the model (learned parameters)
            MODEL_PATH = Path("models")
            MODEL_NAME = f"pytorch02_classification_{MODEL}.pth" 
            MODEL_SAVE_PATH = save_model(model, MODEL_PATH, MODEL_NAME)

        # Instantiate a new model (will have random weights)
        if LOAD_MODEL:
            #is_valid_model = is_valid_model(load_model)
            # Look for .pth file in models\
            path_to_model = path.join("models", f"{LOAD_MODEL}.pth")
            if path.exists(path_to_model):
            #if is_valid_model:
                MODEL_SAVE_PATH = Path(path_to_model)

                loaded_model = get_model(MODEL, get_model_msgs)

                loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
                loaded_model.to(device)  # put model to target device

                print(f"Loaded model:\n{loaded_model}")
                print(f"Model on device:\n{next(loaded_model.parameters()).device}")

                # Make inferences on loaded model
                loaded_model_preds = 0  # variable to hold loaded_model predictions
                loaded_model.eval()
                with torch.inference_mode():
                    # Forward pass
                    loaded_model_logits = loaded_model(X_test).squeeze()
                    loaded_model_preds = torch.round(torch.sigmoid(loaded_model_logits))

                # Compare previous model preds with loaded model preds (should be same)
                print("Are the predictions from previous and loaded model the same? ")
                print(f"{test_pred == loaded_model_preds}")
            else:
                print(f"Could not load model {LOAD_MODEL}")

    except Exception as e:
        print(f"An unexpected exception occured of type {type(e)}")
        print("*** print_exception:")
        print_exception(e, limit=2, file=sys.stdout)

if __name__ == "__main__":
    args = parse_args()
    main(args)
