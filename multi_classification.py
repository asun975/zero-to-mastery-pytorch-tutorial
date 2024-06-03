from os import path
from pathlib import Path
import sys
from traceback import print_exception

from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchmetrics
from torchmetrics import Accuracy

from sklearn.model_selection import train_test_split

from helper_functions import accuracy_fn, plot_decision_boundary, save_model, set_device, set_seeds
from prepare_load_data import sklearn_blobs_data
from models.model_cls import BlobModel


PROG_NAME = "Train/test NN Classification model"
PROG_DESC = "Section 02 PyTorch Neural Networks: Classification models"

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    parser.add_argument('--epochs', type=int, default=100, help='epochs to run')
    parser.add_argument('--epoch_step', type=int, default=10, 
                        help='print train/test loss every n epochs')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='set optimizer learning rate')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save current model')
    
    return parser.parse_args()

def is_valid_model(model_name: str) -> bool:
    # Look for .pth file in models\
    path_to_model = path.join("models", f"{model_name}.pth")
    if path.exists(path_to_model):
        return True
    return False

def main(args) -> None:
    """ARGS."""
    EPOCHS = args.epochs
    EPOCH_STEP = args.epoch_step
    LEARNING_RATE = args.lr
    SAVE = args.save

    """HYPERPARAMETERS."""
    NUM_CLASSES = 4
    NUM_FEATURES = 2
    RANDOM_SEED = 42

    try:
        print(f"Running PyTorch version {torch.__version__}")
        device = set_device()  # set gpu or cpu

        # Create multi-class data
        X_blob, y_blob = sklearn_blobs_data(
        n_samples=1000, 
        n_classes=NUM_CLASSES, 
        n_features=NUM_FEATURES, 
        random_seed=RANDOM_SEED,
        )

        # Split into train and test sets
        X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
            y_blob,
            test_size=0.2,
            random_state=RANDOM_SEED
        )

        # Plot data
        plt.figure(figsize=(10, 7))
        plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
        plt.show()

        # Build model
        model = BlobModel(
            input_features=NUM_FEATURES,
            output_features=NUM_CLASSES,
            hidden_units=8
        ).to(device)
        print(model)

        # Loss and Optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

        # Single forward pass
        model(X_blob_train.to(device))[:5]
        print(f"# of elements in a single prediction sample: {model(X_blob_train.to(device))[0].shape, NUM_CLASSES}")

        y_logits = model(X_blob_test.to(device))  # prediction logits
        # Softmax calculation for prediction probabilities
        y_pred_probs = torch.softmax(input=y_logits, dim=1)
        print(y_logits[:5])
        print(y_pred_probs[:5])

        print(f"Each individual sample sums (close) to 1, ex: first sample sum: {torch.sum(y_pred_probs[0])}")
        # Which class does the model think is *most* likely at the index 0 sample?
        print(y_pred_probs[0])
        print(torch.argmax(y_pred_probs[0]))

        # Fit the model
        set_seeds()
        epochs = EPOCHS
        X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
        X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

        # Training and evaluating loop
        for epoch in range(epochs):
            # Training
            model.train()
            # Forward pass
            y_logits = model(X_blob_train)  # output logits
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            #print(y_logits)
            # Loss and accuracy
            loss = loss_fn(y_logits, y_blob_train)
            acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)
            optimizer.zero_grad()  # Optimizer zero grad
            loss.backward()  # back propagation
            optimizer.step()  # optimizer step

            # Testing
            model.eval()
            with torch.inference_mode():
                # Forward pass
                test_logits = model(X_blob_test)
                test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
                # Test loss and accuracy
                test_loss = loss_fn(test_logits, y_blob_test)
                test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)
            
            # Print train/test loss and acc per epoch
            if epoch % EPOCH_STEP == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%") 
        
        # Make predictions
        model.eval()
        with torch.inference_mode():
            y_logits = model(X_blob_test)
        print(f"First 10 predictions are: {y_logits[:10]}")

        # Predicted logits to prediction probabilities
        y_pred_probs = torch.softmax(y_logits, dim=1)
        # Pred probs to prediction labels
        y_preds = y_pred_probs.argmax(dim=1)
        print(f"Predictions: {y_preds[:10]}\nLabels: {y_blob_test[:10]}")
        print(f"Test accuracy: {accuracy_fn(y_true=y_blob_test, y_pred=y_preds)}%")

        # Plot_decision_boundary() moves data to CPU for matplotlib
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model, X_blob_train, y_blob_train)
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model, X_blob_test, y_blob_test)
        plt.show()

        # Classification evaluation metrics
        # Setup metric and put on target device
        torchmetrics_accuracy = Accuracy(
            task='multiclass', num_classes=NUM_CLASSES).to(device)
        
        # Calculate accuracy
        torchmetrics_accuracy(y_preds, y_blob_test)

        if SAVE:
            # Only load/use saved PyTorch models from trusted sources!
            # save the state_dict of the model (learned parameters)
            MODEL_PATH = Path("models")
            MODEL_NAME = f"pytorch02_multi_classification_model.pth" 
            MODEL_SAVE_PATH = save_model(model, MODEL_PATH, MODEL_NAME)

    except Exception as e:
        print(f"An unexpected exception occured of type {type(e)}")
        print("*** print_exception:")
        print_exception(e, limit=2, file=sys.stdout)

if __name__ == "__main__":
    args = parse_args()
    main(args)
