from argparse import ArgumentParser
from os import path
from pathlib import Path
import sys
from timeit import default_timer as timer
from traceback import print_exception
import random

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
import torchvision
from tqdm.auto import tqdm

from helper_functions import accuracy_fn, set_device, print_train_time, set_seeds
from models.model_cls import FashionMNISTModelV0, FashionMNISTModelV1, FashionMNISTModelV2
from prepare_load_data import FashionMNIST_data

"""Untested code."""
PROG_NAME = "Train/test Computer vision model"
PROG_DESC = "Section 03 PyTorch Computer vision"
# TODO: remove hardcoded names, load settings for saved models
CHOICES = ('v0', 'v1', 'v2')
DEFAULT = CHOICES[0]

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(prog=PROG_NAME, description=PROG_DESC)
    parser.add_argument('--model', type=str, default=DEFAULT, choices=CHOICES, 
                        help='model name')
    parser.add_argument('--epochs', type=int, default=3, help='epochs to run')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='set hyperparameter learning rate for model')
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

def get_model(model_name: str, units: int, num_classes: int, device: torch.device) ->nn.Module:
    if model_name == "v1":
        model = FashionMNISTModelV1(input_shape=784, hidden_units=units, output_shape=num_classes.to(device))
    elif model_name == "v2":
        model = FashionMNISTModelV2(input_shape=1, hidden_units=units, output_shape=num_classes.to(device))
    else:  # default 
        model = FashionMNISTModelV0(input_shape=784, hidden_units=units, output_shape=num_classes.to(device))
    
    return model

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

def make_predictions(model: torch.nn.Module, data: list, device: torch.device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

def main(args):
    RANDOM_SEED = 42
    """ARGS"""
    MODEL = args.model
    EPOCHS = args.epochs
    LOAD_MODEL = False # args.load_model
    SAVE = args.save

    """HYPERPARAMETERS"""
    BATCH_SIZE = 32
    LEARNING_RATE = args.lr

    try:
        print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
        device = set_device()  # set gpu or cpu

        # Get dataset
        train_data, test_data = FashionMNIST_data()
        image, label = train_data[0]
        class_names = train_data.classes

        print(f"First training sample: {image} {label}")
        print(f"Number of training data: {train_data.data}")
        print(f"Number of training targets: {train_data.targets}")
        print(f"Number of testing data: {test_data.data}")
        print(f"Number of testing targets: {test_data.targets}")
        print(f"See Classes: {class_names}")

        # Visualize the data
        print(f"Image shape: {image.shape}")
        plt.imshow(image.squeeze())
        plt.title(class_names[label])
        plt.show()
        
        set_seeds()
        fig = plt.figure(figsize=(9, 9))
        rows, cols = 4, 4
        for i in range(1, rows * cols + 1):
            random_idx = torch.randint(0, len(train_data), size=[1].item())
            img, label = train_data[random_idx]
            fig.add_subplot(rows, cols, i)
            plt.imshow(img.squeeze(), cmap="gray")
            plt.title(class_names[label])
            plt.axis(False)
            plt.show()
        
        # Dataloader
        # Turn dataset into iterables (batches)
        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        # Print Dataloader info (comment out to skip)
        print(f"Dataloaders: {train_dataloader, test_dataloader}")
        print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
        print(f"Length of train dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
        # Print info about train dataloader
        train_features_batch, train_labels_batch = next(iter(train_dataloader))
        print(f"Shape of training features: {train_features_batch.shape}")
        print(f"Shape of training labels: {train_labels_batch.shape}")
        # Show a sample
        set_seeds()
        random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
        plt.imshow(img.squeeze(), cmap="gray")
        plt.axis("Off")
        plt.show()
        print(f"Image size: {img.shape}")
        print(f"Label: {label}, label size: {label.shape}")

        # Flatten layer for image data (comment out to skip)
        flatten_model = nn.Flatten()
        x = train_features_batch[0]  # get one sample
        output = flatten_model(x)  # forward pass
        print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
        print(f"Shape after flattening: {output.shape} -> [color_channels, height * width]")
        print(x)
        print(output)

        # Build model
        set_seeds()
        model = get_model(MODEL, units=10, num_classes=len(class_names))
        model.to(device)

        # Loss function and optimizer
        loss_fn = nn.CrossEntropyLoss  # criteron or cost fn
        optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
        # Set seed and timer
        set_seeds()
        train_time_start = timer()
        epochs = EPOCHS

        # Training and testing loop
        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n-------")
            train_step(
                model=model,
                data_loader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                accuracy_fn=accuracy_fn,
                device=device
            )
            test_step(
                model=model,
                data_loader=test_dataloader,
                loss_fn=loss_fn,
                device=device
            )
        
        # Calculate training time
        train_time_end_on = timer()
        total_train_time_model = print_train_time(
            start=train_time_start,
            end=train_time_end_on,
            device=str(next(model.parameters()).device)
        )

        # Predictions and results
        set_seeds()
        model_results = eval_model(
            model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn, accuracy_fn=accuracy_fn
        )
        print(f"Model results:\n{model_results}")

        # Make and evaluate random predictions
        # TODO: only for best model
        random.seed(RANDOM_SEED)
        test_samples = []
        test_labels = []
        for sample, label in random.sample(list(test_data), k=9):
            test_samples.append(sample)
            test_labels.append(label)
        
        # First test sample and label
        print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

        # Make predictions on test samples
        pred_probs = make_predictions(model=model, data=test_samples)
        print(f"Fist two prediction probablities: {pred_probs[:2]}")
        # Get prediction labels
        pred_classes = pred_probs.argmax(dim=1)
        print(pred_classes)

        # Plot predictions
        plt.figure(figsize=(9, 9))
        nrows = 3
        ncols = 3
        for i, sample in enumerate(test_samples):
            # Create a subplot
            plt.subplot(nrows, ncols, i+1)

            # Plot the target image
            plt.imshow(sample.squeeze(), cmap="gray")

            # Find the prediction label (in text form, e.g. "Sandal")
            pred_label = class_names[pred_classes[i]]

            # Get the truth label (in text form, e.g. "T-shirt")
            truth_label = class_names[test_labels[i]] 

            # Create the title text of the plot
            title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        
            # Check for equality and change title colour accordingly
            if pred_label == truth_label:
                plt.title(title_text, fontsize=10, c="g") # green text if correct
            else:
                plt.title(title_text, fontsize=10, c="r") # red text if wrong
            plt.axis(False)
        plt.show()

        # Confusion matrix
        # make predictions with trained model
        y_preds = []
        model.eval()
        with torch.inference_mode():
            for X, y in tqdm(test_dataloader, desc="Making predictions"):
                # Send data and targets to target device
                X, y = X.to(device), y.to(device)
                # Do the forward pass
                y_logit = model(X)
                # Turn predictions from logits -> prediction probabilities -> predictions labels
                y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) 
                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())
        # Concatenate list of predictions into a tensor
        y_pred_tensor = torch.cat(y_preds)

        # setup confustion matrix
        confmat = ConfusionMatrix(num_classes=len(class_names), 
                                  task='multiclass')
        confmat_tensor = confmat(preds=y_pred_tensor,
                                 target=test_data.targets)
        
        # Plot confusion matrix
        fig, ax = plot_confusion_matrix(
            conf_mat=confmat_tensor.numpy(),
            class_names=class_names,
            figsize=(10, 7)
        )
        if SAVE:
            # 1. Create path
            MODEL_PATH = Path("models")
            MODEL_PATH.mkdir(parents=True, exist_ok=True)

            # 2. Create model 
            # Common convention to save with .pt or .pth
            MODEL_NAME = f"pytorch03_computer_vision_{MODEL}.pth"
            MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

            # 3. Save the model state dict 
            print(f"Saving model to: {MODEL_SAVE_PATH}")
            torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 

        # TODO
        """import pandas as pd
        compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
        compare_results

        # Visualize our model results
        compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
        plt.xlabel("accuracy (%)")
        plt.ylabel("model")
        """
        # TODO: section/pytorch03_computer_vision_model2CNN.py 
        # line 475-509

    except Exception as e:
        print(f"An unexpected exception occured of type {type(e)}")
        print("*** print_exception:")
        print_exception(e, limit=2, file=sys.stdout)

if __name__ == "__main__":
    args = parse_args()
    main(args)