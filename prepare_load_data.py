import matplotlib.pyplot as plt
import pandas as pd

from torch import arange
from torch import float as flt
from torch import from_numpy
from torch import LongTensor, Tensor

from torchvision import datasets
from torchvision.transforms import ToTensor

from sklearn.datasets import make_blobs, make_circles
from typing import Iterable, Tuple


def linear_regression_data(
    weight: float,
    bias: float,
    step: float = 0.02,
) -> Iterable[Tuple[Tensor, Tensor]]:
    # Range values
    start = 0
    end = 1

    # X: features, y: labels
    X = arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    return X, y


def sklearn_circle_data(
    samples: int,
    noise: float,
    random_state: float
) -> Iterable[Tuple]:
    # Create circles
    X, y = make_circles(samples, noise, random_state)
    # keep random state so we get the same values

    print(f"First 5 X features:\n{X[:5]}")
    print(f"\nFirst 5 y labels:\n{y[:5]}")

    # Make DataFrame of circle data
    circles = pd.DataFrame({
        "X1": X[:, 0],
        "X2": X[:, 1],
        "label": y
    })
    circles.head(10)
    """
    It looks like each pair of X features (X1 and X2) has a label (y)
    value of either 0 or 1.

    This tells us that our problem is binary classification since there's
    only two options (0 or 1).
    """
    circles.label.value_counts()    # Check different labels

    # Visualize with a plot
    plt.scatter(x=X[:, 0],
                y=X[:, 1],
                c=y,
                cmap=plt.cm.RdYlBu)
    X.shape, y.shape    # shapes of our features and labels

    # View the first example of features and labels
    X_sample = X[0]
    y_sample = y[0]
    print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
    print(f"Shapes for one sample of X: {X_sample.shape} and the same for y:{y_sample.shape}")
    """
    This tells us the second dimension for X means it has two features (vector)
    where as y has a single feature (scalar).

    We have two inputs for one output
    """
    # Turn data into tensors
    # Otherwise this causes issues with computations later on
    X = from_numpy(X).type(flt)
    y = from_numpy(y).type(flt)

    # View the first five samples
    X[:5], y[:5]

    return X, y


def sklearn_blobs_data(
    n_samples: int,
    n_classes: int,
    n_features: int,
    random_seed: int,
) -> Iterable[Tuple]:
    # 1. Create multi-class data
    X_blob, y_blob = make_blobs(
        n_samples=n_samples,
        n_features=n_features,  # X features
        centers=n_classes,  # y labels
        cluster_std=1.5,
        # give the clusters a little shake up
        # try changing this to 1.0, the default
        random_state=random_seed
    )

    # 2. Turn data into tensors
    X_blob = from_numpy(X_blob).type(flt)
    y_blob = from_numpy(y_blob).type(LongTensor)
    print(X_blob[:5], y_blob[:5])

    return X_blob, y_blob


def FashionMNIST_data() -> Iterable[Tuple]:
    # Setup training data
    train_data = datasets.FashionMNIST(
        root="data",  # where to download data to?
        train=True,  # get training data
        download=True,  # download data if it doesn't exist on disk
        transform=ToTensor(),
        # images come as PIL format, we want to turn into Torch tensors
        target_transform=None  # you can transform labels as well
    )

    # Setup testing data
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,  # get test data
        download=True,
        transform=ToTensor()
    )
    return train_data, test_data
