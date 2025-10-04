import numpy as np
from torchvision.datasets import CIFAR10,MNIST
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
from torchvision.transforms import ToTensor
import os
import torch
from torch import Tensor
import torch.nn.functional as F

DATASETS_FOLDER = os.environ["DATASETS"]

def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)


def load_cifar(loss: str, device: torch.device) -> (TensorDataset, TensorDataset):
    # Assuming DATASETS_FOLDER is defined elsewhere
    # Assuming functions flatten, make_labels, center, standardize, and unflatten are defined elsewhere

    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)

    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
                      make_labels(torch.tensor(cifar10_test.targets), loss)

    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)

    # Convert data to tensors, standardize, unflatten, and then transpose to match PyTorch's expected input format
    # Finally, move them to the specified device
    train_data = torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float().to(device)
    train_labels = y_train.to(device)
    test_data = torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float().to(device)
    test_labels = y_test.to(device)

    train = TensorDataset(train_data, train_labels)
    test = TensorDataset(test_data, test_labels)

    return train, test


def load_mnist(loss: str, device: torch.device) -> (TensorDataset, TensorDataset):
    # MNIST data is loaded with torchvision's MNIST class
    mnist_train = MNIST(root=DATASETS_FOLDER, download=True, train=True, transform=ToTensor())
    mnist_test = MNIST(root=DATASETS_FOLDER, download=True, train=False, transform=ToTensor())

    # Convert data to numpy arrays for preprocessing
    X_train = mnist_train.data.numpy() / 255.0
    X_test = mnist_test.data.numpy() / 255.0
    
    # Flatten the images for certain types of models (e.g., fully connected networks)
    X_train_flattened = flatten(X_train)
    X_test_flattened = flatten(X_test)

    y_train = torch.tensor(mnist_train.targets)
    y_test = torch.tensor(mnist_test.targets)

    # Prepare labels based on the specified loss function
    y_train_prepared = make_labels(y_train, loss)
    y_test_prepared = make_labels(y_test, loss)

    # No need to center or standardize MNIST as it's already in a good range after division by 255

    # Convert flattened and preprocessed data back to tensors and move to the specified device
    # Note: MNIST images are 1 channel (grayscale), so the shape is adjusted accordingly in unflatten
    train_data = torch.from_numpy(unflatten(X_train_flattened, (28, 28, 1)).transpose((0, 3, 1, 2))).float().to(device)
    train_labels = y_train_prepared.to(device)
    test_data = torch.from_numpy(unflatten(X_test_flattened, (28, 28, 1)).transpose((0, 3, 1, 2))).float().to(device)
    test_labels = y_test_prepared.to(device)

    train = TensorDataset(train_data, train_labels)
    test = TensorDataset(test_data, test_labels)

    return train, test    