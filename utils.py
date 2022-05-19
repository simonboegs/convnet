import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

DATA_DIR = "../data"

def create_dataloaders():
    data_train = CIFAR10(
        root="../data",
        train=True,
        download=False,
        transform=ToTensor()
    )
    data_test = CIFAR10(
        root="../data",
        train=False,
        download=False,
        transform=ToTensor()
    )
    train_dl = DataLoader(
            data_train,
            batch_size=BATCH_SIZE,
            shuffle=True
            )
    test_dl = DataLoader(
            data_test,
            batch_size=BATCH_SIZE,
            shuffle=True
            )
    return (train_dl, test_dl)
