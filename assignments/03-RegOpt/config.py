from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 32
    num_epochs = 10
    initial_learning_rate = 0.0008

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "num_of_epoch": num_epochs,
        "ini_lr": initial_learning_rate,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(), lr=CONFIG.initial_learning_rate
    )

    transforms = Compose(
        [
            ToTensor(),
        ]
    )
