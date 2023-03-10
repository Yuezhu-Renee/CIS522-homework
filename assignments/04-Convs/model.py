import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    The CNN model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize the model
        """
        super().__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=num_channels, out_channels=32, kernel_size=3
        )
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc2 = nn.Linear(1152, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Train the model
        Return:
            Output: torch.Tensor
        """
        out = F.relu(self.conv_layer1(x))
        out = self.max_pool1(out)
        out = F.relu(self.conv_layer2(out))
        out = self.max_pool1(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc2(out)

        return out
