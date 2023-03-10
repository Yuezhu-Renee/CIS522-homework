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
            in_channels=num_channels, out_channels=24, kernel_size=3
        )
        self.conv_layer2 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(432, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Train the model
        Return:
            Output: torch.Tensor
        """
        out = F.relu(self.conv_layer1(x))
        out = self.max_pool(out)
        out = F.relu(self.conv_layer2(out))
        out = self.max_pool(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out
