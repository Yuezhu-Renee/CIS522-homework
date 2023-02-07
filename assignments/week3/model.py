from typing import Callable
import torch


class MLP(torch.nn.Module):
    """
    Multilayer perceptron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP with relu and weight as ones.

        Arguments:
            input_size(int): The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.

        Returns:
            Nothing
        """
        super().__init__()
        self.hidden_count = hidden_count
        self.inputlayer = torch.nn.Linear(input_size, hidden_size)
        self.layers = torch.nn.ModuleList()

        for i in range(hidden_count - 1):
            self.layers += [torch.nn.Linear(hidden_size, hidden_size)]

        self.outputlayer = torch.nn.Linear(hidden_size, num_classes)
        self.activation = activation()
        self.norm = torch.nn.BatchNorm1d(hidden_size)
        self.drop = torch.nn.Dropout(0.2)

        initializer(self.inputlayer.weight)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                initializer(layer.weight)

        initializer(self.outputlayer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x(torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the network.
        """
        x = self.inputlayer(x)
        x = self.activation(x)

        for layer in self.layers:
            x = self.activation(self.drop(self.norm(layer(x))))

        x = self.outputlayer(x)
        return x
