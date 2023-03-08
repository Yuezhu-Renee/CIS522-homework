import torch


class View(torch.nn.Module):
    """
    Change the dimension.
    """

    def __init__(self, o):
        super().__init__()
        self.o = o

    def forward(self, x):
        return x.view(-1, self.o)


class Model(torch.nn.Module):
    """
    The CNN model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initialize the model
        """
        c1 = 32
        c2 = 32
        super().__init__()
        dropout_prob = 0.5

        def convbn(channel_in, channel_out, kernel_sz, stride_sz=1, padding=0):
            """
            build a conv + bn block
            operations are in the following order:
            1. conv2d
            2. relu
            3. batchnorm

            use specified input channels (channel_in), output channels (channel_out), kernel size (kernel_sz),
            stride size (stride_sz), padding (padding)
            """
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=channel_in,
                    out_channels=channel_out,
                    kernel_size=kernel_sz,
                    stride=stride_sz,
                    padding=padding,
                ),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(num_features=channel_out),
            )

        self.m = torch.nn.Sequential(
            # Block 1
            convbn(num_channels, c1, 3, 1, 1),
            convbn(c1, c1, 3, 1, 1),
            convbn(c1, c1, 3, 2, 1),
            torch.nn.Dropout(dropout_prob),
            # block 4
            convbn(c1, c2, 3, 1, 1),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, c2, 3, 2, 1),
            torch.nn.Dropout(dropout_prob),
            # block 7
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, c2, 3, 1, 1),
            convbn(c2, num_classes, 1, 1, 1),
            torch.nn.AvgPool2d(kernel_size=8),
            View(10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Train the model
        Return:
            Output: torch.Tensor
        """

        return self.m(x)
