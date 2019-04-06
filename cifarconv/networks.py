import torch
import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_layers = nn.Sequential(
            OrderedDict(
                [
                    ("C1", nn.Conv2d(3, 6, 5)),
                    ("relu1", nn.ReLU()),
                    ("S2", nn.MaxPool2d(2)),
                    ("relu2", nn.ReLU()),
                    ("C3", nn.Conv2d(6, 16, 5)),
                    ("relu3", nn.ReLU()),
                    ("S4", nn.MaxPool2d(2)),
                    ("relu4", nn.ReLU()),
                ]
            )
        )

        self.fully_connected_layers = nn.Sequential(
            OrderedDict(
                [
                    ("C5", nn.Linear(16 * 5 * 5, 120)),
                    ("relu5", nn.ReLU()),
                    ("F6", nn.Linear(120, 84)),
                    ("relu5", nn.ReLU()),
                    ("F7", nn.Linear(84, 10)),
                    ("softmax", nn.Softmax(dim=-1)),
                ]
            )
        )

    def forward(self, x):
        y = self.conv_layers(x)
        y = y.view(-1, 16 * 5 * 5)
        return self.fully_connected_layers(y)
