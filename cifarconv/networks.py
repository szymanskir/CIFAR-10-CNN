import torch
import torch.nn as nn
import torch.nn.functional as F
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
                    ("softmax", nn.LogSoftmax(dim=-1)),
                ]
            )
        )

    def forward(self, x):
        y = self.conv_layers(x)
        y = y.view(-1, 16 * 5 * 5)
        return self.fully_connected_layers(y)


class AllCnn(nn.Module):
    def __init__(self):
        super(AllCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=2)
        self.conv3 = nn.Conv2d(96, 192, 3)
        self.conv4 = nn.Conv2d(192, 192, 3, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=192 * 5 * 5)
        self.fc1 = nn.Linear(192 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.dropout2d(out, p=0.2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.dropout2d(out, p=0.5)
        out = out.view(out.size(0), -1)
        out = self.bn1(out)
        out = F.relu(self.fc1(out))
        out = F.softmax(self.fc2(out))
        return out
