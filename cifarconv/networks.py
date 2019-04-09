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
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.conv7 = nn.Conv2d(192, 192, 3)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(6, 6))

        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        torch.nn.init.kaiming_uniform_(self.conv3.weight)
        torch.nn.init.kaiming_uniform_(self.conv4.weight)
        torch.nn.init.kaiming_uniform_(self.conv5.weight)
        torch.nn.init.kaiming_uniform_(self.conv6.weight)
        torch.nn.init.kaiming_uniform_(self.conv7.weight)
        torch.nn.init.kaiming_uniform_(self.conv8.weight)
        torch.nn.init.kaiming_uniform_(self.conv9.weight)

    def forward(self, x):
        out = self.dropout1(x)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.dropout2(out)
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = self.dropout3(out)
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        out = F.relu(self.conv9(out))
        out = self.avg_pool(out)
        out = F.softmax(out, dim=1)
        return out.view(-1, 10)
