import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from keras.layers.normalization import BatchNormalization


def create_lenet5(input_shape):
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))

    model.add(Conv2D(16, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation("relu"))
    model.add(Dense(84))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    return model


def create_allcnn(input_shape):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=input_shape))
    model.add(Conv2D(96, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            96, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
        )
    )
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            192, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
        )
    )
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (3, 3), padding="same", kernel_initializer="he_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (1, 1), kernel_initializer="he_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(10, (1, 1), kernel_initializer="he_normal"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Activation("softmax"))

    return model


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
        self.batch_norm1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(192)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.conv7 = nn.Conv2d(192, 192, 3)
        self.batch_norm7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.batch_norm8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.batch_norm9 = nn.BatchNorm2d(10)
        self.avg_pool = nn.AvgPool2d(kernel_size=(6, 6))

        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.xavier_normal_(self.conv4.weight)
        torch.nn.init.xavier_normal_(self.conv5.weight)
        torch.nn.init.xavier_normal_(self.conv6.weight)
        torch.nn.init.xavier_normal_(self.conv7.weight)
        torch.nn.init.xavier_normal_(self.conv8.weight)
        torch.nn.init.xavier_normal_(self.conv9.weight)

    def forward(self, x):
        out = self.dropout1(x)
        out = F.relu(self.conv1(out))
        out = self.batch_norm1(out)
        out = F.relu(self.conv2(out))
        out = self.batch_norm2(out)
        out = F.relu(self.conv3(out))
        out = self.batch_norm3(out)
        out = self.dropout2(out)
        out = F.relu(self.conv4(out))
        out = self.batch_norm4(out)
        out = F.relu(self.conv5(out))
        out = self.batch_norm6(out)
        out = F.relu(self.conv6(out))
        out = self.batch_norm6(out)
        out = self.dropout3(out)
        out = F.relu(self.conv7(out))
        out = self.batch_norm7(out)
        out = F.relu(self.conv8(out))
        out = self.batch_norm8(out)
        out = F.relu(self.conv9(out))
        out = self.batch_norm9(out)
        out = self.avg_pool(out)
        out = F.softmax(out, dim=1)
        return out.view(-1, 10)
