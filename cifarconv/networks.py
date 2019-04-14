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
