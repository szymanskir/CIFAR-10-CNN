import keras
from cifarconv.networks import create_allcnn, create_lenet5
from keras.datasets import cifar10
from keras.optimizers import SGD

batch_size = 32
epochs = 180
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = create_allcnn(x_train.shape[1:])
sgd_optimizer = SGD(lr=0.01, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    shuffle=True,
)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
