import click
import keras
import logging
from cifarconv.networks import create_allcnn, create_lenet5
from cifarconv.utils import read_config, write_pickle
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


def update_lr(epoch, current_lr):
    if epoch in {100, 150, 200, 250}:
        return current_lr * 0.1

    return current_lr


@click.group()
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )


def read_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


@main.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output", default="network.hdf5")
def train(config_file, output):
    config = read_config(config_file)["DEFAULT"]

    batch_size = config.getint("BatchSize")
    epochs = config.getint("EpochsCount")

    (x_train, y_train), (x_test, y_test) = read_cifar10()

    model = create_allcnn(x_train.shape[1:])
    sgd_optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    lrate_scheduler = LearningRateScheduler(schedule=update_lr, verbose=1)
    mcp_save = ModelCheckpoint(
        output, save_best_only=True, monitor="val_loss", mode="min"
    )
    model.compile(
        loss="categorical_crossentropy", optimizer=sgd_optimizer, metrics=["accuracy"]
    )

    img_augmentor = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
    )
    img_augmentor.fit(x_train)

    history = model.fit_generator(
        img_augmentor.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) / batch_size,
        callbacks=[lrate_scheduler, mcp_save],
    )

    write_pickle(history, "model-history.pkl")


@main.command()
@click.argument("weights", type=click.Path(exists=True))
def test(weights):
    (x_train, y_train), (x_test, y_test) = read_cifar10()
    model = create_allcnn(x_train.shape[1:])
    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])
    model.load_weights(weights)
    scores = model.evaluate(x_test, y_test, verbose=1)
    logging.info(f"Test loss: {scores[0]}")
    logging.info(f"Test accuracy: {scores[1]}")

if __name__ == '__main__':
    main()
