import click
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cifarconv.networks import LeNet5
from cifarconv.utils import read_config
from cifarconv.data import read_cifar_data
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data


@click.group()
def main():
    torch.manual_seed(44)
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M:%S'
    )


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', default=None)
def train(config_file, output):
    config = read_config(config_file)

    logging.info('Reading CIFAR-10 dataset...')
    train_loader = read_cifar_data(
        config=config,
        train=True
    )
    test_loader = read_cifar_data(
        config=config,
        train=False
    )

    model = LeNet5()
    cost_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    current_cost = 0
    epochs_count = config['DEFAULT'].getint('EpochsCount')
    for epoch in range(epochs_count):
        logging.info(f'Processing epoch {epoch}...')
        for X, y in train_loader:
            optimizer.zero_grad()
            y_predictions = model(X)
            cost = cost_function(y_predictions, y)
            cost.backward()
            optimizer.step()

            logging.info(f'Current cost: {cost.item()}')

    if output:
        torch.save(model.state_dict(), output)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logging.info(
        'Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total)
    )


@main.command()
def test():
    pass


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
def grid_search(config_file):
    config = read_config(config_file)

    logging.info('Reading CIFAR-10 dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform
    )
    X = np.array([x.numpy() for x, _ in data])
    y = np.array([y for _, y in data])
    module = LeNet5()
    net = NeuralNetClassifier(
        module=module,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD
    )
    params = {
        'lr': [0.1, 0.01, 0.001, 0.0001],
        'max_epochs': [2, 4],
        'optimizer__momentum': [0.7, 0.8, 0.9],
        'optimizer__weight_decay': [10, 1, 0.1, 0.01, 0.001]
    }

    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy', verbose=10)

    gs.fit(X, y)
    logging.info(f'\nBest score:{gs.best_score_}')
    logging.info(f'\nBest parameters:{gs.best_params_}')

if __name__ == '__main__':
    main()
