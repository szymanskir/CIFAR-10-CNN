import click
import logging
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data
from cifarconv.networks import LeNet5
from cifarconv.utils import read_config


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
    config = read_config(config_file)['DEFAULT']

    BATCH_SIZE = config.getint('BatchSize')
    WORKERS_COUNT = config.getint('WorkersCount')
    EPOCHS_COUNT = config.getint('EpochsCount')

    logging.info('Reading CIFAR-10 dataset...')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS_COUNT
    )

    test = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS_COUNT
    )

    model = LeNet5()
    cost_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    current_cost = 0
    for epoch in range(EPOCHS_COUNT):
        logging.info(f'Processing epoch {epoch}...')
        for X, y in train_loader:
            optimizer.zero_grad()
            y_predictions = model(X)
            cost = cost_function(y_predictions, y)
            cost.backward()
            optimizer.step()

            logging.info(f'Current cost: {cost.item()}')

    torch.save(model.state_dict(), 'sample-network.pkl')

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


if __name__ == '__main__':
    main()
