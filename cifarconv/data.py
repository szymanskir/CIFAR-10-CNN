import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data
from configparser import ConfigParser


def read_cifar_data(config: ConfigParser, train):
    batch_size = config['DEFAULT'].getint('BatchSize')
    workers_count = config['DEFAULT'].getint('WorkersCount')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data = torchvision.datasets.CIFAR10(
        root='data', train=train, download=True, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers_count
    )

    return data_loader
