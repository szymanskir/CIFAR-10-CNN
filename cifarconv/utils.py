import configparser
import os


def read_config(filepath: str):
    """Reads the given config file.

    Args:
        filepath (str): path to the config file

    Returns (config.ConfigParser):
        ConfigParser object with configuration values.
    """
    assert os.path.isfile(filepath)
    config = configparser.ConfigParser()
    config.read(filepath)

    return config
