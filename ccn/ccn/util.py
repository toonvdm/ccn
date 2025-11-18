"""
Utility functions that can be used in all situations
"""
import pathlib
import yaml
import torch


def read_config(config_path):
    """
    Opens a config file
    :param config_path: path to config file
    :return: attrdict of config
    """
    config_file = pathlib.Path(config_path)
    with open(config_file, "r") as cf:
        opt = yaml.load(cf, Loader=yaml.SafeLoader)
    return opt


def get_data_path():
    return (pathlib.Path(__file__).parent / "../data").resolve()


def to_torch(rgb):
    try:
        rgb = rgb.clone()
    except:
        rgb = torch.tensor(rgb.copy())
    if rgb.max() > 1:
        rgb = rgb / 255.0
    return rgb.permute(2, 0, 1)


def to_img(rgb):
    try:
        rgb = rgb.clone()
    except:
        rgb = torch.tensor(rgb.copy())
    return rgb.permute(1, 2, 0).detach().numpy().clip(0, 1)
