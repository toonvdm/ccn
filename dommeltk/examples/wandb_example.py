import logging
import pathlib
import yaml
import sys

from torchvision import datasets, transforms
import torch.nn

from dommel.train import (
    Trainer,
    loss_factory,
    optimizer_factory,
    scheduler_factory,
)
from dommel.nn import module_factory
from dommel.datasets import PytorchDataset
from dommel.distributions import MultivariateNormal


class MultivariateNormalNN(torch.nn.Module):
    def forward(self, mean, var):
        return MultivariateNormal(mean, var)


# a custom module that we want to load
class ImageDistribution(torch.nn.Module):
    """
    Represents an image explicitly as a MultivariateNormal distribution
    with fixed variance.
    """

    def __init__(self, variance):
        torch.nn.Module.__init__(self)
        self._var = variance

    def forward(self, x):
        return MultivariateNormal(x, self._var * torch.ones_like(x))


# helper function for reading YAML configuration
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # select one of the example configuration files
    if len(sys.argv) > 1 and sys.argv[1] == "vae":
        config = read_config(pathlib.Path(__file__).parent / "mnist_vae.yml")
    elif len(sys.argv) > 1 and sys.argv[1] == "clip_grad_norm":
        # with grad norm it does not reach a NaN value in loss
        config = read_config(
            pathlib.Path(__file__).parent / "mnist_clip_grad_norm.yml"
        )
    else:
        config = read_config(pathlib.Path(__file__).parent / "mnist_ae.yml")

    # create MNIST train and test set
    # we wrap the torch Dataset into our DictDataset
    # to use with our model and trainer
    mnist_train = datasets.MNIST(
        config["trainer"]["data_dir"],
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_dataset = PytorchDataset(mnist_train, keys=["image", "label"])
    mnist_test = datasets.MNIST(
        config["trainer"]["data_dir"],
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_dataset = PytorchDataset(mnist_test, keys=["image", "label"])

    # create model, loss and optimizer from configuration
    # using the factory methods
    model = module_factory(**config["model"])

    loss = loss_factory(**config["loss"])

    optimizer = optimizer_factory(
        list(loss.parameters()) + list(model.parameters()),
        **config["optimizer"],
    )

    # Create a learning rate scheduler
    scheduler = scheduler_factory(optimizer, **config["scheduler"])

    # create the trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        logging_platform="wandb",  # -> select the wandb platform
        experiment_config=config,  # -> store config in wandb as well
        **config["trainer"],
    )

    # train
    trainer.train(config["trainer"]["n_epochs"])

    # load from checkpoint
    trainer.train(config["trainer"]["n_epochs"], start_epoch=5)
