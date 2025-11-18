import click
import logging
import torch
import pathlib
from datetime import datetime
from shutil import copyfile

from dommel.datasets.dataset_factory import dataset_factory
from dommel.nn import module_factory, summary
from dommel.train import (
    loss_factory,
    optimizer_factory,
    scheduler_factory,
    Trainer,
)

from ccn.util import read_config
from ccn.dataset.transforms import (
    RandomAnchorTransform,
    TransformSequence,
    RandomBlackFrame,
    RandomComposeTransform,
    RandomBackgroundTransform,
)

import wandb

from dommel.datasets.dataset_factory import download, unzip
from dommel.datastructs import Dict, TensorDict

from pathlib import Path
import numpy as np

import shutil
import os.path

logger = logging.getLogger(__name__)


class CTrainer(Trainer):
    def _log_visualization(self, train_logs, val_logs, epoch):
        # Re visualize as there is a different transform
        self._visualize_batch["train"] = TensorDict(
            next(iter(self._train_loader))
        ).to(self._device)[0:150, ...]

        self._visualize_batch["val"] = TensorDict(
            next(iter(self._val_loader))
        ).to(self._device)[0:150, ...]

        return Trainer._log_visualization(self, train_logs, val_logs, epoch)


def download_models(models_path, store_path):
    if os.path.exists(str(store_path)):
        shutil.rmtree(store_path)
    models_path = download(models_path, clean=True)
    unzipped_path = unzip(models_path, destination=store_path)
    return unzipped_path


@click.command()
@click.option("--train_config_path")
@click.option("--dataset_config_path")
@click.option("--model_config_path")
@click.option("--models_path")
@click.option("--object_name", default="001_chips_can")
@click.option("--debug", default=False)
@click.option("--background_augmentation", default=True)
def train_cli(
    train_config_path,
    dataset_config_path,
    model_config_path,
    models_path,
    object_name,
    debug,
    background_augmentation,
):
    logging.basicConfig(level=logging.DEBUG)
    logger.info(f"{torch.cuda.device_count()} devices found")

    models_path = Path(download_models(models_path, "/tmp/ccn-models"))

    # =====================================================
    # load configurations
    config = dict()
    for config_path in [
        train_config_path,
        dataset_config_path,
        model_config_path,
    ]:
        config_path = pathlib.Path(__file__).parent.absolute() / config_path
        logger.info(f"Reading config from {config_path}")
        config = {**read_config(config_path), **config}
        logger.info("Config read")

    # Define the path to the download only once...
    config["dataset_name"] = object_name
    dataset_hyperconfig = {
        "object_file": f"{object_name}_textured.h5",
        "location": config["location"],
        "destination": config.get("destination", None),
    }
    for k, v in dataset_hyperconfig.items():
        config["train_dataset"][k] = v
        config["val_dataset"][k] = v
    logger.info(f"Reading config from {config_path}")

    log_dir = (
        pathlib.Path(config["trainer"]["log_dir"])
        / config["dataset_global_name"]
        / config["model"]["name"]
        / f"{config['dataset_name']}"
    )

    idx = len(list(log_dir.glob("*"))) + 1
    # Timestamp log dir!
    log_dir /= f"{datetime.now().strftime('%Y-%m-%d')}-run-{idx:02d}"
    config["trainer"]["log_dir"] = log_dir

    # =====================================================
    # If debug -> do everything on CPU (debug locally)
    if debug:
        config["train_dataset"]["device"] = "cpu"
        config["val_dataset"]["device"] = "cpu"
        config["train_dataset"]["load_device"] = "cpu"
        config["val_dataset"]["load_device"] = "cpu"
        config["trainer"]["device"] = "cpu"

    # =====================================================
    # If load path: copy model and optimizer of final step here...
    start_epoch = config["trainer"].get("start_epoch", 0)

    # =====================================================
    if not background_augmentation:
        # Phase 2
        rct = RandomComposeTransform(
            s_range=[0.4 / np.sqrt(2), 4.0]  # , curriculum=True
        )
        # rbf = RandomBlackFrame(p=0.10)
        tf = TransformSequence(rct)
    else:
        # Phase 3
        rbf = RandomBlackFrame(p=0.30)
        rct = RandomComposeTransform(s_range=[0.4 / np.sqrt(2), 4.0])
        rbt = RandomBackgroundTransform("image_composed")
        rat = RandomAnchorTransform()
        tf = TransformSequence(rbf, rct, rat, rbt)

    clean = True
    # data transform -> is currently in our dataset class
    train_dataset = dataset_factory(
        clean=clean, transform=tf, **config["train_dataset"]
    )
    logger.info(f"Loaded training dataset with length: {len(train_dataset)}")
    val_dataset = None
    if config.get("val_dataset", None) is not None:
        val_dataset = dataset_factory(
            clean=False, transform=tf, **config["val_dataset"]
        )
        logger.info(
            f"Loaded validation dataset with length: {len(val_dataset)}"
        )

    # =====================================================
    # Construct loss
    loss = loss_factory(**config["loss"])
    logger.info("Loss created")

    # Construct model
    dtype = config["train_dataset"].get("dtype", "float32")

    device = config["trainer"].get("device", "cpu")

    model = module_factory(**config["model"]).to(
        device=device,
        dtype=eval(f"torch.{dtype}"),
    )

    # CCN load path; assume CCN is the last of the modules
    idx = len(list(model._modules.keys())) - 1
    if model._modules.get(f"module-{idx}", None) is not None and idx > 0:
        state_dict = torch.load(models_path / f"{object_name}.pt")

        # Hack to only load part of the dommel composable module
        state_dict_fixed = {}
        for k, v in state_dict.items():
            state_dict_fixed[k.replace("module-0.", "")] = v

        model._modules[f"module-{idx}"].load_state_dict(state_dict_fixed)
        model._modules[f"module-{idx}"].freeze = True
        for p in model._modules[f"module-{idx}"].parameters():
            p.requires_grad = False

    # Information
    logger.info("Model created")
    summary(model, train_dataset[0].unsqueeze(0).to(device))

    # construct optimizer
    optimizer = optimizer_factory(
        list(loss.parameters()) + list(model.parameters()),
        **config["optimizer"],
    )
    logger.info("Optimizer created")

    # Create a learning rate scheduler
    scheduler = None
    if config.get("scheduler", None) is not None:
        scheduler = scheduler_factory(optimizer, **config["scheduler"])
        logger.info("Scheduler created")

    # Because wandb is init-ed here, it is not re-initted in the trainer
    wandb.init(
        project=f"SceneCCN",
        name=f"{config['model']['name']}-{object_name}-{log_dir.name}",
        entity="dommel",
        config=config,
    )

    # create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        verbose=False,
        logging_platform="wandb",
        experiment_config=config,
        **config["trainer"],
    )
    logger.info("Trainer created")

    # store config
    copyfile(
        pathlib.Path(__file__).parent / config_path,
        pathlib.Path(config["trainer"]["log_dir"])
        / pathlib.Path(config_path).name,
    )
    logger.info("Stored config")

    # train
    trainer.train(config["trainer"]["n_epochs"], start_epoch=start_epoch)

    logger.info("Trained")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    train_cli()
