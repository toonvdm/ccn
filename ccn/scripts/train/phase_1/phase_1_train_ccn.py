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

import wandb


logger = logging.getLogger(__name__)


@click.command()
@click.option("--train_config_path")
@click.option("--dataset_config_path")
@click.option("--model_config_path")
@click.option("--object_name", default="001_chips_can")
@click.option("--debug", default=False)
def train_cli(
    train_config_path,
    dataset_config_path,
    model_config_path,
    object_name,
    debug,
):
    logging.basicConfig(level=logging.DEBUG)
    logger.info(f"{torch.cuda.device_count()} devices found")

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

    # Timestamp log dir!
    log_dir = (
        pathlib.Path(config["trainer"]["log_dir"])
        / config["dataset_global_name"]
        / config["model"]["name"]
        / f"{config['dataset_name']}"
    )

    idx = len(list(log_dir.glob("*"))) + 1
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
    if config.get("load_path", None):
        load_path = pathlib.Path(config["load_path"])
        model_paths = sorted(list(load_path.glob("models/model-*.pt")))
        optimizer_paths = sorted(list(load_path.glob("optimizers/optim-*.pt")))

        model_path, optimizer_path = None, None
        if start_epoch == 0:
            model_path = model_paths[-1]
            optimizer_path = optimizer_paths[-1]
        else:
            for mp, op in zip(model_paths, optimizer_paths):
                ep = int(str(mp.name).split("-")[-1].split(".")[0])
                if ep == start_epoch:
                    model_path = mp
                    optimizer_path = op

        config["trainer"]["log_dir"].mkdir(exist_ok=True, parents=True)
        model_dir = config["trainer"]["log_dir"] / "models"
        optim_dir = config["trainer"]["log_dir"] / "optimizers"
        model_dir.mkdir(exist_ok=True, parents=True)
        optim_dir.mkdir(exist_ok=True, parents=True)
        copyfile(model_path, model_dir / model_path.name)
        copyfile(optimizer_path, optim_dir / optimizer_path.name)
        start_epoch = int(str(model_path.name).split("-")[-1].split(".")[0])
        logger.info(f"Loaded model at epoch {start_epoch}")

    # =====================================================
    tf = None

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
        logger.info(f"Loaded validation dataset with length: {len(val_dataset)}")

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
        project=f"CCN",
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
        pathlib.Path(config["trainer"]["log_dir"]) / pathlib.Path(config_path).name,
    )
    logger.info("Stored config")

    # train
    trainer.train(config["trainer"]["n_epochs"], start_epoch=start_epoch)

    logger.info("Trained")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")
    train_cli()
