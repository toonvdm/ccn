from ccn.util import get_data_path
from ccn.benchmark.moveto import MoveToBenchmark
from ccn.active_inference.diff_agent import agent_modes
from ccn.active_inference.reval_agent import RevalAgent
from ccn.action import Action

from scene_environments.util import get_data_path as env_data_path
import matplotlib.pyplot as plt

import numpy as np

import logging

from pathlib import Path

import numpy as np
import torch
import random

import click

from pathlib import Path
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"  # GPU

logger = logging.getLogger(__file__)


def agent_factory(agent_mode, objects):
    agent = None
    if agent_mode == "aif":
        load_dir = get_data_path() / "input/phase2-models"
        ccn_dir = get_data_path() / "input/phase1-models"
        agent = RevalAgent(
            objects,
            model_dir=load_dir,
            ccn_dir=ccn_dir,
            particle_filter=True,
            teleport=True,
            debug=False,
            mode=agent_modes.EFE,
            n_samples=5000,
        )
    # elif agent_mode == "lexa":
    #     from ccn.agents.lexa import LexaAgent

    #     agent = LexaAgent()

    # elif agent_mode == "posecnn":
    #     from ccn.agents.posecnn import PoseCNNAgent

    #     agent = PoseCNNAgent(refine=False)

    # elif agent_mode == "posecnn-infogain":
    #     from ccn.agents.posecnn import PoseCNNAgent

    #     agent = PoseCNNAgent(refine=False, explore="aif")

    # elif agent_mode == "posecnn-refine":
    #     from ccn.agents.posecnn import PoseCNNAgent

    #     agent = PoseCNNAgent(refine=True)
    return agent


def main(n_objects=1, agent_mode="aif"):

    store_path = get_data_path() / "working/moveto-benchmark/"
    store_path.mkdir(exist_ok=True, parents=True)

    benchmark_dir = env_data_path() / "evaluation-pickles"

    objects = [
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
    ]

    max_steps = 350

    agent = agent_factory(agent_mode, objects)

    for target_object in objects:
        logger.info(f"Running for {target_object} - {n_objects}")
        benchmark_obj_dir = benchmark_dir / f"{target_object}-{n_objects}"

        store_loc = store_path / f"{agent_mode}/{target_object}-{n_objects}"
        store_loc.mkdir(exist_ok=True, parents=True)

        mvb = MoveToBenchmark(
            benchmark_obj_dir,
            agent,
            max_steps=max_steps,
            store_location=store_loc,
        )

        mvb.run(visualize_runs=False)


if __name__ == "__main__":
    for i in [1, 2, 3, 4, 5]:
        main(i, "aif")
