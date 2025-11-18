import csv

import torch
import random

import matplotlib.pyplot as plt
from tqdm import tqdm
from ccn.visualization import (
    together_particles_plot,
    plot_camera_2d,
    fig2img,
    format_ax,
)
import imageio

import numpy as np

from ccn.active_inference import action_perception_loop
from ccn.active_inference.diff_agent import action_modes
from ccn.active_inference.debug import inject_groundtruth_position_info
from ccn.util import get_data_path, to_img
from ccn.active_inference.object_belief import ObjectBelief

from scene_environments import SceneEnvironment
from scene_environments.util import get_data_path as get_env_path
from scene_environments.errors import rotation_error as re

from skimage.transform import resize

from ccn.geometry import cartesian_to_spherical
from ccn.benchmark.benchmark_vistools import gif

import pickle

from pathlib import Path


def spherical_error(run, idx=-1):
    object_name = run["env_info"][0]["goal_object_name"]
    object_position = run["env_info"][0][object_name][:3, -1]

    spherical_gt = cartesian_to_spherical(
        torch.tensor(run["env_info"][idx]["goal_pose"][:3, -1] - object_position)
    )
    spherical_es = cartesian_to_spherical(
        run["env_info"][idx]["camera_pose"][:3, -1] - object_position
    )
    return torch.abs(spherical_gt - spherical_es)


# Error metrics
def observation_mse(run, idx=-1):
    goal = torch.tensor(run["env_info"][idx]["goal_rgb"].copy(), dtype=torch.float32)
    obs = run["env_info"][idx]["rgb"]
    return torch.linalg.norm(goal - obs).item()


def depth_mse(run, idx=-1):
    goal = torch.tensor(run["env_info"][idx]["goal_depth"].copy(), dtype=torch.float32)
    return torch.linalg.norm(goal - run["env_info"][idx]["depth"]).item()


def translation_error(run, idx=-1):
    goal = torch.tensor(
        run["env_info"][idx]["goal_pose"][:3, -1].copy(), dtype=torch.float32
    )
    current = run["env_info"][idx]["camera_pose"][:3, -1]
    return torch.linalg.norm(goal - current).item()


def rotation_error(run, idx=-1):
    return re(
        run["env_info"][idx]["goal_pose"][:3, :3],
        run["env_info"][idx]["camera_pose"][:3, :3],
    )


class MoveToBenchmark:
    def __init__(
        self,
        benchmark_dir,
        agent,
        max_steps=200,
        store_location=None,
        benchmark_seed=0,
    ):
        """
        :param benchmark_dir: path to .pkl files containing SceneEnvironment
            config files to load in the enviroment
        :param agent: Agent that is benchmarked
        """
        self._envfiles = sorted(list(benchmark_dir.glob("*.pkl")))[:20]
        self._agent = agent
        self._max_steps = max_steps

        self.env = None

        self.store_location = store_location
        if self.store_location is None:
            self.store_location = get_data_path() / "output/movetobenchmark"
            self.store_location.mkdir(exist_ok=True, parents=True)

        self._benchmark_seed = benchmark_seed

        self._runs = None
        self.reset()

    def __len__(self):
        return min(20, len(self._envfiles))

    def reset(self):
        self._runs = dict({})

    def visualize(self, logs, store_dir):
        gif(logs, store_dir)

    def run_one(self, idx, visualize_run=False, stop_when_done=True, from_seed=False):
        # TODO -> observation shape fix?
        if not from_seed:
            self.env = SceneEnvironment.load_pkl(
                self._envfiles[idx],
                self.env,  # Passing the environment we already have does not reload the meshes
                # override={"observation_shape": (64, 64, 3)},
            )
        else:
            # If you don't want to load a pickle but create an env based on the seed
            self.env = SceneEnvironment(
                [
                    "002_master_chef_can",
                    "003_cracker_box",
                    "004_sugar_box",
                    "005_tomato_soup_can",
                    "006_mustard_bottle",
                ],
                mesh_location=Path(get_env_path() / "meshes/ycb_gazebo_sdf"),
                n_objects=5,
                seed=from_seed,
                render_table=True,
                observation_shape=(64, 64, 3),
                eval=eval,
            )

        # Seeds are set in the environment for generation process
        np.random.seed(self._benchmark_seed)
        torch.cuda.manual_seed(self._benchmark_seed)
        torch.manual_seed(self._benchmark_seed)
        random.seed(self._benchmark_seed)

        self._agent.reset()

        # Only for CCN agents!
        # if groundtruth_positions:
        #     inject_groundtruth_position_info(self._agent, self.env)
        # else:
        #     for k in self._agent._models.keys():
        #         self._agent._beliefs[k] = ObjectBelief(k, particle_filter=True)
        #         self._agent._beliefs[k].reset()

        # <stop when done> only evaluates the scene-environment metric
        # Do we also want to evaluate a perception metric or some other stopping
        # criteria? How do we compensate for symmetry, evaluate on depth natch?
        # ... How do then make the comparison w/ other agents such as Lexa fair?
        # We just run it for <self.max_steps> steps, and then we can compute the
        # stopping criteria for each of the runs after the fact
        logs = action_perception_loop(
            self.env,
            self._agent,
            self._max_steps,
            stop_when_done=stop_when_done,
        )

        if visualize_run:
            store_path = self.store_location / f"{idx:02d}"
            store_path.mkdir(exist_ok=True, parents=True)
            self.visualize(logs, store_path)

        self._runs[idx] = logs

        self.to_csv(
            self._metrics(idx),
            str(self.store_location / f"metrics_{idx:02d}.csv"),
        )

        with open(str(self.store_location / f"logs_{idx:02d}.pickle"), "wb") as f:
            pickle.dump(logs, f)

        return logs

    def run(self, visualize_runs=False):
        for idx in range(len(self)):
            # Only run if this has not been run before
            # if self._runs.get(idx, None) is None:
            if not (self.store_location / f"metrics_{idx:02d}.csv").exists():
                self.run_one(idx, visualize_runs)
        return self._runs

    def _metrics(self, idx):
        run = self._runs[idx]

        metrics = dict()
        n = run["n_steps"]

        metrics["step"] = [t for t in range(n)]
        metrics["idx"] = [idx for t in range(n)]
        metrics["envfile"] = [str(self._envfiles[idx].resolve().name) for t in range(n)]
        metrics["n_steps"] = [run["n_steps"] for t in range(n)]
        metrics["observation_mse"] = [observation_mse(run, t) for t in range(n)]
        metrics["depth_mse"] = [depth_mse(run, t) for t in range(n)]
        metrics["translation_error"] = [translation_error(run, t) for t in range(n)]
        metrics["rotation_error"] = [rotation_error(run, t) for t in range(n)]

        se = [spherical_error(run, t) for t in range(n)]
        metrics["azimuth_error"] = [se[t][0].item() for t in range(n)]
        metrics["elevation_error"] = [se[t][1].item() for t in range(n)]
        metrics["range_error"] = [se[t][2].item() for t in range(n)]

        metrics["success"] = [run["reward"] for _ in range(n)]

        metrics = [{k: metrics[k][t] for k in metrics.keys()} for t in range(n)]
        return metrics

    @staticmethod
    def to_csv(metrics, filename):
        with open(filename, "w") as csvfile:
            keys = list(metrics[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for data in metrics:
                writer.writerow(data)
