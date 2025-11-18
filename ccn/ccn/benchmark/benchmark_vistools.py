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
from scene_environments.errors import rotation_error as re

from skimage.transform import resize

from ccn.geometry import cartesian_to_spherical

from pathlib import Path

from ccn.benchmark.moveto import *


def plot_metrics(logs, store_path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    [format_ax(a) for a in ax.flatten()]
    ax[0].set_title("Azimuth")
    spherical_errors = [
        spherical_error(logs, t) for t in range(len(logs["env_info"]))
    ]
    ax[0].plot([s[0] for s in spherical_errors], color="black")
    ax[1].set_title("Elevation")
    ax[1].plot([s[1] for s in spherical_errors], color="black")
    ax[2].set_title("Range")
    ax[2].plot([s[2] for s in spherical_errors], color="black")
    plt.savefig(str(store_path / "metrics.png"), bbox_inches="tight")
    plt.close()


def plot_prefences(benchmark):
    preferences = []
    for i in range(len(benchmark)):
        env = SceneEnvironment.load_pkl(benchmark._envfiles[i])
        preferences += [env.state["goal_rgb"]]
    fig, ax = plt.subplots(5, 4, figsize=(20, 16))
    [a.axis("off") for a in ax.flatten()]
    for i, a in zip(preferences, ax.flatten()):
        a.imshow(i)
    plt.show()


# Visualization tools
def plot_final(logs, store_path):
    ims = []

    state = logs["env_info"][-1]

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    plt.suptitle(f"Step {len(logs['env_info'])}")
    ax[0].imshow(state["goal_rgb"])
    ax[0].set_title("Preference")
    ax[1].imshow(state["rgb"])
    ax[1].set_title("Observation")
    [a.axis("off") for a in ax.flatten()[:-1]]

    envinfo = logs["env_info"][-1]
    l = logs["agent_info"][-1]
    together_particles_plot(l["particles"], envinfo, fig=fig, ax=ax[2])
    # add Camera
    tcp = l.get("target_camera_pose")
    if tcp is not None:
        plot_camera_2d(tcp, ax[2], color="green")
    plot_camera_2d(state["goal_pose"], ax[2], color="purple")
    plot_camera_2d(state["camera_pose"], ax[2], color="black")
    plt.savefig(store_path / f"final.png", bbox_inches="tight")


# Visualization tools
def create_position_belief_gif(logs, store_path):
    ims = []
    for i, l in tqdm(list(enumerate(logs["agent_info"][1:-1]))):
        state = logs["env_info"][i + 1]

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        plt.suptitle(f"Step {i}")
        ax[0].imshow(state["goal_rgb"])
        ax[0].set_title("Preference")
        ax[1].imshow(state["rgb"])
        ax[1].set_title("Observation")
        [a.axis("off") for a in ax.flatten()[:-1]]

        envinfo = logs["env_info"][i + 1]
        together_particles_plot(l["particles"], envinfo, fig=fig, ax=ax[2])
        # add Camera
        tcp = logs["agent_info"][i + 1].get("target_camera_pose")
        if tcp is not None:
            plot_camera_2d(tcp, ax[2], color="green")
        plot_camera_2d(state["goal_pose"], ax[2], color="purple")
        plot_camera_2d(state["camera_pose"], ax[2], color="black")

        if i == len(logs["agent_info"][1:-1]) - 1:
            plt.savefig(store_path / f"final.png", bbox_inches="tight")

        im = fig2img(fig)
        ims.append(im)
        plt.close()

    ims += [ims[-1] for _ in range(15)]

    imageio.mimsave(str(store_path / "position_belief.gif"), ims)


def pad(x):
    y = np.zeros((66, 66, 3))
    y[1:-1, 1:-1, :] = resize(x, (64, 64, 3))
    return y


def gif(logs, store_path):
    seq = [
        (255 * np.hstack([pad(s["image_goal"]), pad(s["image"])])).astype(
            np.uint8
        )
        for s in logs["state"]
    ]
    imageio.mimsave(str(store_path / "sequence.gif"), seq)


def plot_efe_decomposition(logs, store_path):
    pos_epi, pos_ins, dis_ins, efes = [0], [0], [0], [0]
    pose_ins = [0]
    rs = [0]
    highlevels = []
    internal_goals = dict({})
    for step, (info) in enumerate(logs["agent_info"][1:]):
        if info.get("action_mode") == action_modes.HIGHLEVEL.value:
            highlevels += [step]

        if len(info.get("g_info", dict())) > 0:
            g_info = info.get("g_info")
            efe = g_info.get("G")
            idx = efe.argmin()
            efes += [g_info["G"][idx]]
            pos_epi += [g_info["epistemic"]["position"][idx]]
            if g_info.get("instrumental", dict()).get("pose") is not None:
                pos_ins += [g_info["instrumental"]["pos"][idx]]
                pose_ins += [g_info["instrumental"]["pose"][idx]]
                dis_ins += [g_info["instrumental"]["distance"][idx]]
                rs += [g_info["instrumental"]["r"][idx]]
            else:
                pos_ins += [0]
                pose_ins += [0]
                dis_ins += [0]
                rs += [0]

            if idx.item() != 0:
                internal_goals[step] = info.get("x_hat")
        else:
            pos_epi += [pos_epi[-1]]
            pos_ins += [pos_ins[-1]]
            pose_ins += [pose_ins[-1]]
            dis_ins += [dis_ins[-1]]
            rs += [rs[-1]]
            efes += [efes[-1]]

    changed = list(internal_goals.keys())
    fig, ax = plt.subplots(1, 5, figsize=(16, 5))
    plt.suptitle("Expected Free Energy Decomposition (minimum value for G)")
    [format_ax(a) for a in ax.flatten()]
    ax[0].plot(pos_epi)
    ax[0].scatter(
        changed, [pos_epi[hl] for hl in changed], marker="o", color="red"
    )
    ax[0].set_title("Infogain (position)")

    ax[1].plot(np.array(pose_ins))
    ax[1].scatter(
        changed, [pose_ins[hl] for hl in changed], marker="o", color="red"
    )
    ax[1].set_title("Instrumental (pose)")

    ax[2].plot(np.array(dis_ins))
    ax[2].scatter(
        changed, [dis_ins[hl] for hl in changed], marker="o", color="red"
    )
    ax[2].set_title("Instrumental (distance)")

    ax[4].plot(efes)
    ax[4].scatter(
        changed, [efes[hl] for hl in changed], marker="o", color="red"
    )
    ax[4].set_title("Expected Free Energy")

    ax[3].plot(pos_ins)
    ax[3].set_title("Position Instrumental")
    ax[3].scatter(
        changed, [pos_ins[hl] for hl in changed], marker="o", color="red"
    )

    [a.set_xlabel("time") for a in ax.flatten()]
    plt.savefig(str(store_path / "g_decomposition.png"), bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(
        1,
        2 + len(internal_goals.keys()),
        figsize=(4 * (1 + len(internal_goals.keys())), 4),
    )
    plt.suptitle("Internal Goals", fontsize=20)
    ax.flatten()[0].imshow(logs["env_info"][0]["goal_rgb"])
    ax.flatten()[0].set_title("Preference", fontsize=15)
    for a, (k, v) in zip(ax.flatten()[1:], internal_goals.items()):
        a.imshow(to_img(v[0]))
        a.set_title(k, fontsize=15)
    [a.axis("off") for a in ax.flatten()]
    plt.savefig(str(store_path / "internal_goals.png"), bbox_inches="tight")
    plt.close()

    x = np.arange(0, len(highlevels))
    y = [i in highlevels for i in x]
    plt.plot(x, y)
    plt.savefig(str(store_path / "highlevel.png"), bbox_inches="tight")
    plt.close()
