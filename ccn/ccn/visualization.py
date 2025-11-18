import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ccn.util import to_img as torch_to_img

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

import torch
from torch.distributions import MultivariateNormal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from scene_environments.render import ycb_classes

from ccn.util import to_img

from PIL import Image
import io


def format_ax(a):
    a.set_facecolor("whitesmoke")
    a.grid("on", linestyle="dashed")
    a.set_axisbelow(True)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)


def together_particles_plot(
    particles_dict, envinfo, colors=None, fig=None, ax=None, legend=False
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    if colors is None:
        colors = {
            "002_master_chef_can": "blue",
            "003_cracker_box": "crimson",
            "004_sugar_box": "darkorange",
            "005_tomato_soup_can": "sienna",
            "006_mustard_bottle": "gold",
        }

    format_ax(ax)
    ax.set_title("Position belief")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0.5, -0.5])
    for k, pb in particles_dict.items():
        if isinstance(k, int):
            k = ycb_classes[k]
        ax.scatter(*pb[:500, :2].T, c=colors[k], marker=".", alpha=0.25)
    for k, v in envinfo.items():
        if k in colors.keys():
            ax.scatter(
                *v[:2, -1:],
                marker="o",
                color=colors[k],
                edgecolors="black",
                linewidths=2,
                label=k,
            )
    if legend:
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    return fig, ax


def plot_ims(ims, titles=None, suptitle="", savefig_path=None):
    """
    Expects images to be in torch format
    """
    n_ims = len(ims)
    if titles is None:
        titles = []

    titles += ["" for _ in range(n_ims - len(titles))]
    fig, ax = plt.subplots(1, n_ims, figsize=(n_ims * 4, 4))
    for a, i, t in zip(ax, ims, titles):
        a.imshow(to_img(i))
        a.set_title(t)
        a.axis("off")
    plt.suptitle(suptitle)
    if savefig_path is not None:
        plt.savefig(savefig_path, bbox_inches="tight")
    else:
        plt.show()


def mv2mat(x):
    y = torch.eye(4)
    y[:3, :3] = x.covariance_matrix
    y[:3, -1] = x.mean
    return y


def plot_particles(ax, fig, b):
    bel = b.global_pose
    im = ax.scatter(*bel.particles[:, :2].T, c=bel.weights, cmap="viridis", alpha=0.10)
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0.5, -0.5])
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")


def plot_pdf(ax, fig, b):
    if not isinstance(b, MultivariateNormal):
        mvn2d = MultivariateNormal(loc=b[:2, -1], covariance_matrix=b[:2, :2])
    else:
        mvn2d = MultivariateNormal(
            loc=b.mean[:2], covariance_matrix=b.covariance_matrix[:2, :2]
        )

    x = torch.linspace(-0.5, 0.5, 100)
    y = torch.linspace(-0.5, 0.5, 100)
    X, Y = torch.meshgrid(x, y)
    v = torch.cat(
        [X.reshape((len(x), len(x), 1)), Y.reshape((len(y), len(y), 1))],
        dim=2,
    )
    heatmap = torch.exp(mvn2d.log_prob(v)).reshape((len(x), len(y)))
    im = ax.imshow(heatmap.T.numpy(), extent=[-0.5, 0.5, 0.5, -0.5])
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")


def plot_pdfs(ax, fig, bs):
    heatmap = torch.zeros((100, 100))
    for b in bs:
        mvn2d = MultivariateNormal(b.mean[..., :2], b.covariance_matrix[..., :2, :2])

        x = torch.linspace(-0.5, 0.5, 100)
        y = torch.linspace(-0.5, 0.5, 100)
        X, Y = torch.meshgrid(x, y)
        v = torch.cat(
            [X.reshape((len(x), len(x), 1)), Y.reshape((len(y), len(y), 1))],
            dim=2,
        )
        hm = torch.exp(mvn2d.log_prob(v)).reshape((len(x), len(y)))
        # Renormalize
        hm /= hm.max()
        heatmap += hm
    im = ax.imshow(heatmap.T.numpy(), extent=[-0.5, 0.5, 0.5, -0.5])
    # cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax, orientation="vertical")


def plot_gaussian(ax, estimate):
    N = 100
    X = np.linspace(-0.5, 0.5, N)
    Y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(estimate[:2, -1], estimate[:2, :2])
    Z = rv.pdf(pos)
    ax.contour(X, Y, Z, cmap="viridis")


def plot_position_belief(ax, info, state, scene):
    # Plot ground truth object information
    for object_name, pose in info.items():
        x, y = pose[:2, -1]
        ax.plot(
            [x],
            [y],
            marker="x",
            linestyle="none",
            label=" ".join(object_name.split("_")[1:]),
        )

    # Plot current beliefs over the objects
    plot_camera_2d(state["camera_pose"], ax, color="crimson")

    for o, v in scene.objects.items():
        plot_gaussian(ax, v["position_belief"])

        # ray = v["rays"][-1].clone()
        # ax.plot([ray[0][0], ray[1][0]], [ray[0][1], ray[1][1]], linestyle='dashed', marker='o')

        # Plot the last observed camera pose
        plot_camera_2d(v["cam_poses"][-1], ax, color="black")

    # Plot the table contour
    ax.plot([0.5, 0.5], [-0.5, 0.5], color="black")
    ax.plot([0.5, -0.5], [-0.5, -0.5], color="black")
    ax.plot([-0.5, -0.5], [-0.5, 0.5], color="black")
    ax.plot([-0.5, 0.5], [0.5, 0.5], color="black")

    # Set limits
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])

    # Place legends and labels
    ax.legend(loc=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_frames(frames, w, h, store_path):
    to_img = lambda x: x
    if len(frames[0].shape) == 3:
        to_img = torch_to_img

    fig, ax = plt.subplots(w, h, figsize=(h * 4, w * 4))
    [a.axis("off") for a in ax.flatten()]
    for i in range(w):
        for j in range(h):
            ax[i, j].imshow(to_img(frames[i * h + j]))
    plt.savefig(store_path, bbox_inches="tight")
    plt.clf()


def plot_scores(scores, w, h, store_path, categorical=False):
    fig, ax = plt.subplots(w, h, figsize=(h * 4, w * 4))
    [a.axis("off") for a in ax.flatten()]
    for i in range(w):
        for j in range(h):
            if not categorical:
                im = ax[i, j].imshow(
                    scores[i * h + j].reshape(1, 1).detach().numpy(),
                    vmin=0,
                    vmax=1,
                )
            else:
                ax[i, j].imshow(
                    scores[i * h + j].reshape(1, 1).detach().numpy(),
                    vmin=0,
                    vmax=len(categorical),
                    cmap="Accent",
                )

    if not categorical:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.05, 0.10, 0.90])
        cbar_ax.tick_params(labelsize=30)
        fig.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=Normalize(vmin=scores.min(), vmax=scores.max())
            ),
            cax=cbar_ax,
        )

    plt.savefig(store_path, bbox_inches="tight")
    plt.clf()

    if categorical:
        fig, ax = plt.subplots(1, len(categorical), figsize=(len(categorical), 1))
        [a.axis("off") for a in ax.flatten()]
        for i in range(len(categorical)):
            ax[i].imshow(
                np.array([i]).reshape(1, 1),
                vmin=0,
                vmax=len(categorical),
                cmap="Accent",
            )
            ax[i].set_title("\n".join(categorical[i].split("_")[1:]))
        plt.savefig(
            str(store_path).replace(".jpeg", "_legend.jpeg"),
            bbox_inches="tight",
        )


def vec_to_quiver(matrix, vec):
    u, v, w = matrix[:3, :3] @ vec
    x, y, z = matrix[:3, -1]
    return x, y, z, u, v, w


def plot_axis(tf, ax, colors=None, length=0.25):
    if colors is None:
        # colors = ["crimson", "lime", "cyan"]
        colors = ["tab:red", "tab:green", "tab:blue"]
    if not isinstance(length, list):
        length = [length, length, length]
    x, y, z, u, v, w = vec_to_quiver(tf, [1, 0, 0])
    ax.quiver(x, y, z, u, v, w, length=length[0], color=colors[0])
    x, y, z, u, v, w = vec_to_quiver(tf, [0, 1, 0])
    ax.quiver(x, y, z, u, v, w, length=length[1], color=colors[1])
    x, y, z, u, v, w = vec_to_quiver(tf, [0, 0, 1])
    ax.quiver(x, y, z, u, v, w, length=length[2], color=colors[2])


def plot_camera_2d(tf, ax, color="black", length=0.0775):
    vecs = length * np.array([[-1, -1, -3], [1, 1, -3]])
    x1, y1, _, u1, v1, _ = vec_to_quiver(tf, vecs[0])
    x2, y2, _, u2, v2, _ = vec_to_quiver(tf, vecs[1])

    n1 = np.linalg.norm([u1, v1])
    n2 = np.linalg.norm([u2, v2])

    ax.plot([x1, x1 + length * u1 / n1], [y1, y1 + length * v1 / n1], color=color)
    ax.plot([x2, x2 + length * u2 / n2], [y2, y2 + length * v2 / n2], color=color)
    ax.plot(
        [x1 + length * u1 / n1, x2 + length * u2 / n2],
        [y1 + length * v1 / n1, y2 + length * v2 / n2],
        color=color,
    )

    ax.scatter([x1], [y1], marker="o", s=100, color=color)


def plot_camera(tf, ax, color="black", length=0.0775, label=None, opengl=True):
    if opengl:
        vecs = length * np.array([[-1, -1, -3], [-1, 1, -3], [1, 1, -3], [1, -1, -3]])
    else:
        raise NotImplementedError

    new_vec = []
    x, y, z, u, v, w = vec_to_quiver(tf, vecs[-1])
    last_vec = [x + u, y + v, z + w]
    ax.scatter([x], [y], [z], marker="o", s=100, color=color, label=label)
    for v in vecs:
        x, y, z, u, v, w = vec_to_quiver(tf, v)
        ax.plot(
            [x, x + u],
            [y, y + v],
            [z, z + w],
            color=color,
            linestyle="-",
        )

        ax.plot(
            [x + u, last_vec[0]],
            [y + v, last_vec[1]],
            [z + w, last_vec[2]],
            color=color,
            linestyle="-",
        )
        last_vec = [x + u, y + v, z + w]


def plot_points(ground_truths, estimates, camera_pose):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    plot_axis(camera_pose, ax)

    plt.plot(
        ground_truths[:, 0],
        ground_truths[:, 1],
        ground_truths[:, 2],
        marker="o",
        linestyle="none",
        color="black",
    )

    plt.plot(
        estimates[:, 0],
        estimates[:, 1],
        estimates[:, 2],
        marker="x",
        linestyle="none",
        color="crimson",
    )

    plt.plot(
        [estimates[:, 0].mean()],
        [estimates[:, 1].mean()],
        [estimates[:, 2].mean()],
        marker="o",
        linestyle="none",
        color="crimson",
    )

    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Y$")
    ax.set_zlabel(r"$Z$")

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 0.5])
    plt.show()


def remove_axis(ax):
    # Hide grid lines
    ax.grid(True)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


def plot_run(run_dir):
    run_dir = Path(run_dir)
    x = np.load(str(run_dir / "state_dicts.npz"), allow_pickle=True)["arr_0"]
    poses_gt = [i["goal_pose"] for i in x]
    poses_ag = [i["camera_pose"] for i in x]

    object_poses = np.load(str(run_dir / "info_dict.npz"), allow_pickle=True)["arr_0"]
    gt_objects = [i["groundtruth"] for i in object_poses]
    es_objects = [i["estimated"] for i in object_poses]

    seq = []

    trajectory = []
    for pgt, pag, gt_obs, es_obs in zip(poses_gt, poses_ag, gt_objects, es_objects):
        fig = plt.figure(figsize=(6, 5))
        ax = plt.axes(projection="3d")

        # Set up view
        plot_axis(
            np.eye(4),
            ax,
            ["tab:red", "tab:green", "tab:blue"],
            [0.15, 0.15, 0.15],
        )
        ax.set_xlim([-0.5, 0.5])  # 1
        ax.set_ylim([-0.5, 0.5])  # .5
        ax.set_zlim([0, 1.0])  # .3
        ax.view_init(elev=25.0, azim=-30)  # -45)

        # remove_axis(ax)

        # plot camera's
        plot_camera(pgt, ax, "crimson", label="Target pose")
        plot_camera(pag, ax, "skyblue", label="Current pose")

        trajectory += [pag[:3, -1]]
        v = 1.0
        for p1, p2 in zip(trajectory[1:], trajectory[:-1]):
            x = [p1[0], p2[0]]
            y = [p1[1], p2[1]]
            z = [p1[2], p2[2]]
            color = plt.get_cmap("Blues")(v)
            ax.plot(x, y, z, color=color)
            v -= 0.10  # only plot the final 10

        # Plot object poses
        for key, pose in gt_obs.items():
            x, y, z = pose[:3, -1]
            ax.plot(
                [x, x],
                [y, y],
                [0, 2 * z],
                linestyle="dashed",
                marker=".",
                label="Groundtruth " + " ".join(key.split("_")[1:]),
            )

        for key, pose in es_obs.items():
            x, y, z = pose[:3, -1]
            ax.plot(
                [x, x],
                [y, y],
                [0, 2 * z],
                linestyle="dashed",
                marker="x",
                label="Estimated " + " ".join(key.split("_")[1:]),
            )

        ax.set_xlabel(r"$\mathbf{x}$")
        ax.set_ylabel(r"$\mathbf{y}$")
        ax.set_zlabel(r"$\mathbf{z}$")

        # legend
        plt.legend()

        fig.tight_layout()
        fig.canvas.draw()
        plt.clf()

        # grab the pixel buffer and dump it into a numpy array
        seq += [np.array(fig.canvas.renderer.buffer_rgba())]

    io.mimwrite(str(run_dir / "trajectory.gif"), seq)
    return seq
