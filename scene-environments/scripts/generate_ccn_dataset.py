from pathlib import Path
from tqdm import tqdm
import numpy as np

from skimage.transform import resize

from scene_environments.render import Renderer
from scene_environments.geometry import (
    sph2cart,
    look_at_matrix,
    rot_z,
    distance,
)
from scene_environments.constants import realsense_intrinsics

from dommel.datasets.utils import store_h5
import math

from imageio import mimsave


def random_poses(n, r_min, r_max=0.5, z_near=0.05, lookat=None):
    if lookat is None:
        lookat = np.zeros(3)

    r_min = r_min + z_near

    poses = np.zeros((n, 4, 4))
    for i in range(n):
        an = np.random.random() * 2 * np.pi
        en = np.random.random() * np.pi - np.pi / 2
        r = np.random.random() * (r_max - r_min) + r_min
        theta = np.random.random() * 2 * np.pi
        cart = sph2cart(an, en, r)
        poses[i] = look_at_matrix(cart, lookat) @ rot_z(theta)
    return poses


def random_poses_fib(n, r_min, r_max=0.5, z_near=0.05, lookat=None):

    if lookat is None:
        lookat = np.zeros(3)

    r_min = r_min + z_near

    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

    poses = np.zeros((n, 4, 4))
    alphas = np.zeros(n)
    for i in range(n):

        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        r = np.random.random() * (r_max - r_min) + r_min
        cart = r * np.array([x, y, z])

        alphas[i] = np.random.random() * np.pi * 2
        poses[i] = look_at_matrix(cart, lookat) @ rot_z(alphas[i])

    return poses, alphas


def create_fixed_point_dataset(
    mesh,
    result_dir,
    fib=True,
    resolution=(480, 480),
    z_near=0.05,
    r_max=0.5,
    n=10000,
):

    scale = resolution[1] / 480

    renderer = Renderer(
        {str(mesh): np.eye(4)},
        scale * realsense_intrinsics,
        resolution=((int(scale * 640), int(scale * 480))),
        crop_shape=resolution,
        background_color=(0, 0, 0),
        object_centric=True,
    )

    mesh_name, mesh = list(renderer.meshes.items())[0]
    r_min = distance(mesh.centroid, mesh.bounds[0])

    rgbs = []
    ds = []

    if fib:
        poses = random_poses_fib(n, r_min, z_near=z_near, r_max=r_max)
    else:
        poses = random_poses(n, r_min, z_near=z_near, r_max=r_max)

    for i, p in enumerate(poses):
        if i % 100 == 0:
            print(f"{mesh}:\t {i:04d}/{len(poses)}")
        rgb, d, _ = renderer.render(p)
        rgbs += [resize(rgb.copy() / 255.0, (64, 64, 3))]
        ds += [resize(d, (64, 64))]

    data_dict = {
        "intrinsic_matrix": realsense_intrinsics,
        "rgb": np.asarray(rgbs),
        "depth": np.asarray(ds),
        "pose_matrix": np.asarrunconstraineday(poses),
    }

    store_h5(data_dict, str(result_dir / f"{Path(mesh_name).parent.stem}.h5"))
    del data_dict


def create_fixed_point_radius_dataset(
    mesh,
    result_dir,
    resolution=(480, 480),
    z_near=0.05,
    n=10000,
):

    scale = resolution[1] / 480

    renderer = Renderer(
        {str(mesh): np.eye(4)},
        scale * realsense_intrinsics,
        resolution=((int(scale * 640), int(scale * 480))),
        crop_shape=resolution,
        background_color=(0, 0, 0),
        object_centric=True,
    )

    mesh_name, mesh = list(renderer.meshes.items())[0]

    d = 0.35

    rgbs = []
    ds = []

    poses, alphas = random_poses_fib(n, d, z_near=z_near, r_max=d)

    # import matplotlib.pyplot as plt
    # from ccn.experiment_plotting import plot_axis

    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # plt.suptitle(d)
    # for p in poses:
    #     plot_axis(p, ax, colors=["red", "green", "blue"], length=0.125)
    # # ax.set_xlim([-2, 2])
    # # ax.set_ylim([-2, 2])
    # # ax.set_zlim([-2, 2])
    # plt.show()

    for i, p in enumerate(poses):
        if i % 100 == 0:
            print(f"{mesh}:\t {i:04d}/{len(poses)}")

        rgb, d, _ = renderer.render(p)
        rgbs += [resize(rgb.copy() / 255.0, (64, 64, 3))]
        ds += [resize(d, (64, 64))]

    data_dict = {
        "intrinsic_matrix": realsense_intrinsics,
        "rgb": np.asarray(rgbs),
        "depth": np.asarray(ds),
        "alpha": np.asarray(alphas),
        "pose_matrix": np.asarray(poses),
    }

    # Debug stuff
    store_h5(data_dict, str(result_dir / f"{Path(mesh_name).parent.stem}.h5"))
    rgbs_sorted = []
    pos = poses.copy()[:, 2, -1]
    for idx in pos.argsort():
        rgbs_sorted += [rgbs[idx]]

    mimsave(
        str(result_dir / f"{Path(mesh_name).parent.stem}.gif"), rgbs_sorted
    )
    del data_dict


if __name__ == "__main__":
    n = 0.5

    r_max = 1  # 0.5
    z_near = 0.05
    res = (480, 480)
    fib = True

    data_dir = Path(__file__).absolute().parent.parent / "../data"
    result_dir = data_dir / f"output/ccn-dataset-{n}k-{'fib' * fib}-SDD-rot"
    result_dir.mkdir(exist_ok=True, parents=True)

    print(data_dir)

    meshes = sorted(
        list(
            (data_dir / "meshes/ycb_gazebo_sdf/").glob(
                "*_textured/textured.dae"
            )
        )
    )

    already_rendered = [m.stem for m in list(result_dir.glob("*.h5"))]
    meshes = [m for m in meshes if m.parent.stem not in already_rendered]

    for m in tqdm(meshes):
        create_fixed_point_radius_dataset(
            m,
            result_dir,
            resolution=res,
            z_near=z_near,
            n=int(n * 1000),
        )
