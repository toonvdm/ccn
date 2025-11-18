import numpy as np
import torch

from pyquaternion import Quaternion
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from ccn.geometry import look_at_matrix


def normalize(v):
    return v / np.linalg.norm(v)


def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def distance_torch(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


def interpolate_pose(p1, p2, dist, look_at):
    translate = p2[:3, -1] - p1[:3, -1]
    scale = np.min(
        [1, dist / (1e-8 + distance(translate.numpy(), np.zeros(3)))]
    )
    new_pos = p1[:3, -1] + scale * translate
    return look_at_matrix(new_pos.to(dtype=p1.dtype), look_at)


def interpolate_pose_with_orientation(p1, p2, percentage):
    dtype = torch.float64

    pos1 = p1[:3, -1]
    q = matrix_to_quaternion(p1[:3, :3].unsqueeze(0))[0].detach().numpy()
    q1 = Quaternion(q)

    pos2 = p2[:3, -1]
    q = matrix_to_quaternion(p2[:3, :3].unsqueeze(0))[0].detach().numpy()
    q2 = Quaternion(q)

    interpolated = torch.eye(4, dtype=dtype)
    interpolated[:3, -1] = pos1 + percentage * (pos2 - pos1)
    interpolated[:3, :3] = torch.tensor(
        Quaternion.slerp(q1, q2, percentage).rotation_matrix, dtype=dtype
    )

    return interpolated


def interpolate_pose_with_orientation_step_size(p1, p2, step_size):

    norm = torch.linalg.norm(p1[:3, -1] - p2[:3, -1]).item()
    percentage = min([step_size / norm, 1])

    dtype = torch.float64

    pos1 = p1[:3, -1]
    q = matrix_to_quaternion(p1[:3, :3].unsqueeze(0))[0].detach().numpy()
    q1 = Quaternion(q)

    pos2 = p2[:3, -1]
    q = matrix_to_quaternion(p2[:3, :3].unsqueeze(0))[0].detach().numpy()
    q2 = Quaternion(q)

    interpolated = torch.eye(4, dtype=dtype)
    interpolated[:3, -1] = pos1 + percentage * (pos2 - pos1)
    interpolated[:3, :3] = torch.tensor(
        Quaternion.slerp(q1, q2, percentage).rotation_matrix, dtype=dtype
    )

    return interpolated


def rot_x(theta):
    rotate_x = np.eye(4)
    rotate_x[1][1] = np.cos(theta)
    rotate_x[1][2] = -np.sin(theta)
    rotate_x[2][1] = np.sin(theta)
    rotate_x[2][2] = np.cos(theta)
    return rotate_x


def rot_y(theta):
    rotate_y = np.eye(4)
    rotate_y[0][0] = np.cos(theta)
    rotate_y[2][0] = -np.sin(theta)
    rotate_y[0][2] = np.sin(theta)
    rotate_y[2][2] = np.cos(theta)
    return rotate_y


def rot_z(theta):
    rotate_z = np.eye(4)
    rotate_z[0][0] = np.cos(theta)
    rotate_z[0][1] = -np.sin(theta)
    rotate_z[1][0] = np.sin(theta)
    rotate_z[1][1] = np.cos(theta)
    return rotate_z


def trajectory_z(d=0.275, n=25):
    trajectory = np.zeros((n, 4, 4))
    for i, theta in enumerate(np.linspace(-np.pi, np.pi, n)):
        trajectory[i] = look_at_matrix(
            np.array([*sph2cart(0, 0, d)]), np.zeros(3)
        ) @ rot_z(theta)
    return torch.tensor(trajectory, dtype=torch.float32)


def trajectory_azimuth(d=0.275, n=25):
    trajectory = np.zeros((n, 4, 4))
    for i, azimuth in enumerate(np.linspace(0, np.pi, n)):
        trajectory[i] = look_at_matrix(
            np.array([*sph2cart(azimuth, 0, d)]), np.zeros(3)
        )
    return torch.tensor(trajectory, dtype=torch.float32)


def trajectory_elevation(d=0.275, n=25):
    trajectory = np.zeros((n, 4, 4))
    for i, elevation in enumerate(np.linspace(-np.pi / 2, np.pi / 2, n // 2)):
        trajectory[i] = look_at_matrix(
            np.array([*sph2cart(0, elevation, d)]), np.zeros(3)
        )
    for i, elevation in enumerate(
        np.linspace(np.pi / 2, -np.pi / 2, n - n // 2)
    ):
        trajectory[i + n // 2] = look_at_matrix(
            np.array([*sph2cart(np.pi, elevation, d)]), np.zeros(3)
        ) @ rot_z(np.pi)
    return torch.tensor(trajectory, dtype=torch.float32)


def trajectory_r(d_min=0.13, n=25):
    trajectory = np.zeros((n, 4, 4))
    for i, d in enumerate(np.linspace(d_min, 0.5, n // 2)):
        trajectory[i] = look_at_matrix(
            np.array([*sph2cart(np.pi, 0, d)]), np.zeros(3)
        )
    for i, d in enumerate(np.linspace(0.5, d_min, n - n // 2)):
        trajectory[n // 2 + i] = look_at_matrix(
            np.array([*sph2cart(np.pi, 0, d)]), np.zeros(3)
        )
    return torch.tensor(trajectory, dtype=torch.float32)


def get_random_pose(z_pos=True):
    an = np.random.random() * 2 * np.pi
    en = np.random.random() * np.pi / 2
    r = np.random.random() * 0.35 + 0.15
    x, y, z = sph2cart(an, en, r)
    return np.array([x, y, z])


def get_poses_sphere(n, lookat=None, theta=False):
    if lookat is None:
        lookat = np.zeros(3)

    if theta is False:
        poses = np.zeros((n, n, n, 4, 4))
        for i, an in enumerate(np.linspace(0, 2 * np.pi, n)):
            for j, en in enumerate(np.linspace(0, np.pi / 2, n)):
                for k, r in enumerate(np.linspace(0.25, 0.40, n)):
                    cart = sph2cart(an, en, r)
                    poses[i, j, k] = look_at_matrix(cart, lookat)
    else:
        poses = np.zeros((n, n, n, n, 4, 4))
        for i, an in enumerate(np.linspace(0, 2 * np.pi, n)):
            for j, en in enumerate(np.linspace(-np.pi / 2, np.pi / 2, n)):
                for k, r in enumerate(np.linspace(0.25, 0.40, n)):
                    for l, theta in enumerate(np.linspace(-np.pi, np.pi, n)):
                        cart = sph2cart(an, en, r)
                        poses[i, j, k, l] = look_at_matrix(
                            cart, lookat
                        ) @ rot_z(theta)

    return torch.tensor(poses.reshape((-1, 4, 4)), dtype=torch.float64)


def get_random_poses(n, lookat=None, theta=None):
    if lookat is None:
        lookat = np.zeros(3)

    poses = np.zeros((n, 4, 4))
    for i in range(n):
        an = np.random.random() * np.pi
        en = np.random.random() * np.pi - np.pi / 2
        r = np.random.random() * 0.35 + 0.15
        if theta is None:
            theta = np.random.random() * np.pi - np.pi / 2

        cart = sph2cart(an, en, r)
        poses[i] = look_at_matrix(cart, lookat) @ rot_z(theta)
    return poses


def get_random_neighbourhood_poses(n, current_pose, lookat=None, theta=None):
    if lookat is None:
        lookat = np.zeros(3)

    poses = np.zeros((n, 4, 4))
    pos = current_pose[:3, -1]
    for i in range(n):
        n_pos = np.zeros(3)
        n_pos[0] = np.clip(pos[0] + 0.05 * (np.random.random() * 2 - 1), -1, 1)
        n_pos[1] = np.clip(pos[1] + 0.05 * (np.random.random() * 2 - 1), -1, 1)
        n_pos[2] = np.clip(pos[2] + 0.05 * (np.random.random() * 2 - 1), 0, 1)
        if theta is None:
            theta = np.pi / 18 * (np.random.random() * 2 - 1)
        poses[i] = look_at_matrix(n_pos, lookat) @ rot_z(theta)
    return poses


def get_neighbourhood_samples(n, current_pose, lookat=None, theta=False):
    # Do not consider theta to have paper results
    if lookat is None:
        lookat = np.zeros(3)

    cp = current_pose[:3, -1]
    if not theta:
        poses = np.zeros((n, n, n, 4, 4))
        for i, xi in enumerate(np.linspace(-0.1, 0.1, n)):
            for j, yi in enumerate(np.linspace(-0.1, 0.1, n)):
                for k, zi in enumerate(np.linspace(-0.1, 0.1, n)):
                    poses[i, j, k] = look_at_matrix(
                        [cp[0] + xi, cp[1] + yi, cp[2] + zi], lookat
                    )
    else:
        poses = np.zeros((n, n, n, n, 4, 4))
        for i, xi in enumerate(np.linspace(-0.1, 0.1, n)):
            for j, yi in enumerate(np.linspace(-0.1, 0.1, n)):
                for k, zi in enumerate(np.linspace(-0.1, 0.1, n)):
                    for l, theta in enumerate(np.linspace(-0.2, 0.2, n)):
                        poses[i, j, k, l] = look_at_matrix(
                            [cp[0] + xi, cp[1] + yi, cp[2] + zi], lookat
                        ) @ rot_z(theta)

    return torch.tensor(poses.reshape((-1, 4, 4)), dtype=torch.float32)


def is_valid_rotation_matrix(r):
    check1 = np.allclose(r @ r.T, np.eye(r.shape[0]))
    check2 = np.allclose(np.linalg.det(r), 1)
    if not check1:
        print("> r.T does not equal r.inv")
    if not check2:
        print("> Det is not 1")
    return check1 and check2


def recompute_pose(pose):
    position = pose[:3, -1]
    lookat = pose @ torch.tensor([0, 0, -1, 1], dtype=pose.dtype)
    p_after = look_at_matrix(position, lookat[:-1])
    return p_after


def to_vector(mat):
    translations = mat[:, :3, -1]
    orientations = matrix_to_quaternion(mat[:, :3, :3])
    return torch.cat([translations, orientations], dim=1)


def to_mat(vector):
    mat = torch.eye(4).unsqueeze(0).repeat(vector.shape[0], 1, 1)
    mat[:, :3, :3] = quaternion_to_matrix(vector[:, 3:])
    mat[:, :3, -1] = vector[:, :3]
    return mat
