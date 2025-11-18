import numpy as np
import torch

from pytorch3d.transforms import matrix_to_quaternion
from pyquaternion import Quaternion


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


def normalize(v):
    return v / np.linalg.norm(v)


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


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


def look_at_matrix(from_vec, to_vec, earth_normal=None):
    forward = normalize(from_vec - to_vec)

    # random right vector...
    # we want this to be perpendicular to the plane
    # thus we take the product with [0, 1, 0]
    if earth_normal is None:
        earth_normal = normalize([0, 0, 1])
    right = normalize(np.cross(earth_normal, forward))
    up = normalize(np.cross(forward, right))

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = forward
    mat[:3, 3] = from_vec

    det = np.linalg.det(mat[:3, :3])
    mat[:3, :3] /= abs(det) ** (1 / 3)

    return mat


def look_at_matrix_torch(from_vec, to_vec, earth_normal=None):
    forward = from_vec - to_vec
    forward /= torch.linalg.norm(forward)

    # random right vector...
    # we want this to be perpendicular to the plane
    # thus we take the product with [0, 1, 0]
    if earth_normal is None:
        earth_normal = torch.tensor([0, 0, 1], dtype=from_vec.dtype)
        earth_normal /= torch.linalg.norm(earth_normal)
    right = torch.cross(earth_normal, forward)
    right /= torch.linalg.norm(right)
    up = torch.cross(forward, right)
    up /= torch.linalg.norm(up)

    mat = torch.eye(4, dtype=from_vec.dtype)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = forward
    mat[:3, 3] = from_vec

    det = torch.linalg.det(mat[:3, :3])
    mat[:3, :3] /= abs(det) ** (1 / 3)

    return mat


def distance(x, y):
    return np.linalg.norm(x-y) #np.sqrt(np.sum((x - y) ** 2))


def recompute_pose(pose):
    position = pose[:3, -1]
    lookat = pose @ torch.tensor([0, 0, -1, 1], dtype=pose.dtype)
    p_after = look_at_matrix_torch(position, lookat[:-1])
    return p_after
