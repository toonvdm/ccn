import numpy as np

from dataclasses import dataclass


@dataclass
class Transforms:
    """
    transformation matrices useful for transferring between
    different frames.
    - VIEW frame is the frame directly observed by the camera:
        - x is the horizontal axis (left is positive)
        - y is the vertical axis (up is positive)
        - z is the axis going out of the camera (camera points positive z)
    - OPENGL frame is the frame in which OPENGL camera's are described
        - the camera is pointing in the negative z-axis
        - x is the horizontal axis of the image (right is positive)
        - y is the vertical axis of the image (up is positive)
    """

    VIEW_TO_OPENGL = np.asarray(
        [
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )


def invert_transform(transform):
    """
    Compute the inverted transform
    :param transform: homogeneous transformation matrix
    :return: inverted homogeneous transformation matrix
    """
    tf_inv = np.eye(4)
    inv_rot = transform[:3, :3].T
    tf_inv[:3, :3] = inv_rot
    tf_inv[:3, -1] = -inv_rot @ transform[:3, -1]
    return tf_inv


def chain_transforms(transforms):
    """
    Chain a sequence of transforms, the first element of
    the sequence will be applied first.
    :param transforms: list of homogeneous (4x4) transforms
    :return: combination of transforms
    """
    chained = np.eye(4)
    for tf in transforms:
        chained = tf @ chained
    return chained


def apply_transform(transform, point):
    """
    Applies a transform on a point
    :param transform: a homogeneous (4x4) transformation matrix
    :param point: a point or array of points in 3D space, dim: (..., 3) or (3)
    :return: the transformed point or array of points in the same dim as point
    """
    if len(point.shape) == 1:
        point = point.reshape(1, -1)
    new_point = np.ones((*point.shape[:-1], 4))
    new_point[..., :3] = point
    return (
        (transform @ new_point.reshape(-1, 4).T).T[:, :3].reshape(*point.shape)
    )
