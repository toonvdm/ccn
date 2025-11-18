import numpy as np


def normalize(v):
    """
    Normalizes a vector v
    :param v: vector
    :return: normalized vector
    """
    return v / np.linalg.norm(v)


def look_at(source, target, earth_normal=None):
    """
    :param source: vector describing the position of the camera
    :param target: vector describing the position of the object to look at
    :param earth_normal: the normal on the earth plane in local reference
        frame. This makes sure the camera is pointing upright
    :return: a homogeneous transformation matrix describing the pose of the
        camera in view frame (camera points in the positive z-axis)
    """
    forward = normalize(target - source)

    # we want the right vector to be perpendicular to the plane
    # thus we take the product with the earth normal
    if earth_normal is None:
        earth_normal = normalize(np.array([0, 0, 1]))
    right = normalize(np.cross(earth_normal, forward))
    up = normalize(np.cross(forward, right))

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = forward
    mat[:3, 3] = source

    mat[:3, :3] /= np.linalg.det(mat[:3, :3])

    return mat
