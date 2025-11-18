import numpy as np

from dommel.geometry.transforms import apply_transform


def uvd_to_coord_view_frame(u, v, d, intrinsics_k):
    """
    Method to convert pixel depth value to coordinates in
    view frame.
    u, v and d should be in the same shape, can either be
    single pixel are full array
    args:
    - u: width pixels
    - v: height pixels
    - d: depth values
    - intrinsics_k: intrinsic matrix
    returns:
    - x: x values of shape d.shape
    - y: y values of shape d.shape
    - z: z values of shape d.shape
    """
    fx = intrinsics_k[0][0]
    fy = intrinsics_k[1][1]
    cx = intrinsics_k[0][2]
    cy = intrinsics_k[1][2]

    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy

    z = d
    x = x_over_z * z
    y = y_over_z * z

    # invert for mirroring through camera lens
    return -x, -y, z


def depth_to_view_frame(depth_map, intrinsics_k):
    """
    args:
    - depth_map: the depth map of shape (h, w)
    - intrinsics_k: the intrinsic matrix of the camera of shape (3, 3)
    out:
    - coordinates in view space of shape (h, w, 3)
    """
    h, w = depth_map.shape
    u_map, v_map = np.meshgrid(np.arange(w), np.arange(h))

    x, y, z = uvd_to_coord_view_frame(u_map, v_map, depth_map, intrinsics_k)

    coords = np.zeros((*x.shape, 3))
    coords[..., 0] = x
    coords[..., 1] = y
    coords[..., 2] = z

    return coords


def depth_to_target_frame(depth_map, intrinsics_k, view_to_target_frame):
    """
    Convert the depth map to a coordinate frame of choice
    :param depth_map: depth map (2d)
    :param intrinsics_k: intrinsics matrix of the camera
    :param view_to_target_frame: the (chained) transform  to go from
        view frame to the desired coordinate frame
    :return: the transformed points, in the same shape of depth_map (w, h, 3)
    """
    coords_view_frame = depth_to_view_frame(depth_map, intrinsics_k)
    coords_target_frame = apply_transform(
        view_to_target_frame, coords_view_frame
    )
    return coords_target_frame
