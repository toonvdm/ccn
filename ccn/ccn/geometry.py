import torch


def cartesian_to_spherical(coord):
    # Extract coordinates to have shape [..., 1]
    # Which is easy to cat later
    x = coord[..., 0:1]
    y = coord[..., 1:2]
    z = coord[..., 2:3]

    hxy = torch.hypot(x, y)
    r = torch.hypot(hxy, z)

    # Get it in the [0, 2 Pi] ranges
    azimuth = torch.atan2(y, x) % (2 * torch.pi)  # [0, 2 pi]
    elevation = torch.atan2(z, hxy)  # [-pi, pi]

    return torch.cat([azimuth, elevation, r], dim=-1)


def spherical_to_cartesian(spherical):
    azimuth = spherical[..., 0:1]  # azimuth
    elevation = spherical[..., 1:2]  # elevation
    r = spherical[..., 2:3]  # range

    rcos_theta = r * torch.cos(azimuth)
    x = rcos_theta * torch.cos(elevation)
    y = rcos_theta * torch.sin(elevation)
    z = r * torch.sin(azimuth)
    return torch.cat([x, y, z], dim=-1)


def relative_transform(pose_1, pose_2):
    """
    Compute the relative transform to move from pose 1
    to pose 2. This method is applied in batch
    :param pose_1: pose in batched matrix format (B, 4, 4)
    :param pose_2: pose in batched matrix format (B, 4, 4)
    :return:
    """
    rt = torch.eye(4).to(pose_1.device, dtype=pose_1.dtype)
    rt = rt.unsqueeze(0).repeat(pose_1.shape[0], 1, 1)
    rt[..., :3, :3] = torch.bmm(
        pose_1[..., :3, :3].transpose(-1, -2), pose_2[..., :3, :3]
    )
    trans = pose_2[..., :3, -1] - pose_1[..., :3, -1]
    trans = trans.unsqueeze(-1)
    rt[..., :3, -1] = torch.bmm(pose_1[..., :3, :3].transpose(-2, -1), trans,)[
        ..., 0
    ]

    return rt


def look_at_matrix(from_vec, to_vec, earth_normal=None):
    """
    Computes a homogeneous matrix to look from the from_vec to the
    to_vector, both in batched format (B, 3) or normal format (3,)
    The earth_normal should be a tensor of shape (3,)
    """
    remove_batch = False
    if len(from_vec.shape) == 1:
        from_vec = from_vec.unsqueeze(0)
        to_vec = to_vec.unsqueeze(0)
        remove_batch = True

    forward = from_vec - to_vec
    forward /= torch.linalg.norm(forward, dim=1).unsqueeze(-1)

    if earth_normal is None:
        earth_normal = torch.tensor([0, 0, 1], dtype=from_vec.dtype)

    # Add batch dim to the earth normal
    earth_normal = earth_normal.unsqueeze(0).repeat(from_vec.shape[0], 1)
    earth_normal /= torch.linalg.norm(earth_normal, dim=1).unsqueeze(-1)

    right = torch.cross(earth_normal, forward, dim=1)
    right /= torch.linalg.norm(right, dim=1).unsqueeze(-1)

    up = torch.cross(forward, right, dim=1)
    up /= torch.linalg.norm(up, dim=1).unsqueeze(-1)

    mat = torch.eye(4).unsqueeze(0).repeat(from_vec.shape[0], 1, 1)
    mat[:, :3, 0] = right
    mat[:, :3, 1] = up
    mat[:, :3, 2] = forward
    mat[:, :3, 3] = from_vec

    det = torch.linalg.det(mat[:, :3, :3]).abs() ** (1 / 3)
    mat[:, :3, :3] /= det.unsqueeze(-1).unsqueeze(-1)

    if remove_batch:
        mat = mat[0]

    return mat
