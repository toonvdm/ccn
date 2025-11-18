import torch
from torch.distributions import MultivariateNormal

from pytorch3d.transforms import Transform3d

from ccn.camera import opengl2ee
from ccn.geometry import look_at_matrix


def cast_ray(u, v, poses, camera_model):
    remove_batch = False
    if len(poses.shape) <= 2:
        poses = poses.unsqueeze(0)
        remove_batch = True

    ray = camera_model.uvd_to_xyz(
        torch.tensor([u, u]), torch.tensor([v, v]), torch.tensor([0, 1])
    )
    ray = torch.cat([r.reshape(2, 1) for r in ray], dim=1).unsqueeze(0)
    rays = ray.repeat(poses.shape[0], 1, 1)
    viewpoint = torch.bmm(
        poses, opengl2ee.unsqueeze(0).repeat(poses.shape[0], 1, 1)
    )
    rays = Transform3d(matrix=viewpoint.transpose(1, 2)).transform_points(rays)

    if remove_batch:
        rays = rays[0]

    return rays


def ray_estimate(rays, d=0.5, sigma=None):
    """
    Estimate belief over position, given a certain set of rays
    If a single ray is provided, then a non batched version will be
    executed
    """
    remove_batch = False
    if len(rays.shape) == 2:
        rays = rays.unsqueeze(0)
        remove_batch = True

    ray_coord_frame = look_at_matrix(rays[:, 0], rays[:, 1])

    cov = torch.eye(3).unsqueeze(0).repeat(len(rays), 1, 1)
    if sigma:
        cov[:, 0, 0] = sigma
        cov[:, 1, 1] = sigma
        cov[:, 2, 2] = sigma
    else:
        cov[:, 0, 0] = 0.020  # z_99 for .5 cm
        cov[:, 1, 1] = 0.020  # z_99 for .5 cm
        cov[:, 2, 2] = 0.1937 / 2  # z_99 for the .25 m range

    mean = torch.zeros((len(rays), 3, 1))
    mean[:, 2, :] = -d

    R = ray_coord_frame[:, :3, :3]
    new_cov = torch.bmm(torch.bmm(R, cov), R.transpose(1, 2))
    new_mean = torch.bmm(R, mean)[..., 0] + rays[0][0]

    if remove_batch:
        new_mean = new_mean[0]
        new_cov = new_cov[0]

    return MultivariateNormal(new_mean, new_cov)
