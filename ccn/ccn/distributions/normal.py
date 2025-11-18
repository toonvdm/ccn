import torch
from torch.distributions import MultivariateNormal

from ccn.camera import PinholeModel, realsense_intrinsics


def multiply_multivariate_normals(mv1, mv2):
    """
    Expects mv1 and mv2 to be torch.distributions.MultivariateNormal
    Either both in batched format, or both in single format
    """
    mu_1 = mv1.mean
    co_1 = mv1.covariance_matrix
    mu_2 = mv2.mean
    co_2 = mv2.covariance_matrix

    remove_batch = False
    if len(mu_1.shape) <= 1:
        mu_1 = mu_1.unsqueeze(0)
        co_1 = co_1.unsqueeze(0)
        mu_2 = mu_2.unsqueeze(0)
        co_2 = co_2.unsqueeze(0)
        remove_batch = True

    co_inv = torch.inverse(co_1 + co_2)
    co = torch.bmm(torch.bmm(co_1, co_inv), co_2)
    t1 = torch.bmm(torch.bmm(co_2, co_inv), mu_1.unsqueeze(-1))[..., 0]
    t2 = torch.bmm(torch.bmm(co_1, co_inv), mu_2.unsqueeze(-1))[..., 0]
    mean = t1 + t2

    if remove_batch:
        mean = mean[0]
        co = co[0]

    return MultivariateNormal(mean, co)


def empirical_truncated_normal(mv, camera_pose, n=10000):
    """
    Because I don't immediately find a method for computing the mean
    of a truncated normal distribution, I'll approximate it by sampling
    N=10000 points from the distribution, and filtering out the ones
    that should be low-probability.
    :param mv: MultivariateNormal
    :param camera_pose: 4x4 matrix describing the camera pose, from which
        the object is not observed. Meaning that this should be truncated
        from the belief.
    """
    pinhole = PinholeModel(realsense_intrinsics)
    samples = mv.sample((n,))

    # Recompute the mean over all the points that are not in view, using
    # this density
    not_in_view = ~pinhole.coords_in_view(samples, camera_pose)
    samples = samples[not_in_view]

    return MultivariateNormal(samples.mean(dim=0), mv.covariance_matrix)
