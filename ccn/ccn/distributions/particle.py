import numpy as np
import torch
from torch.distributions import MultivariateNormal

from ccn.camera import PinholeModel, realsense_intrinsics


class ParticleFilter:
    def __init__(self, n_particles=1000, ranges=None, resample_std=0.00025):

        self.range = ranges
        if self.range is None:
            self.range = torch.tensor([[-0.5, 0.5], [-0.5, 0.5], [0, 0.5]])

        self.particles = self.get_particles(n_particles)
        self.weights = torch.ones((n_particles,)) / n_particles
        self.pinhole = PinholeModel(realsense_intrinsics)

        self.resample_std = resample_std

    def get_particles(self, n):
        """
        return n uniform particles, in the range provided by self.range
        """
        parts = torch.rand((n, self.range.size(0)))
        for i in range(self.range.size(0)):
            (mini, maxi), _ = self.range[i].sort()
            parts[:, i] = parts[:, i] * (maxi - mini) + mini
        return parts

    def predict(self, offset=None, std=0.1):
        """
        In a particle filter, the predict step would integrate the action
        and predict the output of the particles. However, in our implementation
        we don't really consider action (of the objects)
        """
        # we just add noise to the particles
        if offset is not None:
            self.particles += offset.unsqueeze(0)
        self.particles += torch.randn_like(self.particles) * std
        # self.particles += (torch.rand_like(self.particles) - 0.5) * std

    def update(self, belief):
        """
        Compute the update using the belief state.
        """
        # Integrate observation
        self.weights *= torch.exp(belief.log_prob(self.particles))
        # Renormalize weights
        self.weights += 1e-5
        self.weights /= self.weights.sum()

    def update_negative(self, camera_pose, d_max=2):
        """
        Compute the update by setting the observed particles
        from this camera_pose to the minimal value
        """
        in_view = self.pinhole.coords_in_view(
            self.particles, camera_pose, d_max=d_max
        )
        self.weights[in_view] = 1e-5
        # Fix low values
        self.weights += 1e-5
        # Renormalize weights
        self.weights /= self.weights.sum()

    @property
    def multivariate_normal(self):
        mean = self.mean
        return MultivariateNormal(mean, torch.eye(len(mean)) * self.variance)

    @property
    def mean(self):
        w = (self.weights / self.weights.sum()).unsqueeze(-1)
        mean = (self.particles * w).sum(dim=0)
        return mean

    @property
    def variance(self):
        w = (self.weights / self.weights.sum()).unsqueeze(-1)
        wse = w * (self.particles - self.mean.unsqueeze(0)) ** 2
        return wse.sum(dim=0)

    def resample(self):
        cm = torch.cumsum(self.weights, dim=0)
        cm[-1] = 1.0
        indices = torch.searchsorted(cm, torch.rand(len(self.particles)))
        self.particles = self.particles[indices]

    def add_position_estimate(
        self, belief_or_camera, var=1, observed=True, d_max=2
    ):
        if observed:
            direction = None
            # Resample based on the offset of the predicted mean w.r.t. belief
            # if self.resample_std > 0:
            #     offset = belief_or_camera.mean - self.mean
            #     length = torch.linalg.norm(offset)  # , self.resample_std)
            #     # Take a step in the direction of the expected mean
            #     direction = 0.25 * offset / length

            self.predict(direction, std=self.resample_std)
            self.update(belief_or_camera)
        else:
            self.predict(std=self.resample_std)
            self.update_negative(belief_or_camera, d_max=d_max)
        self.resample()
