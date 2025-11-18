import torch
from torch.distributions import MultivariateNormal

from ccn.distributions.particle import ParticleFilter
from ccn.distributions.normal import multiply_multivariate_normals


class ObjectBelief:
    """
    A class containing the information about a specific object belief
    i.e. position, pose, ...
    """

    def __init__(self, object_name, particle_filter=True):
        self.object_name = object_name
        self._particle_filter = particle_filter

        # Global level i.e. where are the objects
        self.was_updated = False

        self.global_pose = None
        self.local_poses = None
        self.camera_poses = None
        self.z_wheres = None
        self.crops = None
        self.reconstruction_mses = None
        self.reset()

    @property
    def position_multivariate_normal(self):
        if self._particle_filter:
            return self.global_pose.multivariate_normal
        else:
            return self.global_pose

    def reset(self):
        if self._particle_filter:
            self.global_pose = ParticleFilter(10000)
        else:
            self.global_pose = MultivariateNormal(
                torch.zeros(3), 0.125 * torch.eye(3)
            )

        # CCN level
        self.reconstruction_mses = []
        self.local_poses = []
        self.camera_poses = []
        self.z_wheres = []
        self.crops = []
        self.reconstruction_mses = []

    def update_position_belief(
        self, position_or_camera, var, observed, d_max=2
    ):
        if self._particle_filter:
            self.global_pose.add_position_estimate(
                position_or_camera, var, observed, d_max
            )
        elif observed:
            self.global_pose = multiply_multivariate_normals(
                self.global_pose, position_or_camera
            )

    def update_pose_belief(
        self, q_pose, camera_pose, z_where, reco_mse, crop, rgb
    ):
        self.crops += [(crop, rgb)]
        self.reconstruction_mses += [reco_mse]
        self.local_poses += [q_pose]
        self.camera_poses += [camera_pose]
        self.z_wheres += [z_where]
        self.was_updated = True
