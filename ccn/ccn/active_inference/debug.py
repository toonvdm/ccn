import torch
import numpy as np 

from ccn.active_inference.object_belief import ObjectBelief
from ccn.distributions.particle import ParticleFilter
from torch.distributions import MultivariateNormal


class GTObjectBelief(ObjectBelief):
    def __init__(self, object_name, groundtruth_position):
        ObjectBelief.__init__(self, object_name, True)

        # self.global_pose = MultivariateNormal(
        #     groundtruth_position, 0.001 * torch.eye(3)
        # )

        # Super narrow belief
        self.global_pose = ParticleFilter(n_particles=10, resample_std=0)
        self.global_pose.particles = (groundtruth_position.reshape(1, 3) + 
            torch.rand((10, 3)) * 1e-12
        )
    
    def reset(self): 
        # CCN level
        self.local_poses = []
        self.camera_poses = []
        self.z_wheres = []
        self.crops = []
        self.reconstruction_mses = []
        return

    def update_position_belief(self, *args, **kwargs):
        return


def inject_groundtruth_position_info(agent, environment):
    """
    Set the belief for the object positions to the ground truth signal acquired from
    the environment
    """
    for k, bel in agent._beliefs.items():
        pos = torch.tensor(environment.info.get(k, np.eye(4))[:3, -1], dtype=torch.float32)
        agent._beliefs[k] = GTObjectBelief(k, pos)
