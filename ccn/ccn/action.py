"""
An action is defined as a relative transform in 3D space; consisting of a 
translation and an orientation component. This can be represented as a
- homogeneous (4x4) matrix 
- a displacement vector + quaternion orientation 
- ...
"""
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
from ccn.geometry import spherical_to_cartesian, cartesian_to_spherical


class Action:
    def __init__(self, matrix):
        self._matrix = matrix

    def __getitem__(self, idx):
        return Action(self._matrix[idx])

    @staticmethod
    def from_matrix(matrix):
        return Action(matrix)

    @property
    def matrix(self):
        """
        Homogeneous 4x4 matrix representation
        rot = mat[:3, :3]
        trans = mat[:3, -1]
        """
        return self._matrix

    @staticmethod
    def from_spherical(spherical):
        matrix = torch.zeros((*spherical.shape[:-1], 4, 4))
        matrix[..., :4, :4] = torch.eye(4)
        matrix[..., :3, -1] = spherical_to_cartesian(spherical)
        return Action(matrix=matrix)

    @property
    def spherical(self):
        """
        Spherical representation, assumes the agent keeps looking at
        origin, and thus only represents the translation of the
        agent in azimuth, elevation, Range space
        """
        return cartesian_to_spherical(self._matrix[..., :3, -1])

    @staticmethod
    def from_cartesian_quaternion(cart_quat):
        matrix = torch.zeros((*cart_quat.shape[:-1], 4, 4))
        matrix[..., :4, :4] = torch.eye(4)
        matrix[..., :3, -1] = cart_quat[..., :3]
        matrix[..., :3, :3] = quaternion_to_matrix(cart_quat[..., 3:])
        return Action(matrix)

    @property
    def cartesian_quaternion(self):
        """
        Representation of cartesisan coordinates + quaternion representation
        for the orientation
        """
        tra = self._matrix[..., :3, -1]
        ori = matrix_to_quaternion(self._matrix[..., :3, :3])
        return torch.cat([tra, ori], dim=-1)

    @staticmethod
    def from_cartesian_6dof(cart_6dof):
        matrix = torch.zeros((*cart_6dof.shape[:-1], 4, 4))
        matrix[..., :4, :4] = torch.eye(4)
        matrix[..., :3, -1] = cart_6dof[..., :3]
        matrix[..., :3, :3] = rotation_6d_to_matrix(cart_6dof[..., 3:])
        return Action(matrix)

    @property
    def cartesian_6dof(self):
        """
        Representation of cartesian coordinates + 6DOF representaiton for the
        orientation
        """
        tra = self._matrix[..., :3, -1]
        ori = matrix_to_rotation_6d(self._matrix[..., :3, :3])
        return torch.cat([tra, ori], dim=-1)

    @staticmethod
    def from_cartesian_euler(cart_euler):
        matrix = torch.zeros((*cart_euler.shape[:-1], 4, 4))
        matrix[..., :4, :4] = torch.eye(4)
        matrix[..., :3, -1] = cart_euler[..., :3]
        matrix[..., :3, :3] = euler_angles_to_matrix(
            cart_euler[..., 3:], "XYZ"
        )
        return Action(matrix)

    @property
    def cartesian_euler(self):
        tra = self._matrix[..., :3, -1]
        ori = matrix_to_euler_angles(self._matrix[..., :3, :3], "XYZ")
        return torch.cat([tra, ori], dim=-1)

    @staticmethod
    def from_environment_action(env_action):
        tra = env_action[..., :3]
        ori = env_action[..., 3:]
        return Action.from_cartesian_euler(torch.cat([tra, ori], dim=-1))

    @property
    def environment_action(self):
        env_action = self.cartesian_euler
        env_action[..., 3:] = 1 / 2 + env_action[..., 3:] / (2 * torch.pi)
        return env_action

    @property
    def nn_action(self):
        return self.spherical[..., :2]

    def get_step(self, step_size=0.05):
        """
        Create the action representation to take a step in the direction
        of the target action, with step size (in meter)
        """
        cart_euler = self.cartesian_euler
        d = torch.linalg.norm(cart_euler[..., :3], dim=-1)
        fraction = torch.min(step_size / d, torch.ones_like(d))
        return Action.from_cartesian_euler(cart_euler * fraction.unsqueeze(-1))
