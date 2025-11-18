import numpy as np
import torch
import random

from scene_environments.geometry import look_at_matrix, sph2cart, rot_z
from scene_environments.render import Renderer
from scene_environments.constants import realsense_intrinsics

from pytorch3d.transforms import quaternion_to_matrix


def random_pose(lookat, d_min, rn=False):
    if not rn:
        rn = np.random.uniform(d_min + 1.0, d_min + 3.0)
    en = np.random.uniform(-np.pi / 2, np.pi / 2)
    an = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, 2 * np.pi)

    return torch.tensor(
        look_at_matrix(sph2cart(an, en, rn), lookat) @ rot_z(theta),
        dtype=torch.float32,
    )


def to_mat(vector):
    mat = torch.eye(4).unsqueeze(0).repeat(vector.shape[0], 1, 1)
    mat[:, :3, :3] = quaternion_to_matrix(vector[:, 3:])
    mat[:, :3, -1] = vector[:, :3]
    return mat


class ObjectEnvironment:
    def __init__(
        self, scene_object_path, goal_object_path, fixed_r=False, seed=0
    ):

        width, height = 160, 120
        self.window = ((width - height) // 2, -(width - height) // 2)

        self._fixed_r = fixed_r

        self.renderer = self._init_renderer(scene_object_path, width, height)
        self._d_min = np.linalg.norm(
            self.renderer._mesh.centroid - self.renderer._mesh.bounds[0]
        )

        self.goal_renderer = self.renderer
        if goal_object_path != scene_object_path:
            self.goal_renderer = self._init_renderer(
                goal_object_path, width, height
            )

        self.seed = seed
        self.lookat = np.zeros(3)
        self.goal = None
        self._pose = None
        self.d = None
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.goal = self._generate_goal()

        self.pose = random_pose(self.lookat, self._d_min, self._fixed_r)

        return self.state

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, v):
        self._pose = v
        _, d = self.renderer.render(self.pose)
        self.d = d[:, self.window[0] : self.window[1]]

    @property
    def pose_error(self):
        x = self.goal["d_evaluate"].copy()
        y = self.d.copy()
        se = (x.flatten() - y.flatten()) ** 2
        return se.mean()

    @property
    def scaled_pose_error(self):
        x = self.goal["d_evaluate"].copy()
        scale = 1 / (x > 0).sum()
        y = self.d.copy()
        e = x.flatten() - y.flatten()
        return ((scale * e) ** 2).mean()

    @property
    def done(self):
        """
        To evaluate whether the goal was reached, the agent has a render of the
        scene-object in the goal pose.
        """
        threshold = 0.05
        return self.pose_error > threshold

    @property
    def state(self):
        return {
            "goal": self.goal,
            "rgb": self.rgb,
            "d": self.d,
            "pose": self.pose,
        }

    def _init_renderer(self, object_mesh_path, width, height):
        intrinsics = (height / 480) * realsense_intrinsics
        intrinsics[-1, -1] = 1.0
        return Renderer(
            str(object_mesh_path),
            intrinsics=intrinsics,
            resolution=(width, height),
            crop_shape=(height, height),
            object_centric=True,
        )

    def _generate_goal(self):
        agent_pose = random_pose(self.lookat, self._d_min, self._fixed_r)
        rgb, d = self.goal_renderer.render(agent_pose)
        rgb_evaluate, d_evaluate = self.renderer.render(agent_pose)

        return {
            "d": d,
            "rgb": rgb,
            "rgb_evaluate": rgb_evaluate,
            "d_evaluate": d_evaluate,
            "camera_pose": agent_pose,
            "object_pose": np.eye(4),
        }

    def act(self, action):
        action = to_mat(action.unsqueeze(0))[0]
        self.pose = torch.mm(self.pose, action)
        # Make sure determinant is always 1
        det = torch.linalg.det(self.pose[:3, :3])
        self.pose[:3, :3] /= abs(det) ** (1 / 3)

        # Recompute pose to deal with rounding issues / warping of the camera
        # TODO: fix theta angle here
        # self.pose = recompute_pose(self.pose)

        self.rgb, self.d = self.renderer.render(self.pose)

        return self.state
