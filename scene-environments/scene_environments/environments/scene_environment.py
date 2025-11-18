import numpy as np
import torch

import logging
import pickle
import random

from pathlib import Path

import gym
from gym import spaces

from scene_environments.render import Renderer, load_mesh
from scene_environments.constants import realsense_intrinsics
from scene_environments.errors import translation_error, rotation_error
from scene_environments.geometry import (
    look_at_matrix,
    sph2cart,
    rot_x,
    rot_y,
    rot_z,
    distance,
    recompute_pose,
)

from pytorch3d.transforms import (
    quaternion_to_matrix,
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

logger = logging.getLogger()


def to_vector(mat):
    if not isinstance(mat, torch.Tensor):
        mat = torch.from_numpy(mat)
    translations = mat[:3, -1].squeeze()
    orientations = matrix_to_quaternion(mat[:3, :3])
    return np.concatenate([translations, orientations], axis=0)


class SceneEnvironment(gym.Env):
    def __init__(
        self,
        objects,
        seed=0,
        n_objects=1,
        render_table=True,
        mesh_location=None,
        observation_shape=None,
        resample_objects=False,
        resample_table_color=False,
        random_table_color=False,
        eval=False,
        render_wall=False,
        matrix_actions=False,
        target_object=None,
        init_renderers=True,
    ):
        """
        Environment in which objects are placed at given positions in a
        virtual space. The agent is a camera that can move around.
        ---
        :param objects: list of objects to consider
        :param render_table: boolean describing whether or not to render the
            ground plane as a table.
        :param mesh_location: the location to find the meshes, if no value is
            provided, it will default to the location in data/input/meshes
        :param observation_shape: the spatial dimensions of the observation shapes
        """

        self.eval = eval

        self.seed = seed

        # TODO: Clean up everything related to target object
        # is a construct to force the goal object identity
        # to create an evaluation set
        self.target_object = target_object

        self._environment_bounds = np.array(
            [[-0.5, 0.5], [-0.5, 0.5], [0.1, 0.6]]
        )

        # === Objects set up
        self.mesh_location = mesh_location
        if mesh_location is None:
            self.mesh_location = (
                Path(__file__).parent.parent.parent
                / "data/meshes/ycb_gazebo_sdf"
            )

        # Load the object meshes in advance (this is bottleneck, so doing it
        # in reset slows the environment down too much)
        self.resample_objects = resample_objects
        self.n_objects = n_objects

        all_objects = [
            str(self.mesh_location / str(o + "_textured/textured.dae"))
            for o in objects
        ]
        self.all_objects = {o: load_mesh(o) for o in all_objects}

        # store the object radius, avoiding the z-value to have no occlusions
        # when placing them together on the table
        self.object_rs = {
            n: distance(m.centroid[:2], m.bounds[0][:2]).item()
            for n, m in self.all_objects.items()
        }
        self.object_sphere_rs = {
            self.extract_key_from_path(n): distance(
                m.centroid[:3], m.bounds[0][:3]
            ).item()
            for n, m in self.all_objects.items()
        }

        self.chosen_meshes = {}
        if len(objects) > 0:
            self.chosen_meshes = self.sample_objects()

        # Setup table
        self.render_table = render_table
        self.table_color = np.ones(3) * 0.30
        self.resample_table_color = resample_table_color
        if random_table_color:
            self.table_color = (np.random.random(3) * 255).astype(np.uint8)

        # Copy random-ness of table over to the wall
        self.render_wall = render_wall
        self.wall_color = np.ones(3) * 0.30
        self.resample_wall_color = resample_table_color
        if random_table_color:
            self.wall_color = (np.random.random(3) * 255).astype(np.uint8)

        self.background_color = [0, 0, 0]

        self.rgb = None
        self.depth = None
        self.instance_mask = None

        self.goal_idx = 0
        self.goals = None

        # === Rendering set up
        self.observation_shape = observation_shape
        if self.observation_shape is None:
            self.observation_shape = (480, 480)

        if init_renderers:
            self.renderer, self.preference_renderer = self.init_renderer()
        else:
            self.renderer, self.preference_renderer = None, None

        # === Agent state set up
        self.agent_pose = None

        # The action space allows the agent to move by half a meter
        # in the Euler representation of position and rotate to an
        # arbitrary angle using the quaternion representation, sampled
        # from: http://planning.cs.uiuc.edu/node198.html
        step_size = 0.05
        self.action_space = spaces.Box(
            low=np.array(
                [
                    -step_size,
                    -step_size,
                    -step_size,
                    -step_size,
                    -step_size,
                    -step_size,
                ]
            ),
            high=np.array(
                [
                    step_size,
                    step_size,
                    step_size,
                    step_size,
                    step_size,
                    step_size,
                ]
            ),
            dtype=np.float64,
        )

        # === Reset scene
        if len(objects) > 0:
            self.reset()

        if matrix_actions:
            self.action_to_matrix = lambda x: x

    @staticmethod
    def extract_key_from_path(path):
        return path.split("/")[-2].replace("_textured", "")

    def store_pkl(self, store_path):
        config = {
            "seed": self.seed,
            "eval": self.eval,
            "observation_shape": self.observation_shape,
            "background_color": self.background_color,
            "table": {
                "render_table": self.render_table,
                "color": self.table_color,
                "resample": self.resample_table_color,
            },
            "wall": {
                "render_wall": self.render_wall,
                "color": self.wall_color,
                "resample": self.resample_wall_color,
            },
            "objects": {
                "object_paths": list(self.all_objects.keys()),
                "object_radiuses": self.object_rs,
                "object_sphere_radiuses": self.object_sphere_rs,
                "n_objects": self.n_objects,
                "resample_objects": self.resample_objects,
                "objects_dict": self.objects_dict,
                "chosen_objects": list(self.chosen_meshes.keys()),
            },
            "agent_pose": self.agent_pose,
            "goals": self.goals,
            "target_object": self.target_object,
        }

        pickle.dump(config, open(store_path, "wb"))

    @staticmethod
    def load_pkl(load_path, env=None, override=None):
        config = pickle.load(open(load_path, "rb"))

        if override is not None:
            for k, v in override.items():
                config[k] = v

        # Set all values
        use_cache = True
        if env is None:
            env = SceneEnvironment({}, init_renderers=False)
            use_cache = False

        env.target_object = config["target_object"]
        env.seed = config["seed"]
        env.eval = config["eval"]
        env.observation_shape = config["observation_shape"]
        env.background_color = config["background_color"]
        env.render_table = config["table"]["render_table"]
        env.table_color = config["table"]["color"]
        env.resample_table_color = config["table"]["resample"]

        # Backwards compatibility
        if config.get("wall", None):
            env.render_wall = config["wall"]["render_wall"]
            env.wall_color = config["wall"]["color"]
            env.resample_wall_color = config["wall"]["resample"]
        else:
            env.render_wall = False
            env.wall_color = None
            env.resample_wall_color = False

        env.n_objects = config["objects"]["n_objects"]
        env.object_rs = config["objects"]["object_radiuses"]
        env.object_sphere_rs = config["objects"]["object_sphere_radiuses"]
        env.resample_objects = config["objects"]["resample_objects"]

        env.agent_pose = config["agent_pose"]

        if not use_cache:
            env.all_objects = {
                o: load_mesh(o) for o in config["objects"]["object_paths"]
            }

        env.objects_dict = config["objects"]["objects_dict"]
        env.chosen_meshes = {
            k: env.all_objects[k] for k in config["objects"]["chosen_objects"]
        }

        if not use_cache:
            env.renderer, env.preference_renderer = env.init_renderer()
        env.renderer.set_table_color(env.table_color)
        env.preference_renderer.set_table_color(env.table_color)

        env.construct_scene(env.objects_dict)

        # env.goals = config["goals"]
        env.goals = None  # regenerate the goals

        env.reset()
        env.render()

        return env

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        np.random.seed(value)
        random.seed(value)

    @property
    def scale(self):
        s = 1.0
        if self.observation_shape is not None:
            s = self.observation_shape[1] / 480
        return s

    @property
    def intrinsics(self):
        # Intrinsic matrix of an intel realsense D435i
        # set scale to 1 to render in high resolution
        return self.scale * realsense_intrinsics

    @staticmethod
    def action_to_matrix(action):
        matrix = (
            rot_x(2 * np.pi * action[3] - np.pi)
            @ rot_y(2 * np.pi * action[4] - np.pi)
            @ rot_z(2 * np.pi * action[5] - np.pi)
        )
        matrix[:3, -1] = action[:3]
        return torch.tensor(matrix, dtype=torch.float64)

    @property
    def state(self):
        return {
            "camera_pose": self.agent_pose.to(torch.float32),
            "rgb": self.rgb[
                : self.observation_shape[0], : self.observation_shape[1]
            ],
            "depth": self.d[
                : self.observation_shape[0], : self.observation_shape[1]
            ],
            "goal_pose": self.goal["pose"],
            "goal_rgb": self.goal["rgb"][
                : self.observation_shape[0], : self.observation_shape[1]
            ],
            "goal_depth": self.goal["depth"][
                : self.observation_shape[0], : self.observation_shape[1]
            ],
        }

    def _get_obs(self, state):
        obs = {
            "image": state["rgb"],
            "state": to_vector(state["camera_pose"]),
            "image_goal": self.goal["rgb"][
                : self.observation_shape[0], : self.observation_shape[1]
            ],
        }
        # Masking goal's camera_pose
        obs["goal"] = np.zeros_like(obs["state"])
        return obs

    @property
    def info(self):
        info_dict = self._info.copy()
        info_dict.update(
            {
                "instance_mask": self.instance_mask,
                "goal_instance_mask": self.goal["instance_mask"],
                "goal_object_name": self.goal["object_name"],
            }
        )
        info_dict.update(self.state)
        return info_dict

    @property
    def reward(self):
        t_gt = self.goal["pose"][:3, -1]
        r_gt = self.goal["pose"][:3, :3]
        t_est = self.agent_pose.numpy()[:3, -1]
        r_est = self.agent_pose.numpy()[:3, :3]
        te = translation_error(t_est, t_gt)
        re = rotation_error(r_est, r_gt)
        # Below 7.5 cm and .5 rad
        ret = (te < 0.075) and (re < 0.5)
        return ret

    @property
    def done(self):
        return self.reward == 1 or self.rgb.sum() == 0

    @property
    def goal(self):
        return self.goals[self.goal_idx]

    def set_goal_idx(self, idx):
        self.goal_idx = idx

    def get_goal_idx(self):
        return self.goal_idx

    def get_goals(self):
        return self.goals

    def clip_pose_inside_environment(self, pose):
        pose[0, -1] = np.clip(pose[0, -1], *self._environment_bounds[0])
        pose[1, -1] = np.clip(pose[1, -1], *self._environment_bounds[1])
        pose[2, -1] = np.clip(pose[2, -1], *self._environment_bounds[2])
        return pose

    def clip_pose_outside_objects(self, pose):
        z_near = 0.05
        for key, object_pose in self._info.items():
            # Add camera clipping distance + epsilon to it
            radius = self.object_sphere_rs[key] + z_near + 0.01

            object_center = torch.tensor(
                object_pose[:3, -1], dtype=torch.float32
            )
            position = pose[:3, -1]

            if torch.linalg.norm(position - object_center) < radius:
                direction = position - object_center
                direction *= radius / torch.linalg.norm(direction)
                pose[:3, -1] = object_center + direction
        return pose

    def step(self, action):
        """
        Perform the action provided, this will change the camera extrinsic
        matrix
        ---
        :param action: relative transform w.r.t. the current camera pose,
            can either be:
            - tensor of dim [6] as [*translation(xyz), *u-space orientation(XYZ)]
        """

        # Execute one time step within the environment
        action = self.action_to_matrix(action)
        self.agent_pose = torch.mm(self.agent_pose, action)

        det = torch.linalg.det(self.agent_pose[:3, :3])
        self.agent_pose[:3, :3] /= abs(det) ** (1 / 3)

        # make sure agent does not move out of bounds
        self.agent_pose = self.clip_pose_inside_environment(self.agent_pose)
        self.agent_pose = self.clip_pose_outside_objects(self.agent_pose)

        # Recompute pose to deal with rounding issues / warping of the camera
        self.agent_pose = recompute_pose(self.agent_pose)

        self.rgb, self.d = self.render()

        return self._get_obs(self.state), self.reward, self.done, self.info

    def init_renderer(self):
        """
        Initializing the renderer, by placing the chosen objects
        in the scene (in the initial pose)
        """
        resolution = np.array([int(self.scale * 640), int(self.scale * 480)])

        # place all objects in the renderer!
        object_dict = {o: np.eye(4) for o in self.all_objects.keys()}

        scene_renderer = Renderer(
            object_dict,
            intrinsics=self.intrinsics,
            resolution=resolution,
            crop_shape=self.observation_shape,
            plane=self.render_table,
            table_color=self.table_color,
            background_color=self.background_color,
            wall=self.render_wall,
            wall_color=self.wall_color,
        )

        # Same renderer but without a ground plane
        preference_renderer = Renderer(
            object_dict,
            meshes=scene_renderer.meshes,
            intrinsics=self.intrinsics,
            resolution=resolution,
            crop_shape=self.observation_shape,
            plane=self.render_table,
            table_color=self.table_color,
            background_color=self.background_color,
            wall=self.render_wall,
            wall_color=self.wall_color,
        )

        return scene_renderer, preference_renderer

    def generate_scene_constellation(self):
        """
        Randomly choose the objects to place in the scene,
        choose a position to place them that does not allow for
        intersection with themselves
        :returns: objects_dict: a dictionary where the key is the
            mesh path and the value is the mesh pose w.r.t. the
            global reference frame
        """
        objects_dict = {}
        info_dict = {}

        for o, mesh in self.chosen_meshes.items():

            found = False
            while not found:
                pose = np.eye(4)
                pose[0, -1] = np.random.random() * 0.5 - 0.25
                pose[1, -1] = np.random.random() * 0.5 - 0.25

                for o2, pose2 in info_dict.items():
                    # info dict can be used because of only x y coords
                    # also easier for keys
                    d = distance(pose[:2, -1], pose2[:2, -1])
                    o2 = (
                        str(self.mesh_location / o2) + "_textured/textured.dae"
                    )
                    if d < self.object_rs[o] + self.object_rs[o2]:
                        # There is no intersection going on
                        found = False
                        break
                else:
                    found = True

            objects_dict[o] = pose.copy()

            # For the info dict, use a different height
            # - the object center
            # while the objects_dict uses the object
            # bottom for rendering
            val = pose.copy()
            # Look at the center of the object
            val[2, -1] = mesh.centroid[2]
            o = o.split("/")[-2].replace("_textured", "")
            info_dict[o] = val.copy()

        return objects_dict.copy(), info_dict.copy()

    def construct_scene(self, objects_dict):

        # Hide object if it is not in the objects dict
        hidden_pose = np.eye(4)
        hidden_pose[2, -1] = -3

        self.renderer.update_object_positions(
            {
                k: objects_dict.get(k, hidden_pose)
                for k, _ in self.all_objects.items()
            }
        )
        self.preference_renderer.update_object_positions(
            {
                k: objects_dict.get(k, hidden_pose)
                for k, _ in self.all_objects.items()
            }
        )

    def create_goal(self, objects_dict, seed_offset=0, goal_object_index=None):

        np_state = np.random.get_state()
        self.seed = self.seed + seed_offset

        if goal_object_index is None:
            goal_object_index = np.random.randint(
                len(self.chosen_meshes.keys())
            )
        # ability to force a target object
        if self.target_object is not None:
            meshes = list(self.chosen_meshes.keys())
            goal_object_index = [
                i
                for i in range(len(meshes))
                if self.target_object in meshes[i]
            ][0]

        goal_object_key = list(objects_dict.keys())[goal_object_index]

        # hide object below the table for the goal image
        hidden_pose = np.eye(4)
        hidden_pose[2, -1] = -1
        preference_poses = {k: hidden_pose for k in self.all_objects.keys()}
        preference_poses.update(
            {goal_object_key: objects_dict[goal_object_key]}
        )
        self.preference_renderer.update_object_positions(preference_poses)

        # Sample a goal orientation
        goal_object_key = goal_object_key.split("/")[-2].split("_textured")[0]
        target_position = self._info[goal_object_key][:3, -1]

        # Random pose
        an = np.random.random() * 2 * np.pi  # full azimuth
        en = np.random.random() * np.pi / 2  # above table
        r_min = self.object_sphere_rs[goal_object_key] + 0.05
        # outside of object
        r = np.random.random() * (0.50 - r_min) + r_min
        camera_pos = np.array([*sph2cart(an, en, r)])
        camera_pos += target_position

        goal_pose = torch.tensor(
            look_at_matrix(camera_pos, target_position), dtype=torch.float32
        )
        goal_pose = self.clip_pose_inside_environment(goal_pose)
        goal_pose = self.clip_pose_outside_objects(goal_pose).numpy()

        # Render the observation
        rgb, d, instance_mask = self.preference_renderer.render(
            goal_pose, render_instance_mask=True
        )

        np.random.set_state(np_state)

        return {
            "rgb": rgb,
            "depth": d,
            "pose": goal_pose,
            "instance_mask": instance_mask,
            "object_name": goal_object_key,
        }

    def sample_objects(self):
        keys = list(self.all_objects.keys())
        idcs = []
        items = [keys[i] for i in idcs]
        while self.target_object not in items:
            idcs = np.random.choice(
                np.arange(len(keys)), self.n_objects, replace=False,
            )
            items = [
                keys[i].split("/")[-2].replace("_textured", "") for i in idcs
            ]
            if self.target_object is None:
                break
        return {keys[i]: self.all_objects[keys[i]] for i in idcs}

    def reset(self):
        """
        Reset the agent to an initial position. As the environment does not
        change, only the agent pose should be adapted.
        """
        position = sph2cart(az=0, el=np.pi / 4, r=0.65)
        self.agent_pose = torch.tensor(
            look_at_matrix(position, np.zeros(3)), dtype=torch.float64,
        )

        # First sample the color of the table
        if not self.eval and self.resample_table_color:
            self.table_color = (np.random.random(3) * 255).astype(np.uint8)
            self.renderer.set_table_color(self.table_color)
            self.preference_renderer.set_table_color(self.table_color)

            self.wall_color = (np.random.random(3) * 255).astype(np.uint8)
            self.renderer.set_wall_color(self.wall_color)
            self.preference_renderer.set_wall_color(self.wall_color)

        # Then sample the objects from the list
        if not self.eval and self.resample_objects:
            self.chosen_meshes = self.sample_objects()

        # === Set up environment & renderer
        if not self.eval or self.goals is None:
            self.objects_dict, self._info = self.generate_scene_constellation()
            self.construct_scene(self.objects_dict)

            # === Agent goal set up
            self.goals = [
                self.create_goal(
                    self.objects_dict,
                    seed_offset=i,
                    goal_object_index=i % len(self.chosen_meshes),
                )
                for i in range(10)
            ]

        self.render()

        return self._get_obs(self.state)

    def render(self):
        self.rgb, self.d, self.instance_mask = self.renderer.render(
            self.agent_pose, True
        )
        return self.rgb, self.d
