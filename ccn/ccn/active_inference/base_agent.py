from enum import Enum
import numpy as np
import logging

import torch

from dommel.nn import module_factory
from dommel.util.yaml_parser import parse

from ccn.util import to_torch
from ccn.action import Action
from ccn.geometry import (
    relative_transform,
    look_at_matrix,
    spherical_to_cartesian,
)
from ccn.dataset.transforms import pixel_mask
from ccn.distributions import multiply_multivariate_normals
from ccn.models.spatial_transformers import infer_uv, z_where_inv
from ccn.models.spatial_transformers import SpatialTransformer
from ccn.models.networks import Interpolate
from ccn.models.medianpool import MedianPool2d
from ccn.rays import cast_ray, ray_estimate
from ccn.camera import PinholeModel, realsense_intrinsics
from ccn.active_inference.object_belief import ObjectBelief
from ccn.visualization import plot_ims

from torch.distributions import kl_divergence, Normal, MultivariateNormal

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

agent_modes = Enum("Modes", ["INFOGAIN", "INSTRUMENTAL", "EFE"])


class BaseAgent:
    def __init__(
        self,
        object_names,
        model_dir,
        ccn_dir,
        particle_filter=False,
        teleport=True,
        debug=False,
        mode=agent_modes.INFOGAIN,
        sample_mean_lat=False,
    ):
        self._teleport = teleport
        self._sample_mean_lat = sample_mean_lat

        self._beliefs, self._models = {}, {}
        for object_name in object_names:
            ccn, mstn = self._load_models(object_name, model_dir, ccn_dir)
            self._models[object_name] = {"ccn": ccn, "mstn": mstn}
            self._beliefs[object_name] = ObjectBelief(object_name, particle_filter)

        self._pinhole = PinholeModel(realsense_intrinsics, opengl=True)

        # Interpolation modules
        self._interpolate_32 = Interpolate((32, 32), mode="bilinear")
        self._interpolate_64 = Interpolate((64, 64), mode="bilinear")
        self._interpolate_480 = Interpolate((480, 480), mode="bilinear")

        # Other helper models
        self.spatial_transformer_64 = SpatialTransformer(64)
        self.spatial_transformer_480 = SpatialTransformer(480)
        self.median = MedianPool2d(5)
        self._softmax = torch.nn.Softmax(dim=0)

        if mode.value not in [a.value for a in agent_modes]:
            raise ValueError(f"{mode} is not a valid mode type: {agent_modes}")
        self.mode = mode.value

        self.debug = debug

        self._preference = {}

        self._lookat = None
        self.reset()

    @staticmethod
    def _depth_to_scale(depth):
        return depth / (0.35 + 0.05)  # 40

    @staticmethod
    def _scale_to_depth(scale):
        return (0.05 + 0.35) / (scale + 1e-4)

    @staticmethod
    def _load_models(object_name, model_dir, ccn_dir):
        # CCN and ST trained models
        model_config = parse(str(model_dir / f"{object_name}.yml"))

        model = module_factory(**model_config["model"])
        model.load_state_dict(torch.load(model_dir / f"{object_name}.pt"))
        for param in model.parameters():
            param.requires_grad = False

        mstn = model._modules["module-0"]

        model_config = parse(str(ccn_dir / f"{object_name}.yml"))
        model = module_factory(**model_config["model"])
        model.load_state_dict(torch.load(ccn_dir / f"{object_name}.pt"))
        for param in model.parameters():
            param.requires_grad = False
        vanilla_ccn = model._modules["module-0"]

        return vanilla_ccn, mstn

    def _distance_to_closest_visible(self, camera_pose):
        # Compute d-max
        objects = torch.zeros((len(self._beliefs.keys()), 3))
        for i, (_, belief) in enumerate(self._beliefs.items()):
            objects[i] = belief.global_pose.mean
        iv = self._pinhole.coords_in_view(objects, camera_pose)
        objects = objects[iv]
        d_max = 2
        if objects.shape[0] > 0:
            d_max = torch.linalg.norm(
                objects - camera_pose[:3, -1].unsqueeze(0).repeat(objects.shape[0], 1),
            )
        return d_max

    def _update_preference(self, state):
        # Only update if the preference has been changed / not been set
        previous_goal_rgb = self._preference.get(
            "goal_rgb", np.zeros_like(state["goal_rgb"])
        )
        if np.allclose(state["goal_rgb"], previous_goal_rgb) == False:
            # tqdm.write("Updating preference")

            self._preference["goal_rgb"] = state["goal_rgb"].copy()

            rgb_in = to_torch(state["goal_rgb"]).unsqueeze(0)

            c_best = 0
            for i, (model_name, model) in enumerate(self._models.items()):
                crop, rgb_m_64, c, z_where = model["mstn"].forward(rgb_in)

                # update preference every time some more likely observation
                # comes along
                # reco = model['ccn'].decoder(model["ccn"].encoder(crop).mean)
                # r_err = torch.linalg.norm(reco-crop)

                # TODO: fix likelihood in less arbitrary way
                # print(c.item(), model_name)
                if c.item() > c_best:  # and r_err < 10:
                    c_best = c.item()

                    # approximate posterior over pose
                    q = model["ccn"].encoder(crop)
                    x_hat = model["ccn"].decoder(q.mean)

                    self._preference["object_name"] = model_name
                    # self._preference["rgb64"] = self._interpolate_64(rgb_in)
                    self._preference["crop"] = crop
                    self._preference["crop_sum"] = crop.sum()
                    self._preference["q_ccn"] = q
                    self._preference["x_hat"] = x_hat
                    self._preference["d_hat"] = self._scale_to_depth(
                        z_where[0][0].item()
                    )
            # tqdm.write(
            #     f"Setting Preference for: {self._preference['object_name']}"
            # )

    def _update_object_belief(self, state):
        # Compute the belief update for each object
        for object_name, model in self._models.items():
            rgb = to_torch(state["rgb"].copy()).unsqueeze(0).to(torch.float32)

            crop, rgb_m_64, c, z_where = model["mstn"].forward(rgb)

            # approximate posterior over pose
            q = model["ccn"].encoder(crop)
            decoded = model["ccn"].decoder(q.mean)

            observed = c > 0.9

            # Compute update
            u, v = infer_uv(z_where_inv(z_where)[0])
            ray = cast_ray(v, u, state["camera_pose"].clone(), self._pinhole)
            pos_or_cam = ray_estimate(ray, self._scale_to_depth(z_where[0][0]))
            if not observed:
                pos_or_cam = state["camera_pose"].clone()
                d_max = self._distance_to_closest_visible(pos_or_cam)
                self._beliefs[object_name].update_position_belief(
                    pos_or_cam,
                    var=q.variance.sum(),
                    observed=observed,
                    d_max=d_max,
                )

            else:
                self._beliefs[object_name].update_pose_belief(
                    q,
                    state["camera_pose"].clone(),
                    z_where[0],
                    torch.linalg.norm(decoded - crop),
                    crop,
                    rgb,
                )
                self._beliefs[object_name].update_position_belief(
                    pos_or_cam,
                    var=q.variance.sum(),
                    observed=observed,
                    d_max=2,
                )

    def _imagine(self, camera_pose):
        canvas = torch.zeros((3, 480, 480))
        for object_name, belief in self._beliefs.items():
            if len(belief.camera_poses) < 1:
                break

            model = self._models[object_name]

            prior = belief.global_pose

            coord = self._pinhole.coords_to_camera_frame(
                prior.mean.unsqueeze(0), camera_pose
            )
            u, v, d = self._pinhole.xyz_to_uvd(*coord.T)

            s = self._depth_to_scale(d)
            x, y = self.spatial_transformer_480.uv_to_xy(u, v)
            z_where = torch.tensor([[s, x, y]])

            relative = relative_transform(
                belief.camera_poses[-1].unsqueeze(0),
                camera_pose.unsqueeze(0),
            )
            action = Action.from_matrix(relative).nn_action

            q_hat = model["ccn"].transition(belief.local_poses[-1].mean, action)
            x_hat = model["ccn"].decoder(q_hat.mean)
            x_hat_full = self.spatial_transformer_480.inverse(z_where, x_hat)
            canvas += (1 - pixel_mask(x_hat_full[0])) * x_hat_full[0]

        return canvas

    def _reconstruct(self, state):
        return self._imagine(state["camera_pose"].clone())

    def _sample_positions(self, n=300, near=None):
        # Ws parameters
        if near is None:
            ws = torch.tensor([[-0.5, 0.5], [-0.5, 0.5], [0.1, 0.6]])
            mins = ws.min(dim=1).values.unsqueeze(0)
            ranges = (ws.max(dim=1).values - mins[0]).unsqueeze(0)
            return torch.rand((n, 3)) * ranges + mins
        else:
            # near the current position, i.e. refine here
            return torch.randn((n, 3)) * 0.025 + near.unsqueeze(0)

    def _sample_actions(self, current, n=300):
        lookats = torch.zeros((0, 3))
        targets = torch.zeros((0, 4, 4))
        object_idcs = torch.zeros((n))
        n_per_object = n // len(list(self._beliefs.keys()))
        for idx, (object_name, belief) in enumerate(self._beliefs.items()):
            # TODO: remove this if statement
            if object_name == self._preference["object_name"]:
                object_idcs[idx * n_per_object : (idx + 1) * n_per_object] = idx
                # Step 1. Sample lookats
                if self._sample_mean_lat:
                    lat = belief.global_pose.mean.unsqueeze(0).repeat(n_per_object, 1)
                else:
                    bel = belief.position_multivariate_normal
                    lat = bel.sample((n_per_object,))

                # lat = belief.global_pose.mean.repeat(n_per_object, 1)

                n_near = 10
                pos = self._sample_positions(n_per_object - n_near)
                pos = torch.cat(
                    [
                        pos,
                        self._sample_positions(n_near, near=current[:3, -1]),
                    ],
                    dim=0,
                )
                # Rescale the positions w.r.t. the preference distance
                if self.mode == agent_modes.EFE.value:
                    d = self._preference["d_hat"]
                    d = torch.randn_like(pos[:, 0]) * 0.10 + d
                    # d = 0.40
                    norm = torch.linalg.norm(pos - lat, dim=1)
                    scale_factor = (d / norm).unsqueeze(-1)
                    pos = lat + (pos - lat) * scale_factor

                pos[:, 0] = torch.clip(pos[:, 0], -0.5, 0.5)
                pos[:, 1] = torch.clip(pos[:, 1], -0.5, 0.5)
                pos[:, 2] = torch.clip(pos[:, 2], 0.1, 0.6)

                # Also consider staying still
                # pos[-1] = current.clone()[:3, -1]

                # 2. Compute random poses and matrices that look at these points
                target = look_at_matrix(pos, lat)

                # ...
                lookats = torch.cat([lookats, lat], dim=0)
                targets = torch.cat([targets, target], dim=0)

        current = current.unsqueeze(0).repeat(targets.shape[0], 1, 1)
        actions = Action(relative_transform(current.clone(), targets.clone()))

        return {
            "global_target": targets,
            "look_at": lookats,
            "action": actions,
            "object_indices": object_idcs,
        }

    def _expected_position_infogain(self, actions):
        """
        Compute the expected infogain with respect to the global targets. This avoids the problem
        where you have to overcome a peak in the expected
        free energy to achieve the overall lowest G.
        """
        targets = actions["global_target"]
        lookats = actions["look_at"]

        expected_rays = cast_ray(240, 240, targets, self._pinhole)
        distances = torch.linalg.norm(targets[:, :3, -1] - lookats, dim=1)
        positions = ray_estimate(expected_rays, distances.unsqueeze(1))

        position_infogain = torch.zeros((len(targets)))
        for belief in self._beliefs.values():
            prior = belief.position_multivariate_normal
            pm, pcv = prior.mean, prior.covariance_matrix
            pm = pm.unsqueeze(0).repeat(len(targets), 1)
            pcv = pcv.unsqueeze(0).repeat(len(targets), 1, 1)
            prior_b = MultivariateNormal(pm, pcv)
            prob = torch.exp(prior_b.log_prob(positions.mean))
            post = multiply_multivariate_normals(prior_b, positions)
            position_infogain += prob * kl_divergence(prior_b, post)

        return position_infogain

    def _expected_position_instrumental(self, actions):
        lookats = actions["look_at"]
        negative_log_prob = -self._beliefs[
            self._preference["object_name"]
        ].position_multivariate_normal.log_prob(lookats)
        return negative_log_prob

    def _expected_pose_instrumental(self, state, actions):
        bel = self._beliefs[self._preference["object_name"]]
        preference_model = self._models[self._preference["object_name"]]

        if len(bel.camera_poses) > 0:
            neg_pose_log_prob = torch.zeros(len(actions["global_target"]))
            # Only using argmin causes the agent to get stuck sometimes because it will always predict the
            # same action; only the last one will also yield annoying behaviour because the agent will
            # sometimes jumpa away
            # for idx in [-1, np.argmin(bel.reconstruction_mses)]: #[np.argmin([q.variance.sum() for q in bel.local_poses]), -1]:
            indices = np.argsort([z[1:].abs().mean() for z in bel.z_wheres])
            # indices = np.argsort(q.variance.abs().sum() for q in bel.local_poses)
            # idx = np.random.choice(indices[:5])
            idx = -1
            action = Action.from_matrix(
                relative_transform(
                    bel.camera_poses[idx]
                    .unsqueeze(0)
                    .repeat(len(actions["global_target"]), 1, 1),
                    actions["global_target"],
                )
            ).nn_action

            q = bel.local_poses[idx].mean.repeat(action.shape[0], 1)

            q_hat_t = preference_model["ccn"].transition(q, action)
            neg_pose_log_prob -= self._preference["q_ccn"].log_prob(q_hat_t.mean)

            object_pos = self._beliefs[self._preference["object_name"]].global_pose.mean
            d = torch.linalg.norm(
                object_pos.unsqueeze(0) - actions["global_target"][:, :3, -1],
                dim=1,
            )
            distance_log_prob = -Normal(
                loc=torch.ones_like(d) * self._preference["d_hat"],
                scale=torch.ones_like(d) * 0.025,
            ).log_prob(d)

            return (
                neg_pose_log_prob + distance_log_prob,
                {
                    "q_hat_t": q_hat_t,
                    "nll_pose": neg_pose_log_prob,
                    "nll_dist": distance_log_prob,
                    "r": d,
                },
            )
        else:
            # Preference object not in view: don't bother
            return torch.zeros(actions["action"].matrix.shape[0]), dict()

    def reset(self):
        self._steps = 0
        for k in self._models.keys():
            self._beliefs[k].reset()
        self._preference = {}

    def act(self, n):
        raise NotImplementedError
