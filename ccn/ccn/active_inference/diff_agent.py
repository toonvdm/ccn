import torch
from torch.distributions import Categorical

from ccn.models.spatial_transformers import infer_uv, z_where_inv
from ccn.active_inference.base_agent import BaseAgent, agent_modes
from ccn.action import Action
from ccn.geometry import relative_transform

import logging

from enum import Enum

logger = logging.getLogger(__name__)

action_modes = Enum("Action_modes", ["REFINE", "HIGHLEVEL", "STEP"])


class DAgent(BaseAgent):
    """
    Agent that manages attractor points on multiple levels of the hierarchy,
    the resulting action is a smooth differential movement
    """

    def __init__(
        self,
        object_names,
        model_dir,
        ccn_dir,
        particle_filter=False,
        teleport=True,
        debug=False,
        mode=agent_modes.INFOGAIN,
        n_samples=5000,
    ):
        BaseAgent.__init__(
            self,
            object_names,
            model_dir,
            ccn_dir,
            particle_filter,
            teleport,
            debug,
            mode,
        )

        self.action_mode = action_modes.HIGHLEVEL.value
        self._n_samples = n_samples

        self.global_target_reached = True
        self.global_target = dict()
        self.prev_action = torch.zeros((6))

    def _clear_belief_markers(self):
        for b in self._beliefs.values():
            b.was_updated = False

    def reset(self):
        BaseAgent.reset(self)
        self.global_target = dict()
        self.global_target_reached = True
        self.prev_action = torch.zeros((6))

    @staticmethod
    def _add_to_action_dict(action_dict, action):
        for k, v in action.items():
            if isinstance(v, Action):
                action_dict[k] = Action.from_matrix(
                    torch.cat(
                        [v.matrix.unsqueeze(0), action_dict[k].matrix],
                        dim=0,
                    )
                )
            else:
                action_dict[k] = torch.cat([v.unsqueeze(0), action_dict[k]], dim=0)
        return action_dict

    def act(self, state):
        n = self._n_samples

        # TODO make this a Finite State Machine.
        self._clear_belief_markers()
        self._update_preference(state)
        self._update_object_belief(state)

        # Get the action based on which mode you are in
        x_hat = None
        x_hat_adjusted = None
        g_info = dict()

        if self.action_mode == action_modes.HIGHLEVEL.value:
            if self.global_target_reached or self.global_target == dict():
                actions = self._sample_actions(state["camera_pose"].clone(), n)

                epi = self._expected_position_infogain(actions)
                g_info["epistemic"] = {"position": -epi}

                fe = -epi

                if self.mode == agent_modes.EFE.value:
                    ins_position = self._expected_position_instrumental(actions)
                    (
                        ins_pose,
                        pose_inst_info,
                    ) = self._expected_pose_instrumental(state, actions)

                    fe += ins_position + ins_pose

                    g_info["instrumental"] = {
                        "pos": ins_position,
                        "pose": pose_inst_info.get("nll_pose"),
                        "distance": pose_inst_info.get("nll_dist"),
                        "r": pose_inst_info.get("r"),
                    }

                    idx = fe.argmin()
                    if pose_inst_info.get("q_hat_t") is not None:
                        q_hat_t = pose_inst_info.get("q_hat_t")
                        x_hat = self._models[self._preference["object_name"]][
                            "ccn"
                        ].decoder(q_hat_t.mean[idx : idx + 1])

                g_info["G"] = fe
                g_info["actions"] = actions["action"].matrix
                g_info["globals"] = actions["global_target"]
                idx = fe.argmin()

                self.lookat = actions["look_at"][idx]
                self.lookat_object = list(self._beliefs.keys())[
                    int(actions["object_indices"][idx].item())
                ]
                self.global_target = {k: v[idx] for k, v in actions.items()}
                self.global_target_reached = False

            relative_tf = relative_transform(
                state["camera_pose"].clone().unsqueeze(0),
                self.global_target["global_target"].unsqueeze(0),
            )[0]
            action_instance = Action(relative_tf).get_step(0.05)
            action = action_instance.environment_action

            # if these are the same, then the agent will reach the target in the next step
            if torch.allclose(action_instance.matrix[:3, -1], torch.zeros(3)):
                # so that next time a new target is set
                self.global_target_reached = True
                # Swap the action mode depending on whether the agent reached it's action
                self.action_mode = action_modes.REFINE.value
            else:
                # If it gets stuck, choose a new target
                if torch.allclose(self.prev_action, action, 1e-3, 1e-3):
                    self.global_target = dict()

        elif self.action_mode == action_modes.REFINE.value:
            # The spatial-transformer object centric thing
            target_belief = self._beliefs[self.lookat_object]
            action = torch.zeros((6))
            if target_belief.was_updated:
                z = target_belief.z_wheres[-1].unsqueeze(0)
                u, v = infer_uv(z_where_inv(z)[0])

                dv = (v - 240) / 480
                du = (u - 240) / 480

                action[4] = dv / 50
                action[3] = -du / 50

                # Object in center, change action mode
                if max(abs(du), abs(dv)) < 0.05:  # 05:  # Can be 24 pixels off
                    self.action_mode = action_modes.HIGHLEVEL.value
            else:
                # If the refining object is not in view; then just switch to the
                # high level again
                self.action_mode = action_modes.HIGHLEVEL.value

        self.prev_action = action
        return (
            action,
            {
                "action_mode": self.action_mode,
                "action": action,
                "position_beliefs": self._beliefs,
                "particles": {
                    k: v.global_pose.particles for k, v in self._beliefs.items()
                },
                "target_camera_pose": self.global_target.get("global_target", None),
                "lookat": self.lookat,
                "lookat_object": self.lookat_object,
                "x_hat": x_hat,
                "x_hat_distance_adjusted": x_hat_adjusted,
                "g_info": g_info,
            },
        )
