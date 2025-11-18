from ccn.active_inference.diff_agent import DAgent, action_modes, agent_modes
from ccn.action import Action

from ccn.geometry import relative_transform, look_at_matrix

from ccn.models.spatial_transformers import infer_uv, z_where_inv

import torch
from enum import Enum


class RevalAgent(DAgent):
    def _sample_actions(self, current, n=300):
        lookats = torch.zeros((0, 3))
        targets = torch.zeros((0, 4, 4))
        object_idcs = torch.zeros((n))
        n_per_object = n // len(list(self._beliefs.keys()))
        for idx, (object_name, belief) in enumerate(self._beliefs.items()):

            # In EFE mode - only care about the preference
            if self.mode == agent_modes.EFE.value:
                object_name = self._preference["object_name"]
                belief = self._beliefs[object_name]

            object_idcs[idx * n_per_object : (idx + 1) * n_per_object] = idx
            # Step 1. Sample lookats
            if self._sample_mean_lat:
                lat = belief.global_pose.mean.unsqueeze(0).repeat(
                    n_per_object, 1
                )
            else:
                bel = belief.position_multivariate_normal
                lat = bel.sample((n_per_object,))

            # lat = belief.global_pose.mean.repeat(n_per_object, 1)

            n_near = 10
            pos = self._sample_positions(n_per_object - n_near)
            pos = torch.cat(
                [pos, self._sample_positions(n_near, near=current[:3, -1]),],
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

            # Also reconsider previous target
            # pos[-1] = current.clone()[:3, -1]
            pos[-1] = self.global_target.get("global_target", torch.eye(4))[
                :3, -1
            ]

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

    def _expected_free_energy(self, state, actions):
        """
        Compute the expected free energy, if the agent is an infogain agent it will
        only return the epistemic value over the position, if there is a preference 
        it will also include the instrumental terms over pose and position 
        """
        epi = self._expected_position_infogain(actions)

        if self.mode == agent_modes.EFE.value:
            ins_position = self._expected_position_instrumental(actions)
            ins_pose, pose_inst_info = self._expected_pose_instrumental(
                state, actions
            )
        else:
            ins_position, ins_pose = (
                torch.zeros_like(epi),
                torch.zeros_like(epi),
            )

        efe = -epi + ins_position + ins_pose
        return (
            efe,
            {
                "G": efe,
                "position_epistemic": epi,
                "position_instrumental": ins_position,
                "pose_instrumental": ins_pose,
            },
        )

    def reset(self):
        DAgent.reset(self)
        self.n_steps = 0
        self.efe_info = dict(
            {
                "G": 0,
                "position_epistemic": 0,
                "position_instrumental": 0,
                "pose_instrumental": 0,
            }
        )

    def act(self, state):
        return self.act_fsm(state)

    def act_fsm(self, state):
        """
        Finite state implementation of the active inference agent acting in 
        different states: 
        1. The Highlevel state: the agent has to make a decision on the
            reasoning level, i.e. process all observations and decide for 
            a target to move to. 
            
            This should be triggered every  10 steps, as to re-evaluate so 
            often 
        2. The Refinement state, when the agent has reached its target it 
            goes in a low level mode, changing the orientation until the object
            is placed in the center of the observation 
        3. The step stage: the agent is still moving towards its global 
            objective, and does not recompute a goal every time

        """
        self.n_steps += 1

        # Update state information
        self._clear_belief_markers()
        self._update_preference(state)
        self._update_object_belief(state)

        # Set the action mode
        if len(self.global_target) == 0:
            self.action_mode = action_modes.HIGHLEVEL.value
        elif self.global_target_reached:
            self.action_mode = action_modes.HIGHLEVEL.value
        elif self.n_steps % 10 == 0:
            self.action_mode = action_modes.HIGHLEVEL.value

        # =====
        # Finite state machine
        # =====
        # Mode HIGHLEVEL: select a new global target for the agent to move to
        if self.action_mode == action_modes.HIGHLEVEL.value:

            actions = self._sample_actions(
                state["camera_pose"].clone(), self._n_samples
            )
            efe, self.efe_info = self._expected_free_energy(state, actions)
            chosen_index = efe.argmin()

            self.lookat = actions["look_at"][chosen_index]
            self.lookat_object = list(self._beliefs.keys())[
                int(actions["object_indices"][chosen_index].item())
            ]

            self.global_target = {
                k: v[chosen_index] for k, v in actions.items()
            }
            self.global_target_reached = False

            relative_tf = relative_transform(
                state["camera_pose"].clone().unsqueeze(0),
                self.global_target["global_target"].unsqueeze(0),
            )[0]
            action_instance = Action(relative_tf)

            # set the mode to step -> it must now get to the target
            self.action_mode = action_modes.STEP.value

            # If the step is so small, the refinement is the next step
            if torch.linalg.norm(action_instance.matrix[:3, -1]) < 0.05:
                self.global_target_reached = True
                self.action_mode = action_modes.REFINE.value

            # Get an environment action
            action = action_instance.get_step(0.05).environment_action

        # Mode STEP: step in the direction of the global target
        elif self.action_mode == action_modes.STEP.value:
            relative_tf = relative_transform(
                state["camera_pose"].clone().unsqueeze(0),
                self.global_target["global_target"].unsqueeze(0),
            )[0]
            action_instance = Action(relative_tf)

            # If the step is so small, the refinement is the next step
            if torch.linalg.norm(action_instance.matrix[:3, -1]) < 0.05:
                self.global_target_reached = True
                self.action_mode = action_modes.REFINE.value

            # Get an environment action
            action = action_instance.get_step(0.05).environment_action

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
                if max(abs(du), abs(dv)) < 0.05:  # Can be 24 pixels off
                    self.action_mode = action_modes.HIGHLEVEL.value
            else:
                # Can not observe the object -> go back to a high level mode
                self.action_mode = action_modes.HIGHLEVEL.value

        return (
            action,
            dict(
                {
                    "particles": {
                        k: v.global_pose.particles
                        for k, v in self._beliefs.items()
                    },
                    "g_info": self.efe_info,
                }
            ),
        )
