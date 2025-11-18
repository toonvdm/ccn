from tqdm import tqdm
import torch
import copy

from ccn.active_inference.base_agent import agent_modes
from ccn.active_inference.diff_agent import DAgent

__all__ = ["action_perception_loop", "agent_modes", "DAgent"]


def action_perception_loop(environment, agent, steps=10, stop_when_done=False):
    logs = {"state": [], "agent_info": [], "env_info": []}

    state = environment.reset()
    env_info = environment.info

    logs["reward"] = 0

    for step in tqdm(range(steps)):
        # -> provide env-info to actinf agent because workings
        if type(agent).__name__ == "LexaAgent":
            action, agent_info = agent.act(state), dict({})
            # tensorflow to pytorch
            action = torch.tensor(action.numpy(), dtype=torch.float32)
        else:
            action, agent_info = agent.act(env_info)

        state, reward, done, env_info = environment.step(
            action.to(torch.float64)
        )

        logs["reward"] = reward
        logs["state"].append(copy.deepcopy(state))
        logs["env_info"].append(copy.deepcopy(env_info))
        logs["agent_info"].append(agent_info)

        if agent_info.get("failed", False):
            print("Agent failed")
            break

        if reward == 1 and stop_when_done:
            print("Done")
            break

    logs["n_steps"] = step + 1

    return logs
