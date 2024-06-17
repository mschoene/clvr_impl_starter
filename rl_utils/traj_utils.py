import torch
import numpy as np
from dataclasses import dataclass, asdict
from rl_utils.torch_utils import np_to_torch

@dataclass
class episodeStep():
        state: np.array
        action: np.array
        action_probas: np.array
        reward: float
        next_state_obs: np.array 
        done: bool 
        value: float 
        delta: float = 0.
        ret: float = 0.
        advantage: float =0.


def collect_trajectory_step(actor, state, env, episode):
    action, action_proba, value = actor.act(state) 
    next_state_obs, reward, done, _  = env.step(action) # dont need the info, put in _
    next_pos_t = np_to_torch(next_state_obs)   
    episode.append(episodeStep(state, action, action_proba, reward, next_state_obs, done, value ) )
    return next_pos_t, done

def collect_trajectory(actor, env, max_steps):
    episode = []
    state = env.reset()
    state = np_to_torch(state)
    with torch.no_grad():
        actor.eval()
        for st in range(max_steps):
            pos_t, done = collect_trajectory_step(actor, state, env, episode)
            if done:
                break 
            state = pos_t # init state for next step 
    return episode

def collect_n_trajectories(n_traj, replayBuffer, actor, env, n_traj_steps, gamma, lambda_val):
    for _ in range(n_traj):
        episode = collect_trajectory(actor, env, n_traj_steps)
        calc_discd_vals(episode, gamma, lambda_val)
        for e_t in episode:
            replayBuffer.append([v for v in asdict(e_t).values()])


def calc_discd_vals(episode, gamma, lambda_val):
    # Initialize tensors
    advantages = torch.zeros(len(episode), dtype=torch.float32)
    returns = torch.zeros(len(episode), dtype=torch.float32)
    last_adv = 0.0

    for t in reversed(range(len(episode))):
        reward = episode[t].reward
        value = episode[t].value
        if t < len(episode) - 1:
            next_value = episode[t + 1].value
            next_return = returns[t + 1]
            delta = reward + gamma * next_value - value
            advantages[t] = delta + gamma * lambda_val * last_adv
            returns[t] = reward + gamma * next_return
            last_adv = advantages[t]
        else:
            # For the last timestep
            advantages[t] = reward - value
            returns[t] = reward + value

    for i, e_t in enumerate(episode):
        e_t.advantage = advantages[i]
        e_t.ret = returns[i]