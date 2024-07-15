from collections import namedtuple
import torch
import numpy as np
from dataclasses import dataclass, asdict
#from rl_utils.buffer import ReplayBuffer
from rl_utils.torch_utils import np_to_torch
import concurrent.futures
import gym

import multiprocessing as mp


# class to set and update the running mean reward
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var)

#instantiate
running_mean_std = RunningMeanStd(shape=())

def normalize_rewards_with_running_stats(rewards):
    rewards = np.array(rewards)
    running_mean_std.update(rewards)
    normalized_rewards = (rewards - running_mean_std.mean) / (running_mean_std.std + 1e-8)
    return normalized_rewards.tolist()



EpisodeStep = namedtuple('EpisodeStep', ['state', 'action', 'action_probas', 'reward', 'done', 'value', 'ret', 'advantage'])


def collect_trajectory_step(actor, state, env, episode, deterministic=False ):
    action, action_proba, value = actor.act(state, deterministic) 
    next_state_obs, reward, done, _ = env.step(action) # dont need the info, put in _
    episode.append(EpisodeStep(state, action, action_proba, reward, done, value, 0., 0.))

    return np_to_torch(next_state_obs), done

def collect_trajectory(seed, actor, env_name, max_steps , deterministic=False ):
    episode = []
    env = gym.make(env_name)
    env.seed(seed)
    state = env.reset()
    state = np_to_torch(state)
    with torch.no_grad():
        actor.eval()
        for st in range(max_steps):
            pos_t, done = collect_trajectory_step(actor, state, env, episode, deterministic )
            if done:
                break 
            state = pos_t # init state for next step 
    return episode


def collect_n_trajectories(n_traj, replayBuffer, actor, env_name, n_traj_steps, gamma, lambda_val, n_workers = 4, deterministic=False ):
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        # each worker produces n_traj many different env 
        seeds = np.random.randint(0, 20000, size=n_traj)
        for seed in seeds: # collect bundle/vine (nworker many) traj with same starting point
        #for _ in range(n_traj):
            future = executor.submit(collect_trajectory,seed, actor, env_name, n_traj_steps  , deterministic)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            trajectory = future.result()
            trajectory = calc_discd_vals(trajectory, gamma, lambda_val)
            for e_t in trajectory:
                replayBuffer.append(list(e_t)) #replayBuffer.append([v for v in asdict(e_t).values()])


def calc_discd_vals(episode, gamma, lambda_val):

    rewards = [step.reward for step in episode]
    normalized_rewards = normalize_rewards_with_running_stats(rewards)
    
    # Initialize tensors
    advantages = torch.zeros(len(episode), dtype=torch.float32)
    returns = torch.zeros(len(episode), dtype=torch.float32)
    last_adv = 0.0

    for t in reversed(range(len(episode))):
        reward = normalized_rewards[t]
        #reward = episode[t].reward #TODO check if this works properly 
        value = episode[t].value
        if t < len(episode) - 1:
            next_value = episode[t + 1].value
            delta = reward + gamma * next_value - value
            last_adv = delta + gamma * lambda_val * last_adv
        else:
            delta = reward - value
            last_adv = delta

        advantages[t] = last_adv
        returns[t] = advantages[t] + value  # Returns are advantages plus value estimates

    updated_episode = []
    for i in range(len(episode)):
        updated_step = episode[i]._replace(advantage=advantages[i], ret=returns[i])
        updated_episode.append(updated_step)

    return updated_episode

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