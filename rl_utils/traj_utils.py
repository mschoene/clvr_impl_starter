from collections import namedtuple
import torch
import numpy as np
from dataclasses import dataclass, asdict
#from rl_utils.buffer import ReplayBuffer
from rl_utils.torch_utils import np_to_torch
import concurrent.futures
import gym

import multiprocessing as mp


@dataclass
class episodeStep():
        state: np.array
        action: np.array
        action_probas: np.array
        reward: float
        done: bool 
        value: float 
        ret: float = 0.
        advantage: float =0.

EpisodeStep = namedtuple('EpisodeStep', ['state', 'action', 'action_probas', 'reward', 'done', 'value', 'ret', 'advantage'])


def collect_trajectory_step(actor, state, env, episode):
    action, action_proba, value = actor.act(state) 
    next_state_obs, reward, done, _ = env.step(action) # dont need the info, put in _
    #next_pos_t = np_to_torch(next_state_obs)   
    #episode.append(episodeStep(state, action, action_proba, reward, next_state_obs, done, value ) )
    #episode.append(episodeStep(state, action, action_proba, reward, done, value ) )
    episode.append(EpisodeStep(state, action, action_proba, reward, done, value, 0., 0.))

    return np_to_torch(next_state_obs), done

def collect_trajectory(seed, actor, env_name, max_steps):
    episode = []
    env = gym.make(env_name)
    env.seed(seed)
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


#def collect_n_trajectories(n_traj, replayBuffer, actor, env, n_traj_steps, gamma, lambda_val, n_workers = 4):

    #if len(replayBuffer) + n_traj_steps*n_workers*n_traj > replayBuffer.maxlen:
    #        print("popping length away ")
    #        replayBuffer.pop_batch(n_traj_steps*n_workers*n_traj)
    #        print("popping length away ")
    #seeds = np.random.randint(0, 10000, size=n_traj* n_workers)
    #for _ in range(n_traj*n_workers):
    #    print(_)
    #    episode = collect_trajectory(actor, env, n_traj_steps)
         
    #    episode = calc_discd_vals(episode, gamma, lambda_val)
        
    #    for e_t in episode:
    #        replayBuffer.append(list(e_t))
        #for e_t in episode:
        #    print( e_t)
        #    replayBuffer.append([v for v in asdict(e_t).values()])

def collect_n_trajectories(n_traj, replayBuffer, actor, env_name, n_traj_steps, gamma, lambda_val, n_workers = 4):
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        seeds = np.random.randint(0, 10000, size=n_traj)
        for seed in seeds:
        #for _ in range(n_traj):
            future = executor.submit(collect_trajectory,seed, actor, env_name, n_traj_steps)
            #calc_discd_vals(future.result(), gamma, lambda_val)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            trajectory = future.result()
            trajectory = calc_discd_vals(trajectory, gamma, lambda_val)
            for e_t in trajectory:
                replayBuffer.append(list(e_t)) #replayBuffer.append([v for v in asdict(e_t).values()])


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

    #for i, e_t in enumerate(episode):
    #    e_t.advantage = advantages[i]
    #    e_t.ret = returns[i]

    updated_episode = []
    for i in range(len(episode)):
        updated_step = episode[i]._replace(advantage=advantages[i], ret=returns[i])
        updated_episode.append(updated_step)

    return updated_episode