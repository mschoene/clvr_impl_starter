from asyncio import wait
import torch.optim as optim
import gym
import numpy as np
import cv2
import torch
from general_utils import AttrDict
from sprites_env.envs.sprites import SpritesEnv, map_int_to_action
import torch.nn as nn
import torch.nn.functional as F
from  collections import deque, namedtuple
import itertools

from replayBuffer import *
from models import Oracle

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary
from datetime import datetime

from dataclasses import dataclass, asdict

from torchvision.transforms import v2


data_spec = AttrDict(
        resolution=64,
        max_ep_len=40,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True)

def extract_values_from_batch(batched_data, batch_size):

    # Check if the input is batched
    is_batched = (batch_size > 1)
    
    print( batched_data, len(batched_data))
    print("first one ", batched_data[0])
    # Extract the required values from the batched data
    if is_batched:

        ipos_t = batched_data[0]
        iaction_probas_old = batched_data[2]
        ivalue = batched_data[7]
        iadvantage = batched_data[8]
    else:
        ipos_t = batched_data[0]
        iaction_probas_old = batched_data[2]
        ivalue = batched_data[7]
        iadvantage = batched_data[8]
    
    ipos_t = ipos_t.to(torch.float32)
    
    return ipos_t, iaction_probas_old, iadvantage, ivalue


def my_collate_fn(data):
    # TODO: Implement your function
    # But I guess in your case it should be:
    return tuple(data)


env = SpritesEnv(n_distractors = 0)
env.seed(42)
env.set_config(data_spec)

#n_actors = 2 # = N 
n_traj_steps = 19 # T
n_trajectories = 4 
n_episodes = 20  #total number of iterations over the  dataset, this will mean M*nEpisodes = #of steps taken
#  will add N new episodes to the bufffer (kicking the olderst N out of the buffer)
buffer_size = 20  # M = N*T thus defining the number of actors as N = M/T
n_epochs = 1
batch_size = 2


Oracle_model = Oracle(4) #4 = x y of actor and target. could also think to input the velocities too for extra information #TODO test velo impact
optimizer = optim.Adam( Oracle_model.parameters(), lr =2.5e-4 )


@dataclass
class episodeStep():
        state: np.array
        action: np.array
        action_probas: np.array
        reward: float
        next_state_obs: np.array 
        done: bool 
        delta: float 
        value: float 
        advantage: float =0


class NpDataset(Dataset):
  def __init__(self, array):
    self.array = array
  def __len__(self): return len(self.array) #-1 #rm last one because???
  def __getitem__(self, i): return self.array[i]


def torch_pos(i_state):
        pos, _ = np.split(i_state, 2, -1)
        pos = pos.flatten()
        return torch.from_numpy(pos).float().unsqueeze(0)


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/RL_training{}'.format(timestamp))

loss_fn = nn.MSELoss()
repBuf = replayBuffer(buffer_size)

#def do_episodes(env, actor, critic, num_episodes, gamma = 1.):
def do_episodes(env,n_episodes=1,  gamma = 0.99, lambda_val = 0.95):
        #We shall step through the amount of episodes #In each episode we step through the trajectory
        # according to the policy (actor) at hand and add the values to the episode as estimated by the critic
        
        for _ in range(n_trajectories):
                state = env.reset()
                episode = []
                for st in itertools.count():
                        with torch.no_grad():
                                state = env._state
                                pos_t = torch_pos(state)

                                action_proba, value = Oracle_model(pos_t)
                                print("value " , value)
                                action = map_int_to_action(int(torch.argmax(action_proba)))

                                next_state_obs, reward, done, _  = env.step(action) #TODO fix this to the action predicted

                                #obtain value of next state according to current policy
                                next_pos_t =  torch_pos(state)
                                _, next_state_value = Oracle_model(next_pos_t)

                                #print(state, env._state)
                                delta = reward + gamma * next_state_value - value
                                episode.append(episodeStep(pos_t, action, action_proba, reward, next_state_obs, done, delta, value ) )
                                #TODO also keep action probas
                                if st>=n_traj_steps or done:
                                        break 
                                #done with one episode

                #compute advantage from delta
                adv = torch.tensor(0.)
                for i, e_t in enumerate(reversed(episode)):
                        episode[-(i+1)].advantage = adv #fill in reverse but enum starts at 0, so we append it first at the last item and go forward from there
                        adv = adv + e_t.delta[0] *gamma * lambda_val 

                print("before appending  ", len(repBuf))
                #append each timestep at the buffer
                for e_t in episode:
                        repBuf.append([v for v in asdict(e_t).values()])
                        print("adding  " ,len(repBuf))

        for i_epoch in range(n_epochs):   
                print(len(repBuf), " vs count " )

                data= NpDataset( ( [ele for ele in repBuf] ))
                print(len(data.array))
                #for i in data.array:
                       #print( len(i))
                #print(data.array)

                dataloader = DataLoader(data, batch_size=batch_size , collate_fn=my_collate_fn ) # create your dataloader
                for i_batch, sample_batched in enumerate(dataloader):
                        #print(sample_batched)
                        optimizer.zero_grad()
                        pos_t_batched, action_probas_old_batched, advantage_batched, value_batched = extract_values_from_batch(sample_batched, batch_size)

                        #pred actions and values
                        action_probas_prop, value_prop = Oracle_model(pos_t_batched)
                        #calculate the loss
                        ap_ratio = action_probas_prop / action_probas_old_batched 
                        action_loss = (ap_ratio * advantage_batched).mean()
                        value_loss = torch.pow(value_batched-value_prop, 2).mean()

                        total_loss = value_loss - action_loss
                        print('total loss = {}   action_loss = {}  value_loss = {} '.format(total_loss, action_loss, value_loss))

                        total_loss.mean().backward()
                        optimizer.step()

do_episodes(env, n_episodes=n_episodes)

#cv2.imwrite("test_rl.png", 255 * np.expand_dims(obs, -1))
#obs, reward, done, info = env.step([0, 0])
#obs2, reward2, done2, info2 = env.step([0, 0])
#print(obs, reward, done, info)

