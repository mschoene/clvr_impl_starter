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

def print_model_parameters(model):
    print("Model Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Name: {name}, Size: {param.size()}, Type: {param.dtype}")
            print(param.data)

def print_gradients(model):
    print("Gradients:")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"Name: {name}, Size: {param.grad.size()}, Type: {param.grad.dtype}")
            print(param.grad)


data_spec = AttrDict(
        resolution=64,
        max_ep_len=40,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True)

def extract_values_from_batch(batched_data, batch_size):
    # Check if the input is batched
    is_batched = (batch_size > 1)
    # Extract the required values from the batched data
    if is_batched:

        ipos_t = [b[0] for b in batched_data] #batched_data[0]
        iaction_probas_old = [b[2] for b in batched_data] #batched_data[2]
        ivalue = [b[7] for b in batched_data] #batched_data[7]
        iadvantage = [b[8] for b in batched_data] #batched_data[8]
        iret = [b[9] for b in batched_data] #batched_data[8]
    else:
        ipos_t = batched_data[0]
        iaction_probas_old = batched_data[2]
        ivalue = batched_data[7]
        iadvantage = batched_data[8]
        iret = batched_data[9]
    
    ipos_t = torch.stack(ipos_t)
    ipos_t = ipos_t.to(torch.float32)
    
    #print(iadvantage)

    return ipos_t, torch.stack(iaction_probas_old), torch.stack(iadvantage), torch.stack(ivalue), torch.stack(iret),


def my_collate_fn(data):
    return tuple(data)


env = SpritesEnv(n_distractors = 0)
env.seed(41)
env.set_config(data_spec)

#n_actors = 2 # = N 
n_traj_steps = 19 #39#19 # T number is T-1 in fact
n_trajectories = 100
n_episodes = 30  #total number of iterations over the  dataset, this will mean M*nEpisodes = #of steps taken
#  will add N new episodes to the bufffer (kicking the olderst N out of the buffer)
buffer_size = 8*20*n_trajectories #1000000 # M = N*T thus defining the number of actors as N = M/T
n_epochs = 5 #number of optim steps on a given buffer data set
batch_size = 512



Oracle_model = Oracle(4) #4 = x y of actor and target. could also think to input the velocities too for extra information #TODO test velo impact
optimizer = optim.Adam( Oracle_model.parameters(), lr =2.5e-4 )
epsilon= 0.1

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
        advantage: float =0.
        ret: float = 0.


class NpDataset(Dataset):
  def __init__(self, array):
    self.array = array
  def __len__(self): return len(self.array) 
  def __getitem__(self, i): return self.array[i]


def torch_pos(i_state):
        pos, _ = np.split(i_state, 2, -1)
        pos = pos.flatten()
        return torch.from_numpy(pos).float().unsqueeze(0)


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/RL_training{}'.format(timestamp))

loss_fn = nn.MSELoss()
repBuf = replayBuffer(buffer_size)

print(Oracle_model)

#def do_episodes(env, actor, critic, num_episodes, gamma = 1.):
def do_episodes(env, n_episodes,  gamma = 0.99, lambda_val = 0.95):
        #We shall step through the amount of episodes #In each episode we step through the trajectory
        # according to the policy (actor) at hand and add the values to the episode as estimated by the critic
        counter = 0.

        for iteration in range(n_episodes):
                print("buffer length  ", len(repBuf))
                for _ in range(n_trajectories):
                        state = env.reset()
                        episode = []
                        for st in itertools.count():
                                with torch.no_grad():
                                        state = env._state
                                        #print(env.action_space)
                                        pos_t = torch_pos(state)

                                        action, action_proba, value = Oracle_model(pos_t)
                                        #action = map_int_to_action(int(torch.argmax(action_proba)))
                                        #action = action_totake
                                        #print(action)

                                        next_state_obs, reward, done, _  = env.step(action) #TODO fix this to the action predicted

                                        #obtain value of next state according to current policy
                                        next_pos_t =  torch_pos(state)
                                        __, _, next_state_value = Oracle_model(next_pos_t)

                                        #print(state, env._state)
                                        delta = reward + gamma * next_state_value - value
                                        episode.append(episodeStep(pos_t, action, action_proba, reward, next_state_obs, done, delta, value ) )
                                        #TODO also keep action probas
                                        if st>=n_traj_steps or done:
                                                break 
                                        #done with one episode

                        #compute advantage from delta
                        adv = torch.tensor([episode[-1].reward - episode[-1].value ])
                        ret = torch.tensor([episode[-1].value ]) #TODO init with max a Q(s_t, a) 
                        for i, e_t in enumerate(reversed(episode)):
                                episode[-(i+1)].advantage = adv #fill in reverse but enum starts at 0, so we append it first at the last item and go forward from there
                                episode[-(i+1)].ret = ret #fill in reverse but enum starts at 0, so we append it first at the last item and go forward from there
                                adv = adv + e_t.delta[0] *gamma * lambda_val 
                                #print(e_t.delta, e_t.delta[0])
                                ret = e_t.reward  + ret*gamma 

                        #append each timestep at the buffer
                        for e_t in episode:
                                repBuf.append([v for v in asdict(e_t).values()])

                for i_epoch in range(n_epochs):   
                        data= NpDataset( ( [ele for ele in repBuf] ))
                        total_loss = 0.
                        dataloader = DataLoader(data, batch_size=batch_size , collate_fn=my_collate_fn, shuffle=True ) # create your dataloader
                        for i_batch, sample_batched in enumerate(dataloader):
                                optimizer.zero_grad()

                                pos_t_batched, action_probas_old_batched, advantage_batched, value_batched, return_batched = extract_values_from_batch(sample_batched, batch_size)

                                #pred actions and values
                                action, action_probas_prop, value_prop = Oracle_model(pos_t_batched)
                                #calculate the loss
                                #print(action_probas_prop, action_probas_old_batched)
                                #TODO get probabilities instead and divide those
                                ap_ratio = action_probas_prop / action_probas_old_batched 

                                clipped_ratio = torch.clamp(ap_ratio,  (1.-epsilon), (1.+ epsilon) )
                                #action_loss = (ap_ratio * advantage_batched.detach() ).mean()
                                #action_loss = ( torch.min(   clipped_ratio, ap_ratio) * advantage_batched.detach() ).mean()
                                action_loss = ( (torch.min(   clipped_ratio* advantage_batched.detach(), ap_ratio* advantage_batched.detach())  )).mean() 
                                #for iad in advantage_batched:
                                #        if( iad < 0):
                                #                print(iad )
                                #print(( (torch.min(   clipped_ratio* advantage_batched.detach(), ap_ratio* advantage_batched.detach())  )).mean()     , "  vs ", ( torch.min(   clipped_ratio, ap_ratio) * advantage_batched.detach() ).mean())
                                #value_loss = torch.pow(value_batched.detach() -value_prop, 2).mean() 

                                value_loss = torch.pow(return_batched.detach() -value_prop, 2).mean() 

                                total_loss = value_loss - action_loss

                                if(i_batch % 100 == 0):
                                        print('total loss = {}   action_loss = {}  value_loss = {} '.format(total_loss, action_loss, value_loss))

                                total_loss.mean().backward()
                                #print_gradients(Oracle_model)
                                optimizer.step()
                                #optimizer.zero_grad()

                                counter += 1*batch_size

                                reward_b = 0.
                                for i_a in action.detach().numpy():
                                        next_state_obs, reward, done, _ = env.step(i_a) 
                                        reward_b += reward


                                tb_x = i_epoch * len(dataloader) + i_batch + 1
                                writer.add_scalar('Loss/train', total_loss, tb_x)
                                writer.add_scalar('Reward/train', reward_b, tb_x)
                        
        print(counter)


# Assuming you have a PyTorch model named 'model'
#print_model_parameters(Oracle_model)
# Train your model
do_episodes(env, n_episodes=n_episodes)

# After training
#print_model_parameters(Oracle_model)


#cv2.imwrite("test_rl.png", 255 * np.expand_dims(obs, -1))
#obs, reward, done, info = env.step([0, 0])
#obs2, reward2, done2, info2 = env.step([0, 0])
#print(obs, reward, done, info)

