from asyncio import wait
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gym
import numpy as np
import cv2
import torch
from general_utils import AttrDict
from sprites_env.envs.sprites import SpritesEnv 
import torch.nn as nn
import torch.nn.functional as F
from  collections import deque, namedtuple
import itertools

from replayBuffer import *
from models import Oracle

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary
from datetime import datetime

from dataclasses import dataclass, asdict

from torchvision.transforms import v2
from torch.distributions import MultivariateNormal


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
        max_ep_len=400,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True)

def extract_values_from_batch(batched_data, batch_size):
    # Check if the input is batched
    is_batched = (batch_size > 1)
    # Extract the required values from the batched data
    if is_batched:
        ipos_t = [b[0] for b in batched_data] #batched_data[0]
        iaction = [b[1] for b in batched_data] #batched_data[2]
        iaction_probas_old = [b[2] for b in batched_data] #batched_data[2]
        ireward = [b[3] for b in batched_data] #batched_data[2]
        ivalue = [b[6] for b in batched_data] #batched_data[7]
        iadvantage = [b[9] for b in batched_data] #batched_data[8]
        iret = [b[8] for b in batched_data] #batched_data[8]
    else:
        ipos_t = batched_data[0]
        iaction = batched_data[1]
        iaction_probas_old = batched_data[2]
        ireward = batched_data[3] #batched_data[2]
        ivalue = batched_data[6]
        iadvantage = batched_data[9]
        iret = batched_data[8]
    
    ipos_t = torch.stack(ipos_t)
    ipos_t = ipos_t.to(torch.float32)
    
    return ipos_t, torch.stack(iaction), torch.stack(iaction_probas_old), torch.stack(iadvantage), torch.stack(ivalue), torch.stack(iret), ireward

def my_collate_fn(data):
    return tuple(data)

n_distractors = 0 
env = SpritesEnv(n_distractors = n_distractors)
env = gym.make('SpritesState-v0')
env.seed(1)
env.set_config(data_spec)

n_actors = 8 # 1# 4# 4 #2 #1
n_traj_steps = 49 #399 #127 #39 #5 #9 #39#19 #ntraj step is T-1 in. so T steps will be taken (ie ntrajsteps +1)
n_trajectories = 4 #24# 1#1#50  # 10
n_episodes = 500#50 #0#500  #total number of iterations over the  data, this will mean M*nEpisodes = #of steps taken

#  will add N new episodes to the bufffer (kicking the olderst N out of the buffer)
buffer_size = n_actors* (n_traj_steps+1)*n_trajectories #1000000 # M = N*T thus defining the number of actors as N = M/T
n_epochs =  10 #5 #10 #3# 5  #4 #5 #number of optim steps on a given buffer data set
#batch_size = min(n_actors* (n_traj_steps+1)*n_trajectories, 8000) #256 #512 #1024  #64 #32 #64 #128 #32 #128
batch_size = min(n_actors* (n_traj_steps+1)*n_trajectories, 100000) #256 #512 #1024  #64 #32 #64 #128 #32 #128
minibatch_size = 128# 64 #256 #128 # 64 #(n_traj_steps+1)*n_trajectories //4  # 128 #64 #100 # 512 #these are steps not episodes to run over

state_space = 4 + n_distractors*2
actions_space_std = 0.5 # 1# -1 #3 #since we exp() this value, neg values are <0 #05 # 0.4 #5 #0.001 #5 # #5 # data_spec['max_speed'] *1.2  # since actions will be clipped to max speed, we can limit the range here already
Oracle_model = Oracle(state_space, env.action_space, state_space, actions_space_std ) 


do_adv_norm = True
do_a2c = False
ent_coef = 0.05
vf_coef = 0.5
max_grad_norm = 0.5



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


class NpDataset(Dataset):
  def __init__(self, array):
    self.array = array
  def __len__(self): return len(self.array) 
  def __getitem__(self, i): return self.array[i]


def torch_pos(i_state):
        #pos, _ = np.split(i_state, 2, -1)
        #pos = pos.flatten()
        #return torch.from_numpy(pos).float().unsqueeze(0)
        return torch.from_numpy(i_state).float().unsqueeze(0)


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/RL_training{}'.format(timestamp))

#loss_fn = nn.MSELoss()
repBuf = replayBuffer(buffer_size)
buffer_size_eval = n_traj_steps*10 #eval on 10 trajectories
repBuf_eval = replayBuffer(buffer_size)

#print(Oracle_model)
optimizer = optim.Adam( Oracle_model.parameters(), lr = 0.0003 )
epsilon= 0.2
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma =0.8)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor =0.2, patience = 7500, min_lr = 2e-7 )
#scheduler_eps = ReduceLROnPlateau(epsilon, 'min', factor =0.8, patience = 500, min_lr = 0.05 )

def collect_trajectory_step(actor, state, env, episode, episode_num):
    #actor model predicts action and values given the state
    action, action_proba, value = actor.act(state, episode_num) 
    #action = torch.clamp(action, min=-1.0, max=1.0).detach().numpy()
    next_state_obs, reward, done, _  = env.step(action)
    next_pos_t =  torch_pos(next_state_obs)
    #print(state - next_pos_t )
    #print(f"State: {state}, Action: {action}, Next State: {next_state_obs}, Reward: {reward}")
    #print(f"State calc: {state[:,:2]+0.05*action},  Next State: {next_state_obs}, Reward: {reward}")     
    episode.append(episodeStep(state, action, action_proba, reward, next_state_obs, done, value ) )
    return next_pos_t, done

def collect_trajectory(actor, env, max_steps, episode_num):
        episode = []
        state = env.reset()
        state = torch_pos(state)
        with torch.no_grad():
                actor.eval()
                for st in range(max_steps):  # Use range(max_steps) instead of itertools.count for clarity
                #for st in itertools.count():
                      pos_t, done = collect_trajectory_step(actor, state, env, episode, episode_num)
                      if done:
                      #if st >= max_steps or done:
                        #env.reset()
                        break 
                      state = pos_t #init state for next step 
        return episode

def collect_n_trajectories(n_traj, replayBuffer, actor, env, n_traj_steps, gamma, lambda_val, episode_num):
        for _ in range(n_traj):
                episode = collect_trajectory(actor, env, n_traj_steps, episode_num)
                calc_discd_vals(episode, gamma, lambda_val)
                for e_t in episode:
                        #if(e_t.advantage <= 0.):
                        #        print(e_t)
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

    # Normalize advantages
    #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for i, e_t in enumerate(episode):
        e_t.advantage = advantages[i]
        e_t.ret = returns[i]


def print_grad_norms(model, stage):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.data.norm(2).item()
            total_norm += norm ** 2
            print(f"Gradient norm for {name} at {stage}: {norm}")
    total_norm = total_norm ** 0.5
    print(f"Total grad norm at {stage}: {total_norm}")

#def do_episodes(env, actor, critic, num_episodes, gamma = 1.):
def do_episodes(env, n_episodes,  gamma = 0.99 , lambda_val = 0.95):
        # We shall step through the amount of episodes #In each episode we step through the trajectory
        # according to the policy (actor) at hand and add the values to the episode as estimated by the critic
        counter = 0.
        tb_x = 0
        for iteration in range(n_episodes):
                if (iteration % 10 == 0 ):
                    print("buffer length  ", len(repBuf), " at iteration " , iteration, " env steps so far ", counter)
                    for name, param in Oracle_model.named_parameters():
                        if param.requires_grad and "action_std" in name:
                            print( name, param.data)  
                ###  collecting trajectories and appending the episodes to the buffer ###
                collect_n_trajectories(n_trajectories, repBuf, Oracle_model, env, n_traj_steps, gamma, lambda_val, iteration)

                ###
                if (len(repBuf) == buffer_size): #fill buffer fully first and then run
                #if (len(repBuf) ): #start on the first small sample
                    for i_epoch in range(n_epochs):   
                            data= NpDataset( ( [ele for ele in repBuf] ))
                            total_loss = 0.
                            
                            random_sampler = RandomSampler(data, num_samples = len(repBuf) )#int(len(repBuf)/n_actors))
                            dataloader = DataLoader(data, batch_size=minibatch_size , collate_fn=my_collate_fn, sampler=random_sampler )
                            
                            for i_batch, sample_batched in enumerate(dataloader):
                                    Oracle_model.train()
                                    pos_t_batched, actions_batched, action_probas_old_batched, advantage_batched, value_batched, return_batched, reward_batched = extract_values_from_batch(sample_batched, minibatch_size)
                                    
                                    return_batched = return_batched.detach()
                                    pos_t_batched = pos_t_batched.detach()
                                    actions_batched = actions_batched.detach()
                                    advantage_batched.detach()
                                    if(do_adv_norm):
                                        advantage_batched = (advantage_batched - advantage_batched.mean()) / (advantage_batched.std()+ 1e-8)
                                        return_batched = (return_batched - return_batched.mean()) / (return_batched.std() + 1e-8)

                                    #evaluate state action:
                                    action_probas_prop, value_prop, entropy_prop = Oracle_model.evaluate(pos_t_batched, actions_batched)

                                    ap_ratio = torch.exp( action_probas_prop- action_probas_old_batched.detach() )

                                    if(do_a2c):
                                        action_loss = - ( ap_ratio * advantage_batched).mean() 
                                    else:   # do PPO clipping
                                        clipped_ratio = torch.clamp(ap_ratio,  (1.-epsilon), (1.+ epsilon) )
                                        act1 = ap_ratio * advantage_batched 
                                        act2 = clipped_ratio * advantage_batched
                                        action_loss = -torch.min( act1, act2).mean() 
                                    
                                    value_loss = (F.mse_loss(value_prop.squeeze(), return_batched.squeeze(), reduction='mean' ))

                                    entropy_loss = - entropy_prop.mean()
                                    
                                    #just to keep it from exploding and just going random/max action rather than trying to predict the correct mean
                                    log_std_penalty_loss = 0.1 *(Oracle_model.action_std).mean()

                                    total_loss =  vf_coef * value_loss + action_loss  + ent_coef *  entropy_loss   + log_std_penalty_loss
    
                                    optimizer.zero_grad()
                                    total_loss.mean().backward() #retain_graph=True)
                                    torch.nn.utils.clip_grad_norm_(Oracle_model.parameters(), max_norm= max_grad_norm )

                                    optimizer.step()

                                    #env step counter (epoch steps are only counted once)
                                    if( i_epoch % (n_epochs-1)==0  and i_epoch>0): 
                                        counter += minibatch_size
    
                                        #tb_x = i_epoch * len(dataloader) + i_batch + 1
                                        tb_x  += 1 #= (i_epoch+1) * (i_batch + 1)
                                        writer.add_scalar('Loss/train', total_loss.item(), counter)
                                        writer.add_scalar('Loss/Policy_grad', action_loss.detach().numpy() , counter)
                                        writer.add_scalar('Loss/Value', (value_loss.detach().numpy()) , counter)
                                        writer.add_scalar('Loss/Entropy', entropy_loss.detach().numpy(), counter)
                                        writer.add_scalar('Reward/train', np.array(reward_batched).mean(), counter)
                                        writer.add_scalar('LearningRate', scheduler.optimizer.param_groups[0]['lr'] , counter)

                                    if( i_batch % (20 * minibatch_size) == 0 ):

                                            collect_n_trajectories(n_trajectories, repBuf_eval, Oracle_model, env, n_traj_steps, gamma, lambda_val, 0)
                                            pos_t_eval,actions_eval, action_probas_old_eval, advantage_eval, _, return_eval, reward_eval = extract_values_from_batch(repBuf_eval, len(repBuf_eval))
                                            
                                            writer.add_scalar('Reward/eval', np.array(reward_eval).mean(), counter)
                                            
                                            #if( (i_batch % (1000 * minibatch_size)) == 0) and i_batch >0:
                                            if i_batch >0:
                                                print('total loss = {:.4f}    policy_grad_loss = {:.10f}   value_loss = {:.4f}   entropy_loss = {:.4f}       batch reward = {:.4f}     eval reward = {:.4f}     last lr = {}'.format(total_loss.item(), action_loss, value_loss, entropy_loss, np.array(reward_batched).mean(), np.array(reward_eval).mean(), scheduler.get_last_lr()))
                                            
 
                                    #scheduler.step(total_loss)
        
        for name, param in Oracle_model.named_parameters():
            if param.requires_grad and "action_std" in name:
                 print( name, param.data)  
            if param.grad is not None:
                print(f"Gradient for {name}:")
                print(param.grad)
                print(f"Gradient magnitude for {name}: {param.grad.norm()}")
            else:
                print(f"No gradient computed for {name}")

        
        
# Assuming you have a PyTorch model named 'model'
#print_model_parameters(Oracle_model)
# Train your model
do_episodes(env,n_episodes=n_episodes)

# After training



#rollout = repBuf(-10, -1)
#obs = repBuf[-20][4]
#print(obs)
##white line between figures
#white_layer = np.ones((64,2), dtype=np.uint8) * 255
#
#for rolli in range(-20, -1): #rollout[1:]:
#        obs = np.concatenate((obs, white_layer), axis=1)
#        obs = np.concatenate((obs, repBuf[rolli][4]), axis = 1)
#cv2.imwrite("RL_train_test_oracle.png", 255* np.expand_dims(obs, -1))
#
#obs2 = repBuf[-50][4]
#for rolli in range(-50, -30): #rollout[1:]:
#        obs2 = np.concatenate((obs2, white_layer), axis=1)
#        obs2 = np.concatenate((obs2, repBuf[rolli][4]), axis = 1)
#cv2.imwrite("RL_train_test_oracle2.png", 255* np.expand_dims(obs2, -1))

#   obs = env.reset()
#    cv2.imwrite("test_rl.png", 255 * np.expand_dims(obs, -1))
#    obs, reward, done, info = env.step([0, 0])
#    cv2.imwrite("test_rl_1.png", 255 * np.expand_dims(obs, -1))

# actions taken
x_values = []
y_values = []

#print(repBuf[0][1][0])
for item in range(len(repBuf)):
    x, y = repBuf[item][1][0].tolist()
    x_values.append(x)
    y_values.append(y)
x_values = np.array(x_values)
y_values = np.array(y_values)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(x_values, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of X actions values')
plt.xlabel('X')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(y_values, bins=100, color='green', alpha=0.7)
plt.title('Histogram of Y actions values')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.tight_layout()

plt.savefig('histograms_actions.png')







x_values = []
y_values = []

#print(repBuf[0][1][0])
for item in range(len(repBuf)):
    x, y = repBuf[item][0][0][0:2].tolist()
    x_values.append(x)
    y_values.append(y)
x_values = np.array(x_values)
y_values = np.array(y_values)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(x_values, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of X pos agent')
plt.xlabel('X')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(y_values, bins=100, color='green', alpha=0.7)
plt.title('Histogram of Y pos agent')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.tight_layout()

plt.savefig('histograms.png')

#plt.show()

#import pdb; pdb.set_trace()









#def calc_discd_vals(episode, gamma, lambda_val ):
#        adv = torch.tensor([ 0. ])
#        delta0 = torch.tensor([episode[-1].reward - episode[-1].value ])
#        #adv = torch.tensor([episode[-1].reward - episode[-1].value ])
#        ret = torch.tensor([ episode[-1].value ]) #TODO
#        for i, e_t in enumerate(reversed(episode)):
#                episode[-(i+1)].delta = delta0
#                #print(i , " ", e_t.reward, episode[-(i+1)].value, e_t.value,  e_t.delta, e_t.reward + gamma * episode[-(i+1)].value - e_t.value )
#                episode[-(i+1)].ret = ret #fill in reverse but enum starts at 0, so we append it first at the last item and go forward from there
#                
#                episode[-(i+1)].advantage = delta0[0] + gamma * lambda_val * adv  #fill in reverse but enum starts at 0, so we append it first at the last item and go forward from there
#                adv = episode[-(i+1)].advantage
#                delta0 = e_t.reward + gamma * episode[-(i+1)].value - e_t.value
#                #print(adv)
#                ret = e_t.reward  + ret*gamma 
#        #return episode



#
#        for iteration in range(n_episodes):
#                print("buffer length  ", len(repBuf), " at iteration " , iteration, " env steps so far ", counter)
#                #if(iteration > 0):
#                #      n_trajectories = 50
#                for _ in range(n_trajectories):
#                        obs = env.reset()
#                        state = env._state
#                        pos_t = torch_pos(state)
#                        episode = []
#                        with torch.no_grad():
#                                Oracle_model.eval()
#                                for st in itertools.count():
#                                        
#                                        #action_proba_dist, value = Oracle_model(pos_t)
#                                        #action = action_proba_dist.sample()
#                                        #action_proba = action_proba_dist.log_prob(action)
#                                        action, action_proba, value = Oracle_model.act(pos_t) #, deterministic = True) #)
#                                        #action, action_proba, value = Oracle_model(pos_t)
#                                        #action = map_int_to_action(int(torch.argmax(action_proba)))
#                                        #action = action_totake
#                                        #print(action)
#
#                                        next_state_obs, reward, done, _  = env.step(action)
#                                        
#                                        #TODO check if this makes sense and works for another env where we actually have a termination (sprites could run for arbitrarily long seqs.)
#                                        if(done):
#                                              env.reset()
#                                        #obtain value of next state according to current policy
#                                        next_pos_t =  torch_pos(env._state)
#                                        #__, _, next_state_value = Oracle_model(next_pos_t)
#                                        #TODO technically not necessary, can construct it afterwards:
#                                        #TODO also check since we're sampling this will not be the same as the actual value at the next step take in the next loop step
#                                        #__, next_state_value = Oracle_model(next_pos_t) #, deterministic =True)
#                                        #_, __, next_state_value = Oracle_model.act(next_pos_t ) #, deterministic = True)
#
#                                        #print(state, env._state)
#                                        #delta = reward + gamma * next_state_value - value
#
#                                        episode.append(episodeStep(pos_t, action, action_proba, reward, next_state_obs, done, value ) )
#                                        #TODO also keep action probas
#                                        if st>=n_traj_steps or done:
#                                                env.reset()
#                                                break 
#                                        #done with one episode
#                                        pos_t = next_pos_t
#
#
#                        #compute advantage from delta
#                        #adv = torch.tensor([episode[-1].reward  ])
#                        calc_discd_vals(episode , gamma, lambda_val)
#
#                        #for i, e_t in enumerate(episode):
#                                #print(episode[i].ret , episode[i].value)
#                        #        episode[i].adv = torch.tensor(episode[i].ret - episode[i].value )
#                                #print(episode[i].adv )
#
#                        #append each timestep at the buffer
#                        for e_t in episode:
#                                repBuf.append([v for v in asdict(e_t).values()])
#                                #print( e_t.ret)
#                                #print(adv)
#                        #print(ret)











 #               #if (len(repBuf) == buffer_size): #fill buffer fully first and then run
 #               if (len(repBuf) ): #fill buffer fully first and then run
 #                       for i_epoch in range(n_epochs):   
 #                               data= NpDataset( ( [ele for ele in repBuf] ))
 #                               total_loss = 0.
 #                               #dataloader = DataLoader(data, batch_size=batch_size , collate_fn=my_collate_fn, shuffle=True ) # create your dataloader
 #                               dataloader = DataLoader(data, batch_size=minibatch_size , collate_fn=my_collate_fn, shuffle=True ) # create your dataloader
 #                               
 #                               for i_batch, sample_batched in enumerate(dataloader):
 #                                       Oracle_model.train()
 #                                       #pos_t_batched,actions_batched, action_probas_old_batched, advantage_batched, value_batched, return_batched, reward_batched = extract_values_from_batch(sample_batched, batch_size)
 #                                       pos_t_batched,actions_batched, action_probas_old_batched, advantage_batched, value_batched, return_batched, reward_batched = extract_values_from_batch(sample_batched, minibatch_size)
 #                                       
 #                                       #normalizing the advantage apparently makes the convergences more stable. at this point I'll try almost anything
 #                                       #advantage_batched = (advantage_batched - advantage_batched.mean()) / (advantage_batched.std() + 1e-10)
#
 #                                       #return_batched = (return_batched - return_batched.mean()) / (return_batched.std() + 1e-7)
#
 #                                       #pred actions and values
 #                                       #action_probas_prop_distr, value_prop = Oracle_model(pos_t_batched)
 #                                       #action = action_probas_prop_distr.rsample()
 #                                       #action_probas_prop = action_probas_prop_distr.log_prob(action)
 #       
 #                                       #entropy += action_probas_prop.entropy().mean()
 #                                       action_probas_prop, value_prop, entropy_prop = Oracle_model.evaluate(pos_t_batched, actions_batched)
 #                                       #action, action_probas_prop, value_prop = Oracle_model(pos_t_batched)
 #                                       #calculate the loss
 #                                       
 #                                       ap_ratio = torch.exp( action_probas_prop- action_probas_old_batched.detach() )
#
 #                                       #TODO check why this would work better than the T return one
 #                                       #advantage_batched = return_batched.detach() - value_batched.detach()
 #                                       clipped_ratio = torch.clamp(ap_ratio,  (1.-epsilon), (1.+ epsilon) )
 #                                       #print(ap_ratio, clipped_ratio )
 #                                       act1 = ap_ratio * advantage_batched.detach() #  (return_batched.detach()  - value_batched.detach()   )  #advantage_batched
 #                                       act2 = clipped_ratio * advantage_batched.detach() #  (return_batched.detach()  - value_batched.detach()  ) #advantage_batched
 #                                       
 #                                       #print(act1[0], act2[0], act1[0]-act2[0])
 #                                       #print(act1, act2)
 #                                       action_loss = (torch.min( act1, act2  )).mean() 
 #                                       #action_loss = ( value_prop).mean() #TODO undo this, using simpler reward for testing
 #                                       
 #                                       #print(value_batched[0], value_prop[0],  advantage_batched[0])
 #                                       value_loss = ( (value_prop - return_batched.detach())/(return_batched.detach() + 0.00004) ).pow(2).mean() 
 #                                       #entropy aka exploration encouragment
 #                                       #entropy_loss = ((action_probas_prop)).mean()
 #                                       entropy_loss = entropy_prop.mean()
 #       
 #                                       #total_loss =  0.5* value_loss - action_loss  - 0.001*   entropy_loss
 #                                       total_loss =  ( 1.* value_loss - action_loss  - 0.0001*  entropy_loss)
 #                                       #total_loss =  (0.8* value_loss - action_loss  +   0.1*entropy_loss)
 #                                       #total_loss = total_loss.mean().sum()      
 #       
 #                                       optimizer.zero_grad()
#
 #                                       #total_loss.sum().backward()
 #                                       total_loss.backward() #retain_graph=True)
 #                                       optimizer.step()
#
 #                                       #for i, (name, param) in enumerate(Oracle_model.named_parameters()):
 #                                       #        if name.startswith( 'actor') :
 #                                       #              print(name, param,param.requires_grad, param.grad.data)
 #                                       #print_gradients(Oracle_model)
 #                                       #counter += batch_size
 #                                       counter += minibatch_size
 #       
 #                                       reward_b = 0.
 #                                       for ir in reward_batched:
 #                                             reward_b+= ir
 #       
 #                                       #reward_b /= batch_size
 #                                       reward_b /= minibatch_size
 #                                       tb_x = i_epoch * len(dataloader) + i_batch + 1
 #                                       writer.add_scalar('Loss/train', total_loss.item(), tb_x)
 #                                       writer.add_scalar('Reward/train', reward_b, tb_x)
 #       
 #                                       if(i_batch % 5000 == 0):
 #                                               #print(ap_ratio , action_probas_prop  ,action_probas_old_batched )
 #                                               #print(action[0]) #, action_proba)
 #                                               print('total loss = {:.4f}    action_loss = {:.4f}   value_loss = {:.4f}   entropy_loss = {:.4f}       batch reward = {:.4f}     last lr = {}'.format(total_loss.item(), action_loss, value_loss, entropy_loss, reward_b, scheduler.get_last_lr()))
 #                               
 #                                       #if(counter > 0 and counter%(batch_size*n_epochs*100) ==0):
 #                                               #print("lowering step size ")
 #                                       #scheduler.step()
 #                                       scheduler.step(total_loss)
 #       print(counter)