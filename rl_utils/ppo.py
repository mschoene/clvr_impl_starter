import numpy as np
import torch
import torch.nn.functional as F
from replayBuffer import *

from torch.utils.data import DataLoader, RandomSampler
from rl_utils.torch_utils import  get_averaged_tensor # np_to_torch
from rl_utils.traj_utils import *

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
#from replayBuffer import *

from torch.utils.data import DataLoader, RandomSampler
#from torchsummary import summary
from datetime import datetime

#from dataclasses import dataclass, asdict

from rl_utils.torch_utils import  get_averaged_tensor # np_to_torch
from rl_utils.traj_utils import *
from rl_utils.buffer import *



class MimiPPO:
    def __init__(self, 
                model, 
                model_name,
                env,
                env_name,
                gamma=0.99, 
                lambda_val = 0.95,

                ent_coef = 0.005, 
                vf_coef=0.5, 
                std_coef = 2.0, #0.2 for state

                max_grad_norm = 0.5, 
                do_adv_norm=True, 
                do_a2c = False, 
                do_std_penalty = True,

                n_trajectories =  8, #4, #16,# 8,
                n_actors = 4,
                n_traj_steps = 49,
                lr = 0.0003,
                epsilon = 0.2,
                n_episodes = 500,
                n_epochs =  10,
                minibatch_size = 128, #256, #128,
                max_env_steps = 5_000_000
            ): #this is not a sad smiley but a duck with a very wide mouth

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print("--- working on device ", self.device , " ---")

        self.model_name = model_name
        self.max_env_steps = max_env_steps
        self.model = model.to(self.device)
        self.env = env
        self.env_name = env_name
        self.gamma = gamma
        self.lambda_val = lambda_val
        self.do_adv_norm = do_adv_norm
        self.do_a2c = do_a2c
        self.do_std_penalty = do_std_penalty # penalty on choising a large std on the action and thus converging to trivial solution 
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.std_coef = std_coef
        self.max_grad_norm = max_grad_norm
        self.n_trajectories = n_trajectories

        self.n_actors = n_actors # number of parallel workers collecting trajectories
        self.n_traj_steps = n_traj_steps #399 #127 #39 #5 #9 #39#19 #ntraj step is T-1 in. so T steps will be taken (ie ntrajsteps +1)
        self.n_episodes = n_episodes#50 #0#500  #total number of iterations over the  data, this will mean M*nEpisodes = #of steps taken
        self.n_epochs =  n_epochs  #number of optim steps on a given buffer data set

        #self.buffer_size = 4 * n_actors* (self.n_traj_steps+1)*self.n_trajectories # M = N*T thus defining the number of actors as N = M/T
        self.buffer_size = n_actors* (self.n_traj_steps+1)*self.n_trajectories # M = N*T thus defining the number of actors as N = M/T
        #self.buffer_size = 4* 2* 8 * (self.n_traj_steps+1) # M = N*T thus defining the number of actors as N = M/T

        self.batch_size = n_actors* (self.n_traj_steps+1)*self.n_trajectories  #256 #512 #1024  #64 #32 #64 #128 #32 #128

        self.minibatch_size = minibatch_size # size of the batch to average over 
        self.replayBuffer = replayBuffer(self.buffer_size)
        self.lr = lr
        self.epsilon = epsilon

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/RL_training_{}_{}_{}'.format(self.model_name,self.env_name ,self.timestamp))
        self.optimizer = optim.Adam( self.model.parameters(), lr = self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor = 0.2, patience = 7500, min_lr = 2e-7 )

    def train(self):
        # We shall step through the amount of episodes #In each episode we step through the trajectory
        # according to the policy (actor) at hand and add the values to the episode as estimated by the critic
        counter = 0. #this counts the number of environment steps in total
        next_threshold = 1000
        while(counter < self.max_env_steps):
        #for iteration in range(self.n_episodes):
            if counter >= next_threshold:
                next_threshold += 1000
                print("buffer length  ", len(self.replayBuffer), " env steps so far ", counter)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and "action_std" in name:
                        print( name, param.data)  

            #self.model = self.model.to('cpu') 
            ###  collecting trajectories and appending the episodes to the buffer ###
            collect_n_trajectories(self.n_trajectories, self.replayBuffer, self.model, self.env_name, self.n_traj_steps, self.gamma, self.lambda_val, n_workers=self.n_actors)
            ###
            counter += (self.n_trajectories * self.n_actors * (self.n_traj_steps+1))

            if (len(self.replayBuffer) == self.buffer_size): #fill buffer fully first and then run
                for i_epoch in range(self.n_epochs):   
                    data = NpDataset( ( [ele for ele in self.replayBuffer] ))
                    total_loss = 0.

                    #if self.device.type == 'cuda':
                    #   self.model.to(self.device)

                    #random_sampler = RandomSampler(data, num_samples = len(self.replayBuffer) ) 
                    #dataloader = DataLoader(data, batch_size = self.minibatch_size , collate_fn=my_collate_fn, sampler=random_sampler, num_workers=4 )

                    dataloader = DataLoader(data, batch_size=self.minibatch_size, collate_fn=my_collate_fn, shuffle=True) #, num_workers=4)

                    #print(enumerate(dataloader))
                    looper = enumerate(dataloader)
                    for _, sample_batched in looper:

                        self.model.train()
                        pos_t_batched, actions_batched, action_probas_old_batched, advantage_batched, return_batched, reward_batched = extract_values_from_batch(sample_batched, self.minibatch_size)
                        
                        #pos_t_batched = pos_t_batched.to(self.device)
                        #actions_batched = actions_batched.to(self.device)
                        #action_probas_old_batched = action_probas_old_batched.to(self.device)
                        #advantage_batched = advantage_batched.to(self.device)
                        #return_batched = return_batched.to(self.device)
                        #reward_batched = reward_batched.to(self.device)

                        if(self.do_adv_norm):
                            advantage_batched = get_averaged_tensor(advantage_batched)
                            return_batched = get_averaged_tensor(return_batched)

                        #evaluate state action:
                        action_probas_prop, value_prop, entropy_prop = self.model.evaluate(pos_t_batched, actions_batched)
                        ap_ratio = torch.exp( action_probas_prop- action_probas_old_batched )

                        if(self.do_a2c): #do unclipped advantage policy loss
                            action_loss = -( ap_ratio * advantage_batched).mean() 
                        else: # do PPO clipping
                            clipped_ratio = torch.clamp(ap_ratio,  (1.-self.epsilon), (1.+ self.epsilon) )
                            act1 = ap_ratio * advantage_batched 
                            act2 = clipped_ratio * advantage_batched
                            action_loss = -torch.min(act1, act2).mean() 

                        value_loss = F.mse_loss(value_prop.squeeze(), return_batched.squeeze())
                        entropy_loss = - entropy_prop.mean()

                        #to keep it from exploding and just going random/max action rather than trying to predict the correct mean
                        log_std_penalty_loss = self.std_coef * (self.model.action_std).mean()

                        # total loss 
                        total_loss = self.vf_coef * value_loss + action_loss + self.ent_coef * entropy_loss + log_std_penalty_loss

                        self.optimizer.zero_grad()
                        total_loss.mean().backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.max_grad_norm )
                        self.optimizer.step()

                    if( i_epoch % (self.n_epochs-1)==0 and i_epoch>0): 
                        self.writer.add_scalar('Loss/train', total_loss.item(), counter)
                        self.writer.add_scalar('Loss/Policy_grad', action_loss.detach().cpu().numpy(), counter)
                        self.writer.add_scalar('Loss/Value', value_loss.detach().cpu().numpy(), counter)
                        self.writer.add_scalar('Loss/Entropy', entropy_loss.detach().cpu().numpy(), counter)
                        self.writer.add_scalar('Reward/train', reward_batched.mean().cpu().numpy(), counter)
                        #self.writer.add_scalar('LearningRate', self.scheduler.optimizer.param_groups[0]['lr'], counter)

        #if (counter > self.max_env_steps):
        print("DONE " , self.max_env_steps , " steps")
        model_path = 'runs/RL_trained_model_{}_{}_{}'.format(self.model_name,self.env_name, self.timestamp)
        torch.save(self.model.state_dict(), model_path)
                     #scheduler.step(total_loss)


