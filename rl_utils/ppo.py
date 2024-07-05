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

import wandb

import matplotlib.pyplot as plt

def plot_grad_flow(named_parameters, file_path='grad_flow.png'):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
    plt.figure(figsize=(12, 8))  # Increase figure size to accommodate long labels
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads)), layers, rotation=90)  # Rotate labels 90 degrees
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.savefig(file_path)
    #plt.close()
    #plt.show()


def make_histos(buffer):
    # actions taken
    x_values = []
    y_values = []

    #print(repBuf[0][1][0])
    for item in range(len(buffer)):
        x, y = buffer[item][1][0].tolist()
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
    #plt.close('all')

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class MimiPPO:
    def __init__(self, 
                model, 
                model_name,
                env,
                env_name,
                #wandb_inst,
                gamma=0.99, 
                lambda_val = 0.95,

                ent_coef = 0.005, 
                vf_coef=0.5, 
                std_coef =  0.0, #01, #2.0, #0.2 for state

                max_grad_norm = 0.5, 
                do_adv_norm = True, #False, #True, 
                do_a2c = False, 
                #do_std_penalty = True,

                n_trajectories = 10, #4, # 16, #16,  
                n_actors = 4, # 8, #4, #40, #20, #8, 
                n_traj_steps = 40, #49,
                lr = 0.0003,
                epsilon = 0.1, #0.2,
                n_episodes = 500,
                n_epochs = 10,
                minibatch_size = 128, #256, #128,
                max_env_steps = 5_000_000,
                final_log_std = -10,
                off_poli_factor = 1,
                do_wandb = False,
                do_vf_clip = True, 
                do_lin_lr_decay = True,
                verbose = True,
            ): #this is not a sad smiley but a very hungry duck

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
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
        #self.do_std_penalty = do_std_penalty # penalty on choising a large std on the action and thus converging to trivial solution 
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.std_coef = std_coef
        self.max_grad_norm = max_grad_norm
        self.n_trajectories = n_trajectories

        self.n_actors = n_actors # number of parallel workers collecting trajectories
        self.n_traj_steps = n_traj_steps #399 #127 #39 #5 #9 #39#19 #ntraj step is T-1 in. so T steps will be taken (ie ntrajsteps +1)
        self.n_episodes = n_episodes#50 #0#500  #total number of iterations over the  data, this will mean M*nEpisodes = #of steps taken
        self.n_epochs =  n_epochs  #number of optim steps on a given buffer data set

        self.off_poli_factor = off_poli_factor # fraction of steps from old collection ie off policy
        self.buffer_size = int( self.off_poli_factor *  n_actors* (self.n_traj_steps)*self.n_trajectories ) # M = N*T thus defining the number of actors as N = M/T

        #self.batch_size = n_actors * (self.n_traj_steps+1)*self.n_trajectories  #256 #512 #1024  #64 #32 #64 #128 #32 #128

        self.minibatch_size = minibatch_size # size of the batch to average over 
        self.replayBuffer = replayBuffer(self.buffer_size)
        self.lr = lr
        self.epsilon = epsilon

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/RL_training_{}_{}_{}'.format(self.model_name,self.env_name ,self.timestamp))
        #self.wandb_inst = wandb_inst
        #self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-4)
        #self.optimizer = optim.RAdam( self.model.parameters(), betas = (0.9, 0.999))

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor = 0.2, patience = 1500, min_lr = 1e-5 )

        # anneal action standard deviation from init value to -10 => exp(-10) = 0.00004539992
        self.final_log_std = final_log_std
        self.init_log_std = self.model.action_std.clone().detach()
        self.annealing_rate = (self.init_log_std[0] - self.final_log_std) / 5000000.
        self.annealing_rate.to("cpu") 
        #self.annealing_rate = (self.final_log_std  / (self.init_log_std +1e-8)) ** (1. / 5000000.)
        self.do_wandb = do_wandb
        self.do_vf_clip = do_vf_clip
        self.do_lin_lr_decay = do_lin_lr_decay
        self.verbose = verbose
        
    def train(self):
        # We shall step through the amount of episodes #In each episode we step through the trajectory
        # according to the policy (actor) at hand and add the values to the episode as estimated by the critic
        counter = 0. #this counts the number of environment steps in total
        next_threshold = 10000
        #reward_batched = 0

        while(counter < self.max_env_steps):
        #for iteration in range(self.n_episodes):
            if counter >= next_threshold:
                next_threshold += 10000
                print("buffer length  ", len(self.replayBuffer), " env steps so far ", counter)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and "action_std" in name:
                        print( name, param.data)  

            self.model = self.model.to('cpu') 
            ###  collecting trajectories and appending the episodes to the buffer ###
            collect_n_trajectories(self.n_trajectories, self.replayBuffer, self.model, self.env_name, self.n_traj_steps, self.gamma, self.lambda_val, n_workers=self.n_actors)
            ###
            #counter += (self.n_trajectories * self.n_actors * (self.n_traj_steps))
            #counter += int ( len(self.replayBuffer))
            if (len(self.replayBuffer) == self.buffer_size): #fill buffer fully first and then run
                counter += int ( (self.buffer_size)/self.off_poli_factor  )

            #if(counter > 100000):
            #    new_log_std = self.init_log_std  - (self.annealing_rate * counter) 
                #print("init std " , self.init_log_std, self.annealing_rate , counter, new_log_std)
                #new_log_std = self.model.action_std.to('cpu')- (self.annealing_rate.to("cpu")  * counter) 
                #new_log_std = self.model.action_std * (self.annealing_rate ** counter)
            #    self.model.action_std.data = torch.tensor( new_log_std )#.to(self.device)

            if (len(self.replayBuffer) == self.buffer_size): #fill buffer fully first and then run
                for i_epoch in range(self.n_epochs):   
                    data = NpDataset( ( [ele for ele in self.replayBuffer] ))
                    total_loss = 0
                    total_loss_sum = 0
                    action_loss_sum = 0
                    value_loss_sum = 0
                    entropy_loss_sum = 0
                    reward_sum = 0
                    num_batches = 0

                    # for checks 
                    if self.verbose:
                        rewards_for_log = []

                    if self.device.type == 'cuda':
                       self.model.to(self.device)
                       data.to(self.device)

                    dataloader = DataLoader(data, batch_size=self.minibatch_size, collate_fn=my_collate_fn, shuffle=True) #, num_workers=4)
                    
                    self.model.train()

                    for _, sample_batched in enumerate(dataloader):

                        pos_t_batched, actions_batched, action_probas_old_batched, \
                            advantage_batched, return_batched, reward_batched, value_batched \
                                = extract_values_from_batch(sample_batched, self.minibatch_size)
                        
                        if(self.do_adv_norm):
                            advantage_batched = get_averaged_tensor(advantage_batched)
                            #return_batched = get_averaged_tensor(return_batched)

                        #evaluate state action:
                        action_probas_prop, value_prop, entropy_prop = self.model.evaluate(pos_t_batched, actions_batched)

                        #action_probas_prop_notSummed = action_probas_prop
                        #action_probas_old_batched_notSummed = action_probas_old_batched
                        #action_probas_prop = action_probas_prop.sum(dim=-1)
                        #action_probas_old_batched = action_probas_old_batched.sum(dim=-1)

                        #print(action_probas_prop_notSummed.shape, action_probas_old_batched_notSummed.shape, action_probas_prop.shape, action_probas_old_batched.shape)
                        #print(action_probas_prop_notSummed[0], action_probas_old_batched_notSummed[0], action_probas_prop[0], action_probas_old_batched[0])
                        #print( "eval ", action_probas_prop )
                        #print( "batch",  action_probas_old_batched )

                        ap_ratio = torch.exp( action_probas_prop- action_probas_old_batched )

                        if(self.do_a2c): #do unclipped advantage policy loss
                            action_loss = -( ap_ratio * advantage_batched).mean() 
                        else: # do PPO clipping
                            clipped_ratio = torch.clamp(ap_ratio,  (1.-self.epsilon), (1.+ self.epsilon) )
                            act1 = ap_ratio * advantage_batched  #/( ap_ratio.mean() + 10e-8)
                            act2 = clipped_ratio * advantage_batched # /( clipped_ratio.mean() + 10e-8)
                            #action_loss = -torch.min(act1 , act2 )
                            action_loss = -torch.min(act1 , act2 ).mean() 
                            #action_loss = -torch.min(act1 /(act1.mean()), act2 /(act2.mean()) ).mean() 
                            #action_loss = (action_loss/ (ap_ratio.mean()) ).mean()

                        if self.do_vf_clip:
                            #print(value_batched.shape, return_batched.shape, value_prop.shape)
                            value_loss =  (value_batched - return_batched).pow(2)
                            value_clip = value_batched + torch.clamp( value_prop -value_batched, -self.epsilon, self.epsilon)
                            value_loss_clipped = (value_clip - return_batched ).pow(2)
                            value_loss = torch.max(value_loss, value_loss_clipped).mean()
                        else:
                            value_loss = F.mse_loss(value_prop, return_batched)

                        entropy_loss = - entropy_prop.mean()

                        #to keep it from exploding and just going random/max action rather than trying to predict the correct mean
                        log_std_penalty_loss = self.std_coef * (torch.exp(self.model.action_std) ).mean() * ( counter / self.max_env_steps )

                        # total loss 
                        total_loss = self.vf_coef * value_loss + action_loss + log_std_penalty_loss + self.ent_coef * entropy_loss 

                        self.optimizer.zero_grad()
                        total_loss.mean().backward()

                        #plot_grad_flow(self.model.named_parameters())
                        
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.max_grad_norm )
                        self.optimizer.step()

                        total_loss_sum += total_loss.item()
                        action_loss_sum += action_loss.detach().cpu().numpy()
                        value_loss_sum += value_loss.detach().cpu().numpy()
                        entropy_loss_sum += entropy_loss.detach().cpu().numpy()
                        reward_sum += reward_batched.mean().cpu().numpy()
                        num_batches += 1
                        if self.verbose:
                            rewards_for_log.extend(reward_batched.cpu().numpy())
                    #done with one epoch

                    # log last avg epoch results

                    if( i_epoch % (self.n_epochs-1)==0 and i_epoch>0): 
                        avg_total_loss = total_loss_sum / num_batches
                        avg_action_loss = action_loss_sum / num_batches
                        avg_value_loss = value_loss_sum / num_batches
                        avg_entropy_loss = entropy_loss_sum / num_batches
                        avg_reward = reward_sum / num_batches

                        self.writer.add_scalar('Loss/train', avg_total_loss, counter)
                        self.writer.add_scalar('Loss/Policy_grad', avg_action_loss, counter)
                        self.writer.add_scalar('Loss/Value', avg_value_loss, counter)
                        self.writer.add_scalar('Loss/Entropy', avg_entropy_loss, counter)
                        self.writer.add_scalar('Reward/train', avg_reward, counter)
                        self.writer.add_scalar('Param/action_std', self.model.action_std.data[0].cpu(), counter)
                        self.writer.add_scalar('Param/learningRate', self.scheduler.optimizer.param_groups[0]['lr'], counter)

                        if self.verbose:
                            self.writer.add_histogram('Rewards/Distribution', np.array(rewards_for_log), counter)
                            self.writer.add_histogram('Policy/ap_ratio', ap_ratio, counter)
                            self.writer.add_histogram('Policy/ap_ratio_clipped ', clipped_ratio, counter)
                            self.writer.add_histogram('Policy/avd_r_clip', act2, counter)
                            self.writer.add_histogram('Policy/avd_r', act1, counter)
                            self.writer.add_histogram('Policy/action_proba_eval', action_probas_prop, counter)
                            self.writer.add_histogram('Policy/action_proba_batch', action_probas_old_batched, counter)
                            self.writer.add_histogram('Policy/actions_batched', actions_batched, counter)
                            self.writer.add_histogram('Policy/value_prop', value_prop, counter)
                            self.writer.add_histogram('Policy/value_batched', value_batched, counter)
                            self.writer.add_histogram('Policy/return_batched', return_batched, counter)

                            ave_grads = []
                            layers = []
                            for n, p in self.model.named_parameters():
                                if p.requires_grad and ("bias" not in n):
                                    if p.grad is not None:
                                        layers.append(n)
                                        ave_grads.append(p.grad.abs().mean().cpu().item())
                                        # Log gradients to TensorBoard as histograms
                                        self.writer.add_histogram(f'Gradients/{n}', p.grad, counter)

                        print(avg_reward)

                        if self.do_wandb:
                            wandb.log({
                                'epoch': i_epoch + 1,
                                'env_steps': counter,
                                'loss': avg_total_loss,
                                'loss_pg': avg_action_loss,
                                'loss_vf': avg_value_loss,
                                'average_reward': avg_reward,
                                'action_std': self.model.action_std.data[0].cpu(),
                                'Step': counter , 
                                'Rewards/Distribution': wandb.Histogram(np.array(rewards_for_log)),
                                'Policy/ap_ratio': wandb.Histogram(ap_ratio.detach().cpu().numpy()),
                                'Policy/ap_ratio_clipped': wandb.Histogram(clipped_ratio.detach().cpu().numpy()),
                                'Policy/avd_r_clip': wandb.Histogram(act2.detach().cpu().numpy()),
                                'Policy/avd_r': wandb.Histogram(act1.detach().cpu().numpy()),
                                'Policy/action_proba_eval': wandb.Histogram(action_probas_prop.detach().cpu().numpy()),
                                'Policy/action_proba_batch': wandb.Histogram(action_probas_old_batched.detach().cpu().numpy()),
                                'Policy/actions_batched': wandb.Histogram(actions_batched.detach().cpu().numpy()),
                                'Policy/value_prop': wandb.Histogram(value_prop.detach().cpu().numpy()),
                                'Policy/value_batched': wandb.Histogram(value_batched.detach().cpu().numpy()),
                                'Policy/return_batched': wandb.Histogram(return_batched.detach().cpu().numpy()),

                            })
                    #self.scheduler.step(total_loss)
                if self.do_lin_lr_decay:
                    update_linear_schedule(self.optimizer, counter, self.max_env_steps, self.lr)

            #make_histos(self.replayBuffer)
            #print(self.model.action_std.data)


        #if (counter > self.max_env_steps):
        print("DONE " , self.max_env_steps , " steps")
        model_path = 'runs/RL_trained_model_{}_{}_{}'.format(self.model_name,self.env_name, self.timestamp)
        torch.save(self.model.state_dict(), model_path)


