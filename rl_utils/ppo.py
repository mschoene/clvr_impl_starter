import numpy as np
import torch
import torch.nn.functional as F
#from replayBuffer import *
from rl_utils.buffer import ReplayBuffer

from torch.utils.data import DataLoader, RandomSampler
from rl_utils.torch_utils import  get_averaged_tensor # np_to_torch
from rl_utils.traj_utils import *

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler
from datetime import datetime

from rl_utils.torch_utils import  get_averaged_tensor # np_to_torch
from rl_utils.traj_utils import *
from rl_utils.buffer import *

import wandb
import matplotlib.pyplot as plt
import os

torch.autograd.set_detect_anomaly(True)

# Define checkpoints, save every 1M env steps
checkpoints = [1_000_000, 2_000_000, 3_000_000, 4_000_000]

# save checkpoint
def save_checkpoint(model, counter, optimizer, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'counter': counter
    }
    torch.save(checkpoint, filename)

#without clipping it is too sensitive and results in unsable behavior
def adjust_kl_coeff(kl_divergence, kl_coeff, target_kl, kl_increment=2.0, kl_decrement=0.5, kl_min=1e-5, kl_max=10):
    if kl_divergence > target_kl * 1.5:
        kl_coeff = min(kl_coeff * kl_increment, kl_max)
    elif kl_divergence < target_kl / 1.5:
        kl_coeff = max(kl_coeff * kl_decrement, kl_min)
    return kl_coeff
#
#def adjust_kl_coeff(kl_div, kl_coeff, kl_target, beta=2.0):
#    if kl_div > kl_target:
#        kl_coeff *= beta
#    else:
#        kl_coeff /= beta
#    return kl_coeff

def anneal_clip_range(initial_clip_range, final_clip_range, current_step, total_steps):
    """
    Linearly anneal the clipping range from initial_clip_range to final_clip_range.

    Args:
    - initial_clip_range (float): The initial clipping range at the start of training.
    - final_clip_range (float): The final clipping range at the end of training.
    - current_step (int): The current step in the training process.
    - total_steps (int): The total number of steps in the training process.

    Returns:
    - float: The annealed clipping range for the current step.
    """
    clip_range = initial_clip_range - (initial_clip_range - final_clip_range) * (current_step / total_steps)
    return max(clip_range, final_clip_range)  # Ensure the clip range doesn't go below the final value


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lrs):
    """
    Linearly decreases the learning rate for each parameter group based on the epoch.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to update.
        epoch (int): The current epoch or alternatively the number of env steps
        total_num_epochs (int): The total number of epochs or alternatively the number of total env steps
        initial_lrs (list): List of initial learning rates for each parameter group.
    """
    for param_group, initial_lr in zip(optimizer.param_groups, initial_lrs):
        lr = initial_lr * (1 - epoch / float(total_num_epochs))
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
                kl_coef = 0.0,
                kl_target = 0.01,

                max_grad_norm = 0.5,
                do_adv_norm = True, #False, #True,
                do_a2c = False,
                #do_std_penalty = True,

                n_trajectories = 10, #4, # 16, #16,
                n_actors = 4, # 8, #4, #40, #20, #8,
                n_traj_steps = 40, #49,
                lr = 0.0003,
                lr_enc = 0.0003,
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
                verbose = False,
                do_eval = True, #do deterministic eval step
                final_eps = 0.0001,
                prev_train_path = None

            ): #this is not a sad smiley but a very hungry duck

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.std_coef = std_coef
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.final_eps = final_eps

        self.max_grad_norm = max_grad_norm
        self.n_trajectories = n_trajectories

        self.n_actors = n_actors # number of parallel workers collecting trajectories
        self.n_traj_steps = n_traj_steps  #ntraj step is T-1 in. so T steps will be taken (ie ntrajsteps +1)
        self.n_episodes = n_episodes # total number of iterations over the  data, this will mean M*nEpisodes = #of steps taken
        self.n_epochs =  n_epochs  #number of optim steps on a given buffer data set

        self.off_poli_factor = off_poli_factor # fraction of steps from old collection ie off policy
        self.a_t_buff_size = self.n_traj_steps * self.n_trajectories
        self.buffer_size = int( self.off_poli_factor *  n_actors* self.a_t_buff_size ) # M = N*T thus defining the number of actors as N = M/T

        # eval data is added to buffer to make use of eval data for training of next buffer so we make max use of data collected.
        self.do_eval = do_eval

        self.minibatch_size = minibatch_size # size of the batch to average over
        self.replayBuffer = ReplayBuffer(self.buffer_size)

        self.buffer_size_eval = int((self.n_traj_steps)*self.n_trajectories )
        self.replayBuffer_eval = ReplayBuffer(self.buffer_size_eval)

        self.lr = lr
        self.lr_enc = lr_enc
        self.epsilon = epsilon

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/RL_training_{}_{}_{}'.format(self.model_name,self.env_name ,self.timestamp))
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-4)


        encoder_params = set(self.model.encoder.parameters())
        all_params = set(self.model.parameters())
        rest_params = all_params - encoder_params
        self.optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': self.lr, 'weight_decay': 1e-5},
            {'params': list(rest_params), 'lr': self.lr_enc, 'weight_decay': 1e-5}
            ], eps=1e-4)

           # {'params': list(rest_params), 'lr': self.lr_enc, 'weight_decay': 1e-5}
        #self.optimizer = optim.RAdam( self.model.parameters(), betas = (0.9, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor = 0.2, patience = 1500, min_lr = 1e-5 )

        self.final_log_std = final_log_std
        self.init_log_std = self.model.action_std.clone().detach()

        self.do_wandb = do_wandb
        self.do_vf_clip = do_vf_clip
        self.do_lin_lr_decay = do_lin_lr_decay
        self.verbose = verbose
        self.counter = 0

        if prev_train_path:
               self.counter = self.load_checkpoint(prev_train_path)

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            print(f"=> loading checkpoint '{filename}'")
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            counter = checkpoint['counter']
            print(f"=> loaded checkpoint '{filename}' (counter {counter})")
            return counter
        else:
            print(f"=> no checkpoint found at '{filename}'")
            return 0



    def train(self):
        # We shall step through the amount of episodes #In each episode we step through the trajectory
        # according to the policy (actor) at hand and add the values to the episode as estimated by the critic
        counter = self.counter #this counts the number of environment steps in total
        next_threshold = 50000 #you get a log printout every next_threshold step

        while(counter < self.max_env_steps):
        #for iteration in range(self.n_episodes):
            if counter >= next_threshold:
                next_threshold += 50000
                print("buffer length  ", len(self.replayBuffer), " env steps so far ", counter)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and "action_std" in name:
                        print( name, param.data)

            self.model = self.model.to('cpu')
            ###  collecting trajectories and appending the episodes to the buffer ###
            collect_n_trajectories(self.n_trajectories, self.replayBuffer, self.model, self.env_name, self.n_traj_steps, self.gamma, self.lambda_val, n_workers=self.n_actors)
            ###
            if (len(self.replayBuffer) == self.buffer_size):
                counter += int ( (self.buffer_size)/self.off_poli_factor  )

            data = NpDataset([ele for ele in self.replayBuffer])

            if self.device.type == 'cuda':
                self.model.to(self.device)
                data.to(self.device)
            # if we got evaluation data make use of it by appending it to the training data
            if self.do_eval:
                data_eval = NpDataset([ele for ele in self.replayBuffer_eval])
                if self.device.type == 'cuda':
                    data_eval.to(self.device)
                data += data_eval
                counter += (self.buffer_size_eval)

            if (len(self.replayBuffer) == self.buffer_size): #fill buffer fully first and then run
                for i_epoch in range(self.n_epochs):
                    total_loss = 0
                    total_loss_sum = 0
                    action_loss_sum = 0
                    value_loss_sum = 0
                    kl_loss_sum = 0
                    entropy_loss_sum = 0
                    reward_sum = 0
                    num_batches = 0

                    # for checks
                    if self.verbose:
                        rewards_for_log = []

                    if self.device.type == 'cuda':
                        self.model.to(self.device)
                    #   data.to(self.device)

                    dataloader = DataLoader(data, batch_size=self.minibatch_size, collate_fn=my_collate_fn, shuffle=True)

                    self.model.train()

                    current_epsilon = anneal_clip_range(self.epsilon ,  self.final_eps, counter, self.max_env_steps)

                    for _, sample_batched in enumerate(dataloader):

                        pos_t_batched, actions_batched, action_probas_old_batched, \
                            advantage_batched, return_batched, reward_batched, value_batched \
                                = extract_values_from_batch(sample_batched, self.minibatch_size)

                        if(self.do_adv_norm):
                            advantage_batched = get_averaged_tensor(advantage_batched)

                        #evaluate state action:
                        action_probas_prop, value_prop, entropy_prop = self.model.evaluate(pos_t_batched, actions_batched)

                        ap_ratio = torch.exp( action_probas_prop- action_probas_old_batched )

                        if(self.do_a2c): #do unclipped advantage policy loss
                            action_loss = -( ap_ratio * advantage_batched).mean()
                        else: # do PPO clipping
                            clipped_ratio = torch.clamp(ap_ratio,  (1.- current_epsilon ), (1.+  current_epsilon) )
                            act1 = ap_ratio * advantage_batched
                            act2 = clipped_ratio * advantage_batched
                            action_loss = -torch.min(act1 , act2 ).mean()

                        if self.do_vf_clip:
                            value_loss =  (value_batched - return_batched).pow(2)
                            value_clip = value_batched + torch.clamp( value_prop -value_batched, -current_epsilon, current_epsilon)
                            value_loss_clipped = (value_clip - return_batched ).pow(2)
                            value_loss = torch.max(value_loss, value_loss_clipped).mean()
                        else:
                            value_loss = F.mse_loss(value_prop, return_batched)

                        entropy_loss = - entropy_prop.mean()

                        kl_loss = (action_probas_old_batched - action_probas_prop).mean()

                        self.kl_coef = adjust_kl_coeff(kl_loss, self.kl_coef, self.kl_target)

                        #to encourange shrinking of the std dev if it's left as a learnable param
                        #log_std_penalty_loss = self.std_coef * (torch.exp(self.model.action_std) ).mean() * ( counter / self.max_env_steps )

                        # total loss
                        total_loss = self.vf_coef * value_loss + action_loss + self.ent_coef * entropy_loss + self.kl_coef * kl_loss #+ log_std_penalty_loss


                        # Check for NaNs in the forward pass
                        if torch.isnan(action_probas_prop).any() or torch.isnan(value_prop).any() or torch.isnan(entropy_prop).any():
                            print("NaNs detected in the forward pass")
                            continue

                        self.optimizer.zero_grad()
                        total_loss.mean().backward()

                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                if param.grad is None:
                                    pass #print(f"Parameter {name} has no gradient.")
                                elif torch.isnan(param.grad).any():
                                    print(f"Parameter {name} has NaN gradients.")
                                    param.grad.zero_()  # This will reset the gradients for this parameter

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.max_grad_norm )
                        self.optimizer.step()

                        total_loss_sum += total_loss.item()
                        action_loss_sum += action_loss.detach().cpu().numpy()
                        value_loss_sum += value_loss.detach().cpu().numpy()
                        entropy_loss_sum += entropy_loss.detach().cpu().numpy()
                        kl_loss_sum += kl_loss.detach().cpu().numpy()
                        reward_sum += reward_batched.mean().cpu().numpy()
                        num_batches += 1
                        if self.verbose:
                            rewards_for_log.extend(reward_batched.cpu().numpy())
                    #done with one epoch


                    eval_reward = 0
                    if self.do_eval:

                        #if self.device.type == 'cuda':
                        #    self.model.to(self.device)
                        # data.to(self.device)
                        self.model = self.model.to('cpu')

                        collect_n_trajectories(self.n_trajectories, self.replayBuffer_eval, self.model, self.env_name, self.n_traj_steps, self.gamma, self.lambda_val, n_workers=1, deterministic=True)
                        data_eval = NpDataset([ele for ele in self.replayBuffer_eval])

                        dataloader_eval = DataLoader(data_eval, batch_size=len(self.replayBuffer_eval), collate_fn=my_collate_fn)

                        #self.model.eval()
                        for _, sample_batched in enumerate(dataloader_eval):

                            pos_t_batched_eval, actions_batched_eval, action_probas_old_batched_eval, \
                                advantage_batched_eval, return_batched_eval, reward_batched_eval, value_batched_eval \
                                    = extract_values_from_batch(sample_batched, len(self.replayBuffer_eval) )
                            eval_reward=reward_batched_eval.mean().cpu().numpy()
                        #self.model.train()

                    # log last avg epoch results
                    if( i_epoch % (self.n_epochs-1)==0 and i_epoch>0):
                        avg_total_loss = total_loss_sum / num_batches
                        avg_action_loss = action_loss_sum / num_batches
                        avg_value_loss = value_loss_sum / num_batches
                        avg_entropy_loss = entropy_loss_sum / num_batches
                        avg_kl_loss = kl_loss_sum / num_batches
                        avg_reward = reward_sum / num_batches

                        self.writer.add_scalar('Loss/train', avg_total_loss, counter)
                        self.writer.add_scalar('Loss/Policy_grad', avg_action_loss, counter)
                        self.writer.add_scalar('Loss/Value', avg_value_loss, counter)
                        self.writer.add_scalar('Loss/Entropy', avg_entropy_loss, counter)
                        self.writer.add_scalar('Loss/KLdiv', avg_kl_loss, counter)
                        self.writer.add_scalar('Reward/train', avg_reward, counter)
                        self.writer.add_scalar('Reward/eval', eval_reward, counter)
                        #self.writer.add_scalar('Param/action_std', self.model.action_std.data[0].cpu(), counter)
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

                        print("train reward", avg_reward)
                        print("eval reward" , eval_reward)

                        if self.do_wandb:
                            wandb.log({
                                'epoch': i_epoch + 1,
                                'env_steps': counter,
                                'loss': avg_total_loss,
                                'loss_pg': avg_action_loss,
                                'loss_vf': avg_value_loss,
                                'loss_kl': avg_kl_loss,
                                'loss_entropy': avg_entropy_loss,
                                'average_reward': avg_reward,
                                'average_reward_eval': eval_reward,
                                #'action_std': self.model.action_std.data[0].cpu(),
                                'Step': counter
                            })

                if self.do_lin_lr_decay:
                    update_linear_schedule(self.optimizer, counter, self.max_env_steps, [self.lr, self.lr_enc])

                # save a checkpoint ever 1M steps
                for checkpoint in checkpoints:
                    if counter >= checkpoint and (counter - int((checkpoint - 1) / 10_000)) < 10_000:
                        checkpoint_path = 'runs/checkpoint_{}_{}_{}.pth'.format(checkpoint // 1_000_000, self.model_name, self.timestamp)
                        save_checkpoint(self.model, counter, self.optimizer, checkpoint_path)



        print("DONE " , self.max_env_steps , " steps")
        model_path = 'runs/RL_trained_model_{}_{}_{}'.format(self.model_name,self.env_name, self.timestamp)
        torch.save(self.model.state_dict(), model_path)


