import argparse
from datetime import datetime
from sprites_datagen.moving_sprites import MovingSpriteDataset,DistractorTemplateMovingSpritesGenerator,MovingSpriteDataset_DistractorOnly

import gym
from sprites_env.envs.sprites import SpritesEnv
from sprites_datagen.rewards import *

from general_utils import AttrDict
from torch.utils.data import Dataset, DataLoader,IterableDataset
from datasets import load_dataset
from general_utils import * #make_image_seq_strip
import numpy as np
import cv2
import sprites_datagen.moving_sprites
from torch.autograd import Variable

from torchviz import make_dot
import matplotlib as plt

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary

import torch.optim as optim

from models import ImageEncoder, ImageDecoder, RewardPredictor, Predictor, Predictor_seq

#from torchvision import transform, transforms
from torchvision.transforms import v2
import torchvision

from rl_utils.torch_utils import load_pretrained_weights
from contextlib import contextmanager
torch.backends.cudnn.enabled = False

def check_requires_grad(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires gradient: {param.requires_grad}")


# Context manager to have context that requires no grad/grad
@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_for_white=10.0):
        super(WeightedMSELoss, self).__init__()
        self.weight_for_white = weight_for_white  # Weight for white pixels

    def forward(self, input, target):
        # Create weights: higher weight for white pixels
        weights = torch.ones_like(target) + target * (self.weight_for_white - 1)
        squared_error = (input - target) ** 2
        weighted_squared_error = squared_error * weights
        return weighted_squared_error.mean()


def main(args):

    train_type = args.type 
    train_epochs = args.n_epochs
    n_distractors = args.n_distractors
    do_pretrained_enc = args.do_pre_enc
    n_cond_fr = args.n_cond_frames
    do_seqLSTM = args.do_seq

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("DEVICE IS ", device)


    if train_type=="horiz":
        spec = AttrDict(
                resolution=64,
                max_seq_len=40, #30, #30,
                max_speed=0.05,      # total image range [0, 1]
                obj_size=0.2,       # size of objects, full images is 1.0
                shapes_per_traj=1,      # number of shapes per trajectory
                rewards=[HorPosReward]
            )
        
    if train_type=="vert":
        spec = AttrDict(
                resolution=64,
                max_seq_len=40, #30, #30,
                max_speed=0.05,      # total image range [0, 1]
                obj_size=0.2,       # size of objects, full images is 1.0
                shapes_per_traj=1,      # number of shapes per trajectory
                rewards=[VertPosReward]
            )

    if(train_type =="full"):
        spec = AttrDict(
                resolution=64,
                max_seq_len=40, #30, #30,
                max_speed=0.05,      # total image range [0, 1]
                obj_size=0.2,       # size of objects, full images is 1.0
                shapes_per_traj=2 + n_distractors,      # number of shapes per trajectory
                rewards=[TargetXReward, TargetYReward, AgentXReward, AgentYReward]
            )

    n_conditioning_frames = n_cond_fr #3 #10 #0# 10 #3 #10 #3
    n_prediction_frames = spec['max_seq_len'] - n_conditioning_frames #25 
    batch_size = 32 
    n_batches = 16 
    n_samples = batch_size*n_batches

    input_channels = 1 # 3 but we transform images to grey scale first
    #this is the size of the input image
    output_size = spec['resolution'] 

    n_heads = 1
    if(train_type =="full"):
        n_heads = 4


    if do_pretrained_enc:
        print("Loading the pretrained encoder ")
        encoder = ImageEncoder(input_channels=input_channels, output_size=output_size)
        if(train_type =="full"):
            if n_distractors==0:
                pretrained_path = "models/encoder_model_2obj_20240708_223549_149"
            elif n_distractors==1:
                pretrained_path = "models/encoder_model_2obj_nDistr_1_20240709_134803_150"
            else:
                pretrained_path = "models/encoder_model_2obj_nDistr_2_20240709_165324_150" 
        if(train_type =='vert'):
            pretrained_path = "models/repr_encoder_vert_nDistr_0_doPre_0_model_epoch_500_20240713_124542" #models/repr_encoder_vert_nDistr_0_doPre_0_model_epoch_100_20240708_142535"
        if(train_type =='horiz'):
            pretrained_path = "models/repr_encoder_horiz_nDistr_0_doPre_0_model_epoch_100_20240708_144153"

        encoder = load_pretrained_weights(encoder, pretrained_path)
        encoder.eval()  # Set to evaluation mode
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        encoder = ImageEncoder(input_channels=input_channels, output_size=output_size)


    decoder = ImageDecoder(input_channels=output_size, output_size=output_size)

    #if(train_type =='vert'):
    #        pretrained_path = "models/repr_decoder_vert_nDistr_0_doPre_0_model_epoch_500_20240713_124542" #models/repr_encoder_vert_nDistr_0_doPre_0_model_epoch_100_20240708_142535"
    #if(train_type =='horiz'):
    #    pretrained_path = "models/repr_encoder_horiz_nDistr_0_doPre_0_model_epoch_100_20240708_144153"
    #decoder = load_pretrained_weights(decoder, pretrained_path)
     

    # LSTM
    predictor = Predictor( input_channels, output_size, batch_size , n_cond_frames=n_conditioning_frames, n_pred_frames=n_prediction_frames)
    #
    if do_seqLSTM:
        predictor = Predictor_seq( input_channels, output_size, batch_size , n_cond_frames=n_conditioning_frames, n_pred_frames=n_prediction_frames)
    # reward head(s)
    reward_predictor = RewardPredictor(n_pred_frames=n_prediction_frames, n_heads=n_heads, lstm_output_size=64)

    #loss_fn_decoder = nn.MSELoss()
    loss_fn_decoder = WeightedMSELoss(weight_for_white=1.75)

    loss_fn_repr = nn.MSELoss()

    optimizer_deco = optim.RAdam( decoder.parameters(), betas = (0.9, 0.999), weight_decay=1e-5)
    #optimizer = RAdam(model.parameters(), lr=0.001, weight_decay=1e-5)
    if do_pretrained_enc:   # ie fixed encoder from AE
        optimizer_repr = optim.RAdam(list(predictor.parameters()) + list(reward_predictor.parameters()), betas=(0.9, 0.999), weight_decay=1e-5)
    else:                   # ie we train the encoder params
        optimizer_repr = optim.RAdam(list(predictor.parameters()) + list(reward_predictor.parameters()) + list(encoder.parameters()), betas=(0.9, 0.999), weight_decay=1e-5)

    predictor.to(device)
    reward_predictor.to(device)
    decoder.to(device)
    encoder.to(device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    ext_str = '{}_nDistr_{}_nCondFr_{}_doPre_{}_doSeq_{}'.format(train_type, n_distractors, n_cond_fr, do_pretrained_enc, do_seqLSTM)

    writer = SummaryWriter('runs/encoder_repr_trainer_{}_ts_{}'.format(ext_str, timestamp))

    # Trafo from rgb to grey scale img (-1,1)
    trafo = v2.Compose([v2.Grayscale(num_output_channels=1)])



    def train_one_epoch(epoch_index, i_dataloader, doTrain = True):
        running_loss = 0.0
        running_loss_dec = 0.0
        loss = 0.0
        loss_dec = 0.0
        total_head_losses = [0.0] * n_heads  # Initialize total loss for each head

        # Loop over batches
        for i_batch, sample_batched in enumerate(i_dataloader):

            with conditional_no_grad(not doTrain):
                #inputs_pre = sample_batched.images[:, 0:n_conditioning_frames, ...] 
                #label_img = sample_batched.images[:, 0:n_conditioning_frames, ...]                
                inputs_pre = sample_batched.images
                label_img = sample_batched.images
                if train_type =="horiz":
                    labels = sample_batched.rewards['horizontal_position']
                elif train_type=="vert":
                    labels = sample_batched.rewards['vertical_position']
                else:
                    labels = torch.stack((sample_batched.rewards['target_x'], sample_batched.rewards['target_y'], sample_batched.rewards['agent_x'], sample_batched.rewards['agent_y']))
                    labels = labels.permute(1,2,0) #last dimension is number of heads when predicting, ie we reoder to batch, sequence, head
                # Transform to grey scale
                inputs = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in inputs_pre], dim=1).transpose(1,0).squeeze(2)
                label_img = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in label_img], dim=1).transpose(1,0).squeeze(2)
                #limit the labels to the prediction frames only
                labels_reward = labels[:, n_conditioning_frames:n_conditioning_frames + n_prediction_frames]
                inputs = inputs.clone().detach().requires_grad_(True)

                #send them to the device
                inputs = inputs.to(device)
                labels_reward = labels_reward.to(device)
                label_img = label_img.to(device)


                # Encode N conditioning frames
                enc_img =  [encoder(x_timestep.squeeze(1)) for x_timestep in inputs.split(1, dim=1)]
                # LSTM predict T prediction steps
                predictions = predictor(enc_img)
                # Predict rewards from h_ts
                output_rewards = reward_predictor(predictions)
                loss = loss_fn_repr(output_rewards, labels_reward )


                if n_heads>1:
                    for head_idx in range(output_rewards.shape[2]):  # outputs.shape[2] is the number of heads
                        # Extract the outputs for the current head
                        head_output = output_rewards[:, :, head_idx]  # Shape: (batch_size, n_frames, ...)
                        # Compute the loss for the current head
                        loss_head = loss_fn_decoder(head_output, labels_reward[:, :, head_idx])
                        total_head_losses[head_idx] += loss_head.item()

                if doTrain : #and epoch_index < 100:
                    optimizer_repr.zero_grad()
                    loss.backward()
                    optimizer_repr.step()

                enc_img_tensor = torch.stack(enc_img, dim=1)  # Shape will be (batch_size, sequence_length, encoded_dim)
                enc_img_tensor = enc_img_tensor.clone().detach().requires_grad_(False)
                enc_img_tensor = enc_img_tensor.reshape(-1, *enc_img_tensor.shape[2:])  # Shape: (batch_size * sequence_length, prediction_dim)         
                dec_img = decoder(enc_img_tensor)
                in_img_truth = label_img.reshape(-1, *label_img.shape[2:])  # Shape: (batch_size * sequence_length, prediction_dim)         
                
                #enc_img_tensor = torch.stack(predictions, dim=1)  # Shape will be (batch_size, sequence_length, encoded_dim)
                #enc_img_tensor = predictions  # Shape will be (batch_size, sequence_length, encoded_dim)
                #enc_img_tensor = enc_img_tensor.clone().detach().requires_grad_(False)
                #enc_img_tensor = enc_img_tensor.reshape(-1, *enc_img_tensor.shape[2:])  # Shape: (batch_size * sequence_length, prediction_dim)         
                #dec_img = decoder(enc_img_tensor)
                #label_img = label_img[:, n_conditioning_frames:, ...]
                #in_img_truth = label_img.reshape(-1, *label_img.shape[2:])  # Shape: (batch_size * sequence_length, prediction_dim)         

                loss_dec = loss_fn_decoder(dec_img, in_img_truth)

                if doTrain:

                    optimizer_deco.zero_grad()
                    loss_dec.backward()
                    # we give the encoder some time to learn first
                    if True : # epoch_index >  30:
                        optimizer_deco.step()

                    counter = (epoch_index)*len(i_dataloader) + i_batch
                    # Log gradient distributions to TensorBoard
                    #for n, p in predictor.named_parameters():
                    #    if p.requires_grad and ("bias" not in n):
                    #        if p.grad is not None:
                    #            writer.add_histogram(f'Gradients/predictor/{n}', p.grad, counter)
                    #
                    #for n, p in encoder.named_parameters():
                    #    if p.requires_grad and ("bias" not in n):
                    #        if p.grad is not None:
                    #            writer.add_histogram(f'Gradients/encoder/{n}', p.grad, counter)
                    #
                    #for n, p in decoder.named_parameters():
                    #    if p.requires_grad and ("bias" not in n):
                    #        if p.grad is not None:
                    #            writer.add_histogram(f'Gradients/decoder/{n}', p.grad, counter)
                    # Log reward histograms 
                    writer.add_histogram(f'Rewards/Prediction', output_rewards, counter)
                    writer.add_histogram(f'Rewards/Truth', labels_reward, counter)
                    writer.add_histogram(f'Rewards/TruthMinusPred', labels_reward- output_rewards, counter)
                    writer.add_histogram(f'Decoder/Pred', dec_img, counter)
                    writer.add_histogram(f'Decoder/Truth', in_img_truth, counter)
                    writer.add_histogram(f'Decoder/TruthMinusPred', in_img_truth - dec_img, counter)

            running_loss_dec += loss_dec.item()  
            running_loss += loss.item()  

                           
        if (epoch_index%10==0 and epoch_index >0):
            vin_img_truth_1seq= label_img[0, :n_conditioning_frames+n_prediction_frames, ...] #.squeeze(1)
            #display = list(dec_img[ 0:n_conditioning_frames, ...]) + list(vin_img_truth_1seq)
            #display = torchvision.utils.make_grid(display,nrow=n_conditioning_frames)
            display = list(dec_img[ 0:n_conditioning_frames+n_prediction_frames, ...]) + list(vin_img_truth_1seq)
            display = torchvision.utils.make_grid(display,nrow=n_conditioning_frames+n_prediction_frames)
            #torchvision.utils.save_image(display, "models/ae_r_comp_{}_nDistr_{}_nCondFr_{}_doPre_{}_epoch_{}_{}.png".format(train_type, n_distractors, n_cond_fr, do_pretrained_enc,  epoch_index, timestamp) )
            torchvision.utils.save_image(display, "models/ae_r_comp_{}_{}_{}.png".format(ext_str, epoch_index, timestamp) )

        if n_heads>1:
            avg_head_losses = [total_loss / len(i_dataloader) for total_loss in total_head_losses]
            loss_dict = {f'Loss/Head_{head_idx}': avg_lossi for head_idx, avg_lossi in enumerate(avg_head_losses)}
            if doTrain:
                writer.add_scalars('Training Loss/Heads', loss_dict, epoch_index + 1)
            else:
                writer.add_scalars('Validation Loss/Heads', loss_dict, epoch_index + 1)

            if (epoch_index%50==0 and epoch_index >0):
                if doTrain:
                    print(f"Epoch {epoch_index + 1} Training Head Losses:")
                else:
                    print(f"Epoch {epoch_index + 1} Validation Head Losses:")

                for head_idx, avg_loss in enumerate(avg_head_losses):
                    print(f"  Head {head_idx}: {avg_loss:.4f}")

        return running_loss / len(i_dataloader), running_loss_dec / len(i_dataloader)



    def do_epochs(EPOCHS=1000):

        for epoch_number in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Define training and validation data
            train_ds = MovingSpriteDataset_DistractorOnly(spec=spec, num_samples=n_samples)
            valid_ds = MovingSpriteDataset_DistractorOnly(spec=spec, num_samples=4*batch_size) #4 bachtes as validation size
            #valid_ds = MovingSpriteDataset_DistractorOnly(spec=spec, num_samples=n_samples) #4 bachtes as validation size
            if(train_type =="full"):
                train_ds = MovingSpriteDataset(spec=spec, num_samples=n_samples)
                #valid_ds = MovingSpriteDataset(spec=spec, num_samples=n_samples)
                valid_ds = MovingSpriteDataset(spec=spec, num_samples=4*batch_size)
            dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=2)
            dataloader_valid = DataLoader(valid_ds, batch_size=batch_size, num_workers=2)
        
            # Make sure gradient tracking is on, and do a pass over the data
            predictor.train()
            reward_predictor.train()
            decoder.train()
            if not do_pretrained_enc:
                encoder.train()

            ### Training ###
            avg_loss, avg_loss_dec = train_one_epoch(epoch_number, dataloader, doTrain = True)

            # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
            predictor.eval()
            reward_predictor.eval()
            decoder.eval()
            encoder.eval()

            ### Evaluation ###
            avg_vloss, avg_vloss_dec = train_one_epoch(epoch_number, dataloader_valid, doTrain = False)

            print('LOSS reward train {} valid {}'.format(avg_loss, avg_vloss))
            print('LOSS decoder train {} valid {}'.format(avg_loss_dec, avg_vloss_dec))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch_number + 1)
            writer.add_scalars('Decoder Training vs. Validation Loss',
                            { 'Training' : avg_loss_dec, 'Validation' : avg_vloss_dec }, epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            #if (avg_vloss < best_vloss*0.5 and avg_vloss<0.008) or epoch_number==EPOCHS-1:
            if epoch_number==EPOCHS-1 or (epoch_number%50==0 and epoch_number >0):
                model_path = 'models/repr_decoder_{}_epoch_{}_{}'.format(ext_str, epoch_number, timestamp)
                torch.save(decoder.state_dict(), model_path)
                model_path = 'models/repr_encoder_{}_epoch_{}_{}'.format(ext_str, epoch_number, timestamp)
                torch.save(encoder.state_dict(), model_path)                
                model_path = 'models/repr_lstm_{}_epoch_{}_{}'.format(ext_str, epoch_number, timestamp)
                torch.save(encoder.state_dict(), model_path)

            #if  avg_vloss < 0.00001:
            #    break
            epoch_number += 1
        writer.close()

    do_epochs(train_epochs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MimiPPO model with specified parameters.")
    parser.add_argument('--type', type=str, required=True, help="Which rewards to consider ('horiz', 'vert', 'full').")
    parser.add_argument('--n_epochs', type=int, default=501, help="Number of training epochs to run, default 500.")
    parser.add_argument('--n_distractors', type=int, default=0, help="Number of distractors, default 0.")
    parser.add_argument('--do_pre_enc', type=int, default=0, help="Use pretrained encoder")
    parser.add_argument('--do_seq', type=int, default=0, help="Do sequential lstm")
    parser.add_argument('--n_cond_frames', type=int, default=3, help="Number of preconditioning frames")

    args = parser.parse_args()

    main(args)
    