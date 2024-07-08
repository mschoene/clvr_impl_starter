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

from models import ImageEncoder, ImageDecoder, RewardPredictor, Predictor

#from torchvision import transform, transforms
from torchvision.transforms import v2
import torchvision

from rl_utils.torch_utils import load_pretrained_weights

torch.backends.cudnn.enabled = False

def check_requires_grad(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires gradient: {param.requires_grad}")


def main(args):

    train_type = args.type 
    train_epochs = args.n_epochs
    n_distractors = args.n_distractors
    do_pretrained_enc = args.do_pre_enc

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("DEVICE IS ", device)

    spec = AttrDict(
            resolution=64,
            max_seq_len=40, #30, #30,
            max_speed=0.05,      # total image range [0, 1]
            obj_size=0.2,       # size of objects, full images is 1.0
            shapes_per_traj=1,      # number of shapes per trajectory
            rewards=[VertPosReward, HorPosReward]
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



    #gen = DistractorTemplateMovingSpritesGenerator(spec)
    #traj = gen.gen_trajectory()
    #overfitting_image = traj.images[:, None].repeat(3, axis=1).astype(np.float32) / (255./2) - 1.0

    #img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
    #cv2.imwrite("test.png", img[0].transpose(1, 2, 0))

    n_conditioning_frames = 25 #10 #3 #10 #3
    n_prediction_frames = spec['max_seq_len'] - n_conditioning_frames #25 
    batch_size = 32 #128 # 64 #128 #64
    n_batches = 20 #100
    n_samples = batch_size*n_batches

    input_channels = 1 # 3 but we change it to grey scale first
    #this is the size of the input image and also the size of the latent space
    output_size = spec['resolution']  #64

    n_heads = 1
    if(train_type =="full"):
        n_heads = 4
    #dataloader = DataLoader(overfitting_image, 1)

    # use 
    if do_pretrained_enc:
        print("Loading the pretrained encoder ")
        encoder = ImageEncoder(input_channels=input_channels, output_size=output_size)
        pretrained_path = "models/encoder_model_2obj_20240620_153556_299"
        encoder = load_pretrained_weights(encoder, pretrained_path)
        encoder.eval()  # Set to evaluation mode
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        encoder = ImageEncoder(input_channels=input_channels, output_size=output_size)

    decoder = ImageDecoder(input_channels=output_size, output_size=output_size)
    # Create an instance of Autoencoder
    #autoencoder = AE(encoder, decoder)

    #model = autoencoder
    #    def __init__(self, input_channels, output_size, batch_size, n_cond_frames=3, n_pred_frames=25, lstm_output_size=32, n_layers_lstm=1, hidden_size=32):

    predictor = Predictor( input_channels, output_size, batch_size , n_cond_frames=n_conditioning_frames, n_pred_frames=n_prediction_frames)
    reward_predictor = RewardPredictor(n_pred_frames=n_prediction_frames, n_heads=n_heads, lstm_output_size=64)
    #reward_predictor = RewardPredictor(n_pred_frames=25, n_heads=1, lstm_output_size=64)

    print(reward_predictor)

    #loss_fn = nn.MSELoss()
    loss_fn_decoder = nn.MSELoss()
    loss_fn_repr = nn.MSELoss()

    #optimizer_repr = optim.RAdam( list(predictor.parameters()) + list(reward_predictor.parameters()), betas = (0.9, 0.999))
    optimizer_deco = optim.RAdam( decoder.parameters(), betas = (0.9, 0.999))

    if do_pretrained_enc: #ie fixed encoder from AE
        optimizer_repr = optim.RAdam(list(predictor.parameters()) + list(reward_predictor.parameters()), betas=(0.9, 0.999))
    else:   #ie we train the encoder params
        optimizer_repr = optim.RAdam(list(predictor.parameters()) + list(reward_predictor.parameters()) + list(encoder.parameters()), betas=(0.9, 0.999))

    predictor.to(device)
    reward_predictor.to(device)
    decoder.to(device)
    encoder.to(device)

    #print( summary(reward_predictor, (input_channels, output_size, batch_size)) )
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/encoder_repr_trainer__{}_nDistr_{}_doPre_{}_model_epoch_{}'.format(train_type, n_distractors, do_pretrained_enc, timestamp))

    #epoch_number = 0

    trafo = v2.Compose([v2.Grayscale(num_output_channels=1) ])


    def train_one_epoch(epoch_index, tb_writer, train_dataloader):
        running_loss = 0.
        last_loss = 0.
        loss_dec = 0.

        #counter = 0
        for i_batch, sample_batched in enumerate(train_dataloader):
            #inputs_pre = sample_batched.images[:, :n_conditioning_frames, ...] #take only first n images as inputs (tech not even nec since model anyway only uses 3) #TODO make more general
            inputs_pre = sample_batched.images #take only first n images as inputs (tech not even nec since model anyway only uses 3) #TODO make more general
            #labels for image decoding
            #label_img = sample_batched.images[:, :n_conditioning_frames, ...]
            label_img = sample_batched.images[:, n_conditioning_frames:n_conditioning_frames + n_prediction_frames, ...]


            #labels for rewards
            #labels = sample_batched.rewards['target_x']
            #labels = sample_batched.rewards['vertical_position']
            if train_type =="horiz":
                labels = sample_batched.rewards['horizontal_position']
            elif train_type=="vert":
                labels = sample_batched.rewards['vertical_position']
            else:
                labels = torch.stack((sample_batched.rewards['target_x'], sample_batched.rewards['target_y'], sample_batched.rewards['agent_x'], sample_batched.rewards['agent_y']))
                labels = labels.permute(1,2,0) #last dimension is number of heads when predicting, ie we reoder to batch, sequence, head

            inputs_pre = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in inputs_pre], dim=1).transpose(1,0).squeeze(2)
            label_img = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in label_img], dim=1).transpose(1,0).squeeze(2)
            #limit the labels to the prediction frames only
            labels = labels[:, n_conditioning_frames:n_conditioning_frames + n_prediction_frames]
            #print(labels.shape)
            #reshaped_tensor = tensor.view(nb * nf, nc)

            inputs = inputs_pre.clone().detach().requires_grad_(True)
            #send them to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            label_img = label_img.to(device)

            #print(inputs.shape, labels.shape)



            enc_img =  [encoder(x_timestep.squeeze(1)) for x_timestep in inputs.split(1, dim=1)]
            predictions = predictor(enc_img)
            #print(predictions.shape)
            #predictions = predictor(inputs)
            output_rewards = reward_predictor(predictions)
            loss = loss_fn_repr(output_rewards, labels )
            #loss = loss_fn_repr(output_rewards[:, 0:-2, ...], labels[:, 1:-1, ...])

            # zero the parameter gradients
            optimizer_repr.zero_grad()
            loss.backward()
            # Adjust learning weights
            optimizer_repr.step()


            #predictions.detach()
            #nb, nf, ndim = predictions.shape
            #in_img = predictions.view(nb*nf, ndim)
            #print(predictions.shape, in_img.shape)

            #in_img = inputs[:, 0, ...] #.squeeze(1)
            ###in_img_truth = label_img[:, 0, ...] #.squeeze(1) #n_conditioning_frames:n_conditioning_frames + n_prediction_frames
            #in_img_truth = label_img[:, n_conditioning_frames:n_conditioning_frames + n_prediction_frames, ...] #.squeeze(1) #

            #enc_img = encoder(in_img)  
            ###enc_img_tensor = torch.stack(enc_img, dim=1)  # Shape will be (batch_size, sequence_length, encoded_dim)
            #print(enc_img_tensor.shape)
            #in_img.detach()
            ###enc_img_tensor = enc_img_tensor.detach()
            ###in_img_truth = in_img_truth.detach()

            #print(predictions.shape, enc_img.shape)
            ###dec_img = decoder(enc_img_tensor[:, 0, ...])
            #print(in_img_truth.shape, in_img.shape)

            #dec_img = decoder(in_img)


            predictions_reshaped = predictions.view(-1, predictions.shape[-1])  # Shape: (batch_size * sequence_length, prediction_dim)
            in_img_truth = label_img.reshape(-1, *label_img.shape[2:])  # Shape: (batch_size * sequence_length, prediction_dim)            print(predictions_reshaped.shape,in_img_truth.shape )
            predictions_reshaped = predictions_reshaped.detach()
            dec_img = decoder(predictions_reshaped)

            loss_dec = loss_fn_decoder(dec_img, in_img_truth)
            
            optimizer_deco.zero_grad()
            loss_dec.backward()


            #loss_i = loss_fn_decoder(dec_img, in_img_truth)
            #loss_dec += loss_i
            #loss_dec.backward()
            optimizer_deco.step()
            #optimizer_deco.zero_grad()

            # Gather data and report
            running_loss += loss.item()  #+ loss_dec.item()
            if i_batch % n_batches == (n_batches -1):
                print('Last loss of lstm {} and decoder {}'.format( loss.item(), loss_dec.item() ))
                last_loss = running_loss / float(n_batches) # loss per batch
                tb_x = epoch_index * len(train_dataloader) + i_batch + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def do_epochs(EPOCHS=1000):
        best_vloss = 1000.

        for epoch_number in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            #get fresh data each epoch
            train_ds = MovingSpriteDataset_DistractorOnly(spec=spec, num_samples=n_samples)
            valid_ds = MovingSpriteDataset_DistractorOnly(spec=spec, num_samples=batch_size*4)
            if(train_type =="full"):
                train_ds = MovingSpriteDataset(spec=spec, num_samples=n_samples)
                valid_ds = MovingSpriteDataset(spec=spec, num_samples=batch_size*4)

            dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=2)
            dataloader_valid = DataLoader(valid_ds, batch_size=batch_size, num_workers=2)

            # Make sure gradient tracking is on, and do a pass over the data
                   
            predictor.train()
            reward_predictor.train()
            decoder.train()
            if not do_pretrained_enc:
                encoder.train()

            avg_loss = train_one_epoch(epoch_number, writer, dataloader)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            predictor.eval()
            reward_predictor.eval()
            decoder.eval()
            encoder.eval()

            counter = 0 #counts validation batches
            total_head_losses = [0.0] * n_heads  # Initialize total loss for each head

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(dataloader_valid): #validation_loader):
                    counter += 1
                    # vinputs, vlabels = vdata.images[:, 0, ...].squeeze(1), vdata.images[:, 0, ...].squeeze(1)

                    vinputs = vdata.images #[:, :n_conditioning_frames, ...] #take only first n images as inputs (tech not nec since model anyway only uses 3) #TODO make more general
                    ###vlabel_img = vdata.images #[:, :n_conditioning_frames, ...]
                    vlabel_img = vdata.images[:, n_conditioning_frames:n_conditioning_frames + n_prediction_frames, ...]

                    if train_type =="horiz":
                        vlabels = vdata.rewards['horizontal_position']
                    elif train_type=="vert":
                        vlabels = vdata.rewards['vertical_position']
                    else:
                        vlabels = torch.stack((vdata.rewards['target_x'], vdata.rewards['target_y'], vdata.rewards['agent_x'], vdata.rewards['agent_y']))
                        vlabels = vlabels.permute(1,2,0) # last dimension is number of heads when predicting, ie we reoder to batch, sequence, head

                    vinputs = vinputs.to(device)
                    vlabels = vlabels.to(device)
                    vlabel_img = vlabel_img.to(device)

                    vinputs = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in vinputs], dim=1).transpose(1,0).squeeze(2)
                    vlabel_img = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in vlabel_img], dim=1).transpose(1,0).squeeze(2)
                    vlabels = vlabels[:, n_conditioning_frames:n_conditioning_frames + n_prediction_frames]

                    venc_img =  [encoder(x_timestep.squeeze(1)) for x_timestep in vinputs.split(1, dim=1)]
                    vpredictions = predictor(venc_img)
                    #vpredictions = predictor(vinputs)
                    voutputs = reward_predictor(vpredictions)

                    if n_heads>1:
                        for head_idx in range(voutputs.shape[2]):  # outputs.shape[2] is the number of heads
                            # Extract the outputs for the current head
                            head_output = voutputs[:, :, head_idx]  # Shape: (batch_size, n_frames, ...)

                            # Compute the loss for the current head
                            loss_head = loss_fn_decoder(head_output, vlabels[:, :, head_idx])

                            # Append the loss to the list
                            #head_losses.append(loss_head.item())
                            # Accumulate the loss for the current head
                            total_head_losses[head_idx] += loss_head.item()


                            # Print the loss for the current head
                            #print(f"Loss for head {head_idx}: {loss_head.item()}")
                            #writer.add_scalar(f'Loss/Head_{head_idx}', loss_head.item(), epoch_number+1)

                    #vloss = loss_fn_repr(voutputs, vlabels)
                    vloss = loss_fn_repr(voutputs, vlabels)
                    #avg_head_loss = sum(head_losses) / len(head_losses)
                    #print(f"Average loss across all heads: {vloss}")
                    vloss_dec = 0.

                    #vin_img = vinputs[:, 0, ...] #.squeeze(1)
                    #vin_img_truth = vlabel_img[:, 0, ...] #.squeeze(1)

                    #venc_img = encoder(vin_img)
                    #vin_img.detach()
                    ##venc_img_tensor = torch.stack(venc_img, dim=1)  # Shape will be (batch_size, sequence_length, encoded_dim)
                    ##venc_img_tensor = venc_img_tensor.detach()
                    #venc_img.detach()
                    ###vin_img_truth = vin_img_truth.detach()
                    ###vdec_img = decoder(venc_img_tensor[:, 0, ...])


                    predictions_reshaped = vpredictions.view(-1, vpredictions.shape[-1])  # Shape: (batch_size * sequence_length, prediction_dim)
                    vin_img_truth = vlabel_img.reshape(-1, *vlabel_img.shape[2:])  # Shape: (batch_size * sequence_length, prediction_dim)            print(predictions_reshaped.shape,in_img_truth.shape )
                    predictions_reshaped = predictions_reshaped.detach()
                    vdec_img = decoder(predictions_reshaped)

                    vloss_i = loss_fn_decoder(vdec_img, vin_img_truth)
                    vloss_dec += vloss_i

                    running_vloss += vloss #+ vloss_dec

                if (epoch_number%10==0 and epoch_number >0) or epoch_number==EPOCHS-1:
                    #every eval decode one sequence for display          
                    #pred_1seq = vpredictions[0:n_prediction_frames, ...] #get 1 item in batch of predicitons (ie output of pred is shorter by conditioning frames)
                    pred_1seq = vpredictions[0, ...] #get 1 item in batch of predicitons (ie output of pred is shorter by conditioning frames)
                    vin_img_truth_1seq= vlabel_img[0, n_conditioning_frames:n_prediction_frames+n_conditioning_frames, ...] #.squeeze(1)
                    print(pred_1seq.shape, vin_img_truth_1seq.shape )
                    vdec_img = decoder(pred_1seq)
                    print(vdec_img.shape)
                    display = list(vdec_img[0:n_prediction_frames, ...]) + list(vin_img_truth_1seq)
                    display = torchvision.utils.make_grid(display,nrow=n_prediction_frames)
                    torchvision.utils.save_image(display, "models/ae_r_comp_{}_nDistr_{}_doPre_{}_epoch_{}_{}.png".format(train_type, n_distractors,do_pretrained_enc,  epoch_number, timestamp) )

            # add loss per head to summary
            if n_heads>1:
                avg_head_losses = [total_loss / counter for total_loss in total_head_losses]
                loss_dict = {f'Loss/Head_{head_idx}': avg_loss for head_idx, avg_loss in enumerate(avg_head_losses)}
                writer.add_scalars('Validation Loss/Heads', loss_dict, epoch_number + 1)

            avg_vloss = running_vloss /counter # (i + 1.)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            

            writer.flush()

            # Track best performance, and save the model's state
            #if (avg_vloss < best_vloss*0.5 and avg_vloss<0.008) or epoch_number==EPOCHS-1:
            if epoch_number==EPOCHS-1 or (epoch_number%100==0 and epoch_number >0):
                best_vloss = avg_vloss
                model_path = 'models/repr_decoder_{}_nDistr_{}_doPre_{}_model_epoch_{}_{}'.format(train_type, n_distractors, do_pretrained_enc, epoch_number, timestamp)
                torch.save(decoder.state_dict(), model_path)
                model_path = 'models/repr_encoder_{}_nDistr_{}_doPre_{}_model_epoch_{}_{}'.format(train_type, n_distractors, do_pretrained_enc, epoch_number, timestamp)
                torch.save(encoder.state_dict(), model_path)                
                model_path = 'models/repr_lstm_{}_nDistr_{}_doPre_{}_model_epoch_{}_{}'.format(train_type, n_distractors, do_pretrained_enc, epoch_number, timestamp)
                torch.save(encoder.state_dict(), model_path)

            #if  avg_vloss < 0.00001:
            #    break

            epoch_number += 1
        writer.close()

    do_epochs(train_epochs)



#model = autoencoder
#model.load_state_dict(torch.load('/home/myriam/KAIST/code/starter/clvr_impl_starter/model_20240429_025411_65'))
#model.load_state_dict(torch.load('model_20240429_025411_11'))
#model.load_state_dict(torch.load('model_20240429_025411_39'))
#model.load_state_dict(torch.load('model_20240429_025411_65'))
#model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MimiPPO model with specified parameters.")
    parser.add_argument('--type', type=str, required=True, help="Which rewards to consider ('horiz', 'vert', 'full').")
    parser.add_argument('--n_epochs', type=int, default=500, help="Number of training epochs to run, default 500.")
    parser.add_argument('--n_distractors', type=int, default=0, help="Number of distractors, default 0.")
    parser.add_argument('--do_pre_enc', type=int, default=0, help="Use pretrained encoder")


    args = parser.parse_args()

    main(args)
    