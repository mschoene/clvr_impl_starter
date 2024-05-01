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

torch.backends.cudnn.enabled = False

def check_requires_grad(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires gradient: {param.requires_grad}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("DEVICE IS ", device)


spec = AttrDict(
        resolution=64,
        max_seq_len=30, #30,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=1,      # number of shapes per trajectory
        #rewards=[TargetXReward, TargetYReward, AgentXReward, AgentYReward]
        rewards=[VertPosReward, HorPosReward]
    )



#gen = DistractorTemplateMovingSpritesGenerator(spec)
#traj = gen.gen_trajectory()
#overfitting_image = traj.images[:, None].repeat(3, axis=1).astype(np.float32) / (255./2) - 1.0

#img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
#cv2.imwrite("test.png", img[0].transpose(1, 2, 0))

n_conditioning_frames = 3
n_prediction_frames = 25 #TODO change to 25 or w/e
batch_size = 128 #64
n_batches = 100
n_samples = batch_size*n_batches

input_channels = 1 # 3 but we change it to grey scale first
#this is the size of the input image and also the size of the latent space
output_size = spec['resolution']  #64



train_ds = MovingSpriteDataset_DistractorOnly(spec=spec, num_samples=n_samples)
valid_ds = MovingSpriteDataset_DistractorOnly(spec=spec, num_samples=batch_size*4)
#train_ds = MovingSpriteDataset(spec=spec, num_samples=n_samples)
#valid_ds = MovingSpriteDataset(spec=spec, num_samples=batch_size*4)


dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=2)
dataloader_valid = DataLoader(valid_ds, batch_size=batch_size, num_workers=2)
#dataloader = DataLoader(overfitting_image, 1)

encoder = ImageEncoder(input_channels=input_channels, output_size=output_size)
decoder = ImageDecoder(input_channels=output_size, output_size=output_size)
# Create an instance of Autoencoder
#autoencoder = AE(encoder, decoder)



#model = autoencoder
#    def __init__(self, input_channels, output_size, batch_size, n_cond_frames=3, n_pred_frames=25, lstm_output_size=32, n_layers_lstm=1, hidden_size=32):

predictor = Predictor(input_channels, output_size, batch_size)
reward_predictor = RewardPredictor(n_pred_frames=25, n_heads=1, lstm_output_size=64)

print(reward_predictor)

loss_fn = nn.MSELoss()
loss_fn_decoder = nn.MSELoss()
loss_fn_repr = nn.MSELoss()

optimizer_repr = optim.RAdam( list(predictor.parameters()) + list(reward_predictor.parameters()), betas = (0.9, 0.999))
optimizer_deco = optim.RAdam( decoder.parameters(), betas = (0.9, 0.999))

predictor.to(device)
reward_predictor.to(device)
decoder.to(device)
encoder.to(device)

#print( summary(reward_predictor, (input_channels, output_size, batch_size)) )
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

trafo = v2.Compose([v2.Grayscale(num_output_channels=1) ])

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    reward_predictor.train()
    decoder.train()


    #counter = 0
    for i_batch, sample_batched in enumerate(dataloader):
        inputs_pre = sample_batched.images[:, :n_conditioning_frames, ...] #take only first n images as inputs (tech not even nec since model anyway only uses 3) #TODO make more general
        #labels for image decoding
        label_img = sample_batched.images[:, :n_conditioning_frames, ...]
        #labels for rewards
        #labels = sample_batched.rewards['target_x']
        labels = sample_batched.rewards['vertical_position']
        #labels = sample_batched.rewards['horizontal_position']

        inputs_pre = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in inputs_pre], dim=1).transpose(1,0).squeeze(2)
        label_img = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in label_img], dim=1).transpose(1,0).squeeze(2)
        #limit the labels to the prediction frames only
        labels = labels[:, n_conditioning_frames:n_conditioning_frames + n_prediction_frames]

        #reshaped_tensor = tensor.view(nb * nf, nc)


        inputs = inputs_pre.clone().detach().requires_grad_(True)
        #send them to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        label_img = label_img.to(device)


        #print(inputs.shape, labels.shape)
        # zero the parameter gradients
        optimizer_deco.zero_grad()
        optimizer_repr.zero_grad()

        #import pdb; pdb.set_trace()

        predictions = predictor(inputs)
        output_rewards = reward_predictor(predictions)
        loss = loss_fn_repr(output_rewards, labels)
        loss.backward()
        # Adjust learning weights
        optimizer_repr.step()


        loss_dec = 0.

        #predictions.detach()
        #nb, nf, ndim = predictions.shape()
        #in_img = predictions.view(nb*nf, ndim)
        in_img = inputs[:, 0, ...] #.squeeze(1)
        in_img_truth = label_img[:, 0, ...] #.squeeze(1)

        enc_img = encoder(in_img)
        in_img.detach()
        enc_img.detach()
        in_img_truth.detach()
        dec_img = decoder(enc_img)
        loss_i = loss_fn_decoder(dec_img, in_img_truth)
        loss_dec += loss_i
        loss_dec.backward()
        optimizer_deco.step()
        optimizer_deco.zero_grad()

        # Gather data and report
        running_loss += loss.item() + loss_dec.item()
        if i_batch % n_batches == (n_batches -1):
            last_loss = running_loss / float(n_batches) # loss per batch
            #print('  batch {} loss: {}'.format(i_batch + 1, last_loss))
            #print(' l1 loss ', l1_lambda * l1_loss )
            tb_x = epoch_index * len(dataloader) + i_batch + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    #epoch_loss = running_loss/counter
    return last_loss

def do_epochs(EPOCHS=1000):
    best_vloss = 1000.

    for epoch_number in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        reward_predictor.train(True)
        decoder.train(True)

        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        reward_predictor.eval()
        decoder.eval()

        counter = 0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(dataloader_valid): #validation_loader):
                counter += 1
                #vinputs, vlabels = vdata.images[:, 0, ...].squeeze(1), vdata.images[:, 0, ...].squeeze(1)
                

                vinputs = vdata.images#[:, :n_conditioning_frames, ...] #take only first n images as inputs (tech not nec since model anyway only uses 3) #TODO make more general
                vlabel_img = vdata.images#[:, :n_conditioning_frames, ...]
                #vlabels = vdata.rewards['target_x']
                vlabels = vdata.rewards['vertical_position']
                #vlabels = vdata.rewards['horizontal_position']

                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                vlabel_img = vlabel_img.to(device)

                vinputs = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in vinputs], dim=1).transpose(1,0).squeeze(2)
                vlabel_img = torch.stack([torch.unsqueeze(trafo(img), dim=1) for img in vlabel_img], dim=1).transpose(1,0).squeeze(2)
                vlabels = vlabels[:, n_conditioning_frames:n_conditioning_frames + n_prediction_frames]

                vpredictions = predictor(vinputs)
                voutputs = reward_predictor(vpredictions)
                vloss = loss_fn_repr(voutputs, vlabels)

                vloss_dec = 0.

                vin_img = vinputs[:, 0, ...] #.squeeze(1)
                vin_img_truth = vlabel_img[:, 0, ...] #.squeeze(1)

                venc_img = encoder(vin_img)
                vin_img.detach()
                venc_img.detach()
                vin_img_truth.detach()
                vdec_img = decoder(venc_img)
                vloss_i = loss_fn_decoder(vdec_img, vin_img_truth)
                vloss_dec += vloss_i

                running_vloss += vloss + vloss_dec

            #every 
            nb, nf, ndim = vpredictions.shape
            vin_img = vpredictions.view(nb*nf, ndim)            
            #print(vin_img.shape)
            vin_img_truth = vlabel_img[:, n_conditioning_frames:n_prediction_frames+n_conditioning_frames, ...] #.squeeze(1)
            nb, nf_t, nc, nw, nh = vin_img_truth.shape
            assert(nf == nf_t)
            vin_img_truth = vin_img_truth.reshape(nb*nf_t, nc, nw, nh) #.squeeze(1)
            #venc_img = encoder(vin_img)
            vdec_img = decoder(vin_img)
            display = list(vdec_img[0:n_prediction_frames]) + list(vin_img_truth[0:n_prediction_frames])
            display = torchvision.utils.make_grid(display,nrow=25)
            torchvision.utils.save_image(display, "ae_reward_x_comp.png")

 
        avg_vloss = running_vloss /counter # (i + 1.)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if (avg_vloss < best_vloss*0.5 and avg_vloss<0.008) or epoch_number==EPOCHS-1:
            best_vloss = avg_vloss
            #model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
            #torch.save(model.state_dict(), model_path)

        if  avg_vloss < 0.00001:
            break

        epoch_number += 1

do_epochs(400)



#model = autoencoder
#model.load_state_dict(torch.load('/home/myriam/KAIST/code/starter/clvr_impl_starter/model_20240429_025411_65'))
#model.load_state_dict(torch.load('model_20240429_025411_11'))
#model.load_state_dict(torch.load('model_20240429_025411_39'))
#model.load_state_dict(torch.load('model_20240429_025411_65'))
#model.eval()

