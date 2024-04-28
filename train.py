from datetime import datetime
from sprites_datagen.moving_sprites import MovingSpriteDataset,DistractorTemplateMovingSpritesGenerator

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

from models import ImageEncoder, ImageDecoder, AE, Autoencoder #, AE_test

import torchvision

torch.backends.cudnn.enabled = False

def check_requires_grad(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Requires gradient: {param.requires_grad}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("DEVICE IS ", device)


spec = AttrDict(
        resolution=64,
        max_seq_len=1, #30,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=4,      # number of shapes per trajectory
        rewards=[ZeroReward]
        #rewards=[HorPosReward, VertPosReward]
    )


#gen = DistractorTemplateMovingSpritesGenerator(spec)
#traj = gen.gen_trajectory()
#overfitting_image = traj.images[:, None].repeat(3, axis=1).astype(np.float32) / (255./2) - 1.0

#img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
#cv2.imwrite("test.png", img[0].transpose(1, 2, 0))

n_conditioning_frames = 3
n_prediction_frames = 6 #TODO change to 25 or w/e
batch_size = 1024 # 16 #256 #512 # 512 #1024
n_samples = batch_size*100


input_channels = 1 # 3
#this is the size of the input image and also the size of the latent space
output_size = spec['resolution']  #64

train_ds = MovingSpriteDataset(spec=spec, num_samples=n_samples)
valid_ds = MovingSpriteDataset(spec=spec, num_samples=batch_size*4)


dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=4)
dataloader_valid = DataLoader(valid_ds, batch_size=batch_size, num_workers=4)
#dataloader = DataLoader(overfitting_image, 1)

encoder = ImageEncoder(input_channels=input_channels, output_size=output_size)
decoder = ImageDecoder(input_channels=output_size, output_size=output_size)
# Create an instance of Autoencoder
autoencoder = AE(encoder, decoder)

model = autoencoder

#model = AE_test()
#model = Autoencoder() #test one from geesforgeeks
print(model)

loss_fn = nn.MSELoss()

#optimizer = optim.RAdam( model.parameters(), betas = (0.9, 0.999)) # , weight_decay=0.001)
optimizer = optim.AdamW( model.parameters(), lr=0.0001) #, weight_decay=0.001)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

print( summary(model, (1, output_size,output_size)) )


#TODO put in an epoch loop
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    model.train()

    #counter = 0
    for i_batch, sample_batched in enumerate(dataloader):
        #print("batch size ", i_batch, sample_batched['states'].size(),sample_batched['images'].size())
        #if i_batch == 2:
        #    break
        #counter += 1
        #print(sample_batched.shape)

        #inputs_pre = sample_batched
        #labels = sample_batched
        #print(sample_batched)
        inputs_pre = sample_batched.images[:, 0, ...].squeeze(1)
        labels = sample_batched.images[:, 0, ...].squeeze(1)

        inputs_pre = torchvision.transforms.Grayscale(num_output_channels=1)(inputs_pre)
        labels = torchvision.transforms.Grayscale(num_output_channels=1)(labels)
        #print(inputs_pre.shape)

        inputs = inputs_pre.clone().detach().requires_grad_(True)
        #inputs = torch.tensor(inputs, requires_grad=True )
        #inputs.retain_grad = True
        #inputs.requires_grad = True
        #print(inputs.shape)
        inputs = inputs.to(device)
        labels = labels.to(device)
        #labels = (labels+1)/2
        #inputs = (inputs+1)/2
        
        #print(inputs.shape, labels.shape)
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)

        #print(outputs[0][1][5], labels[0][1][5])
        loss = loss_fn(outputs, labels)


        #l1_lambda = 0.01  # Adjust the regularization strength
        #l1_loss = 0
        #for param in model.parameters():
        #    l1_loss += torch.norm(param, p=1)
        #loss += l1_lambda * l1_loss
                
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        #print(outputs.grad)
        #print(outputs.requires_grad ==True)
        #print(outputs.is_leaf ==True)
        #for p in model.parameters():
        #    print("param grad ", p.grad)
        #make_dot(outputs, params=dict(model.named_parameters())).render("graph")


        # Gather data and report
        running_loss += loss.item()
        if i_batch % 100 == 99:
            print(inputs[0][0][5][4], outputs[0][0][5][4])
            print(inputs[0][0][4][5], outputs[0][0][4][5])
            last_loss = running_loss / 100. # loss per batch
            print('  batch {} loss: {}'.format(i_batch + 1, last_loss))
            #print(' l1 loss ', l1_lambda * l1_loss )
            tb_x = epoch_index * len(dataloader) + i_batch + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    #epoch_loss = running_loss/counter
    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 100# 40 #60 #80

#best_vloss = 1_000_000.
best_vloss = 0.05

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    counter = 0
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(dataloader_valid): #validation_loader):
            counter += 1 
            vinputs, vlabels = vdata.images[:, 0, ...].squeeze(1), vdata.images[:, 0, ...].squeeze(1)
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            vinputs = torchvision.transforms.Grayscale(num_output_channels=1)(vinputs)
            vlabels = torchvision.transforms.Grayscale(num_output_channels=1)(vlabels)
            #vlabels = (vlabels+1)/2
            #vinputs = (vinputs+1)/2
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss /counter # (i + 1.)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    if  best_vloss < 0.1:
        break

    epoch_number += 1





for i, vdata  in enumerate(dataloader):
    #this is the way to adopt the same plotting for the dataset images as for the traj images
    # it's a bit hacky but it works, so it's not stupid :,) also I don't need to change the code in multiple places so it's a win...
    # undo this/ (255./2) - 1.0
    #print(vdata.images.shape )

    vinputs, vlabels = vdata.images[:, 0, ...], vdata.images[:, 0, ...].squeeze(1)
    #vinputs, vlabels = vdata, vdata
    #print(vinputs.shape,vdata.images.shape )

    vinputs = vinputs.to(device)
    vinputs = torchvision.transforms.Grayscale(num_output_channels=1)(vinputs)

    voutputs = model(vinputs)
    #print(type(vinputs),type(vdata.images) )
    vinputs = vinputs.to('cpu')
    voutputs = voutputs.cpu().detach()#.numpy()
  
    voutputs = torchvision.transforms.Grayscale(num_output_channels=1)(voutputs)

    #img = make_image_seq_strip([ ((1+vinputs[None, :])*(255/2.))] ,sep_val=255.0)#.astype(np.uint8)
    #cv2.imwrite("test_input.png", img[0].transpose(1, 2, 0))
    #print(voutputs[0]) #, vinputs[0])

    #voutputs = voutputs * 2 - 1
    #img = make_image_seq_strip([ ((1+voutputs[None, :])*(255/2.))] ,sep_val=255.0)#.astype(np.uint8)
    #cv2.imwrite("test_output.png", img[0].transpose(1, 2, 0))

    #cv2.imwrite("test_input_direct.png", vinputs[0] ) #.transpose(1,2,0))
    #cv2.imwrite("test_output_direct.png", voutputs[0]) #.transpose(1,2,0))
    display = list(voutputs[0]) + list(vinputs[0])
    display = torchvision.utils.make_grid(display,nrow=2)
    torchvision.utils.save_image(display, "ae_comp.png")
    display = torchvision.transforms.ToPILImage()(display)
    display = display.save("display.jpg") 
    #display.show()
    if i ==0:
        break
