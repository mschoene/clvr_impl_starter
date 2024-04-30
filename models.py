import torch.nn as nn
import math
import torch

class ImageEncoder(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ImageEncoder, self).__init__()
        self.input_channels = input_channels
        self.output_size = output_size
        self.dropout_prob = 0.15
        self.conv_layers = self._create_conf_layers()

    def _create_conf_layers(self):
        layers = []
        in_channels = self.input_channels
        out_channels = 4
        for _ in range(int(math.log2(self.output_size))):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))  # Add BatchNorm layer after Conv2d
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout2d(self.dropout_prob))  # Add dropout layer
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2
        # Additional convolutional layer to reduce height and width to 1x1
        out_channels /= 2
        out_channels = int(out_channels)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))  # Add BatchNorm layer after Conv2d
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(out_channels,128))
        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 64))
        return nn.Sequential(*layers)
    
 #       self.conv_layers = nn.Sequential(
 #           nn.Conv2d(3, 4, kernel_size=4, stride=2, padding=1),
 #           nn.ReLU(),
 #           nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
 #           nn.ReLU(),
 #           nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
 #           nn.ReLU(),
 #           nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
 #           nn.ReLU(),
 #           nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
 #           nn.ReLU(),
 #           nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
 #           nn.ReLU(),
 #           nn.Flatten(),
 #           nn.Linear(128, 64)
 #       )

    def forward(self, x):
        return self.conv_layers(x)
        
#        self.conv_layers = self._create_conf_layers()
#        #self.batchnorm = nn.BatchNorm2d(num_features=output_size)  # Adjust the number of features
#        self.dropout = nn.Dropout(p=0.5)  # Adjust the dropout probability
#
#    def _create_conf_layers(self):
#        layers = []
#        in_channels = self.input_channels
#        out_channels = 4 #init
#        for _ in range(int(math.log2(self.output_size))):
#            layers.append(nn.Conv2d(  in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding = 1))
#            layers.append(nn.ReLU())
#            in_channels = out_channels
#            out_channels *= 2  # Double the number of channels
#        out_channels /= 2  # undo the last double for output layer
#        layers.append(nn.Linear(int(out_channels), self.output_size) )
#        return nn.Sequential(*layers)
#


class ImageDecoder(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ImageDecoder, self).__init__()
        self.input_channels = input_channels  #64 # this is the laten space size
        self.output_size = output_size # 3 #this is the output of the decoder = input size into enc = image size
        #self.conv_layers = self._create_decoder_layers()

        #self.conv_layers = self._create_decoder_layers()
        self.dropout_prob = 0.15

        self.conv_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            #nn.ReLU(),
            #nn.Linear(64, 64 * 1 * 1),  # Linear layer to map from latent space to 4x4 feature map
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 1, 1)),  # Reshape to 1x1 feature map with 64 channels
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: 
            nn.BatchNorm2d(64),  # Add BatchNorm layer after Conv2d
            nn.ReLU(),            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [-1, 32, 8, 8]
            nn.BatchNorm2d(32),  # Add BatchNorm layer after Conv2d
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [-1, 16, 16, 16]
            nn.BatchNorm2d(16),  # Add BatchNorm layer after Conv2d
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [-1, 8, 32, 32]
            nn.BatchNorm2d(8),  # Add BatchNorm layer after Conv2d
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [-1, 4, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [-1, 3, 64, 64]
            #nn.Sigmoid()  # Apply sigmoid activation to ensure output values are in [0, 1]
            nn.Tanh()  # Apply sigmoid activation to ensure output values are in [0, 1]
        )

#        self.conv_layers = nn.Sequential(
#            nn.Unflatten(1, (self.input_channels, 1, 1)), # Reshape to 1x1 with input_channels channels #nn.Unflatten(dim=1, unflattened_size=(64, 1, 1)),
#            nn.ReLU(),
#            #nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=0),
#            nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),            
#            nn.ReLU(),
#            nn.Dropout2d(self.dropout_prob),  # Add dropout layer
#            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
#            nn.ReLU(),
#            nn.Dropout2d(self.dropout_prob),  # Add dropout layer
#            nn.ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
#            nn.ReLU(),
#            nn.Dropout2d(self.dropout_prob),  # Add dropout layer
#            nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
#            nn.ReLU(),
#            nn.Dropout2d(self.dropout_prob),  # Add dropout layer
#            nn.ConvTranspose2d(8, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
#            nn.ReLU(),
#            nn.Dropout2d(self.dropout_prob),  # Add dropout layer
#            nn.ConvTranspose2d(4, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
#            nn.Sigmoid()
#        )
##
#    def _create_decoder_layers(self):
#        layers = []
#        in_channels = self.input_channels  # Start with the input size of the decoder
#        out_channels = 64  # Initial number of output channels
#        
#        # Add a linear layer to translate the input latent space into 1x1 feature map with 64 channels
#        #layers.append(nn.Linear(in_channels, 64))
#        #layers.append(nn.ReLU())
#        layers.append(nn.Unflatten(1, (self.input_channels, 1, 1)) ) # Reshape to 1x1 with input_channels channels
#        layers.append(nn.ReLU())
#        
#        while out_channels > 3:  # Adjusted condition for decoder
#            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
#            #layers.append(nn.ReLU())
#            layers.append(nn.Tanh())
#            in_channels = out_channels  # Set the number of input channels for the next layer
#            out_channels //= 2  # Halve the number of channels
#        
#        # Add the final convolutional layer to generate the output image
#        layers.append(nn.ConvTranspose2d(in_channels, 1, kernel_size=4, stride=2, padding=1))
#        #layers.append(nn.Sigmoid())  
#        layers.append(nn.Tanh() )
#
#        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # Encode the input image
        latent_space = self.encoder(x)
        # Decode the latent space representation
        reconstructed_image = self.decoder(latent_space)
        #reconstructed_image = torch.sigmoid(reconstructed_image)
        # Apply thresholding to make the values either -1 or 1
        #reconstructed_image = torch.where(reconstructed_image > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
        return reconstructed_image
    
 

#3 layer MLP, takes an input, output and hidden size
class MLP3(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=32):
        super(MLP3, self).__init__()
            
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),        
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
   
    def forward(self, x):
        return self.mlp_layers(x)
 


n_output_steps = 5
lstm_output_size = 32
n_layers_lstm = 1


class RewardPredictor(nn.Module):
    def __init__(self, input_channels, output_size, batch_size, n_cond_frames=3, n_pred_frames=25, lstm_output_size=32, n_layers_lstm=1, hidden_size=32):
        super(RewardPredictor, self).__init__()

        self.lstm_output_size = lstm_output_size
        self.n_layers_lstm = n_layers_lstm
        self.n_cond_frames = n_cond_frames
        self.n_pred_frames = n_pred_frames
        self.batch_size = batch_size
        self.conv_encoder = ImageEncoder(input_channels, output_size)
        #self.fc = nn.Linear(output_size*3, self.lstm_output_size)  # from embedding to lstm 1 layer
        self.lstm = nn.LSTM(input_size=self.lstm_output_size, hidden_size=self.lstm_output_size, num_layers=self.n_layers_lstm, batch_first=True)
        
        self.mlp_enc = nn.Sequential(
            MLP3(output_size*n_cond_frames, output_size=self.lstm_output_size),
        )        
        
        #TODO one per time step and reward head, loop in forward?
        self.mlp_head1 = nn.Sequential(
            MLP3(lstm_output_size, output_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #use n conditioning frames. here we are just encoding all of them but technially only need the first 3.. TODO fix to only 3 frames, check that this doesn't mess up update
        conv_embeddings =  [self.conv_encoder(x_timestep.squeeze(1)) for x_timestep in x.split(1, dim=1)]

        # Concat embeds before feeding into mlp
        #for i_cond_frame in self.n_cond_frames:  
        #merged_embedding = torch.cat( (conv_embeddings[0], conv_embeddings[1], conv_embeddings[2]), dim=1) #TODO fix for batch to dim 1?
        merged_embedding = torch.cat( conv_embeddings[0:self.n_cond_frames], dim=1) #TODO fix for batch to dim 1?
        #print("conv is leaf ", merged_embedding.is_leaf)

        #detached_embeddings = [embedding.detach() for embedding in conv_embeddings]
            #       Concatenate detached tensors along dim=1
        #merged_embedding = torch.cat(detached_embeddings, dim=1)

        #print("merged shape ", merged_embedding.shape)
        # pass into MLP
        mlp_output = self.mlp_enc(merged_embedding) #TODO check if needs to be split in 3?
        #print("mlp size ", mlp_output.shape)
        mlp_output = mlp_output.unsqueeze(1) #TODO check 

        h0 = torch.zeros(n_layers_lstm, self.batch_size, lstm_output_size) # Initial hidden state
        c0 = torch.zeros(n_layers_lstm, self.batch_size, lstm_output_size) # Initial cell state

        #print (output_sequence.shape)
        #print(output_sequence[:, 1, :].shape)
        #TODO fix range to end of traj/max len of traj
        #output_sequence = []
        outputs_list = []
        input_t = mlp_output


        #print("mlp still leaf ", mlp_output.is_leaf) #, mlp_output.grad )
        #print(mlp_output.size())
        for i_step in range(self.n_pred_frames):
            #output, (h0, c0)  = self.lstm(mlp_output, (h0, c0))
            #print("output itself is leaf ", outputs.is_leaf)
            #print(output)
            #output_sequence.append(output)

            lstm_outstep, (h0, c0) = self.lstm(input_t, (h0, c0))

            #reward predition head 1 #TODO add others/make it a larger tensor of dimension n_reward_heads
            output_t = self.mlp_head1(h0[-1]) 

            outputs_list.append(output_t.unsqueeze(1))
            #without teacher forcign #TODO check if teacher forcing is nec.
            input_t = lstm_outstep
            #print("lstm output size ", lstm_outstep.shape)
            # Update input for next prediction (use predicted output)
            #input_t = output_t#.unsqueeze(1)

        # Concatenate predicted outputs along the sequence dimension
        outputs = torch.cat(outputs_list, dim=1).squeeze()
        #outputs.squeeze()
        #outputs.retain_grad()
        #print(outputs.grad)
        return outputs