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
        #layers.append(nn.Linear(out_channels,64))
        #layers.append(nn.BatchNorm1d(64))
        #layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 64))
        #layers.append(nn.Linear(64, 64))
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
        self.input_channels = input_channels  
        self.output_size = output_size 

        self.conv_layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
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
            nn.Tanh()  # Apply sigmoid activation to ensure output values are in [0, 1]
        )
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
##
#    def _create_decoder_layers(self):
#        layers = []
#        in_channels = self.input_channels  # Start with the input size of the decoder
#        out_channels = 64  # Initial number of output channels
#        # Add a linear layer to translate the input latent space into 1x1 feature map with 64 channels
#        #layers.append(nn.Linear(in_channels, 64))
#        #layers.append(nn.ReLU())
#        layers.append(nn.Unflatten(1, (self.input_channels, 1, 1)) ) # Reshape to 1x1 with input_channels channels
#        layers.append(nn.ReLU())
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
#        return nn.Sequential(*layers)




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
 

class Predictor(nn.Module):
    def __init__(self, input_channels, output_size, batch_size, n_cond_frames=3, n_pred_frames=25, lstm_output_size=64, n_layers_lstm=1, hidden_size=32):
        super(Predictor, self).__init__()

        self.lstm_output_size = lstm_output_size
        self.n_layers_lstm = n_layers_lstm
        self.n_cond_frames = n_cond_frames
        self.n_pred_frames = n_pred_frames
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv_encoder = ImageEncoder(input_channels, output_size).to(self.device)
        #self.fc = nn.Linear(output_size*3, self.lstm_output_size)  # from embedding to lstm 1 layer
        self.lstm = nn.LSTM(input_size=self.lstm_output_size, hidden_size=self.lstm_output_size, num_layers=self.n_layers_lstm, batch_first=True).to(self.device)
        
        self.mlp_enc = nn.Sequential(
            MLP3(output_size*n_cond_frames, output_size=self.lstm_output_size),
        ).to(self.device)

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

        h0 = torch.zeros(self.n_layers_lstm, self.batch_size, self.lstm_output_size).to(self.device) # Initial hidden state
        c0 = torch.zeros(self.n_layers_lstm, self.batch_size, self.lstm_output_size).to(self.device) # Initial cell state

        #print (output_sequence.shape)
        #print(output_sequence[:, 1, :].shape)
        #TODO fix range to end of traj/max len of traj
        #output_sequence = []
        outputs_list = []
        input_t = mlp_output

        for i_step in range(self.n_pred_frames):
            lstm_outstep, (h0, c0) = self.lstm(input_t, (h0, c0))
            outputs_list.append(h0[-1].unsqueeze(1))
            #without teacher forcign #TODO check if teacher forcing is nec.
            input_t = lstm_outstep

        # Concatenate predicted outputs along the sequence dimension = [nb, >ns<]
        #outputs = torch.cat(outputs_list, dim=1).squeeze()
        outputs = torch.stack(outputs_list, dim=1).squeeze(2)
        return outputs


class RewardPredictor(nn.Module):
    def __init__(self, n_pred_frames=25, n_heads=4, lstm_output_size=64 ):
        super(RewardPredictor, self).__init__()
        self.n_pred_frames = n_pred_frames
        #self.lstm_out = lstm_out
        self.n_heads = n_heads
        self.heads = nn.ModuleList([  nn.Sequential(MLP3(lstm_output_size, output_size=1), nn.Sigmoid()) for _ in range(self.n_heads)  ])

    def forward(self, x):

        batch_size, n_frames, *other_dims = x.size()
        outputs_list = []
        #for i_step in range(self.n_pred_frames):
        for frame_idx in range(n_frames):
            frame_output = []
            for head_idx in range(self.n_heads):
                head_output = self.heads[head_idx](x[:, frame_idx, ...])
                frame_output.append(head_output)
            outputs_list.append(torch.stack(frame_output, dim=1))  # Shape: (batch_size, n_heads, ...)   
        outputs = torch.stack(outputs_list, dim=1).squeeze()  # Shape: (batch_size, n_frames, n_heads, ...)

        return outputs
    

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride): 
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(input_channels, 16, kernel_size=kernel_size, stride=stride)
        self.layer2 = nn.Conv2d(16, 16, kernel_size=kernel_size, stride=stride)
        self.layer3 = nn.Conv2d(16, 16, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(16*(16-1)*3, 64)  #channel-1 xchannel x n_kernels
        self.fc2 = nn.Linear(64, 64) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        #Flatten
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

class Oracle(nn.Module):
    def __init__(self, input_size):
        super(Oracle,self).__init__()
        self.size = 32
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, self.size),
            nn.ReLU(),
            nn.Linear(self.size, self.size),
            nn.ReLU(),
        )
        self.value_out = nn.Sequential(nn.Linear(self.size, 1), nn.Sigmoid())
        self.action_out = nn.Sequential(nn.Linear(self.size, 9), nn.Softmax(dim=1))
        #self.tanh = nn.Tanh()

        #self.action_pred = nn.Linear(self.size, 9)
    def forward(self, x):
        x = self.layers(x)
        value = self.value_out(x)
        action = self.action_out(x)
        #action_x = self.tanh(x) #TODO check if this makes sense and converges or change to output over the discreet action space 1..9
        #action_y= self.tanh(x)
        return action, value

