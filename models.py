import torch.nn as nn
import math
import torch
from torch.distributions import MultivariateNormal

# helper function to init weights orthogonally 
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)



###############################################################
### MLP for arbitrary many MLP layers ###
###############################################################
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=32, num_layers=3, do_final_activ = False):
        super(MLP, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim))
        # Add a final activation if needed
        if do_final_activ:
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
    
############################################################### #TODO refactor in favor of the more general model above
### Three layer MLP, takes an input, output and hidden size ###
###############################################################
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

###############################################################
### Encoder ###
###############################################################

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
            layers.append(nn.BatchNorm2d(out_channels))  
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2
        out_channels /= 2
        out_channels = int(out_channels)
        # Additional convolutional layer to reduce height and width to 1x1
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))  
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(128, 64))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3: #taking care of unbatched case
            x = x.unsqueeze(0)
        return self.conv_layers(x)

###############################################################
### Decoder ###
###############################################################
class ImageDecoder(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ImageDecoder, self).__init__()
        self.input_channels = input_channels  
        self.output_size = output_size 

        self.conv_layers = nn.Sequential(
            #nn.Linear(64, 64),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
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
            nn.Tanh()  # activation
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
#            out_channels //= 2  # Half the number of channels
#        
#        # Add the final convolutional layer to generate the output image
#        layers.append(nn.ConvTranspose2d(in_channels, 1, kernel_size=4, stride=2, padding=1))
#        #layers.append(nn.Sigmoid())  
#        layers.append(nn.Tanh() )
#        return nn.Sequential(*layers)



###############################################################
### Autoencoder ###
###############################################################
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
    

 
###############################################################
### Predictor model  ###
###############################################################
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
        self.lstm = nn.LSTM(input_size = self.lstm_output_size,
                            hidden_size = self.lstm_output_size, 
                            num_layers = self.n_layers_lstm, 
                            batch_first = True).to(self.device)
        
        self.mlp_enc = nn.Sequential(
            MLP3(output_size , output_size = self.lstm_output_size),
            #MLP3(output_size * n_cond_frames, output_size = self.lstm_output_size),
        ).to(self.device)

    def forward(self, x):
        conv_embeddings =  [self.conv_encoder(x_timestep.squeeze(1)) for x_timestep in x.split(1, dim=1)]
        #merged_embedding = torch.cat( conv_embeddings[0:self.n_cond_frames], dim=1) #TODO fix for batch to dim 1?
        #TODO Fix this to take the frames sequentially in



        h0 = torch.zeros(self.n_layers_lstm, self.batch_size, self.lstm_output_size).to(self.device) # Initial hidden state
        c0 = torch.zeros(self.n_layers_lstm, self.batch_size, self.lstm_output_size).to(self.device) # Initial cell state

        #TODO fix range to end of traj/max len of traj
        #output_sequence = []
        outputs_list = []

        #condition with n conditioning frames
        for i_step in range( self.n_cond_frames):
            mlp_output = self.mlp_enc(conv_embeddings[i_step]) #TODO check if needs to be split in 3?
            mlp_output = mlp_output.unsqueeze(1) #TODO check 
            input_t = mlp_output

            lstm_outstep, (h0, c0) = self.lstm(input_t, (h0, c0))
            #outputs_list.append(h0[-1].unsqueeze(1))
            #without teacher forcign #TODO check if teacher forcing is nec.
            #input_t = lstm_outstep

        #then just roll
        for i_step in range(self.n_pred_frames):
            lstm_outstep, (h0, c0) = self.lstm(input_t, (h0, c0))
            outputs_list.append(h0[-1].unsqueeze(1))
            #without teacher forcign #TODO check if teacher forcing is nec.
            input_t = lstm_outstep

        # Concatenate predicted outputs along the sequence dimension = [nb, >ns<]
        outputs = torch.stack(outputs_list, dim=1).squeeze(2)
        return outputs
    


#    def forward(self, x):
#        conv_embeddings =  [self.conv_encoder(x_timestep.squeeze(1)) for x_timestep in x.split(1, dim=1)]
#        merged_embedding = torch.cat( conv_embeddings[0:self.n_cond_frames], dim=1) #TODO fix for batch to dim 1?
#        #TODO Fix this to take the frames sequentially in
#
#        mlp_output = self.mlp_enc(merged_embedding) #TODO check if needs to be split in 3?
#        mlp_output = mlp_output.unsqueeze(1) #TODO check 
#
#        h0 = torch.zeros(self.n_layers_lstm, self.batch_size, self.lstm_output_size).to(self.device) # Initial hidden state
#        c0 = torch.zeros(self.n_layers_lstm, self.batch_size, self.lstm_output_size).to(self.device) # Initial cell state
#
#        #TODO fix range to end of traj/max len of traj
#        #output_sequence = []
#        outputs_list = []
#        input_t = mlp_output
#
#        for i_step in range(self.n_pred_frames):
#            lstm_outstep, (h0, c0) = self.lstm(input_t, (h0, c0))
#            outputs_list.append(h0[-1].unsqueeze(1))
#            #without teacher forcign #TODO check if teacher forcing is nec.
#            input_t = lstm_outstep
#
#        # Concatenate predicted outputs along the sequence dimension = [nb, >ns<]
#        outputs = torch.stack(outputs_list, dim=1).squeeze(2)
#        return outputs


class RewardPredictor(nn.Module):
    def __init__(self, n_pred_frames=25, n_heads=4, lstm_output_size=64 ):
        super(RewardPredictor, self).__init__()
        self.n_pred_frames = n_pred_frames
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Sequential(MLP3(lstm_output_size, output_size=1), nn.Sigmoid()) for _ in range(self.n_heads)  ])


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
    





###############################################################
### Image-scratch model: use encoder with MLPs for AC ###
###############################################################
class ImgScratch(nn.Module):
    def __init__(self, encoder, obs_dim, action_space, action_std_init):
        super(ImgScratch).__init__()
        self.encoder = encoder
        self.input_size = obs_dim
        self.action_dim = action_space

        self.action_std = nn.Parameter(torch.ones(self.action_dim) * action_std_init, requires_grad=True)

        self.actor = nn.Sequential(MLP(64, self.action_dim, 32, 2, True),  nn.Tanh())
        self.value = nn.Sequential(MLP(64, 1, 32, 2, True))

    def forward(self, x):
        return self.encoder(x)


##############################################
### CNN for continous action spaces ###
##############################################
class CNN(nn.Module):
    def __init__(self, mlp_size=64, input_channels=1 , kernel_size=3, stride=2): 
        super(CNN, self).__init__()
        self.mlp_size = mlp_size
  
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(16),  # Add BatchNorm layer after Conv2d
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(16),  # Add BatchNorm layer after Conv2d
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(16),  # Add BatchNorm layer after Conv2d
            nn.ReLU(),
        )
        #self._initialize_flattened_size(input_channels, 64, 64)
        self.mlp_layers = nn.Sequential(
            nn.Linear(7*7*16, self.mlp_size ),
            #nn.Tanh(), # 
            nn.ReLU(),
            nn.Linear(self.mlp_size , self.mlp_size ),
            #nn.Tanh(), # 
            nn.ReLU(),
        )
        self.conv_layers.apply(init_weights)
        self.mlp_layers.apply(init_weights)
    
    def forward(self, x):
        # unbatched case, we unsqueeze the first dimenson to get a "batch" dim
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.conv_layers(x)
        # Flatten the output from the CNN keeping the batch size
        x = x.view(x.size(0), -1)  
        # Forward pass through MLP
        x = self.mlp_layers(x)
        return x
    

###############################################################
### Policy maker: given an encoder make it into a MimiPPOP  ###
###############################################################
class MimiPPOPolicy(nn.Module):
    def __init__(self, enc, obs_dim, action_space, action_std_init,encoder_output_size= 64, separate_layers=False):
        super(MimiPPOPolicy, self).__init__()

        self.encoder = enc
        self.encoder_output_size = encoder_output_size
        self.input_size = obs_dim
        self.action_dim = action_space

        self.action_std = nn.Parameter(torch.ones(self.action_dim) * action_std_init, requires_grad=True)

        self.critic_layers = nn.Sequential( nn.Linear(self.encoder_output_size , 1) )
        self.actor_layers = nn.Sequential( nn.Linear(self.encoder_output_size, self.action_dim), nn.Tanh())
        if separate_layers:
            self.actor_layers = nn.Sequential(MLP(self.encoder_output_size , self.action_dim, 32, 2),  nn.Tanh())
            self.critic_layers = nn.Sequential(MLP(self.encoder_output_size , 1, 32, 2))

        self.actor_layers.apply(init_weights)
        self.critic_layers.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        value = self.critic_layers(x)
        actor_output = self.actor_layers(x)
        action_cov = torch.diag(self.action_std)
        dist = MultivariateNormal(actor_output, action_cov)
        return dist, value
    
    #return action, action logs probabs and value
    def act(self, state, deterministic=False):
        dist, value = self(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        action_log_probs = dist.log_prob(action).sum(dim=-1)
        return action, action_log_probs, value

    #evaluating model for a given action
    def evaluate(self, state, action):
        dist, value = self(state)
        action_log_probs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        return action_log_probs, value, dist_entropy 



###################################################################
### MLP state space (Oracle) policy for continous action spaces ###
###################################################################
class Oracle(nn.Module):
    def __init__(self, input_dim, hidden_dim = 32):
        super(Oracle,self).__init__()
        self.size = hidden_dim
        self.input_size = input_dim
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_size, self.size),
            nn.Tanh(),
            nn.Linear(self.size, self.size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.shared_layers(x)


#########################################################
### MLP state spcae policy for discreet action spaces ###
#########################################################
class Oracle_discret(nn.Module):
    def __init__(self, input_size, n_classes):
        super(Oracle_discret,self).__init__()
        self.size = 32
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, self.size),
            nn.ReLU(),
            nn.Linear(self.size, self.size),
            nn.ReLU(),
        )
        self.value_out = nn.Sequential(nn.Linear(self.size, 1) )
        self.action_p = nn.Sequential(nn.Linear(self.size, n_classes), nn.Softmax(dim = 0))

    def forward(self, x):
        x = self.layers(x)
        value = self.value_out(x)
        action_p = self.action_p(x) 
        dim_axis=0
        if len(action_p.shape) == 2:
            dim_axis = 1
        else:
            dim_axis = 0

        action_v  = torch.argmax(action_p, dim=dim_axis)
        return action_v, action_p,  value


