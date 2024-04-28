import torch.nn as nn
import math
import torch

class ImageEncoder(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ImageEncoder, self).__init__()
        self.input_channels = input_channels
        self.output_size = output_size
        self.dropout_prob = 0.0
        self.conv_layers = self._create_conf_layers()

    def _create_conf_layers(self):
        layers = []
        in_channels = self.input_channels
        out_channels = 4
        for _ in range(int(math.log2(self.output_size))):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            #layers.append(nn.BatchNorm2d(out_channels))  # Add BatchNorm layer after Conv2d

            layers.append(nn.ReLU())
            #layers.append(nn.Tanh())
            #layers.append(nn.Dropout2d(self.dropout_prob))  # Add dropout layer
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2
        # Additional convolutional layer to reduce height and width to 1x1
        out_channels /= 2
        out_channels = int(out_channels)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(out_channels,128))
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
#
#    def forward(self, x):
#        for layer in self.conv_layers[:-1]:
#            x = layer(x)
#            #x = self.batchnorm(x) 
#            x = self.dropout(x)
#
#        #x = self.conv_layers[:-1](x)
#        #print("first " , x.shape)
#        x = x.view(x.size(0), -1)  # Flatten to 1D tensor
#        #print("sec " , x.shape)
#
#        x = self.conv_layers[-1](x)
#        #print("third  " ,x.shape)
#        return x


class ImageDecoder(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ImageDecoder, self).__init__()
        self.input_channels = input_channels  #64 # this is the laten space size
        self.output_size = output_size # 3 #this is the output of the decoder = input size into enc = image size
        #self.conv_layers = self._create_decoder_layers()

        #self.conv_layers = self._create_decoder_layers()
        self.dropout_prob = 0.15

        self.conv_layers = nn.Sequential(
            nn.Linear(64, 64 * 1 * 1),  # Linear layer to map from latent space to 4x4 feature map
            nn.ReLU(),
            nn.Unflatten(1, (64, 1, 1)),  # Reshape to 1x1 feature map with 64 channels
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: 
            nn.ReLU(),            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [-1, 32, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [-1, 16, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: [-1, 8, 32, 32]
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
#        #layers.append(nn.Tanh())
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
        #x = self.conv_layers[:-1](x)
        #x = x.view(x.size(0), -1)  # Flatten to 1D tensor
        #x = self.conv_layers[-1](x)
        for layer in self.conv_layers:
            x = layer(x)

        #x = x* 2 -1 #to map to -1 1 range
        #x = self.conv_layers(x)
#        print("dec 2 ", x.shape)

        #print("dec 3 ", x)
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
    
    
#input_channels = 3  # Assuming input image has 3 channels (RGB)
#output_size = 64  # Size of the output feature vector
#encoder = ImageEncoder(input_channels, output_size)
#deco = ImageDecoder(64, 3)

#print(encoder)
#<print(deco)


#### Test if this one works at least 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
            
        self.encoder = nn.Sequential(
            
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1), #1x64x64 -> 4x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), # to 8x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            

            #nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(8*8*8 , 64) 
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 1, 1)),  # Unflatten the input tensor
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output size: [32, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output size: [16, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # Output size: [8, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),    # Output size: [4, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),    # Output size: [1, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1),    # Output size: [1, 64, 64]
            nn.Tanh()
        )


    def forward(self, x):
        #print("init shape ", x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        return x
 
 
 

#3 layer MLP, takes an input output and hidden size
class MLP3(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=32):
        super(MLP3, self).__init__()
            
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),            
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
   
    def forward(self, x):
        return self.mlp_layers(x)
 