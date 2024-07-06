import torch
import os

def np_to_torch(i_state):
    return torch.from_numpy(i_state).float().unsqueeze(0)

def get_averaged_tensor(i_tensor):
    return (i_tensor - i_tensor.mean()) / (i_tensor.std()+ 1e-5)

def load_pretrained_weights(model, pretrained_path):
    device = torch.device('cpu')
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    return model