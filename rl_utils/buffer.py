import torch
from  collections import deque
from torch.utils.data import Dataset
from rl_utils.traj_utils import EpisodeStep

#class trajBuffer(deque):
#    def __init__(self, maxlen =10):
#        super().__init__(maxlen=maxlen)
#            
#    def append_step(self, input_arr):
#        assert len(input_arr) == 4
#        self.append( input_arr )

#class replayBuffer(deque):
#    def __init__(self, maxlen ):
#        super().__init__(maxlen=maxlen)


class ReplayBuffer(deque):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)
    
class NpDataset(Dataset):
    def __init__(self, array):
        self.array = array
    def __len__(self): return len(self.array) 
    def __getitem__(self, i): return self.array[i]
    #def to(self, device):
    #    self.array = [ele.to(device) for ele in self.array]
    #    return self
    
    def to(self, device):
        # Move each element of the namedtuple to the device
        self.array = [EpisodeStep(*[ele.to(device) if torch.is_tensor(ele) else ele for ele in step]) for step in self.array]
        return self

def my_collate_fn(data):
    return tuple(data)

def extract_values_from_batch(batched_data, batch_size):
    # Check if the input is batched
    is_batched = (batch_size > 1)
    # Extract the required values from the batched data
    if is_batched:
        ipos_t = torch.stack([b[0] for b in batched_data]).to(torch.float32).detach()
        iaction = torch.stack([b[1] for b in batched_data]).detach()
        iaction_probas_old = torch.stack([b[2] for b in batched_data]).detach()
        ireward = torch.tensor([b[3] for b in batched_data]).to(torch.float32).detach()
        iadvantage = torch.stack([b[7] for b in batched_data]).detach()
        iret = torch.stack([b[6] for b in batched_data]).detach()
        ivalue = torch.stack([b[5] for b in batched_data]).detach()
    else:
        ipos_t = batched_data[0].unsqueeze(0).to(torch.float32)
        iaction = batched_data[1].unsqueeze(0).detach()
        iaction_probas_old = batched_data[2].unsqueeze(0).detach()
        ireward = torch.tensor(batched_data[3]).unsqueeze(0).to(torch.float32)
        iadvantage = batched_data[7].unsqueeze(0).detach()
        iret = batched_data[6].unsqueeze(0).detach()
        ivalue = batched_data[5].unsqueeze(0).detach()
    
    return ipos_t, iaction, iaction_probas_old, iadvantage, iret, ireward, ivalue