import torch
from  collections import deque
from torch.utils.data import Dataset

#class trajBuffer(deque):
#    def __init__(self, maxlen =10):
#        super().__init__(maxlen=maxlen)
#            
#    def append_step(self, input_arr):
#        assert len(input_arr) == 4
#        self.append( input_arr )

class replayBuffer(deque):
    def __init__(self, maxlen ):
        super().__init__(maxlen=maxlen)

class NpDataset(Dataset):
    def __init__(self, array):
        self.array = array
    def __len__(self): return len(self.array) 
    def __getitem__(self, i): return self.array[i]


def my_collate_fn(data):
    return tuple(data)

def extract_values_from_batch(batched_data, batch_size):
    # Check if the input is batched
    is_batched = (batch_size > 1)
    # Extract the required values from the batched data
    if is_batched:
        ipos_t = [b[0] for b in batched_data] 
        iaction = [b[1] for b in batched_data] 
        iaction_probas_old = [b[2] for b in batched_data] 
        ireward = [b[3] for b in batched_data] 
        iadvantage = [b[9] for b in batched_data] 
        iret = [b[8] for b in batched_data] 
    else:
        ipos_t = batched_data[0]
        iaction = batched_data[1]
        iaction_probas_old = batched_data[2]
        ireward = batched_data[3] 
        iadvantage = batched_data[9]
        iret = batched_data[8]
    
    ipos_t = torch.stack(ipos_t)
    ipos_t = ipos_t.to(torch.float32)
    
    return ipos_t, torch.stack(iaction).detach(), torch.stack(iaction_probas_old).detach(), torch.stack(iadvantage).detach(), torch.stack(iret).detach(), ireward

