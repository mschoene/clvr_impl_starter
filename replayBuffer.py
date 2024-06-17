import torch.optim as optim
import gym
import numpy as np
import cv2
import torch
from general_utils import AttrDict
from sprites_env.envs.sprites import SpritesEnv
import torch.nn as nn
import torch.nn.functional as F
from  collections import deque, namedtuple
import itertools


class trajBuffer(deque):
        def __init__(self, maxlen =10):
               super().__init__(maxlen=maxlen)
                
        def append_step(self, input_arr):
                assert len(input_arr) == 4
                self.append( input_arr )

class replayBuffer(deque):
        def __init__(self, maxlen ):
                super().__init__(maxlen=maxlen)




def torch_pos(i_state):
        pos, _ = np.split(i_state, 2, -1)
        pos = pos.flatten()
        return torch.from_numpy(pos).float().unsqueeze(0)



data_spec = AttrDict(
    resolution=64,
    max_ep_len=40,
    max_speed=0.05,      # total image range [0, 1]
    obj_size=0.2,       # size of objects, full images is 1.0
    follow=True,
)
#env = SpritesEnv()
#env.set_config(data_spec)
#obs = env.reset()
#cv2.imwrite("test_rl.png", 255 * np.expand_dims(obs, -1))
#obs, reward, done, info = env.step([0, 1])
#cv2.imwrite("test_rl_1.png", 255 * np.expand_dims(obs, -1))
#state = env._state
#pos = torch_pos(state)
#print(env._state)
#print(pos)