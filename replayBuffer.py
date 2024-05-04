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


