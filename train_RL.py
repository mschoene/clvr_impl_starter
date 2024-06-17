import gym
import numpy as np
from general_utils import AttrDict
from sprites_env.envs.sprites import SpritesEnv 
from replayBuffer import *
from models import Oracle
from rl_utils.traj_utils import *
from rl_utils.ppo import PPO

data_spec = AttrDict(
        resolution=64,
        max_ep_len=400,
        max_speed=0.05,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        follow=True)

n_distractors = 0 
env = SpritesEnv(n_distractors = n_distractors)
env = gym.make('SpritesState-v0')
env.seed(1)
env.set_config(data_spec)

actions_space_std = 0.5 
Oracle_model = Oracle(env.observation_space.shape[0], env.action_space.shape[0], actions_space_std ) 


ppo_trainer = PPO(Oracle_model, env)
print("starting training ")
ppo_trainer.train()
print("done training")




#rollout = repBuf(-10, -1)
#obs = repBuf[-20][4]
#print(obs)
##white line between figures
#white_layer = np.ones((64,2), dtype=np.uint8) * 255
#
#for rolli in range(-20, -1): #rollout[1:]:
#        obs = np.concatenate((obs, white_layer), axis=1)
#        obs = np.concatenate((obs, repBuf[rolli][4]), axis = 1)
#cv2.imwrite("RL_train_test_oracle.png", 255* np.expand_dims(obs, -1))
#
#obs2 = repBuf[-50][4]
#for rolli in range(-50, -30): #rollout[1:]:
#        obs2 = np.concatenate((obs2, white_layer), axis=1)
#        obs2 = np.concatenate((obs2, repBuf[rolli][4]), axis = 1)
#cv2.imwrite("RL_train_test_oracle2.png", 255* np.expand_dims(obs2, -1))

#   obs = env.reset()
#    cv2.imwrite("test_rl.png", 255 * np.expand_dims(obs, -1))
#    obs, reward, done, info = env.step([0, 0])
#    cv2.imwrite("test_rl_1.png", 255 * np.expand_dims(obs, -1))

# actions taken
x_values = []
y_values = []

#print(repBuf[0][1][0])
for item in range(len(ppo_trainer.replayBuffer)):
    x, y = ppo_trainer.replayBuffer[item][1][0].tolist()
    x_values.append(x)
    y_values.append(y)
x_values = np.array(x_values)
y_values = np.array(y_values)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(x_values, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of X actions values')
plt.xlabel('X')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(y_values, bins=100, color='green', alpha=0.7)
plt.title('Histogram of Y actions values')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.tight_layout()

plt.savefig('histograms_actions.png')







x_values = []
y_values = []

#print(repBuf[0][1][0])
for item in range(len(ppo_trainer.replayBuffer)):
    x, y = ppo_trainer.replayBuffer[item][0][0][0:2].tolist()
    x_values.append(x)
    y_values.append(y)
x_values = np.array(x_values)
y_values = np.array(y_values)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(x_values, bins=100, color='blue', alpha=0.7)
plt.title('Histogram of X pos agent')
plt.xlabel('X')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(y_values, bins=100, color='green', alpha=0.7)
plt.title('Histogram of Y pos agent')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.tight_layout()

plt.savefig('histograms.png')

