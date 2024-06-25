import gym
import numpy as np
from general_utils import AttrDict
from sprites_env.envs.sprites import SpritesEnv 
from replayBuffer import *
from models import Oracle, CNN, MimiPPOPolicy, ImageEncoder
from rl_utils.traj_utils import *
from rl_utils.ppo import MimiPPO
import argparse
import torch
import wandb
import time

def load_pretrained_weights(model, pretrained_path):
    device = torch.device('cpu')
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    return model

#Freeze params with True
def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

#wandb.init(
#    project="clvr_starter",
#    config={
#        "dataset": "sprites",
#    })

def main(args):

    actions_space_std = 0.5
    ent_coef = args.ent_coef
    minibatch_size = args.minibatch_size

    model_name = args.model_name.lower()
    policy_input_dim = 64
    env_name = f'Sprites-v{args.n_distractors}'
    separate_ac_mlps = False

    if model_name == 'oracle':
        ent_coef = 0.05
        minibatch_size = 64
        env_name = f'SpritesState-v{args.n_distractors}'
        env = gym.make(env_name)
        encoder = Oracle(env.observation_space.shape[0])
        policy_input_dim =32
    elif model_name == 'cnn':
        ent_coef=0.0005
        encoder = CNN()
        #policy = MimiPPOPolicy(encoder=cnn_encoder, obs_dim=obs_dim, action_space=action_space, action_std_init=action_std_init, encoder_output_size=encoder_output_size)
    elif model_name == 'img':
        encoder = ImageEncoder(1, 64)
        separate_ac_mlps = True
        ent_coef=0.0005 


    elif model_name =="enc": #pretrained encoder, frozen
        pretrained_path = "models/encoder_model_2obj_20240620_153556_299"
        encoder = ImageEncoder(1, 64)
        encoder = load_pretrained_weights(encoder, pretrained_path)
        separate_ac_mlps = True
        ent_coef=0.0006 
        set_parameter_requires_grad(encoder, requires_grad=False)
    elif model_name =="enc_ft": #pretrained encoder, fine tuning
        pretrained_path = "models/encoder_model_2obj_20240620_153556_299"
        encoder = ImageEncoder(1, 64)
        encoder = load_pretrained_weights(encoder, pretrained_path)
        separate_ac_mlps = True
        ent_coef=0.0005
        set_parameter_requires_grad(encoder, requires_grad=True)


    elif model_name == "repr": # pretrained representation encoder
        pretrained_path = "models/repr_encoder_full_model_epoch_499_20240622_063255"
        encoder = ImageEncoder(1, 64)
        encoder = load_pretrained_weights(encoder, pretrained_path)
        separate_ac_mlps = True
        ent_coef=0.0005   
        set_parameter_requires_grad(encoder, requires_grad=False)

    elif model_name =="repr_ft": #pretrained representation encoder, fine tuning
        pretrained_path = "models/repr_encoder_full_model_epoch_499_20240622_063255"
        encoder = ImageEncoder(1, 64)
        encoder = load_pretrained_weights(encoder, pretrained_path)
        separate_ac_mlps = True
        #ent_coef=0.0001 
        set_parameter_requires_grad(encoder, requires_grad=True)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    env = gym.make(env_name)
    env.seed(1)

    #model = model_cls(env.observation_space.shape[0], env.action_space.shape[0], actions_space_std)
    model = MimiPPOPolicy(enc=encoder, 
                          obs_dim=env.observation_space.shape[0], 
                          action_space=env.action_space.shape[0], 
                          action_std_init=actions_space_std, 
                          encoder_output_size=policy_input_dim,
                          separate_layers=separate_ac_mlps,
                          )
    ppo_trainer = MimiPPO(model, 
                          model_name, 
                          env, 
                          env_name, 
                          std_coef=args.std_coef, 
                          ent_coef=ent_coef, 
                          minibatch_size=minibatch_size,
                          n_episodes=args.n_episodes,
                          max_env_steps = args.n_env_steps                     
                          )

    #ppo_trainer = MimiPPO( model, env)
    #ppo_trainer = MimiPPO( model, env, std_coef=0.2, ent_coef= 0.0015 ,  minibatch_size=128)

    start = time.time()
    print("=========== starting training ===========")
    ppo_trainer.train()
    print("===========   done training   ===========")
    end = time.time()
    print("finished training in ", end- start , " seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MimiPPO model with specified parameters.")
    parser.add_argument('--model_name', type=str, required=True, help="Model name to use ('oracle', 'cnn', 'enc', 'enc_ft', 'repr', 'repr_ft').")
    parser.add_argument('--std_coef', type=float, default=0.2, help="Standard deviation coefficient.")
    parser.add_argument('--ent_coef', type=float, default=0.0015, help="Entropy coefficient.")
    parser.add_argument('--minibatch_size', type=int, default=128, help="Minibatch size.")
    parser.add_argument('--n_distractors', type=int, choices=range(4), default=0, help="Number of distractors (0 to 3).")
    parser.add_argument('--n_episodes', type=int, choices=range(5000), default=500, help="Number of episodes, default 500.")
    parser.add_argument('--n_env_steps', type=int,  default=5000000, help="Number of episodes, default 5M.")

    args = parser.parse_args()
    #wandb.config.update(args)

    main(args)


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

def make_histos(ppo_trainer):
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

