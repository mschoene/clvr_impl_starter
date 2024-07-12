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
from rl_utils.torch_utils import load_pretrained_weights


#Freeze params with False
def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad

sweep_config = {
    'name': 'cnn_sweep_kldiv',
    'method': 'bayes',  # You can also use 'random' or 'bayes'
    'metric': {
        'name': 'average_reward_eval',
        'goal': 'maximize'
    },
    'parameters': {
        'n_actors': {
            'values': [4, 6]
        },      
        'n_traj': {
            'values': [10,12] 
        },
        'kl_coef': {
            'values': [0.2, 0.0]
        },       
        'ent_coef': {
            'values': [0.01, 0.0]
        },
        'clip_param': {
            'values': [0.05, 0.1, 0.2]
        },     
        'learning_rate_enc': {
            'values': [ 0.0003, 0.0001, 0.00001, 0.000001 ]
        },
        'kl_target': {
            'values': [0.01, 0.05, 0.1]  # Add the desired values for kl_target here
        }
        # 'seed': {
        #     'values': [0, 1, 2, 3]
        # }
    }
}


def generate_run_name(baseline, num_distractors, name_ext):
    if name_ext:
        return f"{baseline}_d_{num_distractors}_{name_ext}"
    else:
        return f"{baseline}_d_{num_distractors}"


def train(args):

    if args.do_sweep:
        wandb.init(project='clvr_starter')

    if args.do_wandb_exp:
        run_name = generate_run_name(args.model_name.lower() , args.n_distractors, args.name_extension)
        wandb.init(project="clvr_starter", name=run_name, config={
        "baseline": args.model_name.lower(),
        "num_distractors": args.n_distractors,
        })
    
    actions_space_std =  args.action_std_init 
    ent_coef = args.ent_coef
    minibatch_size = args.minibatch_size

    model_name = args.model_name.lower()
    policy_input_dim = 64
    env_name = f'Sprites-v{args.n_distractors}'
    separate_ac_mlps = args.sep_ac 
    ac_hidden_layers = args.n_ac_hl
    ac_hidden_size = 32
    epsilon = args.epsilon
    learning_rate_enc = args.learning_rate_enc
    learning_rate = args.learning_rate 
    final_log_std = args.action_std_final
    n_traj = args.n_traj
    n_actors = args.n_actors
    std_coef = args.std_coef
    vf_coef = args.vf_coef
    kl_target = args.kl_target
    seed = 0

    # Get hyperparameters from wandb.config
    if args.do_sweep:
        #minibatch_size = wandb.config.batch_size
        epsilon = wandb.config.clip_param
        #learning_rate = wandb.config.learning_rate
        learning_rate_enc = wandb.config.learning_rate_enc
        n_traj = wandb.config.n_traj
        n_actors = wandb.config.n_actors
        kl_coef = wandb.config.kl_coef
        kl_target = wandb.config.kl_target
        ent_coef = wandb.config.ent_coef
        #seed = wandb.config.seed
        #std_coef = wandb.config.std_coef
        #separate_ac_mlps = wandb.config.do_sep_ac
        #ac_hidden_layers = wandb.config.ac_hidden_layers
        #kl_target = getattr(wandb.config, 'kl_target', 0.0)
    
    if model_name == 'oracle':
        #ent_coef = 0.05
        #minibatch_size = 64
        separate_ac_mlps = False
        env_name = f'SpritesState-v{args.n_distractors}'
        env = gym.make(env_name)
        encoder = Oracle(env.observation_space.shape[0])
        policy_input_dim = 32

    elif model_name == 'cnn':
        encoder = CNN() 
        ac_hidden_layers = 1 
        ac_hidden_size = 64
        separate_ac_mlps = False

    elif model_name == 'img':
        encoder = ImageEncoder(1, 64)
        separate_ac_mlps = True


    elif model_name =="enc": #pretrained encoder, frozen
        if args.n_distractors==0:
            pretrained_path = "models/encoder_model_2obj_20240708_223549_149"
        elif args.n_distractors==1:
            pretrained_path = "models/encoder_model_2obj_nDistr_1_20240709_134803_150"
        else:
            pretrained_path = "models/encoder_model_2obj_nDistr_2_20240709_165324_150" 
        encoder = ImageEncoder(1, 64)
        encoder = load_pretrained_weights(encoder, pretrained_path)
        separate_ac_mlps = True
        set_parameter_requires_grad(encoder, requires_grad=False)
    elif model_name =="enc_ft": #pretrained encoder, fine tuning
        if args.n_distractors==0:
            pretrained_path = "models/encoder_model_2obj_20240708_223549_149"
        elif args.n_distractors==1:
            pretrained_path = "models/encoder_model_2obj_nDistr_1_20240709_134803_150"
        else:
            pretrained_path = "models/encoder_model_2obj_nDistr_2_20240709_165324_150"
        encoder = ImageEncoder(1, 64)
        encoder = load_pretrained_weights(encoder, pretrained_path)
        separate_ac_mlps = True
        set_parameter_requires_grad(encoder, requires_grad=True)


    elif model_name == "repr": # pretrained representation encoder
        if args.n_distractors == 0:
            pretrained_path = "models/repr_encoder_full_nDistr_0_doPre_0_model_epoch_500_20240708_152431"
        elif args.n_distractors ==1:
            pretrained_path = "models/repr_encoder_full_nDistr_1_doPre_0_model_epoch_500_20240709_002237"
        elif args.n_distractors ==2:
            pretrained_path = "models/repr_encoder_full_nDistr_2_doPre_0_model_epoch_500_20240709_020827"
        else:
            pretrained_path = "models/repr_encoder_full_nDistr_3_doPre_0_model_epoch_500_20240709_033751"
        encoder = ImageEncoder(1, 64)
        encoder = load_pretrained_weights(encoder, pretrained_path)
        separate_ac_mlps = True
        set_parameter_requires_grad(encoder, requires_grad=False)

    elif model_name =="repr_ft": #pretrained representation encoder, fine tuning
        if args.n_distractors == 0:
            pretrained_path = "models/repr_encoder_full_nDistr_0_doPre_0_model_epoch_500_20240708_152431"
        elif args.n_distractors ==1:
            pretrained_path = "models/repr_encoder_full_nDistr_1_doPre_0_model_epoch_500_20240709_002237"
        elif args.n_distractors ==2:
            pretrained_path = "models/repr_encoder_full_nDistr_2_doPre_0_model_epoch_500_20240709_020827"
        else:
            pretrained_path = "models/repr_encoder_full_nDistr_3_doPre_0_model_epoch_500_20240709_033751"
        encoder = ImageEncoder(1, 64)
        encoder = load_pretrained_weights(encoder, pretrained_path)
        separate_ac_mlps = True
        set_parameter_requires_grad(encoder, requires_grad=True)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)

    model = MimiPPOPolicy(enc=encoder, 
                          obs_dim = env.observation_space.shape[0], 
                          action_space = env.action_space.shape[0], 
                          action_std_init = actions_space_std, 
                          encoder_output_size = policy_input_dim,
                          separate_layers = separate_ac_mlps,
                          hidden_layer_dim = ac_hidden_size,
                          num_hidden_layers = ac_hidden_layers
                          )
    ppo_trainer = MimiPPO(model, 
                          model_name, 
                          env, 
                          env_name, 
                          std_coef=std_coef, 
                          ent_coef=ent_coef, 
                          kl_coef=kl_coef,
                          kl_target = kl_target,
                          minibatch_size=minibatch_size,
                          n_episodes=args.n_episodes,
                          max_env_steps = args.n_env_steps,
                          vf_coef=vf_coef,
                          epsilon=epsilon,
                          lr=learning_rate,
                          lr_enc=learning_rate_enc,
                          final_log_std=final_log_std,
                          n_trajectories=n_traj,
                          off_poli_factor=args.off_p_frac,
                          n_actors=n_actors,
                          do_wandb=(args.do_sweep or args.do_wandb_exp)
                          )

    #ppo_trainer = MimiPPO( model, env)
    #ppo_trainer = MimiPPO( model, env, std_coef=0.2, ent_coef= 0.0015 ,  minibatch_size=128)

    start = time.time()
    print("=========== starting training ===========")
    ppo_trainer.train()
    print("===========   done training   ===========")
    end = time.time()
    print("finished training in ", end- start , " seconds")


def main(args):

    if args.do_sweep:
        def sweep_train():
            #config = wandb.config
            #wandb.init(project='clvr_starter', config=config)  # Initialize WandB inside the sweep_train function
            train(args)

        #sweep_id = wandb.sweep(sweep_config, project='clvr_starter')

        #print(sweep_id)
        sweep_id = 'jd18pzt5' # '8vibecax' # 'sh3m6vfo' #'lzeyi0wd' # 'wvekjp8m' #'7nfe067v'
        wandb.agent(sweep_id, function=sweep_train, project="clvr_starter")
        #wandb.agent(sweep_id, function=sweep_train)
    else:
        train(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MimiPPO model with specified parameters.")
    parser.add_argument('--model_name', type=str, required=True, help="Model name to use ('oracle', 'cnn', 'enc', 'enc_ft', 'repr', 'repr_ft').")

    parser.add_argument('--std_coef', type=float, default=1., help="Standard deviation coefficient.")
    parser.add_argument('--ent_coef', type=float, default=0.0, help="Entropy coefficient.")
    parser.add_argument('--vf_coef', type=float, default=0.5, help="Value function coefficient.")
    parser.add_argument('--kl_coef', type=float, default=0.2, help="KL divergence coefficient.")
    parser.add_argument('--kl_target', type=float, default=0.01, help="KL divergence target.")

    parser.add_argument('--minibatch_size', type=int, default=128, help="Minibatch size.")
    parser.add_argument('--n_distractors', type=int, choices=range(4), default=0, help="Number of distractors (0 to 3).")
    parser.add_argument('--n_episodes', type=int, default=500, help="Number of episodes, default 500.")
    parser.add_argument('--n_env_steps', type=int,  default=5000000, help="Number of episodes, default 5M.")
    parser.add_argument('--sep_ac', type=int,  default=1, help="0/1  separate policy & value networks")
    parser.add_argument('--action_std_init', type=float, default=-0.7, help="Initial log action standard deviation.")
    parser.add_argument('--action_std_final', type=float, default=0.0, help="Initial log action standard deviation.")
    parser.add_argument('--epsilon', type=float, default=0.2, help="Advantage clipping factor.")
    parser.add_argument('--n_traj', type=int, default=10, help="Number trajectories to sample.")

    parser.add_argument('--learning_rate', type=float, default=0.0003, help="Learning rate")
    parser.add_argument('--learning_rate_enc', type=float, default=0.0003, help="Learning rate for the encoder. Set this if using a pretrained encoder.")

    parser.add_argument('--off_p_frac', type=float, default=1.0, help="Fraction of off policy events to keep in replay buffer, >1 means going off policy. Warning this currently does currently not have any correction applied!")
    parser.add_argument('--n_actors', type=int, default=4, help="Number of actors ")
    
    parser.add_argument('--n_ac_hl', type=int, default=2, help="Number of hidden layers in MLP AC feature extractor ")

    parser.add_argument('--do_sweep', type=bool, default=False, help="Do a HP sweep")
    parser.add_argument('--do_wandb_exp', type=bool, default=False, help="Do wandb experiment and logging")

    parser.add_argument('--name_extension', type=str, default="", help="Add a descriptive name to the output files and run name")

    args = parser.parse_args()

    main(args)
