import argparse
import os
# workaround to unpickle olf model files
import sys
import torch
import numpy as np
from utils.envs import make_vec_envs
from utils.utils import get_render_func, get_vec_normalize, MMD
import yaml
from models.model import Policy
from models import gail
import utils.utils
import mj_envs

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--expert_dir',
    help='path of experiment')
parser.add_argument(
    '--experiment_dir',
    help='path of experiment')
parser.add_argument(
    '--det',
    action='store_true',
    default=True,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--render',
    action='store_true',
    default=False,
    help='whether to render the rollouts')
parser.add_argument(
    '--save_expert',
    action='store_true',
    default=True,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--num_trajs',
    default=4,
    help='number of trajectories for expert data')

args = parser.parse_args()
#exp_name = args.experiment_dir
exp_name = "/home/kit/anthropomatik/kn6273/Repos/logs/training_06.11/training_door_4/ppo/door-v0/gailTrue-lr_p3.00E-04-lr_v1.00E-03-lr_d3.00E-04-penFalse-g_clipTrue-target_entropy0-s0-batch_size512-clip_irTrue-projFalse-p_typeNone-lr_decTrue-clip_vTrue-n_oTrue-n_rTrue-c_o1.00E+01-c_r1.00E+01cov0.00E+00mean0.00E+00"
load_dir = exp_name + '/models/'
with open(exp_name + '/args.yml') as f:
    # use safe_load instead load
    args_dict = yaml.safe_load(f)

env = make_vec_envs(
    args_dict['env_name'],
    args_dict['seed'],
    1,
    None,
    log_dir=args_dict['log_dir'],
    norm_obs=args_dict['norm_obs'],
    norm_reward=args_dict['norm_reward'],
    clip_obs=args_dict['clip_obs'],
    clip_reward=args_dict['clip_reward'],
    device='cpu',
    allow_early_resets=True)
np.random.seed(args_dict['seed'])
# Get a render function
render_func = None
if args.render:
    render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, obs_rms = \
    torch.load(os.path.join(load_dir, args_dict['env_name'] + ".pt"),
               map_location='cpu')

vec_norm = get_vec_normalize(env)
device = torch.device("cpu")

if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

obs = env.reset()

eval_episode_rewards = []

observations = []
actions = []
rewards = []

episode_observations = None
episode_actions = None
episode_rewards = None

while len(eval_episode_rewards) < int(args.num_trajs):
    with torch.no_grad():
        value, action, dist = actor_critic.act(
            obs, deterministic=args.det)
        
    if args.save_expert:
        # need to add unnormalized observation to the expert data
        original_obs = torch.from_numpy(env.get_original_obs())
        
        if episode_observations is None:
            episode_observations = original_obs[:,0:3]
            episode_actions = action[:,0:3]
        else:
            episode_observations = torch.cat([episode_observations, original_obs[:,0:3]], dim=0)
            episode_actions = torch.cat([episode_actions, action[:,0:3]], dim=0)

    # Obser reward and next obs
    obs, reward, done, infos = env.step(action)

    if args.save_expert:
        if episode_rewards is None:
            episode_rewards = reward
        else:
            episode_rewards = torch.cat([episode_rewards, reward], dim=1)

    
    
    if args.save_expert:
        for info in infos:
            if 'episode' in info.keys():
                print(info['episode'])
                eval_episode_rewards.append(info['episode']['r'])                
                observations.append(episode_observations)
                actions.append(episode_actions)
                
                episode_observations = None
                episode_actions = None
                episode_rewards = None

observations = torch.cat(observations, dim=0)
actions = torch.cat(actions, dim=0)
#rewards = torch.cat(rewards, dim=0)

file_name = args_dict['gail_experts_dir'] + args_dict['env_name'] + '_num_traj_' + str(4) + '.pt'
tracking_expert_dataset = gail.ExpertDataset(file_name, num_trajectories=4, subsample_frequency=1, tracking=True)
expert_trajs = tracking_expert_dataset.get_traj()

expert_observations = []
expert_actions = []
for traj in range(4):
    expert_observations.append(expert_trajs['states'][traj][:,0:3])
    expert_actions.append(expert_trajs['actions'][traj][:,0:3])

expert_observations = torch.cat(expert_observations, dim=0).type(torch.DoubleTensor)
expert_actions = torch.cat(expert_actions, dim=0).type(torch.DoubleTensor)
expert_state_actions = torch.cat([expert_observations, expert_actions], dim=1)

observations = observations.type(torch.DoubleTensor)
actions = actions.type(torch.DoubleTensor)
state_actions = torch.cat([observations, actions], dim=1)

print(MMD(expert_observations, observations, "rbf"))
print(MMD(expert_actions, actions, "rbf"))
print(MMD(expert_state_actions, state_actions, "rbf"))

print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
    len(eval_episode_rewards), np.mean(eval_episode_rewards)))
