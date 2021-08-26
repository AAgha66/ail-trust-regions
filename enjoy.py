import argparse
import os
# workaround to unpickle olf model files
import sys
import torch
import numpy as np
from utils.envs import make_vec_envs
from utils.utils import get_render_func, get_vec_normalize
import yaml

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--experiment_dir',
    help='path of experiment')
parser.add_argument(
    '--det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--render',
    action='store_true',
    default=False,
    help='whether to render the rollouts')
parser.add_argument(
    '--config',
    default='configs/ppo.yaml',
    help='path of config file')
parser.add_argument(
    '--save_expert',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--num_trajs',
    default=4,
    help='number of trajectories for expert data')

args = parser.parse_args()
exp_name = args.experiment_dir
load_dir = exp_name + '/models/'
with open(exp_name + '/args.yml') as f:
    # use safe_load instead load
    args_dict = yaml.safe_load(f)
args_dict['log_dir'] = '/tmp/gym'

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

# Get a render function
render_func = None
if args.render:
    render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, obs_rms = \
    torch.load(os.path.join(load_dir, args_dict['env_name'] + ".pt"),
               map_location='cpu')

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

obs = env.reset()

if render_func is not None:
    render_func('human')

eval_episode_rewards = []

observations = []
actions = []
rewards = []

episode_observations = None
episode_actions = None
episode_rewards = None

while len(eval_episode_rewards) < int(args.num_trajs):
    with torch.no_grad():
        value, action, _ = actor_critic.act(
            obs, deterministic=args.det)
    if args.save_expert:
        # need to add unnormalized observation to the expert data
        original_obs = torch.from_numpy(env.get_original_obs())
        if episode_observations is None:
            episode_observations = original_obs
            episode_actions = action
        else:
            episode_observations = torch.cat([episode_observations, original_obs], dim=0)
            episode_actions = torch.cat([episode_actions, action], dim=0)

    # Obser reward and next obs
    obs, reward, done, infos = env.step(action)

    if args.save_expert:
        if episode_rewards is None:
            episode_rewards = reward
        else:
            episode_rewards = torch.cat([episode_rewards, reward], dim=1)

    if render_func is not None:
        render_func('human')

    if args.save_expert:
        for info in infos:
            if 'episode' in info.keys():
                print(info['episode']['l'])
                if info['episode']['l'] == 50:
                    eval_episode_rewards.append(info['episode']['r'])
                    observations.append(torch.unsqueeze(episode_observations, dim=0))
                    actions.append(torch.unsqueeze(episode_actions, dim=0))
                    rewards.append(episode_rewards)

                    episode_observations = None
                    episode_actions = None
                    episode_rewards = None
                else:
                    episode_observations = None
                    episode_actions = None
                    episode_rewards = None

if args.save_expert:
    observations = torch.cat(observations, dim=0)
    actions = torch.cat(actions, dim=0)
    rewards = torch.cat(rewards, dim=0)
    lengths = torch.full(([int(args.num_trajs)]), 1000)

    m = {'states': observations, 'actions': actions,
         'rewards': rewards, 'lengths': lengths}

    expert_dir = exp_name + '/expert/'
    if os.path.isdir(expert_dir):
        print("expert already exists !")
    else:
        os.makedirs(expert_dir)
    torch.save(m, expert_dir + args_dict['env_name'] + '_num_traj_' + str(args.num_trajs) + '.pt')

print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
    len(eval_episode_rewards), np.mean(eval_episode_rewards)))
