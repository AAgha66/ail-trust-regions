import argparse
import os
# workaround to unpickle olf model files
import sys
import torch
import numpy as np
from utils.envs import make_vec_envs
from utils.utils import get_render_func, get_vec_normalize, MMD
import yaml
from models import gail
import mj_envs

num_trajs = 4
gail_experts_dir = '/home/kit/anthropomatik/kn6273/Repos/gail_experts/'

def eval_mmd(files, envs, length):
    for env in envs:
        file_name = gail_experts_dir + env + '_num_traj_' + str(num_trajs) + '.pt'
        tracking_expert_dataset = gail.ExpertDataset(file_name, num_trajectories=num_trajs, subsample_frequency=1, tracking=True)
        expert_trajs = tracking_expert_dataset.get_traj()

        expert_observations = []
        expert_actions = []
        for traj in range(num_trajs):
            expert_observations.append(expert_trajs['states'][traj])
            expert_actions.append(expert_trajs['actions'][traj])

        expert_observations = torch.cat(expert_observations, dim=0).type(torch.DoubleTensor)
        expert_actions = torch.cat(expert_actions, dim=0).type(torch.DoubleTensor)
        expert_state_actions = torch.cat([expert_observations, expert_actions], dim=1)
        
        lines_list = {}
        for key in files:
            with open(files[key] + env + '/inliers.txt', 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                #lines = glob.glob(files[key] + env + '/*')
                for i, _ in enumerate(lines):
                    lines[i] = lines[i]
                lines_list[key] = lines
        
        for key in lines_list:
            MMD_states = []
            MMD_actions = []
            MMD_state_action_pairs = []
            for file in lines_list[key]:
                model_path = file + '/models/'
                with open(file + '/args.yml') as f:
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

                    # We need to use the same statistics for normalization as used in training
                    actor_critic, obs_rms = \
                        torch.load(os.path.join(model_path, args_dict['env_name'] + ".pt"),
                                map_location='cpu')

                    vec_norm = get_vec_normalize(env)
                    
                    if vec_norm is not None:
                        vec_norm.eval()
                        vec_norm.obs_rms = obs_rms

                    obs = env.reset()

                    eval_episode_rewards = []

                    observations = []
                    actions = []
                    
                    episode_observations = None
                    episode_actions = None
                    episode_rewards = None

                    while len(eval_episode_rewards) < int(num_trajs):
                        with torch.no_grad():
                            _, action, _ = actor_critic.act(
                                obs, deterministic=True)
                            
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

                        if episode_rewards is None:
                            episode_rewards = reward
                        else:
                            episode_rewards = torch.cat([episode_rewards, reward], dim=1)

                        
                        
                        for info in infos:
                            if 'episode' in info.keys():
                                if info['episode']['l'] == length:
                                    eval_episode_rewards.append(info['episode']['r'])                
                                    observations.append(episode_observations)
                                    actions.append(episode_actions)
                                    
                                    episode_observations = None
                                    episode_actions = None
                                    episode_rewards = None
                                else:
                                    episode_observations = None
                                    episode_actions = None
                                    episode_rewards = None

                    observations = torch.cat(observations, dim=0)
                    actions = torch.cat(actions, dim=0)
                    
                    observations = observations.type(torch.DoubleTensor)
                    actions = actions.type(torch.DoubleTensor)
                    state_actions = torch.cat([observations, actions], dim=1)

                    MMD_states.append(MMD(expert_observations, observations, "rbf"))
                    MMD_actions.append(MMD(expert_actions, actions, "rbf"))
                    MMD_state_action_pairs.append(MMD(expert_state_actions, state_actions, "rbf"))
                    
            print("{}, MMD_states, mean: {}, std: {}".format(key, np.mean(MMD_states), np.std(MMD_states)))
            print("{}, MMD_actions, mean: {}, std: {}".format(key, np.mean(MMD_actions), np.std(MMD_actions)))
            print("{}, MMD_state_action_pairs, mean: {}, std: {}".format(key, np.mean(MMD_state_action_pairs), np.std(MMD_state_action_pairs)))


if __name__ == "__main__":
    dirs = {'trl': '/home/kit/anthropomatik/kn6273/Repos/logs/training_05.11/training_w2d_4/Walker2d-v2/trl/',
            'ppo': '/home/kit/anthropomatik/kn6273/Repos/logs/training_05.11/training_w2d_4/Walker2d-v2/ppo/'}
    envs = ['Walker2d-v2']
    length= 1000
    eval_mmd(dirs, envs, length)    