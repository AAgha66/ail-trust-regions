import torch
import numpy as np
import utils.utils
from utils.envs import make_vec_envs
from utils.utils import get_vec_normalize, MMD
import yaml
from models import gail
import os
import mj_envs
import matplotlib.pyplot as plt
import pandas as  pd

num_expert_trajs = 16

def plot_l2(files, envs, save_path):
    for env in envs:
        lines_list = {}
        for key in files:
            with open(files[key] + env + '/inliers.txt', 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                #lines = glob.glob(files[key] + env + '/*')
                for i, _ in enumerate(lines):
                    lines[i] = lines[i]
                lines_list[key] = lines
        
        df_means = {}
        df_stds = {}
        for key in lines_list:
            action_l2_distances = []
            for file in lines_list[key]:
                model_path = file + '/models/'
                with open(file + '/args.yml') as f:
                    # use safe_load instead load
                    args_dict = yaml.safe_load(f)

                    envs_ = make_vec_envs(
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

                    tracking_expert_dataset = None
                    if args_dict['track_vf']:
                        file_name = args_dict['gail_experts_dir'] + args_dict['env_name'] + \
                                    '_num_traj_' + str(num_expert_trajs) + '.pt'
                        tracking_expert_dataset = gail.ExpertDataset(
                            file_name, num_trajectories=num_expert_trajs, subsample_frequency=1, tracking=True)

                    # We need to use the same statistics for normalization as used in training
                    actor_critic, obs_rms = \
                        torch.load(os.path.join(model_path, args_dict['env_name'] + ".pt"),
                                map_location='cpu')

                    vec_norm = get_vec_normalize(envs_)
                    device = torch.device("cpu")

                    if vec_norm is not None:
                        vec_norm.eval()
                        vec_norm.obs_rms = obs_rms

                    
                    tracking_diff_actions_norm = []

                    tracking_trajs = tracking_expert_dataset.get_traj()
                    with torch.no_grad():
                        for traj in range(num_expert_trajs):
                            normalized_expert_state = vec_norm._obfilt(tracking_trajs['states'][traj].type(torch.FloatTensor).numpy(), update=False)
                            normalized_expert_state = torch.FloatTensor(normalized_expert_state).to(device)
                            _, tracking_dist = actor_critic.evaluate_actions(normalized_expert_state)
                            
                            diff_actions = tracking_dist.mode() - tracking_trajs['actions'][traj].type(torch.FloatTensor)
                            tracking_diff_actions_norm.append(torch.linalg.norm(diff_actions,ord=2,dim=1))
                        
                        tracking_diff_actions_norm = torch.cat(tracking_diff_actions_norm, dim=0)
                        action_l2_distances.append(torch.unsqueeze(tracking_diff_actions_norm, dim=1))
            
            action_l2_distances = torch.cat(action_l2_distances, dim=1)
            l2_dist_df = pd.DataFrame(action_l2_distances.numpy())

            df_mean = l2_dist_df.mean(axis=1)
            df_std = l2_dist_df.std(axis=1)
            
            df_means[key] = df_mean
            df_stds[key] = df_std
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots()        
        for key in lines_list:
            plt.plot(df_means[key].interpolate(method='linear'), linewidth=1.5)
            plt.fill_between(df_stds[key].index,
                                df_means[key].interpolate(method='linear') - df_stds[key].interpolate(
                                    method='linear'),
                                df_means[key].interpolate(method='linear') + df_stds[key].interpolate(
                                    method='linear'),
                                alpha=0.35)

        plt.xlabel('iteration', fontsize=14)
        plt.ylabel("L2_distance", fontsize=14)

        plt.legend(fontsize=12)
        plt.legend(lines_list.keys())
        #plt.ylim(0, None) # Or similarly "plt.ylim(0)"
        path = save_path + env + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path + "l2_distance")

if __name__ == "__main__":
    dirs = {'4_trajs': '/home/kit/anthropomatik/kn6273/Repos/logs/training_21.10/training_hum/Humanoid-v2/mean_0.007/',
            '10_trajs': '/home/kit/anthropomatik/kn6273/Repos/logs/training_22.10/training_hum_10_traj/Humanoid-v2/trl/',
            '16_trajs': '/home/kit/anthropomatik/kn6273/Repos/logs/training_22.10/training_hum_16_traj/Humanoid-v2/trl/'}
            
    envs = ['Humanoid-v2']
    save_path = 'plots/plots_26.11/l2_distance/'
    
    plot_l2(dirs, envs, save_path)