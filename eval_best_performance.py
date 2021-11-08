import os
from numpy.core.fromnumeric import mean
import yaml
import torch
import numpy as np
import pandas as pd
from utils.envs import make_vec_envs
from utils.projection_utils import compute_metrics, gaussian_kl
from utils.utils import get_render_func, get_vec_normalize
import mj_envs
from models import gail
import pickle
from models.distributions import FixedNormal

def eval_moving_window(files, env):
    lines_list = {}
    for key in files:
        with open(files[key] + env + '/inliers.txt', 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            #lines = glob.glob(files[key] + env + '/*')
            for i, _ in enumerate(lines):
                lines[i] = lines[i] + "/logs/log.csv"
            lines_list[key] = lines
    
    df_means = {}
    df_stds = {}
    for key in lines_list:
        df = []
        for file in lines_list[key]:
            csv_file = pd.read_csv(file)
            #csv_file['vf_diff'] = csv_file['vf_diff'].apply(lambda x: float(x[8:-2]) if 'tensor' in x else float(x))
            df.append(csv_file)

        df = pd.concat(df)
        by_row_index = df.groupby(df.index)

        df_mean = by_row_index.mean()
        df_std = by_row_index.std()
        df_mean['steps'] = df_mean['total_num_steps'] 
        df_mean = df_mean.set_index('total_num_steps')            
        df_std.index = df_mean.index

        df_means[key] = df_mean
        df_stds[key] = df_std

        filtered_df = df_mean[df_mean['mean_eval_episode_rewards'].notnull()]
        filtered_df['Rolling rewards average'] = filtered_df['mean_eval_episode_rewards'].rolling(10).mean()
        filtered_df['Rolling rewards standard deviation'] = filtered_df['mean_eval_episode_rewards'].rolling(10).std()
        print("rewards: {}: {}".format(key, filtered_df['Rolling rewards average'].max()))
        mask = filtered_df['mean_eval_episode_rewards'] > filtered_df['Rolling rewards average'].max() * 0.9
        #print("sample efficiency: {}: {}".format(key, filtered_df['steps'][mask].iloc[0]))
        print("++++")

        filtered_df = df_mean[df_mean['tracking_diff_actions_norm_mean'].notnull()]
        filtered_df['Rolling l2 action diff average'] = filtered_df['tracking_diff_actions_norm_mean'].rolling(5).mean()
        filtered_df['Rolling l2 action diff standard deviation'] = filtered_df['tracking_diff_actions_norm_mean'].rolling(5).std()
        print("L2 distance: {}: {}".format(key, filtered_df['Rolling l2 action diff average'].max()))
        print("++++")

def eval_best_performance(files, env):
    lines_list = {}
    for key in files:
        with open(files[key] + env + '/inliers.txt', 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            #lines = glob.glob(files[key] + env + '/*')
            lines_list[key] = lines

    for key in lines_list:
        eval_episode_rewards = []
        for file in lines_list[key]:
            model_path = file + '/models/'
            with open(file + '/args.yml') as f:
                # use safe_load instead load
                args_dict = yaml.safe_load(f)
                env = make_vec_envs(args_dict['env_name'],
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
                # We need to use the same statistics for normalization as used in training
                actor_critic, obs_rms = \
                    torch.load(os.path.join(model_path, args_dict['env_name'] + ".pt"),
                            map_location='cpu')

                vec_norm = get_vec_normalize(env)
                if vec_norm is not None:
                    vec_norm.eval()
                    vec_norm.obs_rms = obs_rms

                obs = env.reset()

                while len(eval_episode_rewards) < 50:
                    with torch.no_grad():
                        value, action, _ = actor_critic.act(
                            obs, deterministic=True)

                    # Obser reward and next obs
                    obs, reward, done, infos = env.step(action)                        

                    for info in infos:
                        if 'episode' in info.keys():
                            eval_episode_rewards.append(info['episode']['r'])                                                

        print("{}: Evaluation using {} episodes: mean reward {:.5f}, std reward {:.5f}\n".format(key,
            len(eval_episode_rewards), np.mean(eval_episode_rewards), np.std(eval_episode_rewards)))


def eval_dist(files, env):
    lines_list = {}
    for key in files:
        with open(files[key] + env + '/inliers.txt', 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            #lines = glob.glob(files[key] + env + '/*')
            lines_list[key] = lines

    for key in lines_list:
        mean_kl = []
        mean_reverse_kl = []
        maha_list = []
        reverse_maha_list = []
        for file in lines_list[key]:
            model_path = file + '/models/'
            with open(file + '/args.yml') as f:
                # use safe_load instead load
                args_dict = yaml.safe_load(f)
                file_name = args_dict['gail_experts_dir'] + args_dict['env_name'] + '_num_traj_' + str(4) + '.pt'
                tracking_expert_dataset = gail.ExpertDataset(
                    file_name, num_trajectories=4, subsample_frequency=1, tracking=True)
                env = make_vec_envs(args_dict['env_name'],
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
                # We need to use the same statistics for normalization as used in training
                actor_critic_ail, obs_rms_ail = \
                    torch.load(os.path.join(model_path, args_dict['env_name'] + ".pt"),
                            map_location='cpu')
                actor_critic_expert, obs_rms_expert = \
                    torch.load("/home/kit/anthropomatik/kn6273/Repos/logs/training_20.10/training_rl_ppo/Humanoid-v2/gailFalse-lr_p1.00E-04-lr_v1.00E-04-lr_d1.00E-03-penFalse-g_clipTrue-target_entropy0-s1-batch_size64-clip_irTrue-projFalse-p_typeNone-lr_decTrue-clip_vTrue-n_oTrue-n_rTrue-c_o1.00E+01-c_r1.00E+01cov0.00E+00mean0.00E+00/models/Humanoid-v2.pt",
                            map_location='cpu')

                vec_norm_ail = get_vec_normalize(env)
                vec_norm_expert = get_vec_normalize(env)
                if vec_norm_ail is not None:
                    vec_norm_ail.eval()
                    vec_norm_ail.obs_rms = obs_rms_ail

                    vec_norm_expert.eval()
                    vec_norm_expert.obs_rms = obs_rms_expert

                tracking_trajs = tracking_expert_dataset.get_traj()
                for traj in range(4):
                    normalized_expert_state_ail = vec_norm_ail._obfilt(tracking_trajs['states'][traj].type(torch.FloatTensor).numpy(), update=False)
                    normalized_expert_state_ail = torch.FloatTensor(normalized_expert_state_ail).to("cpu")

                    normalized_expert_state = vec_norm_expert._obfilt(tracking_trajs['states'][traj].type(torch.FloatTensor).numpy(), update=False)
                    normalized_expert_state = torch.FloatTensor(normalized_expert_state).to("cpu")

                    _, tracking_dist_ail = actor_critic_ail.evaluate_actions(normalized_expert_state_ail)
                    _, tracking_dist_expert = actor_critic_expert.evaluate_actions(normalized_expert_state)
                    metrics = compute_metrics(tracking_dist_ail, tracking_dist_expert)
                    reverse_metrics = compute_metrics(tracking_dist_expert, tracking_dist_ail)
                    
                    mean_kl.append(metrics["kl"])
                    mean_reverse_kl.append(reverse_metrics["kl"])

                    maha_part, _ = gaussian_kl(tracking_dist_ail, tracking_dist_expert)
                    reverse_maha_part, _ = gaussian_kl(tracking_dist_expert, tracking_dist_ail)
                    maha_list.append(maha_part.mean().item())
                    reverse_maha_list.append(reverse_maha_part.mean().item())                        
        print("{}: Evaluation using {} files: kl {:.5f} , reverse kl {:.5f}, maha {:.5f}, , reverse maha {:.5f}\n".format(key,
            len(lines_list[key]), np.mean(mean_kl), np.mean(mean_reverse_kl), 
            np.mean(maha_list) , np.mean(reverse_maha_list)))


def eval_dist_ardoit(files, env):
    lines_list = {}
    for key in files:
        with open(files[key] + env + '/inliers.txt', 'r') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            #lines = glob.glob(files[key] + env + '/*')
            lines_list[key] = lines

    for key in lines_list:
        mean_kl = []
        mean_reverse_kl = []
        maha_list = []
        reverse_maha_list = []
        for file in lines_list[key]:
            model_path = file + '/models/'
            with open(file + '/args.yml') as f:
                # use safe_load instead load
                args_dict = yaml.safe_load(f)
                file_name = args_dict['gail_experts_dir'] + args_dict['env_name'] + '_num_traj_' + str(4) + '.pt'
                tracking_expert_dataset = gail.ExpertDataset(
                    file_name, num_trajectories=4, subsample_frequency=1, tracking=True)
                env = make_vec_envs(args_dict['env_name'],
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
                # We need to use the same statistics for normalization as used in training
                actor_critic_ail, obs_rms_ail = \
                    torch.load(os.path.join(model_path, args_dict['env_name'] + ".pt"),
                            map_location='cpu')
                
                policy = "/home/kit/anthropomatik/kn6273/Repos/hand_dapg/dapg/policies/{}.pickle".format(args_dict['env_name'])
                pi = pickle.load(open(policy, 'rb'))
                vec_norm_ail = get_vec_normalize(env)
                vec_norm_expert = get_vec_normalize(env)
                if vec_norm_ail is not None:
                    vec_norm_ail.eval()
                    vec_norm_ail.obs_rms = obs_rms_ail

                tracking_trajs = tracking_expert_dataset.get_traj()
                for traj in range(4):
                    normalized_expert_state_ail = vec_norm_ail._obfilt(tracking_trajs['states'][traj].type(torch.FloatTensor).numpy(), update=False)
                    normalized_expert_state_ail = torch.FloatTensor(normalized_expert_state_ail).to("cpu")

                    _, tracking_dist_ail = actor_critic_ail.evaluate_actions(normalized_expert_state_ail)
                    means=None
                    stds=None
                    
                    for i in range(tracking_trajs['states'][traj].shape[0]):
                        o = tracking_trajs['states'][traj][i,:]
                        if means is None:                                
                            means = torch.from_numpy(pi.get_action(o)[1]['mean']).unsqueeze(0)
                            stds = torch.exp(torch.from_numpy(pi.get_action(o)[1]['log_std'])).unsqueeze(0)
                        else:
                            means = torch.cat([means, torch.from_numpy(pi.get_action(o)[1]['mean']).unsqueeze(0)], dim=0)
                            std = torch.cat([means, torch.exp(torch.from_numpy(pi.get_action(o)[1]['log_std'])).unsqueeze(0)], dim=0)

                    
                    tracking_dist_expert = FixedNormal(means, stds.to(torch.float32))

                    metrics = compute_metrics(tracking_dist_ail, tracking_dist_expert)
                    reverse_metrics = compute_metrics(tracking_dist_expert, tracking_dist_ail)
                    
                    mean_kl.append(metrics["kl"])
                    mean_reverse_kl.append(reverse_metrics["kl"])

                    maha_part, _ = gaussian_kl(tracking_dist_ail, tracking_dist_expert)
                    reverse_maha_part, _ = gaussian_kl(tracking_dist_expert, tracking_dist_ail)
                    maha_list.append(maha_part.mean().item())
                    reverse_maha_list.append(reverse_maha_part.mean().item())                        
        print("{}: Evaluation using {} files: kl {:.5f} , reverse kl {:.5f}, maha {:.5f}, , reverse maha {:.5f}\n".format(key,
            len(lines_list[key]), np.mean(mean_kl), np.mean(mean_reverse_kl), 
            np.mean(maha_list) , np.mean(reverse_maha_list)))

if __name__ == "__main__":
    paths = {'trl_4_old': '/home/kit/anthropomatik/kn6273/Repos/logs/training_05.11/training_door_4/door-v0/trl/',
        'ppo_4_old': '/home/kit/anthropomatik/kn6273/Repos/logs/training_05.11/training_door_4/door-v0/ppo/',
        'trl_4': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.11/training_door_4/trl/',
        'ppo_4': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.11/training_door_4/ppo/',
        'trl_10': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.11/training_door_10/door-v0/trl/',
        'ppo_10': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.11/training_door_10/door-v0/ppo/',
        'gp': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.11/training_door_gp/',
        'sn': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.11/training_door_sn/',
        'td': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.11/training_door_td/'}

        
    
    envs = ['door-v0']    
    for env in envs:
        if env.endswith('-v0'):
            eval_dist_ardoit(paths, env)
        else:
            eval_dist(paths, env)
        eval_moving_window(paths, env)    
        eval_best_performance(paths, env)