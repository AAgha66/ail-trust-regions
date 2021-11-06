import os
import yaml
import torch
import numpy as np
import pandas as pd
from utils.envs import make_vec_envs
from utils.utils import get_render_func, get_vec_normalize
import mj_envs


def eval_moving_window(files, envs):
    for env in envs:
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

            criteria = ['mean_eval_episode_rewards']
            labels = ['rewards']

            filtered_df = df_mean[df_mean['mean_eval_episode_rewards'].notnull()]
            filtered_df['Rolling rewards average'] = filtered_df['mean_eval_episode_rewards'].rolling(10).mean()
            filtered_df['Rolling rewards standard deviation'] = filtered_df['mean_eval_episode_rewards'].rolling(10).std()
            print("{}: {}".format(key, filtered_df['Rolling rewards average'].max()))
            mask = filtered_df['mean_eval_episode_rewards'] > filtered_df['Rolling rewards average'].max() * 0.9
            print("{}: {}".format(key, filtered_df['steps'][mask].iloc[0]))
            print("++++")

def eval_best_performance(files, envs):
    for env in envs:
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


if __name__ == "__main__":
    paths = {'trl': '/home/kit/anthropomatik/kn6273/Repos/logs/archive/training_21.09/kl_training_van/'}
    
    envs = ['door-v0']
    
    eval_moving_window(paths, envs)
    eval_best_performance(paths, envs)