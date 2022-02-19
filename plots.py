import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from utils.utils import compute_kurtosis

expert_rewards = {'HalfCheetah-v2': 4920,
                  'hammer-v0': 16356.39770,
                  'door-v0': 3036.42159,
                  'Walker2d-v2': 4623.24123,
                  'Humanoid-v2': 6235.15560}


def plot_CI(files, envs, save_path):
    for env in envs:
        lines_list = {}
        for key in files:
            with open(files[key] + env + '/inliers.txt', 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                # lines = glob.glob(files[key] + env + '/*')
                for i, _ in enumerate(lines):
                    lines[i] = lines[i] + "/logs/log.csv"
                lines_list[key] = lines

        plt.rcParams['figure.figsize'] = (12, 8)
        plt.style.use('seaborn-darkgrid')

        df_means = {}
        df_stds = {}
        for key in lines_list:
            df = []
            for file in lines_list[key]:
                csv_file = pd.read_csv(file)
                # csv_file['vf_diff'] = csv_file['vf_diff'].apply(lambda x: float(x[8:-2]) if 'tensor' in x else float(x))
                df.append(csv_file)

            df = pd.concat(df)
            by_row_index = df.groupby(df.index)

            df_mean = by_row_index.mean()
            df_std = by_row_index.std()
            df_mean = df_mean.set_index('total_num_steps')
            df_std.index = df_mean.index

            df_means[key] = df_mean
            df_stds[key] = df_std

        criteria = ['mean_eval_episode_rewards', 'kl_mean', 'entropy_mean',
                    'value_loss_epoch', 'action_loss_epoch', 'trust_region_loss_epoch',
                    'vf_diff', 'tracking_log_probs_mean', 'tracking_log_probs_median',
                    'tracking_diff_actions_norm_mean', 'tracking_diff_actions_norm_median',
                    'on_policy_cos_mean', 'off_policy_cos_mean', 'on_policy_cos_median', 'off_policy_cos_median']
        labels = ['Rewards', 'Mean KL Divergence', 'Entropy',
                  'Value loss', 'Action loss', 'Trust region loss', 'vf_difference',
                  'Log probability mean', 'Log probability median',
                  'L2 distance mean', 'L2 distance median',
                  'On-policy cos mean', 'Off-policy cos mean', 'On-policy cos median', 'Off-policy cos median']

        for i, c in enumerate(criteria):
            fig, ax = plt.subplots()

            for key in lines_list:
                plt.plot(df_means[key][c].interpolate(method='linear'), linewidth=1.5)
                plt.fill_between(df_stds[key][c].index,
                                 df_means[key][c].interpolate(method='linear') - df_stds[key][c].interpolate(
                                     method='linear'),
                                 df_means[key][c].interpolate(method='linear') + df_stds[key][c].interpolate(
                                     method='linear'),
                                 alpha=0.35)
            if (i == 0):
                plt.axhline(y=expert_rewards[env], color='b', linestyle='--', linewidth=2)
            if (i == 1):
                plt.ylim(0, 0.15)

            ax = plt.gca()
            ax.xaxis.offsetText.set_fontsize(24)
            ax.xaxis.get_offset_text().set_fontsize(24)
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.xlabel('steps', fontsize="30")
            plt.ylabel(labels[i], fontsize="30")
            plt.xticks(fontsize="35")
            plt.yticks(fontsize="35")
            # plt.legend(lines_list.keys(), fontsize="35")

            path = save_path + env + '/main_stats/'
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(path + str(labels[i]) + '.pdf')
            print(path + str(labels[i]))


def plot_stats_rollout(files, envs, save_path):
    for env in envs:
        lines_list = {}
        for key in files:
            with open(files[key] + env + '/inliers.txt', 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                for i, _ in enumerate(lines):
                    lines[i] += "/logs/log_rollout.csv"
                lines_list[key] = lines

        plt.rcParams['figure.figsize'] = (12, 8)
        plt.style.use('seaborn-darkgrid')

        df_means = {}
        df_stds = {}
        for key in lines_list:
            df = []
            for file in lines_list[key]:
                csv_file = pd.read_csv(file)
                ratios_list = csv_file.groupby("total_num_steps")["ratios"].apply(list)
                adv_list = csv_file.groupby("total_num_steps")["advantages"].apply(list)
                reward_list = csv_file.groupby("total_num_steps")["rewards"].apply(list)
                returns_list = csv_file.groupby("total_num_steps")["returns"].apply(list)
                value_list = csv_file.groupby("total_num_steps")["values"].apply(list)

                d = {'total_num_steps': ratios_list.index,
                     'ratios_mean': ratios_list.apply(lambda x: np.mean(x)),
                     'ratios_var': ratios_list.apply(lambda x: np.var(x)),
                     'ratios_kurtosis': ratios_list.apply(compute_kurtosis),

                     'adv_mean': adv_list.apply(lambda x: np.mean(x)),
                     'adv_var': adv_list.apply(lambda x: np.var(x)),
                     'adv_kurtosis': adv_list.apply(compute_kurtosis),

                     'rewards_mean': reward_list.apply(lambda x: np.mean(x)),
                     'rewards_var': reward_list.apply(lambda x: np.var(x)),
                     'rewards_kurtosis': reward_list.apply(compute_kurtosis),

                     'returns_mean': returns_list.apply(lambda x: np.mean(x)),
                     'returns_var': returns_list.apply(lambda x: np.var(x)),
                     'returns_kurtosis': returns_list.apply(compute_kurtosis),

                     'values_mean': value_list.apply(lambda x: np.mean(x)),
                     'values_var': value_list.apply(lambda x: np.var(x)),
                     'values_kurtosis': value_list.apply(compute_kurtosis),
                     }
                new_df = pd.DataFrame(data=d)
                df.append(new_df)

            df = pd.concat(df)
            by_row_index = df.groupby(df.index)

            df_mean = by_row_index.mean()
            df_std = by_row_index.std()
            df_mean = df_mean.set_index('total_num_steps')
            df_std.index = df_mean.index

            df_means[key] = df_mean
            df_stds[key] = df_std
            print(key)
        criteria = ['ratios_mean', 'ratios_var', 'ratios_kurtosis',
                    'adv_mean', 'adv_var', 'adv_kurtosis',
                    'rewards_mean', 'rewards_var', 'rewards_kurtosis',
                    'returns_mean', 'returns_var', 'returns_kurtosis',
                    'values_mean', 'values_var', 'values_kurtosis']

        labels = ['Mean Likelihood Rartios', 'ratios_var', 'Kurtosis Likelihood Rartios',
                  'adv_mean', 'adv_var', 'adv_kurtosis',
                  'rewards_mean', 'rewards_var', 'rewards_kurtosis',
                  'returns_mean', 'returns_var', 'returns_kurtosis',
                  'values_mean', 'values_var', 'values_kurtosis']

        for i, c in enumerate(criteria):
            fig, ax = plt.subplots()

            for key in lines_list:
                plt.plot(df_means[key][c].interpolate(method='linear'), linewidth=1.5)
                plt.fill_between(df_stds[key][c].index,
                                 df_means[key][c].interpolate(method='linear') - df_stds[key][c].interpolate(
                                     method='linear'),
                                 df_means[key][c].interpolate(method='linear') + df_stds[key][c].interpolate(
                                     method='linear'),
                                 alpha=0.35)

            ax = plt.gca()
            ax.xaxis.offsetText.set_fontsize(24)
            ax.xaxis.get_offset_text().set_fontsize(24)
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.xlabel('steps', fontsize="30")
            plt.ylabel(labels[i], fontsize="30")
            plt.xticks(fontsize="35")
            plt.yticks(fontsize="35")
            # plt.legend(lines_list.keys(), fontsize="35")

            path = save_path + env + '/rollout_stats/'
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig("{}{}{}".format(path, labels[i], '_rollout.pdf'))


def plot_Disc(files, envs, save_path):
    for env in envs:
        lines_list = {}
        for key in files:
            with open(files[key] + env + '/inliers.txt', 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                # lines = glob.glob(files[key] + env + '/*')
                for i, _ in enumerate(lines):
                    lines[i] += "/logs/log_grads.csv"
                lines_list[key] = lines

        plt.rcParams['figure.figsize'] = (12, 8)
        plt.style.use('seaborn-darkgrid')

        df_means = {}
        df_stds = {}
        for key in lines_list:
            print(key)
            df = []
            for file in lines_list[key]:
                csv_file = pd.read_csv(file)
                df.append(csv_file)

            df = pd.concat(df)
            by_row_index = df.groupby(df.index)

            df_mean = by_row_index.mean()
            df_std = by_row_index.std()
            df_mean = df_mean.set_index('iteration')
            df_std.index = df_mean.index

            df_means[key] = df_mean
            df_stds[key] = df_std

        criteria = ['disc_grad_norm', 'acc_expert', 'acc_policy']
        labels = ['disc_grad_norm', 'Discriminator accuracy expert samples', 'Discriminator accuracy policy data']

        for i, c in enumerate(criteria):
            fig, ax = plt.subplots()

            for key in lines_list:
                plt.plot(df_means[key][c].interpolate(method='linear'), linewidth=1.5)
                plt.fill_between(df_stds[key][c].index,
                                 df_means[key][c].interpolate(method='linear') - df_stds[key][c].interpolate(
                                     method='linear'),
                                 df_means[key][c].interpolate(method='linear') + df_stds[key][c].interpolate(
                                     method='linear'),
                                 alpha=0.35)

            ax = plt.gca()
            ax.xaxis.offsetText.set_fontsize(24)
            ax.xaxis.get_offset_text().set_fontsize(24)
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

            plt.xlabel('iterations', fontsize="30")
            plt.ylabel(labels[i], fontsize="30")
            plt.xticks(fontsize="35")
            plt.yticks(fontsize="35")
            plt.legend(lines_list.keys(), fontsize="35")

            path = save_path + env + '/disc/'
            if not os.path.isdir(path):
                os.makedirs(path)

            plt.savefig("{}{}{}".format(path, labels[i], '_rollout.pdf'))
            # plt.savefig(path + str(labels[i]))


if __name__ == "__main__":
    dirs = {'TRL': '/home/kit/anthropomatik/kn6273/Repos/logs/training_22.10/training_hc_4/HalfCheetah-v2/trl/',
            'PPO': '/home/kit/anthropomatik/kn6273/Repos/logs/training_22.10/training_hc_4/HalfCheetah-v2/ppo/'}
    envs = ['HalfCheetah-v2']
    save_path = 'plots/plots_11.02/traj_disc/traj_4/'

    # plot_CI(dirs, envs, save_path)
    # plot_stats_rollout(dirs, envs, save_path)
    plot_Disc(dirs, envs, save_path)
