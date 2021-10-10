import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

"""envs = ['HalfCheetah-v2', 'Walker2d-v2', 'Reacher-v2', 'Ant-v2']
expert_rewards = {'HalfCheetah-v2': 3982.87514, 'Walker2d-v2': 4623.24123,
                  'Reacher-v2': -4.04847, 'Ant-v2': 2299.08875}"""
envs = ['HalfCheetah-v2', 'Walker2d-v2']
expert_rewards = {'HalfCheetah-v2': 3982.87514,
                'hammer-v0': 16312.43210,
                'door-v0': 3036.42159,
                'Walker2d-v2': 4623.24123,
                'Humanoid-v2': 7313}

def plot(files):
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

        plt.rcParams['figure.figsize'] = (12, 8)
        plt.style.use('seaborn-darkgrid')

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
            df_mean = df_mean.set_index('total_num_steps')
            df_std.index = df_mean.index

            df_means[key] = df_mean
            df_stds[key] = df_std

        criteria = ['mean_eval_episode_rewards', 'kl_mean', 'entropy_mean',
                    'value_loss_epoch', 'action_loss_epoch', 'trust_region_loss_epoch',
                    'vf_diff', 'tracking_log_probs_mean','tracking_log_probs_median',
                    'on_policy_kurtosis', 'off_policy_kurtosis','on_policy_value_kurtosis','off_policy_value_kurtosis']
        labels = ['rewards', 'kl', 'entropy',
                  'value_loss', 'action_loss', 'trust_region_loss', 'vf_difference',
                  'expert_log_liklihood_mean', 'expert_log_liklihood_median',
                  'on policy actor gradient heavy-tailedness', 'off policy actor gradient heavy-tailedness', 
                  'on policy critic gradient heavy-tailedness', 'off policy critic gradient heavy-tailedness']

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
                plt.axhline(y=expert_rewards[env], color='b', linestyle='--', linewidth=2)  # walker

            plt.xlabel('steps', fontsize=14)
            plt.ylabel(labels[i], fontsize=14)

            plt.legend(fontsize=12)
            plt.legend(lines_list.keys())
            path = 'plots/plots_07.10/trl_normalized_new/' + env + '/'
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(path + str(labels[i]))
            print(path + str(labels[i]))

if __name__ == "__main__":
    files = {'unnormalized': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.10/training_van_unnormalized/',
            'normalized': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.10/training_van_normalized/'}
    plot(files)
