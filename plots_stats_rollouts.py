import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import compute_kurtosis
import os

envs = ['Walker2d-v2', 'HalfCheetah-v2']

def plot(files):
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
        
        labels = ['ratios_mean', 'ratios_var', 'ratios_kurtosis',
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

            plt.xlabel('steps', fontsize=14)
            plt.ylabel(labels[i], fontsize=14)

            plt.legend(fontsize=12)
            plt.legend(lines_list.keys())

            path = 'plots/plots_07.10/trl_normalized_new/' + env + '/'
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig("{}{}{}".format(path, labels[i], '_rollout'))

if __name__ == "__main__":
    files = {'unnormalized': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.10/training_van_unnormalized/',
        'normalized': '/home/kit/anthropomatik/kn6273/Repos/logs/training_06.10/training_van_normalized/'}
    plot(files)
