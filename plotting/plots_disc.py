import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

envs = ['HalfCheetah-v2']

def plot(files):
    for env in envs:
        lines_list = {}
        for key in files:
            with open(files[key] + env + '/inliers.txt', 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
                #lines = glob.glob(files[key] + env + '/*')
                for i, _ in enumerate(lines):
                    lines[i] += "/logs/log_grads.csv"
                lines_list[key] = lines

        plt.rcParams['figure.figsize'] = (12, 8)
        plt.style.use('seaborn-darkgrid')

        df_means = {}
        df_stds = {}
        for key in lines_list:
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
        labels = ['disc_grad_norm', 'accuracy_expert_data', 'accuracy_policy_data']

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

            plt.xlabel('iteration', fontsize=14)
            plt.ylabel(labels[i], fontsize=14)

            plt.legend(fontsize=12)
            plt.legend(lines_list.keys())
            path = 'plots/plots_04.10/ail_rl/' + env + '/'
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(path + str(labels[i]))
            print(path + str(labels[i]))

if __name__ == "__main__":
    files = {'AIL': '/home/kit/anthropomatik/kn6273/Repos/logs/training_02.10/training_hp/HalfCheetah-v2/cov_bound_0.002_mean_bound_ 0.009/', 
            'RL': '/home/kit/anthropomatik/kn6273/Repos/logs/training_03.10/training_rl_trl/'}
    plot(files)
