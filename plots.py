import pandas as pd
import matplotlib.pyplot as plt
import glob

expert = {'HalfCheetah-v2': 4984.08181, 'Walker2d-v2': 5801.77730}

def plots(dirs, env):
    lines_list = {}
    for key in dirs:
        files_rb = glob.glob(dirs[key] + '/*')
        for i in range(len(files_rb)):
            files_rb[i] += "/logs/log.csv"
        lines_list[key] = files_rb

    plt.rcParams['figure.figsize'] = (10, 5)
    plt.style.use('fivethirtyeight')

    df_means = {}
    df_stds = {}

    for key in lines_list:
        df = []
        for file in lines_list[key]:
            df.append(pd.read_csv(file))
        df = pd.concat(df)

        by_row_index = df.groupby(df.index)
        df_mean = by_row_index.mean()
        df_std = by_row_index.std()

        df_mean = df_mean.set_index('total_num_steps')
        df_std.index = df_mean.index

        df_means[key] = df_mean
        df_stds[key] = df_std

    criteria = ['mean_eval_episode_rewards', 'kl_mean', 'entropy_mean',
                'value_loss_epoch', 'action_loss_epoch', 'trust_region_loss_epoch']
    labels = ['rewards', 'kl', 'entropy',
              'value_loss', 'action_loss', 'trust_region_loss']

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
            plt.axhline(y=expert[env], color='b', linestyle='--', linewidth=2)  # halfcheetah
        if (i == 1):
            ax.set_ylim(0, 0.1)

        plt.xlabel('steps', fontsize=14)
        if i == 1:
            plt.ylabel('mean KL divergence', fontsize=14)
        else:
            plt.ylabel(labels[i], fontsize=14)
        plt.legend(fontsize=12)
        plt.legend(lines_list.keys())
        plt.savefig('plots/ppo_vf_vs_no_vf/' + env + '/' + str(labels[i]))


if __name__ == "__main__":
    env = 'HalfCheetah-v2'
    dirs = {'ppo with vf clipping': '/home/aagha/training/train_ppo_vf_clip/' + env,
            'ppo without vf clipping': '/home/aagha/training/train_ppo_no_vf_clip/' + env}
    plots(dirs, env)
