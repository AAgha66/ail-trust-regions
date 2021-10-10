import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns


def plots(dirs, env):
    lines_list = {}
    folders_rb = None
    for key in dirs:
        folders_rb = glob.glob(dirs[key] + '/*')
        files_rb = folders_rb.copy()
        for i in range(len(files_rb)):
            files_rb[i] += "/logs/log_adv.csv"
        lines_list[key] = files_rb

    plt.rcParams['figure.figsize'] = (10, 5)
    plt.style.use('seaborn')

    dfs = []
    for key in lines_list:
        for file in lines_list[key]:
            dfs.append(pd.read_csv(file))

    criteria = ['advantages', 'rewards', 'values']

    for i, df in enumerate(dfs):
        for _, c in enumerate(criteria):
            pivotted = dfs[i].pivot('total_num_steps', 'pair_id', c)
            sns_plot = sns.heatmap(pivotted)
            sns_fig = sns_plot.get_figure()
            sns_fig.savefig(folders_rb[i] + '/' + str(c))
            sns.reset_defaults()
            plt.clf()


if __name__ == "__main__":
    env = 'Walker2d-v2'
    dirs = {'ppo with vf clipping': '/home/aagha/training/train_gail_ppo_vf/' + env}
    dirs = {'trl kl + gradient_penalties': '/home/kit/anthropomatik/kn6273/Repos/logs/training_28.09/training_w2d/Walker2d-v2/gp/', 
            'trl kl + gradient clipping': '/home/kit/anthropomatik/kn6273/Repos/logs/training_28.09/training_w2d/Walker2d-v2/grad_clip/',
            'trl kl': '/home/kit/anthropomatik/kn6273/Repos/logs/training_28.09/training_w2d/Walker2d-v2/refined/'}
    plots(dirs, env)
