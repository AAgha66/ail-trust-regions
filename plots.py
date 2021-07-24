import pandas as pd
import matplotlib.pyplot as plt

files_ppo = [
    'experiments/HalfCheetah-v2gailTrue-lr0.0003-e_coef0-v_coef0.5-max_grad0.5-seed2-num_batch64-clip_irTrue-envHalfCheetah-v2-projectionFalse-lr_decayTrue-clipped_valueTrue-norm_oTrue-norm_rTrue-clip_o10.0-clip_r10.0/logs/log.csv',
    'experiments/HalfCheetah-v2gailTrue-lr0.0003-e_coef0-v_coef0.5-max_grad0.5-seed12-num_batch64-clip_irTrue-envHalfCheetah-v2-projectionFalse-lr_decayTrue-clipped_valueTrue-norm_oTrue-norm_rTrue-clip_o10.0-clip_r10.0/logs/log.csv',
    'experiments/HalfCheetah-v2gailTrue-lr0.0003-e_coef0-v_coef0.5-max_grad0.5-seed22-num_batch64-clip_irTrue-envHalfCheetah-v2-projectionFalse-lr_decayTrue-clipped_valueTrue-norm_oTrue-norm_rTrue-clip_o10.0-clip_r10.0/logs/log.csv']

files_trl = [
    'experiments/HalfCheetah-v2gailTrue-lr5e-05-e_coef0-v_coef0.5-max_grad0.5-seed2-num_batch32-clip_irFalse-envHalfCheetah-v2-projectionTrue-lr_decayFalse-clipped_valueFalse-norm_oTrue-norm_rFalse-clip_o1000000.0-clip_r1000000.0/logs/log.csv',
    'experiments/HalfCheetah-v2gailTrue-lr5e-05-e_coef0-v_coef0.5-max_grad0.5-seed12-num_batch32-clip_irFalse-envHalfCheetah-v2-projectionTrue-lr_decayFalse-clipped_valueFalse-norm_oTrue-norm_rFalse-clip_o1000000.0-clip_r1000000.0/logs/log.csv',
    'experiments/HalfCheetah-v2gailTrue-lr5e-05-e_coef0-v_coef0.5-max_grad0.5-seed22-num_batch32-clip_irFalse-envHalfCheetah-v2-projectionTrue-lr_decayFalse-clipped_valueFalse-norm_oTrue-norm_rFalse-clip_o1000000.0-clip_r1000000.0/logs/log.csv']

plt.rcParams['figure.figsize'] = (10, 5)
plt.style.use('fivethirtyeight')

dfs_ppo = []
dfs_trl = []

for filename in files_ppo:
    dfs_ppo.append(pd.read_csv(filename))

for filename in files_trl:
    dfs_trl.append(pd.read_csv(filename))

df_concat_ppo = pd.concat(dfs_ppo)
df_concat_trl = pd.concat(dfs_trl)

by_row_index_ppo = df_concat_ppo.groupby(df_concat_ppo.index)
df_means_ppo = by_row_index_ppo.mean()

by_row_index_trl = df_concat_trl.groupby(df_concat_trl.index)
df_means_trl = by_row_index_trl.mean()

df_means_ppo = df_means_ppo.set_index('total_num_steps')
df_means_trl = df_means_trl.set_index('total_num_steps')

criteria = ['mean_eval_episode_rewards', 'kl_mean', 'entropy_mean']
labels = ['rewards', 'kl', 'entropy']

for i, c in enumerate(criteria):
    # Print the summary statistics of the DataFrame
    ax_ppo = df_means_ppo[c].plot(linewidth=2, fontsize=12)
    ax_trl = df_means_trl[c].plot(linewidth=2, fontsize=12)
    if (i == 1):
        ax_ppo.set_ylim(0, 0.1)

    # Additional customizations
    ax_ppo.set_xlabel('steps')
    ax_ppo.set_ylabel(labels[i])

    ax_ppo.legend(fontsize=12)
    ax_ppo.legend(["PPO", "TRL"])

    plt.show()
