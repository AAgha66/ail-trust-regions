import glob
import argparse
import pandas as pd
import numpy as np
import random
import os

def get_samples(path):
    try:
        os.remove(path + '/inliers.txt')
    except OSError:
        pass

    samples = glob.glob(path + '/*')
    dfs = []
    scores = []
    for sample in samples:
        log = sample + '/logs/log.csv'
        dfs.append(pd.read_csv(log))
        df_means = dfs[-1].mean()
        scores.append(df_means['mean_eval_episode_rewards'])
    means = []
    for i in range(500):
        means.append(np.mean(random.sample(scores, 3)))

    print('50th percentile (median) = %.3f' % np.median(means))
    # calculate 95% confidence intervals (100 - alpha)
    alpha = 5.0
    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    lower = max(-np.inf, np.percentile(means, lower_p))
    print('%.1fth percentile = %.3f' % (lower_p, lower))
    # calculate upper percentile (e.g. 97.5)
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    upper = min(np.max(means), np.percentile(means, upper_p))
    print('%.1fth percentile = %.3f' % (upper_p, upper))
    with open(path + '/inliers.txt', 'w') as f:
        for i, score in enumerate(scores):
            if lower <= score <= upper:            
                f.write(samples[i] + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CI')
    parser.add_argument(
        '--path', default='/home/aagha/training_27.08/kl_3e+6/HalfCheetah-v2', help='directory containing logs')
    args = parser.parse_args()

    paths = ['/home/kit/anthropomatik/kn6273/Repos/logs/training_06.10/training_van_normalized/HalfCheetah-v2',
'/home/kit/anthropomatik/kn6273/Repos/logs/training_06.10/training_van_normalized/Walker2d-v2',
'/home/kit/anthropomatik/kn6273/Repos/logs/training_06.10/training_van_unnormalized/HalfCheetah-v2',
'/home/kit/anthropomatik/kn6273/Repos/logs/training_06.10/training_van_unnormalized/Walker2d-v2']
    for path in paths:
        get_samples(path=path)
