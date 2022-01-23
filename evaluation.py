import numpy as np
import torch

from utils.envs import make_vec_envs
from utils import utils


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir, norm_obs, norm_reward,
             clip_obs, clip_reward, device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, norm_obs, norm_reward,
                              clip_obs, clip_reward, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    while len(eval_episode_rewards) < 5:
        with torch.no_grad():
            action = actor_critic.get_action(obs, deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    return np.mean(eval_episode_rewards)
