import os
from numpy.core.fromnumeric import mean
import yaml
import torch
import numpy as np
from utils.envs import make_vec_envs
from utils.projection_utils import compute_metrics
from utils.utils import get_vec_normalize
import mj_envs
from models import gail
import pickle
from models.distributions import FixedNormal
import glob

expert = {
    'HalfCheetah-v2': '/home/kit/anthropomatik/kn6273/Repos/ail-trust-regions/train_expert/HalfCheetah-v2/gailFalse-lr_p3.00E-04-lr_v3.00E-04-lr_d1.00E-03-penFalse-g_clipTrue-max_grad0.5-s0-batch_size64-clip_irTrue-projFalse-p_typeNone-lr_decTrue-clip_vTrue-n_oTrue-n_rTrue-c_o1.00E+01-c_r1.00E+01cov0.00E+00/models/HalfCheetah-v2.pt',
    'Humanoid-v2': '/home/kit/anthropomatik/kn6273/Repos/logs/training_20.10/training_rl_ppo/Humanoid-v2/gailFalse-lr_p1.00E-04-lr_v1.00E-04-lr_d1.00E-03-penFalse-g_clipTrue-target_entropy0-s1-batch_size64-clip_irTrue-projFalse-p_typeNone-lr_decTrue-clip_vTrue-n_oTrue-n_rTrue-c_o1.00E+01-c_r1.00E+01cov0.00E+00mean0.00E+00/models/Humanoid-v2.pt',
    'Walker2d-v2': '/home/kit/anthropomatik/kn6273/Repos/ail-trust-regions/train_expert/Walker2d-v2/gailFalse-lr_p3.00E-04-lr_v3.00E-04-lr_d1.00E-03-penFalse-g_clipTrue-max_grad0.5-s0-batch_size64-clip_irTrue-projFalse-p_typeNone-lr_decTrue-clip_vTrue-n_oTrue-n_rTrue-c_o1.00E+01-c_r1.00E+01cov0.00E+00/models/Walker2d-v2.pt'}


def CI(dataset):
    # seed the random number generator
    np.random.seed(0)
    # bootstrap
    scores = list()  #
    for _ in range(100):
        # bootstrap sample
        indices = np.random.randint(0, len(dataset), len(dataset))
        sample = np.asarray(dataset)[indices]
        # calculate and store statistic
        statistic = mean(sample)
        scores.append(statistic)
    # print('50th percentile (median) = %.3f' % np.median(scores))
    # calculate 95% confidence intervals (100 - alpha)
    alpha = 5.0
    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    lower = np.percentile(scores, lower_p)
    # print('%.1fth percentile = %.3f' % (lower_p, lower))
    # calculate upper percentile (e.g. 97.5)
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    upper = np.percentile(scores, upper_p)
    # print('%.1fth percentile = %.3f' % (upper_p, upper))
    return lower, upper


def eval_best_performance(files):
    lines_list = {}
    for key in files:
        lines = glob.glob(files[key] + '/*')
        for l in lines:
            if l.endswith(".txt"):
                os.remove(os.path.join(files[key], l))
        lines = glob.glob(files[key] + '/*')
        lines_list[key] = lines

    eval_dict_rewards = {}
    eval_dict_success_rate = {}
    for key in lines_list:
        eval_dict_rewards[key] = []
        eval_dict_success_rate[key] = []
        for file in lines_list[key]:
            eval_episode_rewards = []
            eval_episode_success_rate = []
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
                        _, action, _ = actor_critic.act(
                            obs, deterministic=True)

                    # Obser reward and next obs
                    obs, _, _, infos = env.step(action)
                    for info in infos:
                        if 'episode' in info.keys():
                            eval_episode_rewards.append(info['episode']['r'])
                            if args_dict['env_name'].endswith('-v0'):
                                eval_episode_success_rate.append(infos[0]['goal_achieved'])
            eval_dict_rewards[key].extend(eval_episode_rewards)
            if args_dict['env_name'].endswith('-v0'):
                eval_dict_success_rate[key].append(np.mean(eval_episode_success_rate))

        upper, lower = CI(eval_dict_rewards[key])
        eval_dict_rewards[key] = [item for item in eval_dict_rewards[key] if item > lower or item < upper]
        if args_dict['env_name'].endswith('-v0'):
            print("{}: Evaluation using {} episodes: mean reward {:.5f} success_rate {:.5f}\n".format(key, len(eval_dict_rewards[key]),
                                                                                                      np.mean(eval_dict_rewards[key]),
                                                                                                      np.mean(eval_dict_success_rate[key])))
        else:
            print("{}: Evaluation using {} episodes: mean reward {:.5f}\n".format(key, len(eval_dict_rewards[key]),
                                                                                  np.mean(eval_dict_rewards[key])))


def eval_dist(files):
    lines_list = {}
    for key in files:
        lines = glob.glob(files[key] + '/*')
        for l in lines:
            if l.endswith(".txt"):
                os.remove(os.path.join(files[key], l))
        lines = glob.glob(files[key] + '/*')
        lines_list[key] = lines

    for key in lines_list:
        mean_reverse_kl = []
        l2_list = []

        for file in lines_list[key]:
            model_path = file + '/models/'
            with open(file + '/args.yml') as f:
                # use safe_load instead load
                args_dict = yaml.safe_load(f)
                file_name = args_dict['gail_experts_dir'] + args_dict['env_name'] + '_num_traj_' + str(4) + '.pt'
                tracking_expert_dataset = gail.ExpertDataset(
                    file_name, num_trajectories=4, subsample_frequency=1, tracking=True)
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
                actor_critic_ail, obs_rms_ail = \
                    torch.load(os.path.join(model_path, args_dict['env_name'] + ".pt"),
                               map_location='cpu')
                actor_critic_expert, obs_rms_expert = torch.load(expert[args_dict['env_name']], map_location='cpu')
                vec_norm_ail = get_vec_normalize(env)
                vec_norm_expert = get_vec_normalize(env)
                if vec_norm_ail is not None:
                    vec_norm_ail.eval()
                    vec_norm_ail.obs_rms = obs_rms_ail

                    vec_norm_expert.eval()
                    vec_norm_expert.obs_rms = obs_rms_expert

                tracking_trajs = tracking_expert_dataset.get_traj()
                for traj in range(4):
                    normalized_expert_state_ail = vec_norm_ail._obfilt(
                        tracking_trajs['states'][traj].type(torch.FloatTensor).numpy(), update=False)
                    normalized_expert_state_ail = torch.FloatTensor(normalized_expert_state_ail).to("cpu")

                    normalized_expert_state = vec_norm_expert._obfilt(
                        tracking_trajs['states'][traj].type(torch.FloatTensor).numpy(), update=False)
                    normalized_expert_state = torch.FloatTensor(normalized_expert_state).to("cpu")

                    _, tracking_dist_ail = actor_critic_ail.evaluate_actions(normalized_expert_state_ail)
                    _, tracking_dist_expert = actor_critic_expert.evaluate_actions(normalized_expert_state)

                    diff_actions = tracking_dist_ail.mode() - tracking_trajs['actions'][traj].type(torch.FloatTensor)
                    l2_list.append(torch.mean(torch.linalg.norm(diff_actions, ord=2, dim=1)).item())

                    # reverse_metrics = compute_metrics(tracking_dist_expert, tracking_dist_ail)
                    metrics = compute_metrics(tracking_dist_ail, tracking_dist_expert)
                    mean_reverse_kl.append(metrics["kl"])

        upper, lower = CI(l2_list)
        l2_list = [item for item in l2_list if item > lower or item < upper]

        upper, lower = CI(mean_reverse_kl)
        mean_reverse_kl = [item for item in mean_reverse_kl if item > lower or item < upper]
        print("{}: Evaluation using {} files: reverse kl {:.5f} , l2 {:.5f}\n".format(key,
                                                                                      len(lines_list[key]),
                                                                                      np.mean(mean_reverse_kl),
                                                                                      np.mean(l2_list)))


def eval_dist_ardoit(files):
    lines_list = {}
    for key in files:
        lines = glob.glob(files[key] + '/*')
        for l in lines:
            if l.endswith(".txt"):
                os.remove(os.path.join(files[key], l))
        lines = glob.glob(files[key] + '/*')
        lines_list[key] = lines

    for key in lines_list:
        mean_reverse_kl = []
        l2_list = []

        for file in lines_list[key]:
            model_path = file + '/models/'
            with open(file + '/args.yml') as f:
                # use safe_load instead load
                args_dict = yaml.safe_load(f)
                file_name = args_dict['gail_experts_dir'] + args_dict['env_name'] + '_num_traj_' + str(4) + '.pt'
                tracking_expert_dataset = gail.ExpertDataset(
                    file_name, num_trajectories=4, subsample_frequency=1, tracking=True)
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
                actor_critic_ail, obs_rms_ail = \
                    torch.load(os.path.join(model_path, args_dict['env_name'] + ".pt"),
                               map_location='cpu')

                policy = "/home/kit/anthropomatik/kn6273/Repos/hand_dapg/dapg/policies/{}.pickle".format(
                    args_dict['env_name'])
                pi = pickle.load(open(policy, 'rb'))
                vec_norm_ail = get_vec_normalize(env)
                vec_norm_expert = get_vec_normalize(env)
                if vec_norm_ail is not None:
                    vec_norm_ail.eval()
                    vec_norm_ail.obs_rms = obs_rms_ail

                tracking_trajs = tracking_expert_dataset.get_traj()
                for traj in range(4):
                    normalized_expert_state_ail = vec_norm_ail._obfilt(
                        tracking_trajs['states'][traj].type(torch.FloatTensor).numpy(), update=False)
                    normalized_expert_state_ail = torch.FloatTensor(normalized_expert_state_ail).to("cpu")

                    _, tracking_dist_ail = actor_critic_ail.evaluate_actions(normalized_expert_state_ail)
                    means = None
                    stds = None

                    for i in range(tracking_trajs['states'][traj].shape[0]):
                        o = tracking_trajs['states'][traj][i, :]
                        if means is None:
                            means = torch.from_numpy(pi.get_action(o)[1]['mean']).unsqueeze(0)
                            stds = torch.exp(torch.from_numpy(pi.get_action(o)[1]['log_std'])).unsqueeze(0)
                        else:
                            means = torch.cat([means, torch.from_numpy(pi.get_action(o)[1]['mean']).unsqueeze(0)],
                                              dim=0)
                            stds = torch.cat(
                                [stds, torch.exp(torch.from_numpy(pi.get_action(o)[1]['log_std'])).unsqueeze(0)], dim=0)

                    tracking_dist_expert = FixedNormal(means, stds.to(torch.float32))
                    diff_actions = tracking_dist_ail.mode() - tracking_trajs['actions'][traj].type(torch.FloatTensor)

                    l2_list.append(torch.mean(torch.linalg.norm(diff_actions, ord=2, dim=1)).item())
                    metrics = compute_metrics(tracking_dist_ail, tracking_dist_expert)
                    mean_reverse_kl.append(metrics["kl"])

        upper, lower = CI(l2_list)
        l2_list = [item for item in l2_list if item > lower or item < upper]
        upper, lower = CI(mean_reverse_kl)
        mean_reverse_kl = [item for item in mean_reverse_kl if item > lower or item < upper]
        print("{}: Evaluation using {} files: reverse kl {:.5f} , l2 {:.5f}\n".format(key,
                                                                                      len(lines_list[key]),
                                                                                      np.mean(mean_reverse_kl),
                                                                                      np.mean(l2_list)))


if __name__ == "__main__":
    paths_mujoco = {
        'hum_4_trl': '/home/kit/anthropomatik/kn6273/Repos/logs/training_21.10/training_hum/Humanoid-v2/mean_0.007/Humanoid-v2',
        'hum_10_trl': '/home/kit/anthropomatik/kn6273/Repos/logs/training_22.10/training_hum_10_traj/Humanoid-v2/trl/Humanoid-v2'}

    paths_ardoit = {}
    # evaluate reverse KL and L2 distance
    eval_dist_ardoit(paths_ardoit)
    eval_dist(paths_mujoco)
    # evaluate performance based on simulation rewards
    eval_best_performance(paths_ardoit)
    eval_best_performance(paths_mujoco)
