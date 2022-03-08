import argparse
import os
import shutil
from collections import deque
import csv
import numpy as np
import torch
import yaml
import utils.utils
from models import gail, ppo, trpo
from utils.arguments import get_args_dict
from utils.envs import make_vec_envs
from models.model import Policy
from evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter
import mj_envs
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

def main(config=None, args_dict=None, overwrite=False):
    if args_dict is None:
        args_dict = get_args_dict(config=config)

    exp_name = "BC"
    log_dir_ = args_dict['logging_dir'] + args_dict['env_name'] + '/' + exp_name

    f = None
    if args_dict['logging']:
        if os.path.isdir(log_dir_):
            if overwrite:
                shutil.rmtree(log_dir_)
            else:
                print("experiment already exists !")
                return

        os.makedirs(log_dir_)
        os.makedirs(log_dir_ + '/summary')
        os.makedirs(log_dir_ + '/logs')
        os.makedirs(log_dir_ + '/models')

        f = open(log_dir_ + '/logs/log.csv', 'w')        

    fnames = ['total_num_steps', 'mean_eval_episode_rewards', 'action_loss_epoch',
              'tracking_log_probs_mean', 'tracking_log_probs_median', 'tracking_diff_actions_norm_mean', 'tracking_diff_actions_norm_median']

    csv_writer = None
    if args_dict['logging']:
        csv_writer = csv.DictWriter(f, fieldnames=fnames)
        csv_writer.writeheader()        

        with open(log_dir_ + '/args.yml', 'w') as outfile:
            yaml.dump(args_dict, outfile, default_flow_style=False)

    torch.manual_seed(args_dict['seed'])
    torch.cuda.manual_seed_all(args_dict['seed'])

    if args_dict['cuda'] and torch.cuda.is_available() and args_dict['cuda_deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args_dict['cuda'] else "cpu")

    envs = make_vec_envs(args_dict['env_name'], args_dict['seed'], args_dict['num_processes'],
                         args_dict['gamma'], args_dict['log_dir'], args_dict['norm_obs'],
                         args_dict['norm_reward'], args_dict['clip_obs'], args_dict['clip_reward'],
                         device, False)
    num_updates = int(args_dict['num_env_steps'] // args_dict['num_steps'] // args_dict['num_processes'])

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)
    actor_critic.to(device)
    
    gail_train_loader = None
    expert_dataset = None
    tracking_expert_dataset = None
    if args_dict['track_vf']:
        file_name = args_dict['gail_experts_dir'] + args_dict['env_name'] + \
                    '_num_traj_' + str(4) + '.pt'
        tracking_expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=1, tracking=True)

    file_name = args_dict['gail_experts_dir'] + args_dict['env_name'] + \
                '_num_traj_' + str(args_dict['num_trajectories']) + '.pt'
    subsample_frequency = None
    if args_dict['env_name'] == "Reacher-v2":
        subsample_frequency = 1
    elif args_dict['env_name'] == "door-v0" or args_dict['env_name'] == "hammer-v0" or args_dict[
        'env_name'] == "relocate-v0":
        subsample_frequency = 4
    elif args_dict['env_name'] == "pen-v0":
        subsample_frequency = 2
    else:
        subsample_frequency = 20
    expert_dataset = gail.ExpertDataset(
        file_name, num_trajectories=args_dict['num_trajectories'], subsample_frequency=subsample_frequency)

    assert len(envs.observation_space.shape) == 1        
    drop_last = len(expert_dataset) > args_dict['gail_batch_size']
    gail_train_loader = torch.utils.data.DataLoader(
        dataset=expert_dataset,
        batch_size=args_dict['gail_batch_size'],
        shuffle=True,
        drop_last=drop_last)

    policy_params = list(actor_critic.base.actor.parameters()) + list(actor_critic.dist.parameters())
    optimizer = optim.Adam(policy_params, lr=args_dict['lr_policy'], eps=args_dict['eps'])

    writer = None
    if args_dict['summary']:
        writer = SummaryWriter(log_dir_ + '/summary')
    list_eval_rewards = []
    best_eval = 0.0
    # variables needed for tracking the gradients values in the tensorboard
    for j in range(num_updates):
        action_loss_epoch = []    
        for data in gail_train_loader:
            exp_state, exp_action = data
            exp_state = utils.utils.get_vec_normalize(envs)._obfilt(exp_state.numpy(), update=False)
            exp_state = torch.FloatTensor(exp_state)
            exp_state = Variable(exp_state).to(device)
            exp_action = Variable(exp_action).to(device)
            _, expert_dist = actor_critic.evaluate_actions(exp_state)
            action_log_probs = expert_dist.log_probs(exp_action)
            
            loss = -action_log_probs.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(),
                                    args_dict['max_grad_norm'])
            optimizer.step()
            action_loss_epoch.append(loss.data.cpu().numpy())
        
        mean_eval_episode_rewards = None
        
        if (args_dict['eval_interval'] is not None and (j % args_dict['eval_interval'] == 0 or j == num_updates - 1)):
            obs_rms = utils.utils.get_vec_normalize(envs).obs_rms
            mean_eval_episode_rewards = evaluate(actor_critic, obs_rms, args_dict['env_name'], args_dict['seed'],
                                                 args_dict['num_processes'], args_dict['log_dir'],
                                                 args_dict['norm_obs'], args_dict['norm_reward'], args_dict['clip_obs'],
                                                 args_dict['clip_reward'], device)
            list_eval_rewards.append(mean_eval_episode_rewards)
            print("Epoch: {}, Evaluation: {}".format(j, np.mean(mean_eval_episode_rewards)))
            if args_dict['summary']:
                writer.add_scalar('mean_eval_episode_rewards',
                                  mean_eval_episode_rewards, j)
            # save for every interval-th episode or for the last epoch
            if args_dict['save_model'] and mean_eval_episode_rewards >= best_eval:
                torch.save([
                    actor_critic,
                    getattr(utils.utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(log_dir_ + '/models', args_dict['env_name'] + ".pt"))

                best_eval = mean_eval_episode_rewards

        tracking_log_probs = []
        tracking_diff_actions_norm = []

        tracking_log_probs_mean = None
        tracking_log_probs_median = None
        tracking_diff_actions_norm_mean = None
        tracking_diff_actions_norm_median = None
        if args_dict['track_vf'] and j % args_dict['track_interval'] == 0:
            tracking_trajs = tracking_expert_dataset.get_traj()
            with torch.no_grad():
                for traj in range(4):
                    normalized_expert_state = utils.utils.get_vec_normalize(envs)._obfilt(
                        tracking_trajs['states'][traj].type(torch.FloatTensor).numpy(), update=False)
                    normalized_expert_state = torch.FloatTensor(normalized_expert_state).to(device)
                    _, tracking_dist = actor_critic.evaluate_actions(normalized_expert_state)
                    tracking_log_probs.append(
                        tracking_dist.log_probs(tracking_trajs['actions'][traj].type(torch.FloatTensor)))
                    diff_actions = tracking_dist.mode() - tracking_trajs['actions'][traj].type(torch.FloatTensor)
                    tracking_diff_actions_norm.append(torch.linalg.norm(diff_actions, ord=2, dim=1))

                tracking_log_probs = torch.cat(tracking_log_probs, dim=0)
                tracking_diff_actions_norm = torch.cat(tracking_diff_actions_norm, dim=0)

                tracking_log_probs_mean = tracking_log_probs.mean().item()
                tracking_log_probs_median = tracking_log_probs.median().item()

                tracking_diff_actions_norm_mean = tracking_diff_actions_norm.mean().item()
                tracking_diff_actions_norm_median = tracking_diff_actions_norm.median().item()

        if args_dict['summary']:
            writer.add_scalar('action_loss_epoch',
                              np.mean(action_loss_epoch), j)                        
            if args_dict['track_vf'] and j % 30 == 0:
                writer.add_scalar('tracking_log_probs_mean',
                                  tracking_log_probs_mean, j)
                writer.add_scalar('tracking_log_probs_median',
                                  tracking_log_probs_median, j)
                writer.add_scalar('tracking_diff_actions_norm_mean',
                                  tracking_diff_actions_norm_mean, j)
                writer.add_scalar('tracking_diff_actions_norm_median',
                                  tracking_diff_actions_norm_median, j)
        if args_dict['logging']:
            csv_writer.writerow({'total_num_steps': j,
                                 'mean_eval_episode_rewards': mean_eval_episode_rewards,
                                 'action_loss_epoch': np.mean(action_loss_epoch),
                                 'tracking_log_probs_mean': tracking_log_probs_mean,
                                 'tracking_log_probs_median': tracking_log_probs_median,
                                 'tracking_diff_actions_norm_mean': tracking_diff_actions_norm_mean,
                                 'tracking_diff_actions_norm_median': tracking_diff_actions_norm_median})

            f.flush()            
    if args_dict['summary']:
        writer.close()
    print('Finished Training: ' + str(sum(list_eval_rewards) / len(list_eval_rewards)))
    return sum(list_eval_rewards) / len(list_eval_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--config', default='configs/ppo.yaml', help='config file with training parameters')
    parser.add_argument('-o', dest='overwrite', action='store_true')

    args = parser.parse_args()
    main(config=args.config, overwrite=args.overwrite)
