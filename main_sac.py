import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from models.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from models.replay_buffer import ReplayMemory
from utils.arguments import get_args_dict
from models import gail
import utils.utils
import os
import shutil
import yaml
import csv
from collections import deque


def main(config=None, args_dict=None, overwrite=False):
    if args_dict is None:
        args_dict = get_args_dict(config=config)

    exp_name = "logs_gail_sac/"
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

    fnames = ['total_num_steps', 'mean_training_episode_reward', 'mean_eval_episode_rewards']
    csv_writer = None
    if args_dict['logging']:
        csv_writer = csv.DictWriter(f, fieldnames=fnames)
        csv_writer.writeheader()

        with open(log_dir_ + '/args.yml', 'w') as outfile:
            yaml.dump(args_dict, outfile, default_flow_style=False)

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args_dict['env_name'])
    env.seed(args_dict['seed'])
    env.action_space.seed(args_dict['seed'])

    torch.manual_seed(args_dict['seed'])
    np.random.seed(args_dict['seed'])

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args_dict)

    # Tesnorboard
    writer = SummaryWriter(
        'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args_dict['env_name'],
                                      "Gaussian", "autotune" if args_dict['automatic_entropy_tuning'] else ""))

    # Memory
    memory = ReplayMemory(1000000, args_dict['seed'])

    # Training Loop
    total_numsteps = 0
    updates = 0
    device = 'cpu'

    discr = None
    gail_train_loader = None
    if args_dict['use_gail']:
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
        assert len(env.observation_space.shape) == 1
        discr = gail.Discriminator(
            input_dim=(env.observation_space.shape[0] + env.action_space.shape[0]),
            hidden_dim=100, device=device, gradient_penalty=args_dict['gradient_penalty'], lr_disc=args_dict['lr_disc'],
            spectral_norm=args_dict['spectral_norm'],
            airl_reward=args_dict['airl_reward'])
        drop_last = len(expert_dataset) > args_dict['gail_batch_size']
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args_dict['gail_batch_size'],
            shuffle=True,
            drop_last=drop_last)

    episode_rewards = deque(maxlen=5)
    best_eval = -np.inf
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            if args_dict['start_steps'] > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args_dict['mini_batch_size']:
                if args_dict['use_gail']:
                    for _ in range(args_dict['updates_per_step']):
                        # expert_batch = expert_dataset.sample_batch(batch_size=args_dict['gail_batch_size'])
                        _, disc_grad_norm_epoch, acc_policy_epoch, acc_expert_epoch = \
                            discr.update_sac(gail_train_loader, memory)
                # Number of updates per step in environment
                for i in range(args_dict['updates_per_step']):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args_dict[
                                                                                                             'mini_batch_size'],
                                                                                                         updates,
                                                                                                         discr=discr)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        if total_numsteps > args_dict['num_env_steps']:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        episode_rewards.append(episode_reward)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      np.mean(episode_rewards), 2))

        if i_episode % 10 == 0:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

            if args_dict['save_model'] and avg_reward >= best_eval:
                torch.save([
                    agent.policy,
                    getattr(utils.utils.get_vec_normalize(env), 'obs_rms', None)
                ], os.path.join(log_dir_ + '/models', args_dict['env_name'] + ".pt"))
                best_eval = avg_reward

            if args_dict['logging']:
                csv_writer.writerow({'total_num_steps': total_numsteps,
                                     'mean_training_episode_reward': np.mean(episode_rewards),
                                     'mean_eval_episode_rewards': avg_reward})
                f.flush()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--config', default='configs/ppo.yaml', help='config file with training parameters')
    parser.add_argument('-o', dest='overwrite', action='store_true')

    args_ = parser.parse_args()
    main(config=args_.config, overwrite=args_.overwrite)
