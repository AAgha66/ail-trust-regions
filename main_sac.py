import numpy as np
import torch
import argparse
from models.sac import ReplayBuffer, SAC
from utils.arguments import get_args_dict
import utils.utils
from collections import deque
from utils.envs import make_vec_envs
from evaluation import evaluate


def main(config=None, args_dict=None, overwrite=False):
    if args_dict is None:
        args_dict = get_args_dict(config=config)

    exp_name = "sac_gail/"
    log_dir_ = args_dict['logging_dir'] + args_dict['env_name'] + '/' + exp_name

    if args_dict is None:
        args_dict = get_args_dict(config=config)

    torch.manual_seed(args_dict['seed'])
    torch.cuda.manual_seed_all(args_dict['seed'])

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args_dict['cuda'] else "cpu")

    env = make_vec_envs(args_dict['env_name'], args_dict['seed'], args_dict['num_processes'],
                        args_dict['gamma'], args_dict['log_dir'], args_dict['norm_obs'],
                        args_dict['norm_reward'], args_dict['clip_obs'], args_dict['clip_reward'],
                        device, False)
    env.action_space.seed(args_dict['seed'])
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    agent = SAC(env)
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(args_dict['replay_size']))

    # Prepare for interaction with environment
    o = env.reset()
    episode_rewards = deque(maxlen=5)
    # Main loop: collect experience in env and update/log each epoch
    for t in range(int(args_dict['num_env_steps'])):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        with torch.no_grad():
            if t > args_dict['start_steps']:
                a = agent.get_action(o)
            else:
                a = env.action_space.sample()
                a = torch.from_numpy(a)

        # Step the env
        o2, r, d, infos = env.step(a)
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # Update handling
        if t >= args_dict['update_after'] and t % args_dict['update_every'] == 0:
            for j in range(args_dict['update_every']):
                batch = replay_buffer.sample_batch(args_dict['mini_batch_size'])
                agent.update(data=batch)

        if t % args_dict['log_interval'] == 0 and len(episode_rewards) > 1:
            print(
                "num timesteps {}, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(t,
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards)))

        if t % args_dict['eval_interval'] == 0 and t > 0:
            obs_rms = utils.utils.get_vec_normalize(env).obs_rms
            mean_eval_episode_rewards = evaluate(agent, obs_rms, args_dict['env_name'], args_dict['seed'],
                                                 args_dict['num_processes'], args_dict['log_dir'],
                                                 args_dict['norm_obs'], args_dict['norm_reward'], args_dict['clip_obs'],
                                                 args_dict['clip_reward'], device)
            print("step: {}, Evaluation: {}".format(t, mean_eval_episode_rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--config', default='configs/ppo.yaml', help='config file with training parameters')
    parser.add_argument('-o', dest='overwrite', action='store_true')

    args = parser.parse_args()
    main(config=args.config, overwrite=args.overwrite)
