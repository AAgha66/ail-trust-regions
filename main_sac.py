import numpy as np
import torch
import argparse
from models.sac import ReplayBuffer, SAC
from utils.arguments import get_args_dict
import utils.utils
from collections import deque
from utils.envs import make_vec_envs
from evaluation import evaluate
from models import gail


def main(config=None, args_dict=None, overwrite=False):
    if args_dict is None:
        args_dict = get_args_dict(config=config)

    exp_name = "sac_gail/"
    log_dir_ = args_dict['logging_dir'] + args_dict['env_name'] + '/' + exp_name

    if args_dict is None:
        args_dict = get_args_dict(config=config)

    torch.manual_seed(args_dict['seed'])
    torch.cuda.manual_seed_all(args_dict['seed'])
    np.random.seed(args_dict['seed'])

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args_dict['cuda'] else "cpu")

    env = make_vec_envs(args_dict['env_name'], args_dict['seed'], args_dict['num_processes'],
                        args_dict['gamma'], args_dict['log_dir'], args_dict['norm_obs'],
                        args_dict['norm_reward'], args_dict['clip_obs'], args_dict['clip_reward'],
                        device, False)
    env.action_space.seed(args_dict['seed'])
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

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
                a = torch.unsqueeze(torch.from_numpy(a), dim=0)

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
            if args_dict['use_gail']:
                gail_epoch = args_dict['gail_epoch']
                if t < 10 * args_dict['update_after']:
                    gail_epoch = 100  # Warm up
                for _ in range(gail_epoch):
                    # expert_batch = expert_dataset.sample_batch(batch_size=args_dict['gail_batch_size'])
                    _, disc_grad_norm_epoch, acc_policy_epoch, acc_expert_epoch = \
                        discr.update_sac(gail_train_loader, replay_buffer,
                                         utils.utils.get_vec_normalize(env)._obfilt)
            for j in range(args_dict['update_every']):
                batch = replay_buffer.sample_batch(args_dict['mini_batch_size'])
                if args_dict['use_gail']:
                    batch['rew'] = discr.predict_reward(
                        batch['obs'], batch['act'], gamma=None, masks=d, update_rms=False,
                        use_disc_as_adv=False)
                    batch['rew'] = torch.squeeze(batch['rew'], dim=-1)
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
