import argparse
import os
import time
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
from utils.storage import RolloutStorage
from evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter


def main(config=None, args_dict=None):
    if args_dict is None:
        args_dict = get_args_dict(config=config)

    exp_name = utils.utils.get_exp_name(args_dict)
    log_dir_ = args_dict['logging_dir'] + args_dict['env_name'] + '/' + exp_name

    f = None
    f_grads = None
    f_adv = None
    if args_dict['logging']:
        if os.path.isdir(log_dir_):
            print("experiment already exists !")
            return
        else:
            os.makedirs(log_dir_)
            os.makedirs(log_dir_ + '/summary')
            os.makedirs(log_dir_ + '/logs')

        f = open(log_dir_ + '/logs/log.csv', 'w')
        f_grads = open(log_dir_ + '/logs/log_grads.csv', 'w')
        f_adv = open(log_dir_ + '/logs/log_adv.csv', 'w')

    fnames = ['total_num_steps', 'mean_training_episode_reward', 'mean_eval_episode_rewards', 'value_loss_epoch',
              'action_loss_epoch', 'trust_region_loss_epoch', 'kl_mean', 'entropy_mean', 'entropy_diff_mean']
    fnames_grads = ['total_num_steps', 'norm_grad_disc', 'acc_expert', 'acc_policy']
    fnames_adv = ['total_num_steps', 'pair_id', 'advantages', 'rewards', 'values']

    csv_writer = None
    csv_writer_grads = None
    csv_writer_adv = None
    if args_dict['logging']:
        csv_writer = csv.DictWriter(f, fieldnames=fnames)
        csv_writer.writeheader()
        csv_writer_grads = csv.DictWriter(f_grads, fieldnames=fnames_grads)
        csv_writer_grads.writeheader()
        csv_writer_adv = csv.DictWriter(f_adv, fieldnames=fnames_adv)
        csv_writer_adv.writeheader()
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

    if args_dict['algo'] == 'ppo':
        agent = ppo.PPO(
            actor_critic=actor_critic,
            clip_param=args_dict['clip_param'],
            policy_epoch=args_dict['policy_epoch'],
            mini_batch_size=args_dict['mini_batch_size'],
            value_loss_coef=args_dict['value_loss_coef'],
            entropy_coef=args_dict['entropy_coef'],
            num_steps=args_dict['num_steps'],
            use_kl_penalty=args_dict['use_kl_penalty'],
            use_rollback=args_dict['use_rollback'],
            use_tr_ppo=args_dict['use_tr_ppo'],
            use_truly_ppo=args_dict['use_truly_ppo'],
            rb_alpha=args_dict['rb_alpha'],
            beta=args_dict['start_beta'],
            kl_target=args_dict['kl_target'],
            lr_value=args_dict['lr_value'],
            lr_policy=args_dict['lr_policy'],
            eps=args_dict['eps'],
            proj_type=args_dict['proj_type'],
            max_grad_norm=args_dict['max_grad_norm'],
            use_clipped_value_loss=args_dict['use_clipped_value_loss'],
            use_projection=args_dict['use_projection'],
            action_space=envs.action_space,
            total_train_steps=num_updates,
            entropy_schedule=args_dict['entropy_schedule'],
            scale_prec=args_dict['scale_prec'],
            entropy_eq=args_dict['entropy_eq'],
            entropy_first=args_dict['entropy_first'],
            clip_importance_ratio=args_dict['clip_importance_ratio'],
            gradient_clipping=args_dict['gradient_clipping'],
            mean_bound=args_dict['mean_bound'],
            cov_bound=args_dict['cov_bound'],
            trust_region_coeff=args_dict['trust_region_coeff'])
    elif args_dict['algo'] == 'trpo':
        agent = trpo.TRPO(actor_critic=actor_critic,
                          vf_epoch=args_dict['vf_epoch'],
                          lr_value=args_dict['lr_value'],
                          eps=args_dict['eps'],
                          action_space=envs.action_space,
                          obs_space=envs.observation_space,
                          num_steps=args_dict['num_steps'],
                          max_kl=args_dict['max_kl'],
                          cg_damping=args_dict['cg_damping'],
                          cg_max_iters=args_dict['cg_max_iters'],
                          line_search_coef=args_dict['line_search_coef'],
                          line_search_max_iter=args_dict['line_search_max_iter'],
                          line_search_accept_ratio=args_dict['line_search_accept_ratio'],
                          mini_batch_size=args_dict['mini_batch_size'])
    else:
        raise NotImplementedError("Policy optimization method not implemented!")



    discr = None
    gail_train_loader = None
    if args_dict['use_gail']:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device, args_dict['gradient_penalty'], lr_disc=args_dict['lr_disc'])
        file_name = os.path.join(
            args_dict['gail_experts_dir'], "trajs_{}.pt".format(
                args_dict['env_name'].split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=args_dict['num_trajectories'], subsample_frequency=20)
        drop_last = len(expert_dataset) > args_dict['gail_batch_size']
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args_dict['gail_batch_size'],
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args_dict['num_steps'], args_dict['num_processes'],
                              envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    tracking_file_name = os.path.join(
        args_dict['gail_experts_dir'], "trajs_{}.pt".format(
            args_dict['env_name'].split('-')[0].lower()))
    tracking_dataset = gail.ExpertDataset(
        tracking_file_name, num_trajectories=args_dict['num_trajectories'],
        subsample_frequency=1)

    episode_rewards = deque(maxlen=5)

    start = time.time()

    writer = None
    if args_dict['summary']:
        writer = SummaryWriter(log_dir_ + '/summary')

    # variables needed for tracking the gradients values in the tensorboard
    gail_iters = 0
    list_eval_rewards = []
    best_eval = -np.inf

    for j in range(num_updates):
        if args_dict['use_linear_lr_decay']:
            # decrease learning rate linearly
            utils.utils.update_linear_schedule(
                agent.optimizer_policy, j, num_updates, args_dict['lr_policy'])
            utils.utils.update_linear_schedule(
                agent.optimizer_vf, j, num_updates, args_dict['lr_value'])

        for step in range(args_dict['num_steps']):
            # Sample actions
            with torch.no_grad():
                value, action, dist = actor_critic.act(rollouts.obs[step])
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, action, dist.mean.detach(), dist.stddev.detach(),
                            value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()
        gail_norm_grad = []
        acc_policy = []
        acc_expert = []
        if args_dict['use_gail']:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args_dict['gail_epoch']
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                _, gail_norm_grad_epoch, acc_policy_epoch, acc_expert_epoch = \
                    discr.update(gail_train_loader, rollouts,
                                 utils.utils.get_vec_normalize(envs)._obfilt)

                gail_norm_grad.extend(gail_norm_grad_epoch)
                acc_policy.extend(acc_policy_epoch)
                acc_expert.extend(acc_expert_epoch)

            for step in range(args_dict['num_steps']):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args_dict['gamma'],
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args_dict['use_gae'], args_dict['gamma'],
                                 args_dict['gae_lambda'], args_dict['use_proper_time_limits'])

        metrics = agent.update(rollouts)

        rollouts.after_update()
        total_num_steps = (j + 1) * args_dict['num_processes'] * args_dict['num_steps']
        if j % args_dict['log_interval'] == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards)))
            if args_dict['summary']:
                writer.add_scalar('mean_training_episode_reward',
                                  np.mean(episode_rewards), total_num_steps)

        mean_eval_episode_rewards = None
        if (args_dict['eval_interval'] is not None and len(episode_rewards) > 1
                and (j % args_dict['eval_interval'] == 0 or j == num_updates - 1)):
            obs_rms = utils.utils.get_vec_normalize(envs).obs_rms
            mean_eval_episode_rewards = evaluate(actor_critic, obs_rms, args_dict['env_name'], args_dict['seed'],
                                                 args_dict['num_processes'], args_dict['log_dir'],
                                                 args_dict['norm_obs'], args_dict['norm_reward'], args_dict['clip_obs'],
                                                 args_dict['clip_reward'], device)
            list_eval_rewards.append(mean_eval_episode_rewards)
            print("Evaluation: " + str(mean_eval_episode_rewards))
            if args_dict['summary']:
                writer.add_scalar('mean_eval_episode_rewards',
                                  mean_eval_episode_rewards, total_num_steps)
            # save for every interval-th episode or for the last epoch
            if args_dict['save_model'] and mean_eval_episode_rewards >= best_eval:
                save_path = log_dir_ + '/models'
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    actor_critic,
                    getattr(utils.utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args_dict['env_name'] + ".pt"))

                best_eval = mean_eval_episode_rewards

        tracking_trajs = tracking_dataset.get_traj()
        tracking_adv = []
        tracking_rewards = []
        tracking_values = []

        with torch.no_grad():
            for traj in range(args_dict['num_trajectories']):
                values, _ = actor_critic.evaluate_actions(
                    tracking_trajs['states'][traj].type(torch.FloatTensor))
                tracking_values.append(values)

                if args_dict['use_gail']:
                    disc_rewards = discr.predict_reward(
                        tracking_trajs['states'][traj].type(torch.FloatTensor),
                        tracking_trajs['actions'][traj].type(torch.FloatTensor),
                        args_dict['gamma'],
                        torch.ones(1000, 1))
                    tracking_rewards.append(disc_rewards)
                else:
                    tracking_rewards.append(tracking_trajs['rewards'][traj])

                gae = 0
                adv = torch.zeros(tracking_rewards[-1].shape[0] - 1, 1)
                for step in reversed(range(tracking_rewards[-1].shape[0] - 1)):
                    rewards = tracking_rewards[-1]
                    values = tracking_values[-1]

                    delta = rewards[step] + args_dict['gamma'] * values[step + 1] - \
                            values[step]
                    gae = delta + args_dict['gamma'] * args_dict['gae_lambda'] * gae
                    adv[step] = gae

                tracking_adv.append(adv)

        tracking_adv = torch.cat(tracking_adv, dim=0)
        tracking_adv = (tracking_adv - tracking_adv.mean()) / (
                tracking_adv.std() + 1e-5)
        tracking_adv = tracking_adv.type(torch.HalfTensor)
        tracking_rewards = torch.cat(tracking_rewards, dim=0).type(torch.HalfTensor)
        tracking_values = torch.cat(tracking_values, dim=0).type(torch.HalfTensor)

        if args_dict['summary']:
            writer.add_scalar('value_loss_epoch',
                              metrics['value_loss_epoch'], total_num_steps)
            writer.add_scalar('action_loss_epoch',
                              metrics['action_loss_epoch'], total_num_steps)
            if args_dict['use_projection']:
                writer.add_scalar('trust_region_loss_epoch',
                                  metrics['trust_region_loss_epoch'], total_num_steps)
            writer.add_scalar('kl_mean',
                              metrics['kl'], total_num_steps)
            writer.add_scalar('entropy_mean',
                              metrics['entropy'], total_num_steps)

            writer.add_histogram("expert_adv", tracking_adv, total_num_steps)
            writer.add_histogram("expert_rewards", tracking_rewards, total_num_steps)
            writer.add_histogram("expert_values", tracking_values, total_num_steps)

        if args_dict['logging']:
            for i in range(tracking_adv.shape[0]):
                csv_writer_adv.writerow({'total_num_steps': total_num_steps,
                                         'pair_id': i,
                                         'advantages': tracking_adv[i].item(),
                                         'rewards': tracking_rewards[i].item(),
                                         'values': tracking_values[i].item()})

            csv_writer.writerow({'total_num_steps': total_num_steps,
                                 'mean_training_episode_reward': np.mean(episode_rewards),
                                 'mean_eval_episode_rewards': mean_eval_episode_rewards,
                                 'value_loss_epoch': metrics['value_loss_epoch'],
                                 'action_loss_epoch': metrics['action_loss_epoch'],
                                 'trust_region_loss_epoch': metrics['trust_region_loss_epoch'],
                                 'kl_mean': metrics['kl'].item(),
                                 'entropy_mean': metrics['entropy'].item(),
                                 'entropy_diff_mean': metrics['entropy_diff'].item()})

            for i, _ in enumerate(gail_norm_grad):
                csv_writer_grads.writerow({'total_num_steps': total_num_steps,
                                           'norm_grad_disc': gail_norm_grad[i].item(),
                                           'acc_expert': acc_expert[i].item(),
                                           'acc_policy': acc_policy[i].item()})
                if args_dict['summary']:
                    writer.add_scalar('norm_grad_disc',
                                      gail_norm_grad[i], gail_iters)
                    writer.add_scalar('acc_policy',
                                      acc_policy[i], gail_iters)
                    writer.add_scalar('acc_expert',
                                      acc_expert[i], gail_iters)
                gail_iters += 1
            f.flush()
            f_adv.flush()
            f_grads.flush()

    if args_dict['summary']:
        writer.close()
    print('Finished Training: ' + str(sum(list_eval_rewards) / len(list_eval_rewards)))
    return sum(list_eval_rewards) / len(list_eval_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--config', default='configs/ppo.yaml', help='config file with training parameters')
    args = parser.parse_args()
    main(config=args.config)
