import numpy as np
import torch
import gym
import argparse
from models.sac import ReplayBuffer, SAC
from utils.arguments import get_args_dict


def test_agent(agent, num_test_episodes, test_env, max_ep_len):
    ret_lst = []
    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            o, r, d, _ = test_env.step(agent.get_action(o, True))
            ep_ret += r
            ep_len += 1
        ret_lst.append(ep_ret)
    return ret_lst


def main(config=None, args_dict=None, overwrite=False):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    if args_dict is None:
        args_dict = get_args_dict(config=config)
    torch.manual_seed(args_dict['seed'])
    np.random.seed(args_dict['seed'])

    env, test_env = gym.make(args_dict['env_name']), gym.make(args_dict['env_name'])
    env.seed(args_dict['seed'])
    test_env.seed(args_dict['seed'])

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    agent = SAC(env)
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(args_dict['replay_size']))

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(int(args_dict['num_env_steps'])):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > args_dict['start_steps']:
            a = agent.get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == args_dict['max_ep_len'] else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == args_dict['max_ep_len']):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= args_dict['update_after'] and t % args_dict['update_every'] == 0:
            for j in range(args_dict['update_every']):
                batch = replay_buffer.sample_batch(args_dict['mini_batch_size'])
                agent.update(data=batch)

        # End of epoch handling
        if (t + 1) % args_dict['steps_per_epoch'] == 0:
            # Test the performance of the deterministic version of the agent.
            ret_lst = test_agent(agent, args_dict['num_test_episodes'], test_env, args_dict['max_ep_len'])
            print("step: {},  Evaluation {} episodes: {}".format(t, args_dict['num_test_episodes'], np.mean(ret_lst)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--config', default='configs/ppo.yaml', help='config file with training parameters')
    parser.add_argument('-o', dest='overwrite', action='store_true')

    args = parser.parse_args()
    main(config=args.config, overwrite=args.overwrite)
