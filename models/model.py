import numpy as np
import torch.nn as nn
import torch
from models.distributions import DiagGaussian
from utils.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        base = MLPBase
        self.base = base(obs_shape[0], **base_kwargs)
        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(self.base.output_size, num_outputs)
        self.entropy_bound = None

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        batched_bound = None
        if self.entropy_bound is not None:
            batched_bound = self.entropy_bound * torch.ones(inputs.shape[0])
        dist = self.dist(actor_features, batched_bound)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return value, action, dist

    def get_action(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        batched_bound = None
        if self.entropy_bound is not None:
            batched_bound = self.entropy_bound * torch.ones(inputs.shape[0])
        dist = self.dist(actor_features, batched_bound)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return action

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs):
        value, actor_features = self.base(inputs)
        batched_bound = None
        if self.entropy_bound is not None:
            batched_bound = self.entropy_bound * torch.ones(inputs.shape[0])
        dist = self.dist(actor_features, batched_bound)

        return value, dist


class NNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(NNBase, self).__init__()
        self._num_inputs = num_inputs
        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLPBase, self).__init__(num_inputs, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.train()

    def forward(self, inputs):
        values = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return values, hidden_actor
