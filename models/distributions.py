import torch
import torch.nn as nn
from utils.utils import AddBias, init
from utils.distribution_utils import FixedNormal
from utils.projection_utils import get_entropy_schedule
from projections.base_projection_layer import entropy_equality_projection
"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


#
# Standardize distribution interfaces
#

# Normal
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, target_entropy, temperature, entropy_schedule, total_train_steps):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), gain=0.01)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.target_entropy = target_entropy
        self.temperature = temperature
        if  entropy_schedule is not "None":
            self.entropy_schedule = get_entropy_schedule(entropy_schedule, total_train_steps, dim=num_outputs)
        self.initial_entropy = None

    def forward(self, x, global_steps):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        if self.entropy_schedule and global_steps is not None:
            dist = FixedNormal(action_mean, action_logstd.exp())
            if self.initial_entropy == None:
                self.initial_entropy = dist.entropy()
            entropy_bound = self.entropy_schedule(self.initial_entropy, self.target_entropy,
                                                  self.temperature, global_steps) * dist.mean.new_ones(dist.mean.shape[0])
            return entropy_equality_projection(dist, entropy_bound)
        else:
            return FixedNormal(action_mean, action_logstd.exp())
