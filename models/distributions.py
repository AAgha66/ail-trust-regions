import torch
import torch.nn as nn
from utils.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


#
# Standardize distribution interfaces
#

class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

    def log_determinant(self):
        """
        Returns the log determinant of a diagonal matrix
        Returns:
            The log determinant of std, aka log sum the diagonal
        """
        return 2 * self.stddev.log().sum(-1)

    def maha(self, mean_other: torch.Tensor):
        diff = self.mean - mean_other
        return (diff / self.stddev).pow(2).sum(-1)

    def precision(self):
        return (1 / self.covariance()).diag_embed()

    def covariance(self):
        return self.stddev.pow(2)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), gain=0.01)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())