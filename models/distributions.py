import torch
import torch.nn as nn
from utils.utils import AddBias, init
from utils.distribution_utils import FixedNormal
from projections.base_projection_layer import entropy_equality_projection
"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


#
# Standardize distribution interfaces
#

# Normal
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), gain=0.01)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x, entropy_bound=None):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)

        if entropy_bound is not None:
            dist = FixedNormal(action_mean, action_logstd.exp())
            return entropy_equality_projection(dist, entropy_bound)
        else:
            return FixedNormal(action_mean, action_logstd.exp())
