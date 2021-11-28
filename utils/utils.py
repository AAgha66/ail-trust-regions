import glob
import os

import torch.nn as nn
from utils.envs import VecNormalize

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)


def compute_kurtosis(norms):
    m = sum(norms) / len(norms)
    # calculate variance using a list comprehension
    fourth_moment = sum((xi - m) ** 4 for xi in norms) / len(norms)
    second_moment = (sum((xi - m) ** 2 for xi in norms) / len(norms)) ** 2
    return (fourth_moment / second_moment) ** 0.25


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def initialize_weights(model: nn.Module, initialization_type: str, scale: float = 2 ** 0.5, init_w=3e-3):
    """
    Weight initializer for the layer or model.
    Args:
        model: module to initialize
        initialization_type: type of inialization
        scale: gain value for orthogonal init
        init_w: init weight for normal and uniform init
    Returns:
    """

    for p in model.parameters():
        if initialization_type == "normal":
            if len(p.data.shape) >= 2:
                p.data.normal_(init_w)  # 0.01
            else:
                p.data.zero_()
        elif initialization_type == "uniform":
            if len(p.data.shape) >= 2:
                p.data.uniform_(-init_w, init_w)
            else:
                p.data.zero_()
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_normal_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                nn.init.orthogonal_(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError(
                "Not a valid initialization type. Choose one of 'normal', 'uniform', 'xavier', and 'orthogonal'")


def get_exp_name(params):
    exp_name = f"{'gail' + str(params['use_gail']) + '-'}" \
               f"lr_p{params['lr_policy']:.2E}-" \
               f"lr_v{params['lr_value']:.2E}-" \
               f"lr_d{params['lr_disc']:.2E}-" \
               f"{'pen' + str(params['gradient_penalty']) + '-'}" \
               f"{'g_clip' + str(params['gradient_clipping']) + '-'}" \
               f"{'target_entropy' + str(params['target_entropy']) + '-'}" \
               f"{'s' + str(params['seed']) + '-'}" \
               f"{'batch_size' + str(params['mini_batch_size']) + '-'}" \
               f"{'clip_ir' + str(params['clip_importance_ratio']) + '-'}" \
               f"{'proj' + str(params['use_projection']) + '-'}" \
               f"{'p_type' + str(params['proj_type']) + '-'}" \
               f"{'lr_dec' + str(params['use_linear_lr_decay']) + '-'}" \
               f"{'clip_v' + str(params['use_clipped_value_loss']) + '-'}" \
               f"{'n_o' + str(params['norm_obs']) + '-'}" \
               f"{'n_r' + str(params['norm_reward']) + '-'}" \
               f"c_o{params['clip_obs']:.2E}-" \
               f"c_r{params['clip_reward']:.2E}" \
               f"cov{params['cov_bound']:.2E}" \
               f"mean{params['mean_bound']:.2E}"

    return exp_name
