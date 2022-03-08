#   Copyright (c) 2021 Robert Bosch GmbH
#   Author: Fabian Otto
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torch as ch
from typing import Tuple
from models.distributions import FixedNormal
from utils.torch_utils import torch_batched_trace


def mean_distance(p_dist, q_dist, scale_prec=False):
    """
    Compute mahalanobis distance for mean or euclidean distance
    Args:
        policy: policy instance
        mean: current mean vectors
        mean_other: old mean vectors
        std_other: scaling covariance matrix
        scale_prec: True computes the mahalanobis distance based on std_other for scaling. False the Euclidean distance.

    Returns:
        Mahalanobis distance or Euclidean distance between mean vectors
    """

    if scale_prec:
        # maha objective for mean
        mean_part = q_dist.maha(p_dist.mean)
    else:
        # euclidean distance for mean
        # mean_part = ch.norm(mean_other - mean, ord=2, axis=1) ** 2
        mean_part = ((q_dist.mean - p_dist.mean) ** 2).sum(1)

    return mean_part


def gaussian_kl(p_dist: FixedNormal, q_dist: FixedNormal) -> Tuple[ch.Tensor, ch.Tensor]:
    """
    Get the expected KL divergence between two sets of Gaussians over states -
    Calculates E KL(p||q): E[sum p(x) log(p(x)/q(x))] in closed form for Gaussians.

    Args:
        policy: policy instance
        p: first distribution tuple (mean, var)
        q: second distribution tuple (mean, var)

    Returns:

    """

    mean = p_dist.mean
    k = mean.shape[-1]

    det_term = p_dist.log_determinant()
    det_term_other = q_dist.log_determinant()

    cov = p_dist.covariance().diag_embed()
    prec_other = q_dist.precision()

    maha_part = .5 * q_dist.maha(mean)
    # trace_part = (var * precision_other).sum([-1, -2])
    trace_part = torch_batched_trace(prec_other @ cov)
    cov_part = .5 * (trace_part - k + det_term_other - det_term)

    return maha_part, cov_part


def compute_metrics(p_dist: FixedNormal, q_dist: FixedNormal) -> dict:
    """
    Returns dict with constraint metrics.
    Args:
        policy: policy instance
        p: current distribution
        q: old distribution

    Returns:
        dict with constraint metrics
    """
    with ch.no_grad():
        entropy_old = q_dist.entropy()
        entropy = p_dist.entropy()

        mean_kl, cov_kl = gaussian_kl(p_dist, q_dist)
        kl = mean_kl + cov_kl

        entropy_diff = entropy_old - entropy

    return {'kl': kl.detach().mean(),
            'entropy': entropy.mean(),
            'entropy_diff': entropy_diff.mean(),
            'kl_max': kl.max(),
            'entropy_max': entropy.max(),
            'entropy_diff_max': entropy_diff.max()
            }


def gaussian_wasserstein_commutative(p_dist: FixedNormal, q_dist: FixedNormal, scale_prec=False) -> Tuple[
    ch.Tensor, ch.Tensor]:
    """
    Compute mean part and cov part of W_2(p || q) with p,q ~ N(y, SS).
    This version DOES assume commutativity of both distributions, i.e. covariance matrices.
    This is less general and assumes both distributions are somewhat close together.
    When scale_prec is true scale both distributions with old precision matrix.
    Args:
        policy: current policy
        p: mean and sqrt of gaussian p
        q: mean and sqrt of gaussian q
        scale_prec: scale objective by old precision matrix.
                    This penalizes directions based on old uncertainty/covariance.

    Returns: mean part of W2, cov part of W2

    """
    sqrt = p_dist.stddev.diag_embed()
    sqrt_other = q_dist.stddev.diag_embed()

    mean_part = mean_distance(p_dist, q_dist, scale_prec)
    cov = p_dist.covariance().diag_embed()
    if scale_prec:
        # cov constraint scaled with precision of old dist
        batch_dim, dim = p_dist.mean.shape
        identity = ch.eye(dim, dtype=sqrt.dtype, device=sqrt.device)
        sqrt_inv_other = ch.solve(identity, sqrt_other)[0]
        c = sqrt_inv_other @ cov @ sqrt_inv_other

        cov_part = torch_batched_trace(identity + c - 2 * sqrt_inv_other @ sqrt)

    else:
        # W2 objective for cov assuming normal W2 objective for mean
        cov_other = q_dist.covariance().diag_embed()
        cov_part = torch_batched_trace(cov_other + cov - 2 * sqrt_other @ sqrt)

    return mean_part, cov_part


def get_entropy_schedule(schedule_type, total_train_steps, dim):
    """
    return entropy schedule callable with interface f(old_entropy, initial_entropy_bound, train_step)
    Args:
        schedule_type: which type of entropy schedule to use, one of [None, 'linear', or 'exp'].
        total_train_steps: total number of training steps to compute appropriate decay over time.
        dim: number of action dimensions to scale exp decay correctly.

    Returns:
        f(initial_entropy, target_entropy, temperature, step)
    """
    if schedule_type == "linear":
        return lambda initial_entropy, target_entropy, temperature, step: step * (
                target_entropy - initial_entropy) / total_train_steps + initial_entropy
    elif schedule_type == "exp":
        return lambda initial_entropy, target_entropy, temperature, step: target_entropy + (
                initial_entropy - target_entropy) * temperature ** (10 * step / total_train_steps)
    else:
        return lambda initial_entropy, target_entropy, temperature, step: initial_entropy.new([-np.inf])
