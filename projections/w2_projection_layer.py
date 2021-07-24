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

import torch as ch
from models.distributions import FixedNormal
from projections.base_projection_layer import BaseProjectionLayer, mean_projection
from utils.projection_utils import gaussian_wasserstein_commutative


class WassersteinProjectionLayer(BaseProjectionLayer):

    def _trust_region_projection(self, p_dist: FixedNormal, q_dist: FixedNormal, eps: ch.Tensor, eps_cov: ch.Tensor, **kwargs):
        """
        Runs commutative Wasserstein projection layer and constructs sqrt of covariance
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part
            **kwargs:

        Returns:
            mean, cov sqrt
        """
        mean = p_dist.mean
        sqrt = p_dist.stddev.diag_embed()

        old_mean = q_dist.mean
        old_sqrt = q_dist.stddev.diag_embed()

        batch_shape = mean.shape[:-1]

        ####################################################################################################################
        # precompute mean and cov part of W2, which are used for the projection.
        # Both parts differ based on precision scaling.
        # If activated, the mean part is the maha distance and the cov has a more complex term in the inner parenthesis.
        mean_part, cov_part = gaussian_wasserstein_commutative(p_dist, q_dist, self.scale_prec)

        ####################################################################################################################
        # project mean (w/ or w/o precision scaling)
        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        ####################################################################################################################
        # project covariance (w/ or w/o precision scaling)

        cov_mask = cov_part > eps_cov

        if cov_mask.any():
            # gradient issue with ch.where, it executes both paths and gives NaN gradient.
            eta = ch.ones(batch_shape, dtype=sqrt.dtype, device=sqrt.device)
            eta[cov_mask] = ch.sqrt(cov_part[cov_mask] / eps_cov) - 1.
            eta = ch.max(-eta, eta)

            new_sqrt = (sqrt + ch.einsum('i,ijk->ijk', eta, old_sqrt)) / (1. + eta + 1e-16)[..., None, None]
            proj_sqrt = ch.where(cov_mask[..., None, None], new_sqrt, sqrt)
        else:
            proj_sqrt = sqrt

        return FixedNormal(proj_mean, proj_sqrt.diagonal(dim1=-2, dim2=-1))

    def trust_region_value(self, p_dist, q_dist):
        """
        Computes the Wasserstein distance between two Gaussian distributions p and q.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
        Returns:
            mean and covariance part of Wasserstein distance
        """
        return gaussian_wasserstein_commutative(p_dist, q_dist, scale_prec=self.scale_prec)