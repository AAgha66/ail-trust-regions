import cpp_projection
import numpy as np
import torch as ch
from typing import Any

from models.distributions import FixedNormal
from projections.base_projection_layer import BaseProjectionLayer, mean_projection
from utils.projection_utils import gaussian_kl
from utils.torch_utils import get_numpy


class KLProjectionLayer(BaseProjectionLayer):

    def _trust_region_projection(self, p_dist: FixedNormal, q_dist: FixedNormal, eps: ch.Tensor, eps_cov: ch.Tensor,
                                 **kwargs):
        """
        Runs KL projection layer and constructs cholesky of covariance
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part
            **kwargs:

        Returns:
            projected mean, projected cov cholesky
        """
        mean = p_dist.mean
        cov = p_dist.stddev.pow(2).diag_embed()

        old_mean = q_dist.mean
        old_cov = q_dist.stddev.pow(2).diag_embed()

        # only project first one to reduce number of numerical optimizations
        cov = cov[:1]
        old_cov = old_cov[:1]

        ################################################################################################################
        # project mean with closed form
        mean_part, _ = gaussian_kl(p_dist=p_dist, q_dist=q_dist)
        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        proj_cov = KLProjectionGradFunctionDiagCovOnly.apply(cov.diagonal(dim1=-2, dim2=-1),
                                                             old_cov.diagonal(dim1=-2, dim2=-1),
                                                             eps_cov)
        proj_std = proj_cov.sqrt().diag_embed()

        # scale first std back to batchsize
        proj_std = proj_std.expand(mean.shape[0], -1, -1)

        return FixedNormal(proj_mean, proj_std.diagonal(dim1=-2, dim2=-1))


class KLProjectionGradFunctionDiagCovOnly(ch.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=100):
        if not KLProjectionGradFunctionDiagCovOnly.projection_op:
            KLProjectionGradFunctionDiagCovOnly.projection_op = \
                cpp_projection.BatchedDiagCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionDiagCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        std, old_std, eps_cov = args

        batch_shape = std.shape[0]
        dim = std.shape[-1]

        cov_np = get_numpy(std)
        old_std = get_numpy(old_std)
        eps = get_numpy(eps_cov) * np.ones(batch_shape)

        # p_op = cpp_projection.BatchedDiagCovOnlyProjection(batch_shape, dim)
        # ctx.proj = projection_op
        p_op = KLProjectionGradFunctionDiagCovOnly.get_projection_op(batch_shape, dim)

        ctx.proj = p_op

        proj_std = p_op.forward(eps, old_std, cov_np)

        return std.new(proj_std)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_std, = grad_outputs

        d_std_np = get_numpy(d_std)
        d_std_np = np.atleast_2d(d_std_np)
        df_stds = projection_op.backward(d_std_np)
        df_stds = np.atleast_2d(df_stds)

        return d_std.new(df_stds), None, None
