import torch
import torch.nn as nn
import torch.optim as optim
from utils.projection_utils import compute_metrics
from projections.projection_factory import get_projection_layer
from models.distributions import FixedNormal
from torch import autograd

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 policy_epoch,
                 vf_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr_policy=None,
                 lr_value=None,
                 eps=None,
                 proj_type="kl",
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 use_projection=True,
                 clip_importance_ratio=True,
                 gradient_clipping=False,
                 mean_bound=0.03,
                 cov_bound=0.001,
                 trust_region_coeff=8.0):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.policy_epoch = policy_epoch
        self.vf_epoch = vf_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.gradient_clipping = gradient_clipping
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_importance_ratio = clip_importance_ratio

        self.mean_bound = mean_bound
        self.cov_bound = cov_bound
        self.trust_region_coeff = trust_region_coeff

        self.proj = None
        if use_projection:
            self.proj = get_projection_layer(proj_type=proj_type, mean_bound=mean_bound,
                                             cov_bound=cov_bound, trust_region_coeff=trust_region_coeff,
                                             scale_prec=True, entropy_schedule=None,
                                             target_entropy=0.0, temperature=0.5, entropy_eq=False,
                                             entropy_first=False, do_regression=False,
                                             cpu=True, dtype=torch.float32)

        self.policy_params = list(actor_critic.base.actor.parameters()) + list(actor_critic.dist.parameters())
        self.vf_params = list(actor_critic.base.critic.parameters())

        self.optimizer_policy = optim.Adam(self.policy_params, lr=lr_policy, eps=eps)
        self.optimizer_vf = optim.Adam(self.vf_params, lr=lr_value, eps=eps)

        self.global_steps = 0

    def policy_update(self, advantages, rollouts):
        action_loss_epoch = 0
        trust_region_loss_epoch = 0
        norm_grad = []

        for e in range(self.policy_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, mini_batch_size=self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, _, _, _, adv_targ, old_means, old_stddevs = sample
                # Reshape to do in a single forward pass for all steps
                _, dist = self.actor_critic.evaluate_actions(obs_batch)

                old_dist = FixedNormal(old_means, old_stddevs)
                # set initial entropy value in first step to calculate appropriate entropy decay
                new_dist = None
                if self.proj is not None:
                    if self.proj.initial_entropy is None:
                        self.proj.initial_entropy = old_dist.entropy().mean()
                    new_dist = self.proj(dist, old_dist, self.global_steps)
                else:
                    new_dist = dist

                old_action_log_probs_batch = old_dist.log_probs(actions_batch)
                action_log_probs = new_dist.log_probs(actions_batch)
                dist_entropy = new_dist.entropy().mean()

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)

                action_loss = None
                surr1 = ratio * adv_targ
                if self.clip_importance_ratio:
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()
                else:
                    action_loss = -surr1.mean()

                # Trust region loss
                trust_region_loss = None
                if self.proj is not None:
                    trust_region_loss = self.proj.get_trust_region_loss(dist, new_dist)

                self.optimizer_policy.zero_grad()

                loss = action_loss - dist_entropy * self.entropy_coef
                # calculating gradient of loss relative to the policy
                ones = torch.ones(loss.size()).to(loss.device)
                grad = autograd.grad(
                    outputs=loss,
                    inputs=dist.mean,
                    grad_outputs=ones,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
                norm_grad_batch = torch.mean(torch.norm(grad, p=2, dim=1))
                norm_grad.append(torch.unsqueeze(norm_grad_batch, dim=0))

                if self.proj is not None:
                    loss += trust_region_loss
                loss.backward()

                if self.gradient_clipping:
                    nn.utils.clip_grad_norm_(self.policy_params,
                                             self.max_grad_norm)

                self.optimizer_policy.step()
                action_loss_epoch += action_loss.item()
                if trust_region_loss:
                    trust_region_loss_epoch += trust_region_loss.item()

        return action_loss_epoch, trust_region_loss_epoch, norm_grad

    def value_update(self, advantages, rollouts):
        value_loss_epoch = 0

        for e in range(self.vf_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, mini_batch_size=self.num_mini_batch)

            for sample in data_generator:
                obs_batch, _, value_preds_batch, return_batch, _, _, _, _ = sample
                # Reshape to do in a single forward pass for all steps
                values, _ = self.actor_critic.evaluate_actions(obs_batch)

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer_vf.zero_grad()
                value_loss.backward()
                self.optimizer_vf.step()

                value_loss_epoch += value_loss.item()

        return value_loss_epoch

    def update(self, rollouts):
        self.global_steps += 1
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = self.value_update(advantages=advantages, rollouts=rollouts)
        action_loss_epoch, trust_region_loss_epoch, norm_grad_policy = \
            self.policy_update(advantages=advantages, rollouts=rollouts)

        # TODO: Find a nicer way to get all obs and old means and stddev
        metrics = None
        data_generator_metric = rollouts.feed_forward_generator(
            advantages, mini_batch_size=2048)
        for set in data_generator_metric:
            obs_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, adv_targ, old_means, old_stddevs = set
            # Reshape to do in a single forward pass for all steps
            _, dist = self.actor_critic.evaluate_actions(obs_batch)
            old_dist = FixedNormal(old_means, old_stddevs)
            metrics = compute_metrics(dist, old_dist)

        num_updates_policy = self.policy_epoch * self.num_mini_batch
        num_updates_value = self.vf_epoch * self.num_mini_batch

        metrics['value_loss_epoch'] = value_loss_epoch / num_updates_value
        metrics['action_loss_epoch'] = action_loss_epoch / num_updates_policy
        metrics['trust_region_loss_epoch'] = trust_region_loss_epoch / num_updates_policy
        metrics['norm_grad_policy'] = norm_grad_policy
        metrics['advantages'] = advantages
        return metrics
