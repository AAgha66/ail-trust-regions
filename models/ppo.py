import torch
import torch.nn as nn
import torch.optim as optim
from utils.projection_utils import compute_metrics, gaussian_kl
from projections.projection_factory import get_projection_layer
from models.distributions import FixedNormal
from torch.autograd import Variable


class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 policy_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 num_steps,
                 lr_policy=None,
                 lr_value=None,
                 eps=None,
                 proj_type="kl",
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 use_projection=True,
                 action_space=None,
                 total_train_steps=None,
                 entropy_schedule='None',
                 scale_prec=True,
                 entropy_eq=False,
                 entropy_first=True,
                 clip_importance_ratio=True,
                 gradient_clipping=False,
                 mean_bound=0.03,
                 cov_bound=0.001,
                 trust_region_coeff=8.0,
                 target_entropy=0,
                 decay=10,
                 gailgamma=1,
                 use_bcgail=False):

        assert sum([clip_importance_ratio, use_projection]) == 1

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.policy_epoch = policy_epoch
        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.gradient_clipping = gradient_clipping
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_importance_ratio = clip_importance_ratio

        self.mean_bound = mean_bound
        self.cov_bound = cov_bound
        self.trust_region_coeff = trust_region_coeff
        self.num_steps = num_steps

        self.proj = None        
        self.cos = None
        self.action_space = action_space

        if use_projection:
            self.proj = get_projection_layer(proj_type=proj_type, mean_bound=mean_bound,
                                             cov_bound=cov_bound, trust_region_coeff=trust_region_coeff,
                                             scale_prec=scale_prec, action_dim=action_space,
                                             entropy_schedule=entropy_schedule,
                                             total_train_steps=total_train_steps,
                                             target_entropy=target_entropy, temperature=0.5, entropy_eq=entropy_eq,
                                             entropy_first=entropy_first, do_regression=False,
                                             cpu=True, dtype=torch.float32)

        self.policy_params = list(actor_critic.base.actor.parameters()) + list(actor_critic.dist.parameters())
        self.vf_params = list(actor_critic.base.critic.parameters())

        self.optimizer_policy = optim.Adam(self.policy_params, lr=lr_policy, eps=eps)
        self.optimizer_vf = optim.Adam(self.vf_params, lr=lr_value, eps=eps)

        self.global_steps = 0

        self.decay = decay
        self.gailgamma = gailgamma
        self.use_bcgail = use_bcgail

    def train(self, advantages, rollouts, expert_dataset=None, obfilt=None):
        action_loss_epoch = 0
        trust_region_loss_epoch = 0
        value_loss_epoch = 0

        for _ in range(self.policy_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, mini_batch_size=self.mini_batch_size)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, _, _, adv_targ, old_means, old_stddevs = sample
                # Reshape to do in a single forward pass for all steps
                values, dist = self.actor_critic.evaluate_actions(obs_batch)

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
                    action_loss = -torch.min(surr1, surr2)
                else:
                    action_loss = -surr1

                # Trust region loss
                trust_region_loss = None
                if self.proj is not None:
                    trust_region_loss = self.proj.get_trust_region_loss(dist, new_dist)
                bcloss = None
                if self.use_bcgail:
                    for exp_state, exp_action in expert_dataset:
                        if obfilt:
                            exp_state = obfilt(exp_state.numpy(), update=False)
                            exp_state = torch.FloatTensor(exp_state)
                        exp_state = Variable(exp_state).to(action_loss.device)
                        exp_action = Variable(exp_action).to(action_loss.device)
                        _, expert_dist = self.actor_critic.evaluate_actions(exp_state)
                        expert_action_log_probs = expert_dist.log_probs(exp_action)
                        bcloss = -expert_action_log_probs.mean()
                        # Multiply this coeff with decay factor
                        break
                value_loss = None
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2)

                value_loss = value_loss.mean()
                self.optimizer_policy.zero_grad()

                if self.use_bcgail:
                    action_loss = self.gailgamma * bcloss + (1 - self.gailgamma) * action_loss.mean()
                else:
                    action_loss = action_loss.mean()

                loss = action_loss - dist_entropy * self.entropy_coef
                if self.proj is not None:
                    loss += trust_region_loss
                loss.backward()

                if self.gradient_clipping:
                    nn.utils.clip_grad_norm_(self.policy_params,
                                             self.max_grad_norm)
                self.optimizer_policy.step()

                self.optimizer_vf.zero_grad()
                value_loss.backward()
                if self.gradient_clipping:
                    nn.utils.clip_grad_norm_(self.vf_params,
                                                self.max_grad_norm)
                self.optimizer_vf.step()
                value_loss_epoch += value_loss.item()

                action_loss_epoch += action_loss.mean().item()

                if trust_region_loss:
                    trust_region_loss_epoch += trust_region_loss.item()

        if self.use_bcgail:
            self.gailgamma *= self.decay
            print('Gamma: {}'.format(self.gailgamma))
        return action_loss_epoch, value_loss_epoch, trust_region_loss_epoch

    def update(self, rollouts, expert_dataset=None, obfilt=None):
        self.global_steps += 1

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        action_loss_epoch, value_loss_epoch, trust_region_loss_epoch = self.train(advantages=advantages, rollouts=rollouts,
                                                                        expert_dataset=expert_dataset, obfilt=obfilt)

        # TODO: Find a nicer way to get all obs and old means and stddev
        metrics = None
        data_generator_metric = rollouts.feed_forward_generator(
            advantages, mini_batch_size=self.num_steps)
        for batch in data_generator_metric:
            obs_batch, _, _, _, _, _, _, old_means, old_stddevs = batch
            # Reshape to do in a single forward pass for all steps
            _, dist = self.actor_critic.evaluate_actions(obs_batch)
            old_dist = FixedNormal(old_means, old_stddevs)
            metrics = compute_metrics(old_dist, dist)

        num_updates_policy = self.policy_epoch * (self.num_steps / self.mini_batch_size)
        num_updates_value = self.policy_epoch * (self.num_steps / self.mini_batch_size)

        metrics['value_loss_epoch'] = value_loss_epoch / num_updates_value
        metrics['action_loss_epoch'] = action_loss_epoch / num_updates_policy
        metrics['trust_region_loss_epoch'] = trust_region_loss_epoch / num_updates_policy

        return metrics
