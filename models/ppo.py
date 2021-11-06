import torch
import torch.nn as nn
import torch.optim as optim
from utils.projection_utils import compute_metrics, gaussian_kl
from projections.projection_factory import get_projection_layer
from models.distributions import FixedNormal
from utils.utils import compute_kurtosis
import numpy as np

class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 policy_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 num_steps,
                 use_kl_penalty=False,
                 use_rollback=False,
                 use_tr_ppo=False,
                 use_truly_ppo=False,
                 beta=0.5,
                 kl_target=0.01,
                 lr_policy=None,
                 lr_value=None,
                 eps=None,
                 proj_type="kl",
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 use_projection=True,
                 track_grad_kurtosis=False,
                 action_space=6,
                 total_train_steps=None,
                 entropy_schedule='None',
                 scale_prec=True,
                 entropy_eq=False,
                 entropy_first=True,
                 clip_importance_ratio=True,
                 use_gmom=False,
                 weiszfeld_iterations=10,
                 gradient_clipping=False,
                 mean_bound=0.03,
                 cov_bound=0.001,
                 rb_alpha=0.3,
                 trust_region_coeff=8.0,
                 target_entropy=0):

        assert sum([use_kl_penalty, clip_importance_ratio, use_projection,
                    use_rollback, use_tr_ppo, use_truly_ppo]) == 1

        self.use_gmom = use_gmom
        self.num_blocks = 8
        self.weiszfeld_iterations = weiszfeld_iterations

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

        self.beta = beta
        self.kl_target = kl_target
        self.use_kl_penalty = use_kl_penalty
        self.use_rollback = use_rollback
        self.use_tr_ppo = use_tr_ppo
        self.use_truly_ppo = use_truly_ppo

        self.rb_alpha = rb_alpha
        self.proj = None
        self.track_grad_kurtosis = track_grad_kurtosis
        
        self.cos = None
        if self.track_grad_kurtosis:
            self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
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

    def train(self, advantages, rollouts, track_kurtosis_flag, use_disc_as_adv):
        action_loss_epoch = 0
        trust_region_loss_epoch = 0
        value_loss_epoch = 0

        on_policy_norms = []
        off_policy_norms = []
        on_policy_cos_gradients = []
        off_policy_cos_gradients = []

        on_policy_value_norms = []
        off_policy_value_norms = []
        policy_grad_norms = []
        critic_grad_norms = []
        ratios_list = []
        for e in range(self.policy_epoch):
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

                maha_part, cov_part = gaussian_kl(old_dist, new_dist)
                kl = (maha_part + cov_part).unsqueeze(-1)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)

                action_loss = None
                surr1 = None
                if use_disc_as_adv:
                    surr1 = torch.exp(action_log_probs) * adv_targ
                else:
                    surr1 = ratio * adv_targ
                if self.clip_importance_ratio:
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    if self.use_kl_penalty:
                        surr1 -= self.beta * kl
                        surr2 -= self.beta * kl
                    action_loss = -torch.min(surr1, surr2)
                elif self.use_kl_penalty:
                    surr1 -= self.beta * kl
                    action_loss = -surr1
                elif self.use_rollback:
                    mask_a = ratio >= 1.0 + self.clip_param
                    mask_b = ratio <= 1.0 - self.clip_param
                    mask_c = torch.logical_not(torch.logical_or(mask_a, mask_b))

                    surr_loss = torch.ones(surr1.shape, dtype=surr1.dtype, device=surr1.device)
                    surr2_a = (- self.rb_alpha * ratio + (1 + self.rb_alpha) * (1 + self.clip_param)) * adv_targ
                    surr2_b = (- self.rb_alpha * ratio + (1 + self.rb_alpha) * (1 - self.clip_param)) * adv_targ

                    surr_loss[mask_a] = torch.min(surr1[mask_a], surr2_a[mask_a])
                    surr_loss[mask_b] = torch.min(surr1[mask_b], surr2_b[mask_b])
                    surr_loss[mask_c] = surr1[mask_c]

                    action_loss = -surr_loss
                elif self.use_tr_ppo:
                    mask = kl >= self.kl_target
                    ratio_old = torch.exp(old_action_log_probs_batch -
                                          old_action_log_probs_batch)
                    surr2 = ratio_old * adv_targ

                    surr_loss = torch.ones(surr1.shape, dtype=surr1.dtype, device=surr1.device)
                    surr_loss[mask] = torch.min(surr1[mask], surr2[mask])
                    surr_loss[~mask] = surr1[~mask]
                    action_loss = -surr_loss
                elif self.use_truly_ppo:
                    ratio_old = torch.exp(old_action_log_probs_batch -
                                          old_action_log_probs_batch)
                    surr2 = ratio_old * adv_targ

                    mask = torch.logical_and(kl >= self.kl_target, surr1 >= surr2)

                    surr_loss = torch.ones(surr1.shape, dtype=surr1.dtype, device=surr1.device)
                    surr_loss[mask] = surr1[mask] - self.rb_alpha * kl[mask]
                    surr_loss[~mask] = surr1[~mask] - self.kl_target

                    action_loss = -surr_loss
                else:
                    action_loss = -surr1

                gradients = []
                if track_kurtosis_flag:
                    if e == self.policy_epoch - 1:
                        ratios_list.extend(ratio.squeeze(-1).tolist())
                    if e == 0 or e == self.policy_epoch - 1:
                        for batch_elt in range(action_loss.shape[0]):
                            gradient = torch.tensor([])
                            action_loss[batch_elt].backward(retain_graph=True)
                            for p in self.policy_params:
                                gradient = torch.cat([gradient,torch.flatten(p.grad.data)])              
                            if e == 0:
                                on_policy_norms.append(gradient.norm(2))                                
                            elif e == self.policy_epoch - 1:
                                off_policy_norms.append(gradient.norm(2))                                
                            gradients.append(gradient)
                            self.optimizer_policy.zero_grad()                                        
                        if e == 0:
                            on_policy_cos_gradients = [self.cos(p1, p2).item() for p1 in gradients for p2 in gradients if not torch.equal(p1, p2)]
                        elif e == self.policy_epoch - 1:
                            off_policy_cos_gradients = [self.cos(p1, p2).item() for p1 in gradients for p2 in gradients if not torch.equal(p1, p2)]
                    
                # Trust region loss
                trust_region_loss = None
                if self.proj is not None:
                    trust_region_loss = self.proj.get_trust_region_loss(dist, new_dist)

                value_loss = None
                if not use_disc_as_adv:
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                                value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped)
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2)

                if track_kurtosis_flag:
                    if e == 0 or e == self.policy_epoch - 1:
                        for batch_elt in range(value_loss.shape[0]):
                            total_norm = 0
                            value_loss[batch_elt].backward(retain_graph=True)
                            for p in self.vf_params:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                            if e == 0:
                                on_policy_value_norms.append(total_norm ** (1. / 2))
                            elif e == self.policy_epoch - 1:
                                off_policy_value_norms.append(total_norm ** (1. / 2))
                            self.optimizer_vf.zero_grad()

                if not use_disc_as_adv:
                    value_loss = value_loss.mean()
                self.optimizer_policy.zero_grad()

                if self.use_gmom:
                    grads = []
                    size_b = action_loss.shape[0] / self.num_blocks
                    for block in range(self.num_blocks):
                        block_loss = action_loss[int(block * size_b): int((block + 1) * size_b - 1)].mean() - \
                                     dist_entropy * self.entropy_coef
                        if self.proj is not None:
                            block_loss += trust_region_loss
                        block_loss.backward(retain_graph=True)
                        flattened_grads = torch.tensor([])
                        for p in self.policy_params:
                            flattened_grads = torch.cat([flattened_grads, torch.flatten(p.grad.clone())])
                        grads.append(flattened_grads)
                        self.optimizer_policy.zero_grad()

                    flattened_mu = torch.mean(torch.stack(grads), dim=0)
                    #WEISZFELD Algorithm (https://arxiv.org/pdf/2102.10264.pdf page 15)
                    for w_iter in range(self.weiszfeld_iterations):
                        d_j = []
                        flattened_mu_old = flattened_mu.clone()
                        flattened_mu = torch.zeros(flattened_mu.shape)
                        for block in range(self.num_blocks):
                            d_j.append(1.0 / (flattened_mu_old - grads[block]).norm(2).item())
                            flattened_mu += grads[block] * d_j[block]
                        flattened_mu = flattened_mu / sum(d_j)
                    k = 0
                    for p in self.policy_params:
                        p.grad = torch.reshape(flattened_mu[k:k + p.grad.numel()].clone(),
                                               p.grad.shape)
                        k += p.grad.numel()
                    action_loss = action_loss.mean()
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
                total_policy_norm = 0
                total_critic_norm = 0

                if track_kurtosis_flag:
                    for p in self.policy_params:
                        param_norm = p.grad.data.norm(2)
                        total_policy_norm += param_norm.item() ** 2
                    policy_grad_norms.append(total_policy_norm ** (1. / 2))

                if not use_disc_as_adv:
                    self.optimizer_vf.zero_grad()
                    value_loss.backward()
                    if self.gradient_clipping:
                        nn.utils.clip_grad_norm_(self.vf_params,
                                                self.max_grad_norm)
                    self.optimizer_vf.step()
                    value_loss_epoch += value_loss.item()
                
                
                if track_kurtosis_flag:
                    for p in self.vf_params:
                        param_norm = p.grad.data.norm(2)
                        total_critic_norm += param_norm.item() ** 2
                    critic_grad_norms.append(total_critic_norm ** (1. / 2))

                
                action_loss_epoch += action_loss.mean().item()

                if trust_region_loss:
                    trust_region_loss_epoch += trust_region_loss.item()
        
        on_policy_kurtosis = None
        off_policy_kurtosis = None
        on_policy_value_kurtosis = None
        off_policy_value_kurtosis = None
        
        on_policy_cos_mean = None 
        off_policy_cos_mean = None 
        on_policy_cos_median = None 
        off_policy_cos_median = None
        
        if track_kurtosis_flag:
            on_policy_kurtosis = compute_kurtosis(on_policy_norms)
            off_policy_kurtosis = compute_kurtosis(off_policy_norms)
            on_policy_value_kurtosis = compute_kurtosis(on_policy_value_norms)
            off_policy_value_kurtosis = compute_kurtosis(off_policy_value_norms)
            
            on_policy_cos_mean = np.mean(on_policy_cos_gradients)
            on_policy_cos_median = np.median(on_policy_cos_gradients)            
            off_policy_cos_mean = np.mean(off_policy_cos_gradients)
            off_policy_cos_median = np.median(off_policy_cos_gradients)

        return action_loss_epoch, value_loss_epoch, trust_region_loss_epoch, \
               on_policy_kurtosis, off_policy_kurtosis, on_policy_value_kurtosis, \
               off_policy_value_kurtosis, policy_grad_norms, critic_grad_norms, ratios_list, \
               on_policy_cos_mean, off_policy_cos_mean, on_policy_cos_median, off_policy_cos_median

    def update(self, rollouts, iteration, use_disc_as_adv):
        self.global_steps += 1
        advantages = None
        
        if use_disc_as_adv:
            advantages = rollouts.rewards
            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5)
        else:
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

            advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-5)        
        track_kurtosis_flag = False
        if self.track_grad_kurtosis:
            track_kurtosis_flag = (iteration % 30 == 0)
        action_loss_epoch, value_loss_epoch, trust_region_loss_epoch, on_policy_kurtosis, \
        off_policy_kurtosis, on_policy_value_kurtosis, off_policy_value_kurtosis, \
        policy_grad_norms, critic_grad_norms, ratios_list, \
        on_policy_cos_mean, off_policy_cos_mean, on_policy_cos_median, off_policy_cos_median = \
            self.train(advantages=advantages, rollouts=rollouts,
                       track_kurtosis_flag=track_kurtosis_flag, use_disc_as_adv=use_disc_as_adv)

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

        if self.use_kl_penalty:
            if metrics['kl'] > self.kl_target * 1.5:
                self.beta = self.beta * 2.0
            elif metrics['kl'] < self.kl_target / 1.5:
                self.beta = self.beta / 2.0

        metrics['value_loss_epoch'] = value_loss_epoch / num_updates_value
        metrics['action_loss_epoch'] = action_loss_epoch / num_updates_policy
        metrics['trust_region_loss_epoch'] = trust_region_loss_epoch / num_updates_policy

        metrics['on_policy_kurtosis'] = on_policy_kurtosis
        metrics['off_policy_kurtosis'] = off_policy_kurtosis

        metrics['on_policy_value_kurtosis'] = on_policy_value_kurtosis
        metrics['off_policy_value_kurtosis'] = off_policy_value_kurtosis

        metrics['policy_grad_norms'] = policy_grad_norms
        metrics['critic_grad_norms'] = critic_grad_norms

        metrics['on_policy_cos_mean'] = on_policy_cos_mean
        metrics['off_policy_cos_mean'] = off_policy_cos_mean
        metrics['on_policy_cos_median'] = on_policy_cos_median
        metrics['off_policy_cos_median'] = off_policy_cos_median

        metrics['ratios_list'] = ratios_list
        return metrics
