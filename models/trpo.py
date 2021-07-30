import torch
import torch.optim as optim
from utils.projection_utils import gaussian_kl
from models.model import Policy
from models.distributions import FixedNormal

class TRPO():
    def __init__(self,
                 actor_critic,
                 policy_epoch,
                 vf_epoch,
                 lr_value,
                 eps,
                 batch_size,
                 action_space,
                 obs_space):

        self.actor_critic = actor_critic

        self.policy_epoch = policy_epoch
        self.vf_epoch = vf_epoch
        self.batch_size = batch_size

        self.policy_params = list(actor_critic.base.actor.parameters()) + list(actor_critic.dist.parameters())
        self.vf_params = list(actor_critic.base.critic.parameters())

        self.optimizer_vf = optim.Adam(self.vf_params, lr=lr_value, eps=eps)

        self.global_steps = 0
        self.max_kl = 0.04

        self.action_space = action_space
        self.obs_space = obs_space

    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    def flat_params(self, model_params):
        params = []
        for param in model_params:
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model_params, new_params):
        index = 0

        for params in model_params:
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

    def train_critic(self, advantages, rollouts):
        value_loss_epoch = 0
        for e in range(self.vf_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, mini_batch_size=self.batch_size)

            for sample in data_generator:
                obs_batch, _, value_preds_batch, return_batch, _, _, _, _ = sample
                # Reshape to do in a single forward pass for all steps
                values, _ = self.actor_critic.evaluate_actions(obs_batch)
                value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer_vf.zero_grad()
                value_loss.backward()
                self.optimizer_vf.step()

                value_loss_epoch += value_loss.item()

            return value_loss_epoch

    def fisher_vector_product(self, obs_batch, p):
        p.detach()

        _, dist = self.actor_critic.evaluate_actions(obs_batch)
        detached_dist = FixedNormal(dist.mean.detach(), dist.stddev.detach())

        mean_kl, cov_kl = gaussian_kl(dist, detached_dist)
        kl = mean_kl + cov_kl
        kl = kl.mean()

        kl_grad = torch.autograd.grad(kl, self.policy_params, create_graph=True)
        kl_grad = self.flat_grad(kl_grad)  # check kl_grad == 0

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.policy_params)
        kl_hessian_p = self.flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    # from openai baseline code
    # https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    def conjugate_gradient(self, states, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(states, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def update(self, rollouts):
        self.global_steps += 1
        # ----------------------------
        # step 1: get returns and GAEs
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        # ----------------------------
        # step 2: train critic several steps with respect to returns
        self.train_critic(advantages=advantages, rollouts=rollouts)

        # ----------------------------
        # step 3: get gradient of loss and hessian of kl
        data_generator_policy = rollouts.feed_forward_generator(
            advantages, mini_batch_size=4096)
        for batch in data_generator_policy:
            obs_batch, actions_batch, \
            value_preds_batch, return_batch, _, adv_targ, _, _ = batch

            _, dist = self.actor_critic.evaluate_actions(obs_batch)
            action_log_probs = dist.log_probs(actions_batch)
            ratio = torch.exp(action_log_probs -
                              action_log_probs.detach())
            loss = (ratio * adv_targ).mean()

            loss_grad = torch.autograd.grad(loss, self.policy_params)
            loss_grad = self.flat_grad(loss_grad)

            step_dir = self.conjugate_gradient(obs_batch, loss_grad.data, nsteps=10)
            loss = loss.data.numpy()

            # ----------------------------
            # step 4: get step direction and step size and full step
            params = self.flat_params(self.policy_params)
            shs = 0.5 * (step_dir * self.fisher_vector_product(obs_batch, step_dir)
                         ).sum(0, keepdim=True)
            step_size = 1 / torch.sqrt(shs / self.max_kl)[0]
            full_step = step_size * step_dir
            # ----------------------------
            # step 5: do backtracking line search for n times
            # old_actor = Actor(actor.num_inputs, actor.num_outputs)
            old_actor_critic = Policy(
                self.obs_space.shape,
                self.action_space)
            old_policy_params = list(old_actor_critic.base.actor.parameters()) + \
                                list(old_actor_critic.dist.parameters())
            self.update_model(old_policy_params, params)
            expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
            expected_improve = expected_improve.data.numpy()

            flag = False
            fraction = 1.0

            for i in range(10):
                new_params = params + fraction * full_step
                self.update_model(self.policy_params, new_params)

                _, new_dist = self.actor_critic.evaluate_actions(obs_batch)
                new_action_log_probs = new_dist.log_probs(actions_batch)

                ratio = torch.exp(new_action_log_probs -
                                  action_log_probs.detach())
                new_loss = (ratio * adv_targ).mean()


                new_loss = new_loss.data.numpy()
                loss_improve = new_loss - loss
                expected_improve *= fraction

                detached_old_dist = FixedNormal(dist.mean.detach(), dist.stddev.detach())
                mean_kl, cov_kl = gaussian_kl(new_dist, detached_old_dist)
                kl = mean_kl + cov_kl
                kl = kl.mean()

                """print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
                      'number of line search: {}'
                      .format(kl.data.numpy(), loss_improve, expected_improve[0], i))"""

                # see https: // en.wikipedia.org / wiki / Backtracking_line_search
                if kl < self.max_kl and (loss_improve / expected_improve) > 0.1:
                    flag = True
                    break

                fraction *= 0.9

            if not flag:
                params = self.flat_params(old_policy_params)
                self.update_model(self.policy_params, params)
                print('policy update does not impove the surrogate')
