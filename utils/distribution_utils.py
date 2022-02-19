import torch

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