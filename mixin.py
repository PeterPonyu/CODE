import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from typing import Optional


class ScviMixin:
    """Mixin class for scVI-related loss functions."""

    def _normal_kl(self, mu1, lv1, mu2, lv2):
        """
        Calculates the KL divergence between two normal distributions.

        Args:
            mu1: Mean of the first distribution.
            lv1: Log variance of the first distribution.
            mu2: Mean of the second distribution.
            lv2: Log variance of the second distribution.

        Returns:
            The KL divergence.
        """
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.0
        lstd2 = lv2 / 2.0
        kl = lstd2 - lstd1 + (v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2) - 0.5
        return kl

    def _log_nb(self, x, mu, theta, eps=1e-8):
        """
        Calculates the log probability under a negative binomial distribution.

        Args:
            x: The data.
            mu: The mean of the distribution.
            theta: The dispersion parameter.
            eps: A small constant for numerical stability.

        Returns:
            The log probability.
        """
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        return res

    def _log_zinb(self, x, mu, theta, pi, eps=1e-8):
        """
        Calculates the log probability under a zero-inflated negative binomial distribution.

        Args:
            x: The data.
            mu: The mean of the distribution.
            theta: The dispersion parameter.
            pi: The logits of the zero-inflation mixing weight.
            eps: A small constant for numerical stability.

        Returns:
            The log probability.
        """
        softplus_pi = F.softplus(-pi)
        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

        res = mul_case_zero + mul_case_non_zero
        return res


class NODEMixin:
    """Mixin class for Neural ODE related functionalities."""

    @staticmethod
    def get_step_size(step_size, t0, t1, n_points):
        """
        Gets the step size configuration for the ODE solver.

        Args:
            step_size: The step size. If None, it is calculated automatically.
            t0: The start time.
            t1: The end time.
            n_points: The number of time points.

        Returns:
            A dictionary with the ODE solver options.
        """
        if step_size is None:
            return {}
        else:
            if step_size == "auto":
                step_size = (t1 - t0) / (n_points - 1)
            return {"step_size": step_size}

    def solve_ode(
        self,
        ode_func: torch.nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        method: str = "rk4",
        step_size: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Solves the ODE using torchdiffeq.

        Args:
            ode_func: The ODE function model.
            z0: The initial state.
            t: The time points.
            method: The solving method.
            step_size: The step size.

        Returns:
            The ODE solution.
        """
        options = self.get_step_size(step_size, t[0], t[-1], len(t))

        cpu_z0 = z0.to("cpu")
        cpu_t = t.to("cpu")

        pred_z = odeint(ode_func, cpu_z0, cpu_t, method=method, options=options)

        pred_z = pred_z.to(z0.device)

        return pred_z


class BetaTCMixin:
    """Mixin class for beta-TCVAE loss."""

    def _betatc_compute_gaussian_log_density(self, samples, mean, log_var):
        import math

        pi = torch.tensor(math.pi, requires_grad=False)
        normalization = torch.log(2 * pi)
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

    def _betatc_compute_total_correlation(self, z_sampled, z_mean, z_logvar):
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(dim=1),
            z_mean.unsqueeze(dim=0),
            z_logvar.unsqueeze(dim=0),
        )
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        return (log_qz - log_qz_product).mean()


class InfoMixin:
    """Mixin class for InfoVAE (MMD) loss."""

    def _compute_mmd(self, z_posterior_samples, z_prior_samples):
        mean_pz_pz = self._compute_unbiased_mean(
            self._compute_kernel(z_prior_samples, z_prior_samples), unbaised=True
        )
        mean_pz_qz = self._compute_unbiased_mean(
            self._compute_kernel(z_prior_samples, z_posterior_samples), unbaised=False
        )
        mean_qz_qz = self._compute_unbiased_mean(
            self._compute_kernel(z_posterior_samples, z_posterior_samples),
            unbaised=True,
        )
        mmd = mean_pz_pz - 2 * mean_pz_qz + mean_qz_qz
        return mmd

    def _compute_unbiased_mean(self, kernel, unbaised):
        N, M = kernel.shape
        if unbaised:
            sum_kernel = kernel.sum(dim=(0, 1)) - torch.diagonal(
                kernel, dim1=0, dim2=1
            ).sum(dim=-1)
            mean_kernel = sum_kernel / (N * (N - 1))
        else:
            mean_kernel = kernel.mean(dim=(0, 1))
        return mean_kernel

    def _compute_kernel(self, z0, z1):
        batch_size, z_size = z0.shape
        z0 = z0.unsqueeze(-2)
        z1 = z1.unsqueeze(-3)
        z0 = z0.expand(batch_size, batch_size, z_size)
        z1 = z1.expand(batch_size, batch_size, z_size)
        kernel = self._kernel_rbf(z0, z1)
        return kernel

    def _kernel_rbf(self, x, y):
        z_size = x.shape[-1]
        sigma = 2 * 2 * z_size
        kernel = torch.exp(-((x - y).pow(2).sum(dim=-1) / sigma))
        return kernel


class DipMixin:
    """Mixin class for DIP-VAE loss."""

    def _dip_loss(self, q_m, q_s):
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        dip_loss_d = torch.sum((cov_diag - 1) ** 2)
        dip_loss_od = torch.sum(cov_off_diag**2)
        dip_loss = 10 * dip_loss_d + 5 * dip_loss_od
        return dip_loss

    def _dip_cov_matrix(self, q_m, q_s):
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.diag(F.softplus(q_s).exp()), dim=0)
        cov_matrix = cov_q_mean + E_var
        return cov_matrix


class EnvMixin:
    """Mixin class for environment-related functionalities."""

    def _calc_score(self, latent):
        labels = self._calc_label(latent)
        scores = self._metrics(latent, labels)
        return scores

    def _calc_label(self, latent):
        labels = KMeans(latent.shape[1]).fit_predict(latent)
        return labels

    def _calc_corr(self, latent):
        acorr = abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1

    def _metrics(self, latent, labels):
        ARI = adjusted_mutual_info_score(self.labels[self.idx], labels)
        NMI = normalized_mutual_info_score(self.labels[self.idx], labels)
        ASW = silhouette_score(latent, labels)
        C_H = calinski_harabasz_score(latent, labels)
        D_B = davies_bouldin_score(latent, labels)
        P_C = self._calc_corr(latent)
        return ARI, NMI, ASW, C_H, D_B, P_C