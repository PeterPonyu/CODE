import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal
from .mixin import NODEMixin
import numpy as np


class Encoder(nn.Module):
    """
    Variational encoder network that maps an input state to a latent distribution.

    Args:
        state_dim: Dimension of the input state.
        hidden_dim: Dimension of the hidden layers.
        action_dim: Dimension of the latent space.
        use_ode: Whether to use the ODE mode, which adds a time parameter output.
    """

    def __init__(
        self, state_dim: int, hidden_dim: int, action_dim: int, use_ode: bool = False
    ):
        super().__init__()
        self.use_ode = use_ode

        self.base_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.latent_params = nn.Linear(hidden_dim, action_dim * 2)

        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initializes the network weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def forward(
        self, x: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass of the encoder.

        Args:
            x: Input tensor of shape (batch_size, state_dim).

        Returns:
            If use_ode=False:
                A tuple containing:
                    - Sampled latent vector.
                    - Mean of the latent distribution.
                    - Log variance of the latent distribution.
            If use_ode=True:
                A tuple containing:
                    - Sampled latent vector.
                    - Mean of the latent distribution.
                    - Log variance of the latent distribution.
                    - Predicted time parameter (in the range [0, 1]).
        """
        hidden = self.base_network(x)

        latent_output = self.latent_params(hidden)
        q_m, q_s = torch.split(latent_output, latent_output.size(-1) // 2, dim=-1)

        std = F.softplus(q_s)

        dist = Normal(q_m, std)
        q_z = dist.rsample()

        if self.use_ode:
            t = self.time_encoder(hidden).squeeze(-1)
            return q_z, q_m, q_s, t

        return q_z, q_m, q_s


class Decoder(nn.Module):
    """
    Decoder network that maps a latent vector back to the original space.

    Supports three loss modes:
    - 'mse': Mean squared error loss for continuous data.
    - 'nb': Negative binomial loss for discrete count data.
    - 'zinb': Zero-inflated negative binomial loss for count data with excess zeros.

    Args:
        state_dim: Dimension of the original space.
        hidden_dim: Dimension of the hidden layers.
        action_dim: Dimension of the latent space.
        loss_mode: The loss function mode. Defaults to 'nb'.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
    ):
        super().__init__()
        self.loss_mode = loss_mode

        self.base_network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if loss_mode in ["nb", "zinb"]:
            self.disp = nn.Parameter(torch.randn(state_dim))
            self.mean_decoder = nn.Sequential(
                nn.Linear(hidden_dim, state_dim), nn.Softmax(dim=-1)
            )
        else:
            self.mean_decoder = nn.Linear(hidden_dim, state_dim)

        if loss_mode == "zinb":
            self.dropout_decoder = nn.Linear(hidden_dim, state_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initializes the network weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Latent vector of shape (batch_size, action_dim).

        Returns:
            For 'mse' and 'nb' modes:
                Reconstructed output.
            For 'zinb' mode:
                A tuple containing:
                    - Reconstructed mean.
                    - Logits of the zero-inflation parameter.
        """
        hidden = self.base_network(x)

        mean = self.mean_decoder(hidden)

        if self.loss_mode == "zinb":
            dropout_logits = self.dropout_decoder(hidden)
            return mean, dropout_logits

        return mean


class LatentODEfunc(nn.Module):
    """
    Latent space ODE function model.

    Args:
        n_latent: Dimension of the latent space.
        n_hidden: Dimension of the hidden layers.
    """

    def __init__(
        self,
        n_latent: int = 10,
        n_hidden: int = 25,
    ):
        super().__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient at time t and state x.

        Args:
            t: Time point.
            x: Latent state.

        Returns:
            The gradient.
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class MoCo(nn.Module):
    """
    Momentum Contrast for Unsupervised Visual Representation Learning.
    """

    def __init__(
        self,
        encoder_q,
        encoder_k,
        state_dim,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        device=torch.device("cuda"),
    ):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.aug_prob = 0.5
        self.n_genes = state_dim

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, exp_q):
        exp_k = self._augment(exp_q)
        q = self.encoder_q(exp_q)[1]
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(exp_k)[1]
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

    def _augment(self, profile):
        if isinstance(profile, torch.Tensor):
            profile_np = profile.cpu().numpy()
        else:
            profile_np = profile.copy()

        if np.random.rand() < self.aug_prob:
            # Masking
            mask = np.random.choice([True, False], self.n_genes, p=[0.2, 0.8])
            profile_np[:, mask] = 0
            # Gaussian Noise
            mask = np.random.choice([True, False], self.n_genes, p=[0.7, 0.3])
            noise = np.random.normal(0, 0.2, (profile_np.shape[0], np.sum(mask)))
            profile_np[:, mask] += noise
        if isinstance(profile, torch.Tensor):
            return torch.from_numpy(profile_np).to(profile.device)
        else:
            return profile_np


class VAE(nn.Module, NODEMixin):
    """
    Variational Autoencoder with support for both linear and ODE-based latent dynamics.

    Args:
        state_dim: Dimension of the input state space.
        hidden_dim: Dimension of the hidden layers.
        action_dim: Dimension of the action/latent space.
        i_dim: Dimension of the information bottleneck.
        use_ode: Whether to use the ODE-based model.
        use_moco: Whether to use Momentum Contrast (MoCo).
        loss_mode: The loss function mode.
        device: The device to run the model on.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        i_dim: int,
        use_ode: bool,
        use_moco: bool,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        super().__init__()
        self.use_moco = use_moco
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode).to(device)

        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim)
        if self.use_moco:
            self.encoder_k = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(
                device
            )
            self.moco = MoCo(
                self.encoder, self.encoder_k, state_dim, dim=action_dim, device=device
            )
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor.

        Returns:
            A tuple of tensors containing the VAE outputs.
        """
        if self.encoder.use_ode:
            q_z, q_m, q_s, t = self.encoder(x)

            idxs = torch.argsort(t)
            t, q_z, q_m, q_s, x = t[idxs], q_z[idxs], q_m[idxs], q_s[idxs], x[idxs]

            unique_mask = torch.ones_like(t, dtype=torch.bool)
            unique_mask[1:] = t[1:] != t[:-1]

            t, q_z, q_m, q_s, x = (
                t[unique_mask],
                q_z[unique_mask],
                q_m[unique_mask],
                q_s[unique_mask],
                x[unique_mask],
            )

            z0 = q_z[0]
            q_z_ode = self.solve_ode(self.ode_solver, z0, t)

            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            le_ode = self.latent_encoder(q_z_ode)
            ld_ode = self.latent_decoder(le_ode)

            if self.decoder.loss_mode == "zinb":
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                pred_x_ode, dropout_logits_ode = self.decoder(q_z_ode)
                pred_xl_ode, dropout_logitsl_ode = self.decoder(ld_ode)
                if self.use_moco:
                    logits, labels = self.moco(x)
                    return (
                        q_z,
                        q_m,
                        q_s,
                        x,
                        pred_x,
                        dropout_logits,
                        le,
                        le_ode,
                        pred_xl,
                        dropout_logitsl,
                        q_z_ode,
                        pred_x_ode,
                        dropout_logits_ode,
                        pred_xl_ode,
                        dropout_logitsl_ode,
                        logits,
                        labels,
                    )
                else:
                    return (
                        q_z,
                        q_m,
                        q_s,
                        x,
                        pred_x,
                        dropout_logits,
                        le,
                        le_ode,
                        pred_xl,
                        dropout_logitsl,
                        q_z_ode,
                        pred_x_ode,
                        dropout_logits_ode,
                        pred_xl_ode,
                        dropout_logitsl_ode,
                    )
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                pred_x_ode = self.decoder(q_z_ode)
                pred_xl_ode = self.decoder(ld_ode)
                if self.use_moco:
                    logits, labels = self.moco(x)
                    return (
                        q_z,
                        q_m,
                        q_s,
                        x,
                        pred_x,
                        le,
                        le_ode,
                        pred_xl,
                        q_z_ode,
                        pred_x_ode,
                        pred_xl_ode,
                        logits,
                        labels,
                    )
                else:
                    return (
                        q_z,
                        q_m,
                        q_s,
                        x,
                        pred_x,
                        le,
                        le_ode,
                        pred_xl,
                        q_z_ode,
                        pred_x_ode,
                        pred_xl_ode,
                    )

        else:
            q_z, q_m, q_s = self.encoder(x)
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            if self.decoder.loss_mode == "zinb":
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                if self.use_moco:
                    logits, labels = self.moco(x)
                    return (
                        q_z,
                        q_m,
                        q_s,
                        pred_x,
                        dropout_logits,
                        le,
                        pred_xl,
                        dropout_logitsl,
                        logits,
                        labels,
                    )
                else:
                    return (
                        q_z,
                        q_m,
                        q_s,
                        pred_x,
                        dropout_logits,
                        le,
                        pred_xl,
                        dropout_logitsl,
                    )
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                if self.use_moco:
                    logits, labels = self.moco(x)
                    return q_z, q_m, q_s, pred_x, le, pred_xl, logits, labels
                else:
                    return q_z, q_m, q_s, pred_x, le, pred_xl