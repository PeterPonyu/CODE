from .model import CODEVAE
from .mixin import EnvMixin
import numpy as np
from sklearn.cluster import KMeans


class Env(CODEVAE, EnvMixin):
    """
    The environment class that encapsulates the data and the model.
    """

    def __init__(
        self,
        adata,
        layer,
        percent,
        recon,
        irecon,
        beta,
        dip,
        tc,
        info,
        hidden_dim,
        latent_dim,
        i_dim,
        use_ode,
        use_moco,
        loss_mode,
        lr,
        vae_reg,
        ode_reg,
        moco_weight,
        device,
        *args,
        **kwargs,
    ):
        self._register_anndata(adata, layer, latent_dim)
        self.batch_size = int(percent * self.n_obs)
        super().__init__(
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            state_dim=self.n_var,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            use_moco=use_moco,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            moco_weight=moco_weight,
            device=device,
        )
        self.score = []

    def load_data(
        self,
    ):
        data, idx = self._sample_data()
        self.idx = idx
        return data

    def step(self, data):
        self.update(data)
        latent = self.take_latent(data)
        score = self._calc_score(latent)
        self.score.append(score)

    def _sample_data(
        self,
    ):
        idx = np.random.permutation(self.n_obs)
        idx_ = np.random.choice(idx, self.batch_size)
        data = self.X[idx_, :]
        return data, idx_

    def _register_anndata(self, adata, layer: str, latent_dim):
        self.X = np.log1p(adata.layers[layer].toarray())
        self.n_obs = adata.shape[0]
        self.n_var = adata.shape[1]
        self.labels = KMeans(latent_dim).fit_predict(self.X)
        return