from .model import CODEVAE
from .mixin import EnvMixin
import numpy as np
from sklearn.cluster import KMeans


class Env(CODEVAE, EnvMixin):
    """
    Environment class that encapsulates data handling and model training.
    
    This class bridges the data (from AnnData) and the model (CODEVAE), providing
    methods for data loading, batch sampling, and score computation during training.
    It inherits both model functionality (CODEVAE) and environment utilities (EnvMixin).
    
    Attributes:
        X (np.ndarray): Log-normalized gene expression matrix (n_obs × n_var)
        n_obs (int): Number of observations (cells)
        n_var (int): Number of variables (genes)
        labels (np.ndarray): KMeans cluster labels for evaluation
        batch_size (int): Number of cells per training batch
        idx (np.ndarray): Indices of current batch
        score (list): History of evaluation metrics
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
        """
        Sample a batch of data for training.
        
        Returns:
            np.ndarray: Batch of data with shape (batch_size, n_var).
        """
        data, idx = self._sample_data()
        self.idx = idx
        return data

    def step(self, data):
        """
        Perform one training step: update model and compute metrics.
        
        Args:
            data (np.ndarray): Batch of training data.
        """
        self.update(data)
        latent = self.take_latent(data)
        score = self._calc_score(latent)
        self.score.append(score)

    def _sample_data(
        self,
    ):
        """
        Randomly sample a batch of cells from the dataset.
        
        Returns:
            tuple: (data, idx) containing:
                - data: Batch data of shape (batch_size, n_var)
                - idx: Indices of sampled cells
        """
        idx = np.random.permutation(self.n_obs)
        idx_ = np.random.choice(idx, self.batch_size)
        data = self.X[idx_, :]
        return data, idx_

    def _register_anndata(self, adata, layer: str, latent_dim):
        """
        Register and preprocess data from AnnData object.
        
        Args:
            adata (AnnData): Annotated data object
            layer (str): Layer name to extract (e.g., 'counts')
            latent_dim (int): Dimension of latent space for KMeans initialization
        """
        # Log1p transform the counts - handle both sparse and dense arrays
        layer_data = adata.layers[layer]
        if hasattr(layer_data, 'toarray'):
            # Sparse matrix
            self.X = np.log1p(layer_data.toarray())
        else:
            # Dense array
            self.X = np.log1p(layer_data)
        self.n_obs = adata.shape[0]
        self.n_var = adata.shape[1]
        # Initialize cluster labels for evaluation
        self.labels = KMeans(n_clusters=latent_dim).fit_predict(self.X)
        return