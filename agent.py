from .environment import Env
from .utils import quiver_autoscale, l2_norm
import scanpy as sc
from anndata import AnnData
import torch
import tqdm
from typing import Optional, Literal
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors


class Agent(Env):
    """
    Main agent class for training and inference with single-cell data.
    
    This class provides a high-level interface for training variational autoencoders
    (VAE) with optional Neural ODE components on single-cell data. It handles data
    loading, model training, and extraction of various representations including
    latent embeddings, pseudotime, and velocity fields.
    
    Inherits from Env which combines the model (CODEVAE) and environment functionality.
    
    Attributes:
        adata (AnnData): The annotated data object
        X (np.ndarray): Preprocessed data matrix (log1p transformed)
        n_obs (int): Number of observations (cells)
        n_var (int): Number of variables (genes)
        batch_size (int): Batch size for training
        loss (list): Training loss history
        score (list): Evaluation metrics history
    """

    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
        percent: float = 0.01,
        recon: float = 1.0,
        irecon: float = 0.0,
        beta: float = 1.0,
        dip: float = 0.0,
        tc: float = 0.0,
        info: float = 0.0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        use_ode: bool = False,
        use_moco: bool = False,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        lr: float = 1e-4,
        vae_reg: float = 0.5,
        ode_reg: float = 0.5,
        moco_weight: float = 1,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        super().__init__(
            adata=adata,
            layer=layer,
            percent=percent,
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
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

    def fit(self, epochs: int = 1000):
        """
        Train the VAE model on the single-cell data.
        
        Performs iterative training with mini-batch sampling. Progress is displayed
        with a progress bar showing loss and evaluation metrics (updated every 10 epochs).
        
        Args:
            epochs (int): Number of training epochs. Default is 1000.
        
        Returns:
            Agent: Self reference for method chaining.
        
        Metrics displayed:
            - Loss: Total training loss
            - ARI: Adjusted Rand Index (clustering similarity)
            - NMI: Normalized Mutual Information (clustering similarity)
            - ASW: Average Silhouette Width (cluster separation)
            - C_H: Calinski-Harabasz score (cluster density)
            - D_B: Davies-Bouldin score (cluster separation, lower is better)
            - P_C: Pearson Correlation (latent dimension correlation)
        """
        with tqdm.tqdm(total=int(epochs), desc="Fitting", ncols=150) as pbar:
            for i in range(int(epochs)):
                data = self.load_data()
                self.step(data)
                if (i + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "Loss": f"{self.loss[-1][0]:.2f}",
                            "ARI": f"{(self.score[-1][0]):.2f}",
                            "NMI": f"{(self.score[-1][1]):.2f}",
                            "ASW": f"{(self.score[-1][2]):.2f}",
                            "C_H": f"{(self.score[-1][3]):.2f}",
                            "D_B": f"{(self.score[-1][4]):.2f}",
                            "P_C": f"{(self.score[-1][5]):.2f}",
                        }
                    )
                pbar.update(1)
        return self

    def get_iembed(
        self,
    ):
        """
        Extract information bottleneck embeddings from the trained model.
        
        The information bottleneck is a lower-dimensional representation (i_dim)
        that compresses the latent space further, encouraging more compact and
        interpretable representations.
        
        Returns:
            np.ndarray: Information bottleneck embeddings of shape (n_obs, i_dim).
        """
        iembed = self.take_iembed(self.X)
        return iembed

    def get_latent(
        self,
    ):
        """
        Extract latent space representations from the trained model.
        
        For ODE models, this combines both VAE and ODE predictions weighted by
        vae_reg and ode_reg parameters. For standard VAE, returns the encoder output.
        
        Returns:
            np.ndarray: Latent representations of shape (n_obs, latent_dim).
        """
        latent = self.take_latent(self.X)
        return latent

    def get_time(
        self,
    ):
        """
        Extract predicted pseudotime for each cell.
        
        Only available when use_ode=True. Pseudotime represents the inferred
        developmental or temporal progression of cells, ranging from 0 to 1.
        
        Returns:
            np.ndarray: Pseudotime values of shape (n_obs,).
        
        Raises:
            AttributeError: If model was not trained with use_ode=True.
        """
        time = self.take_time(self.X)
        return time

    def get_impute(
        self, top_k: int = 30, alpha: float = 0.9, steps: int = 3, decay: float = 0.99
    ):
        """
        Impute gene expression values using learned cell-cell transition probabilities.
        
        This method uses the ODE-derived velocity field to construct a transition matrix
        between cells, then performs multi-step imputation by propagating information
        through the cell neighborhood graph.
        
        Args:
            top_k (int): Number of nearest neighbors to consider for sparse transition
                matrix. Default is 30.
            alpha (float): Blending weight between original and imputed data. 
                0 = use original data, 1 = use fully imputed data. Default is 0.9.
            steps (int): Number of diffusion steps for multi-step imputation. 
                Default is 3.
            decay (float): Decay factor for multi-step imputation weighting. 
                Default is 0.99.
        
        Returns:
            np.ndarray: Imputed data matrix of same shape as input (n_obs, n_var).
        
        Note:
            Requires use_ode=True during model initialization.
        """
        T = self.take_transition(self.X, top_k)

        def multi_step_impute(T, X, steps, decay):
            """
            Perform multi-step diffusion through transition matrix.
            
            Args:
                T (np.ndarray): Transition probability matrix
                X (np.ndarray): Input data matrix
                steps (int): Number of diffusion steps
                decay (float): Decay factor for weighting
            
            Returns:
                np.ndarray: Imputed data after multi-step diffusion
            """
            X_current = X.copy()
            X_imputed = X.copy()
            for i in range(steps):
                X_current = T @ X_current
                X_imputed = X_imputed + decay**i * X_current
            X_imputed = X_imputed / (1 + sum(decay**i for i in range(steps)))
            return X_imputed

        def balanced_impute(T, X, alpha=0.5, steps=3, decay=0.9):
            """
            Blend original and imputed data.
            
            Args:
                T (np.ndarray): Transition probability matrix
                X (np.ndarray): Input data matrix
                alpha (float): Blending weight
                steps (int): Number of diffusion steps
                decay (float): Decay factor
            
            Returns:
                np.ndarray: Balanced imputed data
            """
            X_imputed = multi_step_impute(T, X, steps, decay)
            X_balanced = (1 - alpha) * X + alpha * X_imputed
            return X_balanced

        return balanced_impute(T, self.X, alpha, steps, decay)

    def get_vfres(
        self,
        adata: AnnData,
        zs_key: str,
        E_key: str,
        vf_key: str = "X_vf",
        T_key: str = "cosine_similarity",
        dv_key: str = "X_dv",
        reverse: bool = False,
        run_neigh: bool = True,
        use_rep_neigh: Optional[str] = None,
        t_key: Optional[str] = None,
        n_neigh: int = 20,
        var_stabilize_transform: bool = False,
        scale: int = 10,
        self_transition: bool = False,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ):
        """
        Compute velocity field results for visualization and analysis.
        
        This method computes the velocity field (vector field) in a 2D embedding space
        (e.g., UMAP, t-SNE) based on the ODE gradients in the latent space. It includes
        computing similarity matrices, transition probabilities, and gridded vector fields.
        
        Args:
            adata (AnnData): Annotated data object to store results
            zs_key (str): Key in adata.obsm for latent space coordinates
            E_key (str): Key in adata.obsm for 2D embedding (e.g., 'X_umap')
            vf_key (str): Key to store velocity field gradients. Default 'X_vf'
            T_key (str): Key to store transition similarity matrix. Default 'cosine_similarity'
            dv_key (str): Key to store displacement vectors. Default 'X_dv'
            reverse (bool): Whether to reverse velocity direction. Default False
            run_neigh (bool): Whether to recompute neighbors. Default True
            use_rep_neigh (Optional[str]): Representation for neighbor computation. Default uses zs_key
            t_key (Optional[str]): Key in adata.obs for pseudotime-based neighbors
            n_neigh (int): Number of neighbors. Default 20
            var_stabilize_transform (bool): Apply variance stabilization. Default False
            scale (int): Scaling factor for transition probabilities. Default 10
            self_transition (bool): Include self-transitions. Default False
            smooth (float): Smoothing parameter for grid interpolation. Default 0.5
            stream (bool): Format output for streamplot. Default True
            density (float): Grid density for vector field. Default 1.0
        
        Returns:
            tuple: (E_grid, V_grid) containing:
                - E_grid: Grid coordinates for embedding space
                - V_grid: Velocity vectors at grid points
        
        Note:
            Requires use_ode=True during model initialization.
        """
        # Compute gradients in latent space from ODE
        grads = self.take_grad(self.X)
        adata.obsm[vf_key] = grads
        
        # Compute cosine similarity-based transition matrix
        adata.obsp[T_key] = self.get_similarity(
            adata,
            zs_key=zs_key,
            vf_key=vf_key,
            reverse=reverse,
            run_neigh=run_neigh,
            use_rep_neigh=use_rep_neigh,
            t_key=t_key,
            n_neigh=n_neigh,
            var_stabilize_transform=var_stabilize_transform,
        )
        
        # Project velocity field onto 2D embedding
        adata.obsm[dv_key] = self.get_vf(
            adata,
            T_key=T_key,
            E_key=E_key,
            scale=scale,
            self_transition=self_transition,
        )
        
        # Create gridded vector field for visualization
        E = np.array(adata.obsm[E_key])
        V = adata.obsm[dv_key]
        E_grid, V_grid = self.get_vfgrid(
            E=E,
            V=V,
            smooth=smooth,
            stream=stream,
            density=density,
        )
        return E_grid, V_grid

    def get_similarity(
        self,
        adata: AnnData,
        zs_key: str,
        vf_key: str = "X_vf",
        reverse: bool = False,
        run_neigh: bool = True,
        use_rep_neigh: Optional[str] = None,
        t_key: Optional[str] = None,
        n_neigh: int = 20,
        var_stabilize_transform: bool = False,
    ):
        """
        Compute cell-cell similarity matrix based on velocity field alignment.
        
        This method calculates cosine similarity between the velocity field direction
        and the displacement vectors between cells in latent space. High similarity
        indicates that a cell is likely to transition toward another cell.
        
        Args:
            adata (AnnData): Annotated data object
            zs_key (str): Key in adata.obsm for latent space coordinates
            vf_key (str): Key in adata.obsm for velocity field. Default 'X_vf'
            reverse (bool): Reverse velocity direction. Default False
            run_neigh (bool): Whether to compute neighbors. Default True
            use_rep_neigh (Optional[str]): Representation for neighbor computation
            t_key (Optional[str]): Key for pseudotime-based neighbor selection
            n_neigh (int): Number of neighbors. Default 20
            var_stabilize_transform (bool): Apply sqrt variance stabilization. Default False
        
        Returns:
            scipy.sparse.csr_matrix: Similarity matrix of shape (n_obs, n_obs) with
                values in range [-1, 1] representing cosine similarity.
        """
        Z = np.array(adata.obsm[zs_key])
        V = np.array(adata.obsm[vf_key])
        if reverse:
            V = -V
        if var_stabilize_transform:
            V = np.sqrt(np.abs(V)) * np.sign(V)

        ncells = adata.n_obs

        # Compute or use existing neighbor graph
        if run_neigh or ("neighbors" not in adata.uns):
            if use_rep_neigh is None:
                use_rep_neigh = zs_key
            else:
                if use_rep_neigh not in adata.obsm:
                    raise KeyError(
                        f"`{use_rep_neigh}` not found in `.obsm` of the AnnData. Please provide valid `use_rep_neigh` for neighbor detection."
                    )
            sc.pp.neighbors(adata, use_rep=use_rep_neigh, n_neighbors=n_neigh)
        n_neigh = adata.uns["neighbors"]["params"]["n_neighbors"] - 1

        # Compute pseudotime-based neighbors if provided
        if t_key is not None:
            if t_key not in adata.obs:
                raise KeyError(
                    f"`{t_key}` not found in `.obs` of the AnnData. Please provide valid `t_key` for estimated pseudotime."
                )
            ts = adata.obs[t_key].values
            indices_matrix2 = np.zeros((ncells, n_neigh), dtype=int)
            for i in range(ncells):
                idx = np.abs(ts - ts[i]).argsort()[: (n_neigh + 1)]
                idx = np.setdiff1d(idx, i) if i in idx else idx[:-1]
                indices_matrix2[i] = idx

        # Compute cosine similarity for each cell with its neighbors
        vals, rows, cols = [], [], []
        for i in range(ncells):
            # Get first and second-order neighbors
            idx = adata.obsp["distances"][i].indices
            idx2 = adata.obsp["distances"][idx].indices
            idx2 = np.setdiff1d(idx2, i)
            idx = (
                np.unique(np.concatenate([idx, idx2]))
                if t_key is None
                else np.unique(np.concatenate([idx, idx2, indices_matrix2[i]]))
            )
            
            # Compute displacement vectors
            dZ = Z[idx] - Z[i, None]
            if var_stabilize_transform:
                dZ = np.sqrt(np.abs(dZ)) * np.sign(dZ)
            
            # Compute cosine similarity between displacement and velocity
            cos_sim = np.einsum("ij, j", dZ, V[i]) / (
                l2_norm(dZ, axis=1) * l2_norm(V[i])
            )
            cos_sim[np.isnan(cos_sim)] = 0
            vals.extend(cos_sim)
            rows.extend(np.repeat(i, len(idx)))
            cols.extend(idx)

        res = coo_matrix((vals, (rows, cols)), shape=(ncells, ncells))
        res.data = np.clip(res.data, -1, 1)
        return res.tocsr()

    def get_vf(
        self,
        adata: AnnData,
        T_key: str,
        E_key: str,
        scale: int = 10,
        self_transition: bool = False,
    ):
        """
        Project velocity field from latent space to 2D embedding space.
        
        This method uses the transition probability matrix to compute expected
        displacement vectors in the embedding space (e.g., UMAP, t-SNE).
        
        Args:
            adata (AnnData): Annotated data object
            T_key (str): Key in adata.obsp for transition similarity matrix
            E_key (str): Key in adata.obsm for 2D embedding coordinates
            scale (int): Exponential scaling factor for transition probabilities. 
                Higher values emphasize strong transitions. Default 10
            self_transition (bool): Include self-transition probabilities. Default False
        
        Returns:
            np.ndarray: Displacement vectors in embedding space of shape (n_obs, 2).
        """
        T = adata.obsp[T_key].copy()

        # Optionally add self-transitions for stability
        if self_transition:
            max_t = T.max(1).A.flatten()
            ub = np.percentile(max_t, 98)
            self_t = np.clip(ub - max_t, 0, 1)
            T.setdiag(self_t)

        # Apply exponential scaling and normalize to probabilities
        T = T.sign().multiply(np.expm1(abs(T * scale)))
        T = T.multiply(csr_matrix(1.0 / abs(T).sum(1)))
        if self_transition:
            T.setdiag(0)
            T.eliminate_zeros()

        E = np.array(adata.obsm[E_key])
        V = np.zeros(E.shape)

        # Compute expected displacement for each cell
        for i in range(adata.n_obs):
            idx = T[i].indices
            dE = E[idx] - E[i, None]
            dE /= l2_norm(dE)[:, None]
            dE[np.isnan(dE)] = 0
            prob = T[i].data
            # Weighted average displacement minus mean to center
            V[i] = prob.dot(dE) - prob.mean() * dE.sum(0)

        # Normalize for visualization
        V /= 3 * quiver_autoscale(E, V)
        return V

    def get_vfgrid(
        self,
        E: np.ndarray,
        V: np.ndarray,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ):
        """
        Interpolate velocity field onto a regular grid for visualization.
        
        Creates a gridded representation of the velocity field suitable for
        matplotlib's streamplot or quiver functions.
        
        Args:
            E (np.ndarray): Embedding coordinates of shape (n_obs, 2)
            V (np.ndarray): Velocity vectors of shape (n_obs, 2)
            smooth (float): Smoothing bandwidth for Gaussian kernel. Default 0.5
            stream (bool): Format output for streamplot (True) or quiver (False). Default True
            density (float): Grid density multiplier. Default 1.0
        
        Returns:
            tuple: (E_grid, V_grid)
                - If stream=True: E_grid is (2, n, n) coordinate arrays, V_grid is (2, n, n) velocity
                - If stream=False: E_grid is (m, 2) grid points, V_grid is (m, 2) velocities
        """
        # Create grid along each dimension
        grs = []
        for i in range(E.shape[1]):
            m, M = np.min(E[:, i]), np.max(E[:, i])
            diff = M - m
            m = m - 0.01 * diff
            M = M + 0.01 * diff
            gr = np.linspace(m, M, int(50 * density))
            grs.append(gr)

        meshes = np.meshgrid(*grs)
        E_grid = np.vstack([i.flat for i in meshes]).T

        # Find nearest neighbors for each grid point
        n_neigh = int(E.shape[0] / 50)
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1)
        nn.fit(E)
        dists, neighs = nn.kneighbors(E_grid)

        # Gaussian kernel weighting based on distance
        scale = np.mean([g[1] - g[0] for g in grs]) * smooth
        weight = norm.pdf(x=dists, scale=scale)
        weight_sum = weight.sum(1)

        # Interpolate velocities with weighted average
        V_grid = (V[neighs] * weight[:, :, None]).sum(1)
        V_grid /= np.maximum(1, weight_sum)[:, None]

        if stream:
            # Format for streamplot
            E_grid = np.stack(grs)
            ns = E_grid.shape[1]
            V_grid = V_grid.T.reshape(2, ns, ns)

            # Filter out low-magnitude and low-confidence regions
            mass = np.sqrt((V_grid * V_grid).sum(0))
            min_mass = 1e-5
            min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
            cutoff1 = mass < min_mass

            length = np.sum(np.mean(np.abs(V[neighs]), axis=1), axis=1).reshape(ns, ns)
            cutoff2 = length < np.percentile(length, 5)

            cutoff = cutoff1 | cutoff2
            V_grid[0][cutoff] = np.nan
        else:
            # Format for quiver plot - filter by weight
            min_weight = np.percentile(weight_sum, 99) * 0.01
            E_grid, V_grid = (
                E_grid[weight_sum > min_weight],
                V_grid[weight_sum > min_weight],
            )
            V_grid /= 3 * quiver_autoscale(E_grid, V_grid)

        return E_grid, V_grid