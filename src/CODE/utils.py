import numpy as np
from numpy import ndarray
import pandas as pd
import scib
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.sparse import csr_matrix, csgraph, issparse


def get_dfs(mode, agent_list):
    """
    Aggregate evaluation scores across multiple agents.
    
    Args:
        mode (str): Aggregation mode - 'mean' or 'std'
        agent_list (list): List of lists of Agent objects
    
    Returns:
        generator: DataFrames with aggregated scores (ARI, NMI, ASW, C_H, D_B, P_C)
    """
    if mode == "mean":
        ls = list(
            map(
                lambda x: zip(
                    *(
                        np.array(b).mean(axis=0)
                        for b in zip(*((zip(*a.score)) for a in x))
                    )
                ),
                list(zip(*agent_list)),
            )
        )
    elif mode == "std":
        ls = list(
            map(
                lambda x: zip(
                    *(
                        np.array(b).std(axis=0)
                        for b in zip(*((zip(*a.score)) for a in x))
                    )
                ),
                list(zip(*agent_list)),
            )
        )
    return map(
        lambda x: pd.DataFrame(x, columns=["ARI", "NMI", "ASW", "C_H", "D_B", "P_C"]),
        ls,
    )


def moving_average(a, window_size):
    """
    Compute moving average with pandas rolling window.
    
    Args:
        a (array-like): Input array
        window_size (int): Size of the rolling window
    
    Returns:
        np.ndarray: Smoothed array with moving average
    """
    series = pd.Series(a)
    return (
        series.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    )


def fetch_score(adata1, q_z, label_true, label_mode="KMeans", batch=False):
    """
    Compute clustering and embedding quality metrics.
    
    Evaluates the quality of latent representations using various clustering
    and separation metrics. Supports batch effect evaluation.
    
    Args:
        adata1 (AnnData): Annotated data object
        q_z (np.ndarray): Latent representations
        label_true (np.ndarray): True labels for evaluation
        label_mode (str): Clustering method - 'KMeans', 'Max', or 'Min'. Default 'KMeans'
        batch (bool): Whether to compute batch-related metrics. Default False
    
    Returns:
        tuple: Evaluation metrics (NMI, ARI, ASW, C_H, D_B, [G_C, clisi, ilisi, bASW])
            - NMI: Normalized Mutual Information
            - ARI: Adjusted Rand Index (or Adjusted Mutual Information)
            - ASW: Average Silhouette Width
            - C_H: Calinski-Harabasz score
            - D_B: Davies-Bouldin score
            - G_C: Graph connectivity (only if returned)
            - clisi: cLISI score (only if batch=True)
            - ilisi: iLISI score (only if batch=True)
            - bASW: Batch ASW (only if batch=True)
    """
    # Subsample for efficiency if dataset is large
    if adata1.shape[0] > 3e3:
        idxs = np.random.choice(
            np.random.permutation(adata1.shape[0]), 3000, replace=False
        )
        adata1 = adata1[idxs, :]
        q_z = q_z[idxs, :]
        label_true = label_true[idxs]
    
    # Determine labels based on mode
    if label_mode == "KMeans":
        # Use the number of unique true labels as the number of clusters
        n_clusters = len(np.unique(label_true))
        labels = KMeans(n_clusters=n_clusters).fit_predict(q_z)
    elif label_mode == "Max":
        labels = np.argmax(q_z, axis=1)
    elif label_mode == "Min":
        labels = np.argmin(q_z, axis=1)
    else:
        raise ValueError("Mode must be one of: KMeans, Max, or Min")

    adata1.obsm["X_qz"] = q_z
    adata1.obs["label"] = pd.Categorical(labels)

    # Compute clustering metrics
    NMI = normalized_mutual_info_score(label_true, labels)
    ARI = adjusted_mutual_info_score(label_true, labels)
    ASW = silhouette_score(q_z, labels)
    if label_mode != "KMeans":
        ASW = abs(ASW)
    C_H = calinski_harabasz_score(q_z, labels)
    D_B = davies_bouldin_score(q_z, labels)

    # Subsample again for graph-based metrics if needed
    if adata1.shape[0] > 5e3:
        idxs = np.random.choice(
            np.random.permutation(adata1.shape[0]), 5000, replace=False
        )
        adata1 = adata1[idxs, :]
    
    # Compute graph connectivity
    G_C = graph_connection(
        kneighbors_graph(adata1.obsm["X_qz"], 15), adata1.obs["label"].values
    )
    clisi = scib.metrics.clisi_graph(adata1, "label", "embed", "X_qz", n_cores=-2)
    
    if batch:
        # Compute batch effect metrics
        ilisi = scib.metrics.ilisi_graph(adata1, "batch", "embed", "X_qz", n_cores=-2)
        bASW = scib.metrics.silhouette_batch(adata1, "batch", "label", "X_qz")
        return NMI, ARI, ASW, C_H, D_B, G_C, clisi, ilisi, bASW
    return NMI, ARI, ASW, C_H, D_B


def graph_connection(graph: csr_matrix, labels: ndarray):
    """
    Compute graph connectivity score for clustering quality.
    
    Measures how well connected each cluster is by computing the fraction of
    cells in the largest connected component for each cluster.
    
    Args:
        graph (csr_matrix): Cell-cell connectivity graph
        labels (ndarray): Cluster labels
    
    Returns:
        float: Mean connectivity score across all clusters (higher is better)
    """
    cg_res = []
    for l in np.unique(labels):
        mask = np.where(labels == l)[0]
        subgraph = graph[mask, :][:, mask]
        _, lab = csgraph.connected_components(subgraph, connection="strong")
        tab = np.unique(lab, return_counts=True)[1]
        cg_res.append(tab.max() / tab.sum())
    return np.mean(cg_res)


def quiver_autoscale(
    E: np.ndarray,
    V: np.ndarray,
):
    """
    Compute automatic scaling factor for quiver/arrow plots.
    
    Uses matplotlib's quiver autoscaling to determine appropriate arrow lengths
    for velocity field visualization.
    
    Args:
        E (np.ndarray): Embedding coordinates
        V (np.ndarray): Velocity vectors
    
    Returns:
        float: Scaling factor for velocity vectors
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()

    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles="xy",
        scale=None,
        scale_units="xy",
    )
    Q._init()
    fig.clf()
    plt.close(fig)
    return Q.scale / scale_factor


def l2_norm(x, axis=-1):
    """
    Compute L2 (Euclidean) norm of array.
    
    Handles both dense and sparse matrices efficiently.
    
    Args:
        x (np.ndarray or sparse matrix): Input array
        axis (int): Axis along which to compute norm. Default -1
    
    Returns:
        np.ndarray: L2 norms
    """
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    else:
        return np.sqrt(np.sum(x * x, axis=axis))