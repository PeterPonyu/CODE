"""
Advanced Example: Trajectory Inference with Neural ODE

This example demonstrates using CODE with Neural ODE for trajectory inference
and pseudotime estimation in single-cell data.
"""

import scanpy as sc
from CODE import Agent
import numpy as np

# Load your single-cell data
print("Loading single-cell data...")
# Replace with your actual data file
adata = sc.datasets.pbmc3k()  # Example dataset

# Preprocess the data
print("Preprocessing...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Store raw counts
adata.layers["counts"] = adata.X.copy()

# Basic preprocessing
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Initialize CODE agent with ODE
print("\nInitializing CODE agent with Neural ODE...")
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    hidden_dim=128,
    i_dim=2,
    use_ode=True,         # Enable ODE for trajectory modeling
    loss_mode="nb",
    vae_reg=0.5,          # Balance between VAE and ODE
    ode_reg=0.5,
    lr=1e-4,
)

# Train the model
print("\nTraining model with ODE...")
agent.fit(epochs=1000)

# Extract results
print("\nExtracting results...")
latent = agent.get_latent()
adata.obsm["X_code"] = latent

# Get pseudotime (only available when use_ode=True)
pseudotime = agent.get_time()
adata.obs["pseudotime"] = pseudotime

# Get information bottleneck embedding
iembed = agent.get_iembed()
adata.obsm["X_iembed"] = iembed

# Compute velocity field
print("\nComputing velocity field...")
E_grid, V_grid = agent.get_vfres(
    adata=adata,
    zs_key="X_code",      # Latent embedding key
    E_key="X_umap",       # 2D embedding for visualization
    vf_key="X_vf",        # Output velocity field key
    stream=True,          # Use streamplot format
    density=1.0,          # Grid density
)

# Visualize results
print("\nVisualizing results...")
sc.pl.umap(adata, color=["louvain"], title="Original UMAP")
sc.pl.umap(adata, color=["pseudotime"], title="Pseudotime", cmap="viridis")
sc.pl.embedding(adata, basis="code", color=["pseudotime"], title="CODE latent space with pseudotime")

# Data imputation example
print("\nPerforming data imputation...")
imputed_data = agent.get_impute(
    top_k=30,      # Number of neighbors
    alpha=0.9,     # Blending weight
    steps=3,       # Multi-step imputation
    decay=0.99,    # Decay factor
)
adata.layers["imputed"] = imputed_data

print("\nDone! Results saved in adata:")
print("  - Latent representations: adata.obsm['X_code']")
print("  - Pseudotime: adata.obs['pseudotime']")
print("  - Imputed data: adata.layers['imputed']")
print("  - Velocity field: adata.obsm['X_vf']")
