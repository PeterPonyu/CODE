"""
Basic Example: Training CODE on Single-Cell Data

This example demonstrates the basic usage of CODE for single-cell RNA-seq analysis
using a VAE without ODE modeling.
"""

import scanpy as sc
from CODE import Agent

# Note: This is a template example. Replace with your actual data file.
# Download example dataset from: https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html

# Load example data
print("Loading single-cell data...")
# For this example, we'll use scanpy's built-in PBMC dataset
adata = sc.datasets.pbmc3k()

# Store raw counts before any transformations
adata.layers["counts"] = adata.X.copy()

# Preprocess the data
print("Preprocessing...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Basic preprocessing for visualization
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# Initialize CODE agent
print("\nInitializing CODE agent...")
agent = Agent(
    adata=adata,
    layer="counts",        # Use the raw counts
    latent_dim=10,         # Dimension of latent space
    hidden_dim=128,        # Hidden layer dimension
    i_dim=2,              # Information bottleneck dimension
    use_ode=False,        # Disable ODE for basic example
    loss_mode="nb",       # Negative binomial loss for count data
    lr=1e-4,              # Learning rate
)

# Train the model
print("\nTraining model...")
agent.fit(epochs=500)

# Extract latent representations
print("\nExtracting results...")
latent = agent.get_latent()
adata.obsm["X_code"] = latent

# Get information bottleneck embedding
iembed = agent.get_iembed()
adata.obsm["X_iembed"] = iembed

# Visualize results
print("\nVisualizing results...")
sc.pl.umap(adata, color=["louvain"], title="Original UMAP with Louvain clusters")
sc.pl.embedding(adata, basis="code", color=["louvain"], title="CODE latent space")

print("\nDone! Latent representations saved in adata.obsm['X_code']")
