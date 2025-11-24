# CODE Examples

This directory contains example scripts demonstrating how to use CODE for single-cell RNA-seq analysis.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the fundamental usage of CODE with a standard VAE (without ODE):

- Loading single-cell data
- Basic preprocessing
- Training the VAE model
- Extracting latent representations
- Extracting information bottleneck embeddings

**Key Features:**
- Simple VAE setup
- Suitable for dimensionality reduction and clustering
- No trajectory inference

**Usage:**
```bash
python basic_usage.py
```

### 2. Trajectory Inference (`trajectory_inference.py`)

Demonstrates advanced trajectory inference using Neural ODE:

- Loading and preprocessing data
- Training with ODE enabled
- Computing pseudotime
- Extracting latent representations
- Computing velocity fields
- Data imputation

**Key Features:**
- Neural ODE integration
- Pseudotime estimation
- Velocity field computation
- Multi-step imputation

**Usage:**
```bash
python trajectory_inference.py
```

## Prerequisites

Before running the examples, ensure you have installed CODE and its dependencies:

```bash
pip install -e ..
```

Or if installing from PyPI:
```bash
pip install scCODE
```

## Data Requirements

The examples use the PBMC3k dataset from scanpy, which is automatically downloaded when running the scripts. For your own data:

1. Ensure your data is in AnnData format (`.h5ad` file)
2. Store raw counts in a layer (e.g., `adata.layers["counts"]`)
3. Replace the data loading section in the examples

## Customization

You can modify the following parameters to suit your data:

### Model Parameters
- `latent_dim`: Dimension of latent space (default: 10)
- `hidden_dim`: Hidden layer size (default: 128)
- `i_dim`: Information bottleneck dimension (default: 2)
- `loss_mode`: Loss function - "mse", "nb", or "zinb" (default: "nb")

### Training Parameters
- `epochs`: Number of training epochs (default: 500-1000)
- `lr`: Learning rate (default: 1e-4)
- `percent`: Batch size as percentage of data (default: 0.01)

### Regularization Parameters
- `beta`: KL divergence weight (Beta-VAE)
- `dip`: DIP-VAE regularization
- `tc`: Total correlation penalty (Beta-TC-VAE)
- `info`: MMD regularization (InfoVAE)

### ODE Parameters (when use_ode=True)
- `vae_reg`: Weight for VAE component (default: 0.5)
- `ode_reg`: Weight for ODE component (default: 0.5)

## Expected Output

Both examples will:
1. Display training progress with loss and evaluation metrics
2. Generate visualizations (if display is available)
3. Save results to the AnnData object

Results are stored in:
- `adata.obsm["X_code"]`: Latent representations
- `adata.obsm["X_iembed"]`: Information bottleneck embeddings
- `adata.obs["pseudotime"]`: Pseudotime (ODE only)
- `adata.layers["imputed"]`: Imputed data (ODE only)
- `adata.obsm["X_vf"]`: Velocity field (ODE only)

## Tips

1. **For large datasets**: Reduce `percent` parameter to speed up training
2. **For sparse data**: Use `loss_mode="zinb"` (Zero-Inflated Negative Binomial)
3. **For normalized data**: Use `loss_mode="mse"`
4. **For trajectory analysis**: Set `use_ode=True` and train for more epochs (1000+)
5. **GPU acceleration**: Install PyTorch with CUDA support for faster training

## Troubleshooting

### Out of Memory
- Reduce `percent` (batch size)
- Reduce `hidden_dim` or `latent_dim`
- Use a GPU with more memory

### Poor Results
- Increase training epochs
- Try different loss functions
- Adjust regularization parameters
- Ensure data is properly preprocessed

## Further Reading

For more details, see:
- [Main README](../README.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [API Reference](../README.md#-api-reference)
