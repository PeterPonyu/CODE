# CODE

**C**orrelated Latent Space Learning and C**O**ntinuum Mo**DE**ling of Single Cell Data

A Python package for advanced single-cell data analysis using Variational Autoencoders (VAE) with optional Neural Ordinary Differential Equations (ODE) for modeling cellular dynamics and trajectories.

## Features

- **Variational Autoencoder (VAE)**: Learn low-dimensional latent representations of single-cell data
- **Neural ODE Integration**: Model continuous cell state transitions and trajectories
- **Multiple Loss Functions**: Support for MSE, Negative Binomial (NB), and Zero-Inflated Negative Binomial (ZINB) losses
- **Advanced Regularization**: Beta-VAE, DIP-VAE, Beta-TC-VAE, and InfoVAE (MMD) regularization options
- **Momentum Contrast (MoCo)**: Unsupervised contrastive learning for improved representations
- **Velocity Field Analysis**: Compute and visualize vector fields for cell state transitions
- **Trajectory Inference**: Infer pseudotime and developmental trajectories
- **Data Imputation**: Impute missing values using learned transition matrices

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU support)

### Install from source

```bash
git clone https://github.com/PeterPonyu/CODE.git
cd CODE
pip install -e .
```

### Dependencies

The package requires the following main dependencies:
- `torch` - PyTorch framework
- `numpy` - Numerical computations
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning utilities
- `scanpy` - Single-cell analysis toolkit
- `anndata` - Annotated data structures
- `torchdiffeq` - ODE solvers for PyTorch
- `tqdm` - Progress bars

## Quick Start

```python
import scanpy as sc
from CODE import Agent

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize the agent
agent = Agent(
    adata=adata,
    layer="counts",           # Data layer to use
    latent_dim=10,            # Latent space dimension
    hidden_dim=128,           # Hidden layer dimension
    i_dim=2,                  # Information bottleneck dimension
    use_ode=True,             # Enable ODE modeling
    loss_mode="nb",           # Loss function: "mse", "nb", or "zinb"
)

# Train the model
agent.fit(epochs=1000)

# Extract latent representations
latent = agent.get_latent()
adata.obsm["X_code"] = latent

# Extract pseudotime (if use_ode=True)
pseudotime = agent.get_time()
adata.obs["pseudotime"] = pseudotime

# Get information bottleneck embedding
iembed = agent.get_iembed()
adata.obsm["X_iembed"] = iembed
```

## Usage Examples

### Example 1: Basic VAE without ODE

```python
from CODE import Agent
import scanpy as sc

# Load data
adata = sc.read_h5ad("data.h5ad")

# Create agent without ODE
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    hidden_dim=128,
    use_ode=False,  # Disable ODE
    loss_mode="nb",
)

# Train
agent.fit(epochs=500)

# Get embeddings
latent = agent.get_latent()
```

### Example 2: ODE-based Trajectory Inference

```python
from CODE import Agent
import scanpy as sc

# Load data
adata = sc.read_h5ad("data.h5ad")

# Create agent with ODE
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    use_ode=True,    # Enable ODE for trajectory modeling
    ode_reg=0.5,     # ODE regularization weight
    vae_reg=0.5,     # VAE regularization weight
)

# Train
agent.fit(epochs=1000)

# Get pseudotime
pseudotime = agent.get_time()
adata.obs["pseudotime"] = pseudotime

# Get latent representation (combines VAE and ODE)
latent = agent.get_latent()
adata.obsm["X_code"] = latent
```

### Example 3: Velocity Field Computation

```python
# After training with use_ode=True
import scanpy as sc

# Compute velocity field
E_grid, V_grid = agent.get_vfres(
    adata=adata,
    zs_key="X_code",         # Latent embedding key
    E_key="X_umap",          # 2D embedding for visualization
    vf_key="X_vf",           # Output velocity field key
    stream=True,             # Use streamplot format
    density=1.0,             # Grid density
)

# Visualize with scanpy
sc.pl.embedding(adata, basis="umap", color="pseudotime")
```

### Example 4: Data Imputation

```python
# Get imputed data
imputed_data = agent.get_impute(
    top_k=30,      # Number of neighbors for transition matrix
    alpha=0.9,     # Blending weight (0=original, 1=imputed)
    steps=3,       # Multi-step imputation
    decay=0.99,    # Decay factor for multi-step
)

# Use imputed data
adata.layers["imputed"] = imputed_data
```

## API Reference

### Agent Class

The main interface for training and inference.

**Parameters:**
- `adata` (AnnData): Annotated data object containing single-cell data
- `layer` (str): Layer name in adata.layers to use (default: "counts")
- `percent` (float): Percentage of data to use per batch (default: 0.01)
- `latent_dim` (int): Dimension of latent space (default: 10)
- `hidden_dim` (int): Dimension of hidden layers (default: 128)
- `i_dim` (int): Dimension of information bottleneck (default: 2)
- `use_ode` (bool): Whether to use ODE modeling (default: False)
- `use_moco` (bool): Whether to use Momentum Contrast (default: False)
- `loss_mode` (str): Loss function type - "mse", "nb", or "zinb" (default: "nb")
- `recon` (float): Reconstruction loss weight (default: 1.0)
- `irecon` (float): Information bottleneck reconstruction weight (default: 0.0)
- `beta` (float): KL divergence weight (default: 1.0)
- `dip` (float): DIP-VAE regularization weight (default: 0.0)
- `tc` (float): Total correlation regularization weight (default: 0.0)
- `info` (float): InfoVAE (MMD) regularization weight (default: 0.0)
- `vae_reg` (float): VAE component weight when combining with ODE (default: 0.5)
- `ode_reg` (float): ODE component weight when combining with ODE (default: 0.5)
- `lr` (float): Learning rate (default: 1e-4)
- `device` (torch.device): Computing device (default: auto-detect)

**Methods:**
- `fit(epochs)`: Train the model
- `get_latent()`: Extract latent representations
- `get_iembed()`: Extract information bottleneck embeddings
- `get_time()`: Extract pseudotime (requires use_ode=True)
- `get_impute(top_k, alpha, steps, decay)`: Get imputed data
- `get_vfres(adata, ...)`: Compute velocity field results

## Advanced Features

### Regularization Options

CODE supports multiple VAE regularization techniques:

- **Beta-VAE** (`beta`): Controls the weight of KL divergence for disentanglement
- **DIP-VAE** (`dip`): Encourages decorrelated latent dimensions
- **Beta-TC-VAE** (`tc`): Penalizes total correlation in the latent space
- **InfoVAE** (`info`): Uses Maximum Mean Discrepancy (MMD) for regularization

### Loss Functions

- **MSE** (`loss_mode="mse"`): For normalized continuous data
- **Negative Binomial** (`loss_mode="nb"`): For count data (recommended)
- **Zero-Inflated NB** (`loss_mode="zinb"`): For sparse count data with excess zeros

## Citation

If you use CODE in your research, please cite:

```bibtex
@article{Fu2025,
  title = {Correlated latent space learning for structural differentiation modeling in single cell RNA data},
  journal = {Computers in Biology and Medicine},
  volume = {198},
  pages = {111115},
  year = {2025},
  issn = {0010-4825},
  doi = {10.1016/j.compbiomed.2025.111115},
  author = {Zeyu Fu and Chunlin Chen}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/PeterPonyu/CODE).

