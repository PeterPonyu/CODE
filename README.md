# CODE

<div align="center">

**C**orrelated Latent Space Learning and C**O**ntinuum Mo**DE**ling of Single Cell Data

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

A powerful Python package for single-cell RNA sequencing data analysis using **Variational Autoencoders (VAE)** with optional **Neural Ordinary Differential Equations (ODE)** for modeling cellular dynamics and developmental trajectories.

---

## 🌟 Key Features

- 🧬 **Variational Autoencoder (VAE)**: Learn low-dimensional latent representations of single-cell data
- 🔄 **Neural ODE Integration**: Model continuous cell state transitions and trajectories
- 📊 **Multiple Loss Functions**: Support for MSE, Negative Binomial (NB), and Zero-Inflated Negative Binomial (ZINB) losses
- 🎯 **Advanced Regularization**: Beta-VAE, DIP-VAE, Beta-TC-VAE, and InfoVAE (MMD) regularization options
- 🔗 **Momentum Contrast (MoCo)**: Unsupervised contrastive learning for improved representations
- 🌊 **Velocity Field Analysis**: Compute and visualize vector fields for cell state transitions
- 📈 **Trajectory Inference**: Infer pseudotime and developmental trajectories
- 🔧 **Data Imputation**: Impute missing values using learned transition matrices

---

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.9 or higher
- CUDA (optional, for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/PeterPonyu/CODE.git
cd CODE

# Install the package
pip install -e .
```

### Install from PyPI (when available)

```bash
pip install scCODE
```

### Dependencies

The package automatically installs the following dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥1.9.0 | Deep learning framework |
| `numpy` | ≥1.19.0 | Numerical computations |
| `scipy` | ≥1.5.0 | Scientific computing |
| `scikit-learn` | ≥0.23.0 | Machine learning utilities |
| `scanpy` | ≥1.7.0 | Single-cell analysis toolkit |
| `anndata` | ≥0.7.0 | Annotated data structures |
| `torchdiffeq` | ≥0.2.0 | ODE solvers for PyTorch |
| `tqdm` | ≥4.50.0 | Progress bars |

---

## 🚀 Quick Start

```python
import scanpy as sc
from CODE import Agent

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize the agent
agent = Agent(
    adata=adata,
    layer="counts",      # Data layer to use
    latent_dim=10,       # Latent space dimension
    hidden_dim=128,      # Hidden layer dimension
    use_ode=True,        # Enable ODE modeling
    loss_mode="nb",      # Loss function: "mse", "nb", or "zinb"
)

# Train the model
agent.fit(epochs=1000)

# Extract results
latent = agent.get_latent()              # Latent representations
pseudotime = agent.get_time()            # Pseudotime (requires use_ode=True)
iembed = agent.get_iembed()              # Information bottleneck embedding

# Save to AnnData
adata.obsm["X_code"] = latent
adata.obs["pseudotime"] = pseudotime
adata.obsm["X_iembed"] = iembed
```

For more examples, see the [examples](examples/) directory.

---

## 📚 Usage Examples

### Example 1: Basic VAE Analysis

```python
from CODE import Agent
import scanpy as sc

# Load and preprocess data
adata = sc.read_h5ad("data.h5ad")

# Create agent without ODE
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    hidden_dim=128,
    use_ode=False,       # Disable ODE
    loss_mode="nb",
)

# Train and extract embeddings
agent.fit(epochs=500)
latent = agent.get_latent()
```

### Example 2: Trajectory Inference with ODE

```python
from CODE import Agent
import scanpy as sc

# Load data
adata = sc.read_h5ad("data.h5ad")

# Create agent with ODE for trajectory modeling
agent = Agent(
    adata=adata,
    layer="counts",
    latent_dim=10,
    use_ode=True,        # Enable ODE
    ode_reg=0.5,         # ODE regularization weight
    vae_reg=0.5,         # VAE regularization weight
)

# Train
agent.fit(epochs=1000)

# Get pseudotime and latent representation
pseudotime = agent.get_time()
latent = agent.get_latent()

adata.obs["pseudotime"] = pseudotime
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

# Visualize
sc.pl.embedding(adata, basis="umap", color="pseudotime")
```

### Example 4: Data Imputation

```python
# Get imputed data
imputed_data = agent.get_impute(
    top_k=30,        # Number of neighbors
    alpha=0.9,       # Blending weight (0=original, 1=imputed)
    steps=3,         # Multi-step imputation
    decay=0.99,      # Decay factor
)

adata.layers["imputed"] = imputed_data
```

---

## 🔧 API Reference

### Agent Class

The main interface for training and inference.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | AnnData | *required* | Annotated data object containing single-cell data |
| `layer` | str | `"counts"` | Layer name in adata.layers to use |
| `percent` | float | `0.01` | Percentage of data to use per batch |
| `latent_dim` | int | `10` | Dimension of latent space |
| `hidden_dim` | int | `128` | Dimension of hidden layers |
| `i_dim` | int | `2` | Dimension of information bottleneck |
| `use_ode` | bool | `False` | Whether to use ODE modeling |
| `use_moco` | bool | `False` | Whether to use Momentum Contrast |
| `loss_mode` | str | `"nb"` | Loss function type: `"mse"`, `"nb"`, or `"zinb"` |
| `recon` | float | `1.0` | Reconstruction loss weight |
| `irecon` | float | `0.0` | Information bottleneck reconstruction weight |
| `beta` | float | `1.0` | KL divergence weight (Beta-VAE) |
| `dip` | float | `0.0` | DIP-VAE regularization weight |
| `tc` | float | `0.0` | Total correlation regularization weight (Beta-TC-VAE) |
| `info` | float | `0.0` | InfoVAE (MMD) regularization weight |
| `vae_reg` | float | `0.5` | VAE component weight when combining with ODE |
| `ode_reg` | float | `0.5` | ODE component weight when combining with ODE |
| `lr` | float | `1e-4` | Learning rate |
| `device` | torch.device | *auto* | Computing device (auto-detects CUDA) |

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `fit(epochs)` | Train the model | `Agent` |
| `get_latent()` | Extract latent representations | `np.ndarray` |
| `get_iembed()` | Extract information bottleneck embeddings | `np.ndarray` |
| `get_time()` | Extract pseudotime (requires `use_ode=True`) | `np.ndarray` |
| `get_impute(top_k, alpha, steps, decay)` | Get imputed data | `np.ndarray` |
| `get_vfres(adata, ...)` | Compute velocity field results | `tuple` |

---

## 🎓 Advanced Features

### Regularization Options

CODE supports multiple VAE regularization techniques:

| Regularization | Parameter | Description |
|----------------|-----------|-------------|
| **Beta-VAE** | `beta` | Controls KL divergence weight for disentanglement |
| **DIP-VAE** | `dip` | Encourages decorrelated latent dimensions |
| **Beta-TC-VAE** | `tc` | Penalizes total correlation in latent space |
| **InfoVAE** | `info` | Uses Maximum Mean Discrepancy (MMD) for regularization |

### Loss Functions

| Loss Function | Parameter Value | Best For |
|---------------|-----------------|----------|
| **MSE** | `loss_mode="mse"` | Normalized continuous data |
| **Negative Binomial** | `loss_mode="nb"` | Count data (recommended) |
| **Zero-Inflated NB** | `loss_mode="zinb"` | Sparse count data with excess zeros |

---

## 📖 Citation

If you use CODE in your research, please cite our paper:

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

---

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- Report bugs
- Suggest features
- Submit pull requests
- Set up your development environment

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/PeterPonyu/CODE/issues)
- **Repository**: [https://github.com/PeterPonyu/CODE](https://github.com/PeterPonyu/CODE)

---

## 🙏 Acknowledgments

We thank the open-source community for the excellent tools and libraries that made this project possible, including:

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Scanpy](https://scanpy.readthedocs.io/) - Single-cell analysis toolkit
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) - Neural ODE implementation

---

<div align="center">
Made with ❤️ by the CODE team
</div>
