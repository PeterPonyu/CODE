"""
CODE: Correlated Latent Space Learning and Continuum Modeling of Single Cell Data

A Python package for advanced single-cell RNA sequencing data analysis using
Variational Autoencoders (VAE) with optional Neural Ordinary Differential Equations (ODE)
for modeling cellular dynamics and developmental trajectories.

Main Components:
    - Agent: High-level interface for model training and inference
    - VAE: Variational autoencoder with multiple loss function options
    - ODE: Neural ODE for continuous trajectory modeling
    - MoCo: Momentum contrast for unsupervised contrastive learning

Example:
    >>> from CODE import Agent
    >>> import scanpy as sc
    >>> 
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> agent = Agent(adata, layer="counts", use_ode=True)
    >>> agent.fit(epochs=1000)
    >>> latent = agent.get_latent()
"""

from .agent import Agent

__all__ = ['Agent']

__version__ = '0.1.0'
