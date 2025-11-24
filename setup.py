"""
Setup configuration for CODE package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="scCODE",
    version="0.1.0",
    author="Zeyu Fu, Chunlin Chen",
    author_email="",
    description="Correlated Latent Space Learning and Continuum Modeling of Single Cell Data",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/PeterPonyu/CODE",
    project_urls={
        "Bug Tracker": "https://github.com/PeterPonyu/CODE/issues",
        "Documentation": "https://github.com/PeterPonyu/CODE",
        "Source Code": "https://github.com/PeterPonyu/CODE",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=20.8b1",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    keywords=[
        "single-cell",
        "RNA-seq",
        "VAE",
        "neural-ODE",
        "trajectory-inference",
        "deep-learning",
        "bioinformatics",
    ],
)