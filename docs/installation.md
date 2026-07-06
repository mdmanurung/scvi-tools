# Installation

## Quick install

scvi-tools can be installed via `conda` or `pip`.
We recommend installing into a fresh virtual environment to avoid conflicts with other packages
and compatibility issues.

For the basic CPU version run:

```bash
pip install -U scvi-tools
```
or
```bash
conda install scvi-tools -c conda-forge
```

To install scvi-tools with Nvidia GPU CUDA support, for Linux Systems
(such as Ubuntu or RedHat) use:

```bash
pip install -U scvi-tools[cuda]
```
And for Apple Silicon metal (MPS) support:
```bash
pip install -U scvi-tools[metal]
```

Note that for some OS you will have to use quotes in order to install with dependencies,e.g.:
```bash
pip install -U "scvi-tools[cuda]"
```

Don't know how to get started with virtual environments or `conda`/`pip`? Check out the
[prerequisites](#prerequisites) section.

## Prerequisites

### Virtual environment

A virtual environment can be created with either `conda` or `venv`. We recommend using a fresh `conda` environment.
We currently support Python 3.12 - 3.14.


For `conda`, we recommend using the [Miniforge](https://github.com/conda-forge/miniforge) or
[Mamba](https://mamba.readthedocs.io/en/latest/) distribution, which are generally lighter and
faster than the official distribution and comes with conda-forge as the default channel
(where scvi-tools is hosted).

```bash
conda create -n scvi-env python=3.13  # any python 3.12 to 3.14
conda activate scvi-env
```

For `venv`, we recommend using [uv](https://github.com/astral-sh/uv), which is a high-performance
Python package manager and installer written in Rust.

```bash
pip install -U uv
uv venv .scvi-env
source .scvi-env/bin/activate  # for macOS and Linux
.scvi-env\Scripts\activate  # for Windows
```

### GPU support with PyTorch and JAX

scvi-tools depends on PyTorch for accelerated computing (and optionally on Jax). If you don't plan
on using an accelerated device, we recommend installing scvi-tools directly and letting these
dependencies be installed automatically by your package manager of choice.

If you plan on taking advantage of an accelerated device (e.g., Nvidia GPU or Apple Silicon),
which is likely, scvi-tools supports it, and you should install with the GPU support dependency of scvi-tools.

However, there might be cases where the GPU HW is not supporting the latest installation of PyTorch and Jax.
In this case we recommend installing PyTorch and JAX _before_ installing scvi-tools.
Please follow the respective installation instructions for [PyTorch](https://pytorch.org/get-started/locally/) and
[JAX](https://jax.readthedocs.io/en/latest/installation.html) compatible with your system and device type.

## Optional dependencies

scvi-tools is installed in its lightest form by default.
It has many optional dependencies which expand its capabilities:

- _autotune_ - in order to run scvi.autotune
- _hub_ - in order to use scvi.hub
- _regseq_ - in order to run scvi.data.add_dna_sequence
- _file_sharing_ - for convenient files sharing
- _parallel_ - for parallelization engine
- _interpretability_ - for supervised models interpretability
- _dataloaders_ - for custom dataloaders use
- _jax_ - for Jax support
- _mlflow_ - for MLflow support
- _tests_ - in order to be able to perform tests
- _editing_ - for code editing
- _dev_ - for development purposes
- _cuda_ - for Linux-based OS GPU support
- _metal_ - for Apple Silicon metal (MPS) support
- _docsbuild_ - in order to create docs

The easiest way to install this is with `pip`.
To install capability X run: _pip install scvi-tools[X]_

You can install several capabilities together, e.g:
To install scvi-tools with JAX support for GPU on Ubuntu: _pip install scvi-tools[cuda,jax]_

To install all tutorial dependencies:

```bash
pip install -U scvi-tools[tutorials]
```

### CytoANVI optional dependencies

:::{warning}
**Not yet published to PyPI; install from source into a clean environment.** The `cytoanvi`
distribution bundles a *modified* `scvi` (CytoVI lives at `scvi.external.cytovi`), so it ships both
the `scvi` and `cytoanvi` import packages. It must **not** be installed alongside the upstream
`scvi-tools` package — both provide the `scvi` import and would overwrite each other. Install it
alone, in a fresh virtualenv/conda env.
:::

CytoANVI's core label-transfer, uncertainty, save/load, and panel-aware query-mapping APIs install
with the base package (installed from a checkout of this repository):

```bash
pip install .
python -c "from cytoanvi import CytoANVI; print(CytoANVI.__name__)"
```

Install only the optional CytoANVI backend you need:

```bash
pip install ".[cytoanvi-hierarchy]"     # scHPL/treeArches helpers
pip install ".[cytoanvi-mapping-qc]"    # mapQC query-mapping QC
pip install ".[cytoanvi-annbatch]"      # experimental benchmark loader
pip install ".[cytoanvi-baselines]"     # FlowSOM benchmark baseline
```

The RAPIDS graph baseline is a benchmark dependency, not part of the stable CytoANVI model API:

```bash
pip install ".[rapids]"
```

For large CytoANVI cytometry atlases, use a CUDA-capable Linux environment when possible. The
publication benchmark environment used Python 3.13, PyTorch CUDA wheels, and an A100 40 GB GPU.
On conda-based HPC installations, set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` if importing PyTorch
fails with a `GLIBCXX` symbol error.

To install all optional dependencies (_e.g._ jax support, custom dataloaders, autotune, criticism, model hub):


```bash
pip install -U scvi-tools[optional]
```

To install development dependencies, including `pre-commit` and testing dependencies:

```bash
pip install -U scvi-tools[dev]
```

To install the common development, documentation, and tutorial dependency bundle, run:

```bash
pip install -U scvi-tools[all]
```

`scvi-tools[all]` follows the core project bundle and does not install every optional backend.
Install CytoANVI extras explicitly when you need hierarchy learning, mapping QC, AnnBatch, FlowSOM,
or RAPIDS benchmark support.

## Docker

If you plan on running scvi-tools in a containerized environment, we provide various Docker
[images](https://github.com/scverse/scvi-tools/pkgs/container/scvi-tools) hosted on GHCR.

For local CytoANVI release-candidate reproduction from this checkout, the repository Dockerfile has
separate CPU and CUDA targets:

```bash
docker build --target cpu -t scvi-tools-cytoanvi:cpu .
docker build --target cuda -t scvi-tools-cytoanvi:cuda .
docker run --gpus all --rm -it scvi-tools-cytoanvi:cuda
```

The CUDA target installs the CytoANVI hierarchy, mapping-QC, AnnBatch, and FlowSOM benchmark
extras by default. Publication-scale Roider benchmarks still require mounting the local or archived
`data/` cache into the container.

## R

scvi-tools can be called from R via Reticulate.

This is only recommended for basic functionality (getting the latent space, normalized expression,
differential expression). For more involved analyses with scvi-tools, we highly recommend using it
from Python.

The easiest way to install scvi-tools for R is via conda.

1. Install conda prerequisites.

2. Install R and reticulate in the conda environment:

    ```bash
    conda install -c conda-forge r-base r-essentials r-reticulate
    ```

3. Then in your R code:

    ```R
    library(reticulate)
    ```
See rest of R tutorials for further examples.
