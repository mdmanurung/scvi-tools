> **This is the CytoANVI fork of [scvi-tools](https://github.com/scverse/scvi-tools).**
> CytoANVI adds semi-supervised, annotation-aware variational inference for antibody-based
> single-cell cytometry (mass cytometry, flow cytometry, CITE-seq protein).
> The upstream scvi-tools package is maintained by the [scverse community](https://scverse.org).
>
> **Status: research/internal use — not yet on PyPI.** This fork bundles a *modified* `scvi`
> (CytoVI lives at `scvi.external.cytovi`), so it ships both the `scvi` and `cytoanvi` import
> packages. Install it **from source into a clean environment** and do **not** install it alongside
> the upstream `scvi-tools` package (they collide on the `scvi` import). Public packaging is
> deferred pending resolution of that namespace decision.

<a href="https://scvi-tools.org/">
  <img
    src="https://github.com/scverse/scvi-tools/blob/main/docs/_static/scvi-tools-horizontal.svg?raw=true"
    width="400"
    alt="scvi-tools"
  >
</a>

[![PyPI][pypi-badge]][pypi-link]
[![PyPIDownloads][pepy-badge]][pepy-link]

[scvi-tools] (single-cell variational inference tools) is a package for probabilistic modeling and
analysis of single-cell omics data, built on top of [PyTorch] and [AnnData].

# Analysis of single-cell omics data

scvi-tools is composed of models that perform many analysis tasks across single-cell, multi, and
spatial omics data:

- Dimensionality reduction
- Data integration
- Automated annotation
- Cytometry label transfer with CytoANVI
- Factor analysis
- Doublet detection
- Spatial deconvolution
- and more!

In the [user guide], we provide an overview of each model. All model implementations have a
high-level API that interacts with [Scanpy] and includes standard save/load functions, GPU
acceleration, etc.

CytoANVI is available as a release-candidate top-level package for semi-supervised cytometry label
transfer:

```python
from cytoanvi import CytoANVI
```

See the CytoANVI user guide for current benchmark status, optional dependencies, and preprocessing
requirements before using it for publication-scale cytometry analyses.

# Rapid development of novel probabilistic models

scvi-tools contains the building blocks to develop and deploy novel probabilistic models. These
building blocks are powered by popular probabilistic and machine learning frameworks such as
[PyTorch Lightning] and [Pyro]. For an overview of how the scvi-tools package is structured, you
may refer to the [codebase overview] page.

We recommend checking out the [skeleton repository] as a starting point for developing and
deploying new models with scvi-tools.

# Basic installation

For pip,

```bash
pip install cytoanvi
```

Please be sure to install a version of [PyTorch] that is compatible with your GPU (if applicable).

# Resources

- Tutorials, API reference, and installation guides are available in the [documentation].
- For discussion of usage, check out our [forum].
- Please use the [issues] to submit bug reports.
- If you'd like to contribute, check out our [contributing guide].
- If you find a model useful for your research, please consider citing the corresponding
    publication.

# Reference

If you use `scvi-tools` in your work, please cite

> **A Python library for probabilistic analysis of single-cell omics data**
>
> Adam Gayoso, Romain Lopez, Galen Xing, Pierre Boyeau, Valeh Valiollah Pour Amiri, Justin Hong,
> Katherine Wu, Michael Jayasuriya, Edouard Mehlman, Maxime Langevin, Yining Liu, Jules Samaran,
> Gabriel Misrachi, Achille Nazaret, Oscar Clivio, Chenling Xu, Tal Ashuach, Mariano Gabitto,
> Mohammad Lotfollahi, Valentine Svensson, Eduardo da Veiga Beltrame, Vitalii Kleshchevnikov,
> Carlos Talavera-López, Lior Pachter, Fabian J. Theis, Aaron Streets, Michael I. Jordan,
> Jeffrey Regier & Nir Yosef
>
> _Nature Biotechnology_ 2022 Feb 07. doi: [10.1038/s41587-021-01206-w](https://doi.org/10.1038/s41587-021-01206-w).

along with the publication describing the model used.

You can cite the scverse publication as follows:

> **The scverse project provides a computational ecosystem for single-cell omics data analysis**
>
> Isaac Virshup, Danila Bredikhin, Lukas Heumos, Giovanni Palla, Gregor Sturm, Adam Gayoso,
> Ilia Kats, Mikaela Koutrouli, Scverse Community, Bonnie Berger, Dana Pe’er, Aviv Regev,
> Sarah A. Teichmann, Francesca Finotello, F. Alexander Wolf, Nir Yosef, Oliver Stegle &
> Fabian J. Theis
>
> _Nature Biotechnology_ 2023 Apr 10. doi: [10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8).

scvi-tools is part of the scverse® project ([website](https://scverse.org),
[governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).

If you like scverse® and want to support our mission, please consider making a tax-deductible
[donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time,
professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

Copyright (c) 2020, The scvi-tools development team
Copyright (c) 2026, CytoANVI contributors (Mikhael Manurung)

[anndata]: https://anndata.readthedocs.io/en/latest/
[codebase overview]: https://docs.scvi-tools.org/en/stable/user_guide/background/codebase_overview.html
[contributing guide]: https://github.com/mdmanurung/scvi-tools/blob/main/CONTRIBUTING.md
[documentation]: https://github.com/mdmanurung/scvi-tools
[forum]: https://discourse.scvi-tools.org
[issues]: https://github.com/mdmanurung/scvi-tools/issues
[pepy-badge]: https://static.pepy.tech/badge/cytoanvi
[pepy-link]: https://pepy.tech/project/cytoanvi
[pypi-badge]: https://img.shields.io/pypi/v/cytoanvi.svg
[pypi-link]: https://pypi.org/project/cytoanvi
[pyro]: https://pyro.ai/
[pytorch]: https://pytorch.org
[pytorch lightning]: https://lightning.ai/docs/pytorch/stable/
[scanpy]: http://scanpy.readthedocs.io/
[scvi-tools]: https://scvi-tools.org/
[skeleton repository]: https://github.com/scverse/simple-scvi
[user guide]: https://docs.scvi-tools.org/en/stable/user_guide/index.html
