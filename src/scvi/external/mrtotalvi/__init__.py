"""MrTotalVI — TotalVI with per-donor hierarchical latent space (MrVI-style)."""

from ._model import MrTotalVI
from ._module import MrTotalVAE
from ._seed import combine_mrtotalvi_seed_results

__all__ = [
    "MrTotalVI",
    "MrTotalVAE",
    "combine_mrtotalvi_seed_results",
]
