"""MrTotalVI — TotalVI with per-donor hierarchical latent space (MrVI-style)."""

from ._model import MrTotalVI
from ._module import MrTotalVAE

__all__ = ["MrTotalVI", "MrTotalVAE"]
