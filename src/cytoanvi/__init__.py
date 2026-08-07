from importlib.metadata import PackageNotFoundError, version

from . import hierarchy, mapping_qc
from ._model import CytoANVI
from ._module import CytoANVAE

try:
    __version__ = version("cytoanvi")
except PackageNotFoundError:  # package not installed (e.g. running from a source checkout)
    __version__ = "0.0.0.dev0"

# __version__ is intentionally not in __all__ (dunder, accessible as cytoanvi.__version__).
__all__ = ["CytoANVI", "CytoANVAE", "hierarchy", "mapping_qc"]
