from importlib.metadata import PackageNotFoundError, version

from . import hierarchy, mapping_qc
from ._model import CytoANVI
from ._module import CytoANVAE
from ._uncertainty import get_uncertainty_threshold

try:
    __version__ = version("cytoanvi")
except PackageNotFoundError:  # package not installed (e.g. running from a source checkout)
    __version__ = "0.0.0.dev0"

# __version__ is intentionally not in __all__ (dunder, accessible as cytoanvi.__version__);
# keeping the public-export lock (tests/cytoanvi/test_public_api.py) unchanged.
__all__ = ["CytoANVI", "CytoANVAE", "get_uncertainty_threshold", "hierarchy", "mapping_qc"]
