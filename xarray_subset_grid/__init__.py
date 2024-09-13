# module
from . import grid  # noqa
from . import grids  # noqa
from . import utils  # noqa
from . import accessor  # noqa
from .selector import Selector

__all__ = ['Selector']

try:
    from ._version import __version__
except ImportError:
    __version__ = "version_unknown"

