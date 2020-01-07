from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import event
from . import apply_doy
from . import io
