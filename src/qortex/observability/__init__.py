"""Compatibility shim: re-exports from qortex_observe.

All observability code now lives in the qortex-observe package.
This module preserves backwards-compatible imports.
"""

from qortex_observe import *  # noqa: F401,F403
from qortex_observe import __all__  # noqa: F401
