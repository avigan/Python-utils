__all__ = ['proper_motion']

from . import proper_motion

try:
    from .skycalc import compute_sky_model
except ImportError:
    pass
