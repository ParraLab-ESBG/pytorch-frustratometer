from .awsem import AWSEM, _register_mode_mixins  # noqa: F401
from .decoy_statistics import (  # noqa: F401
    decoy_aa_frequencies, decoy_aa_weights, decoy_statistics,
)


def __getattr__(name: str):
    # Lazy-load lightning-dependent classes to avoid paying the import cost
    # at package init time.
    if name in ("FrustratometerBase", "_FromPDBMixin"):
        import importlib
        mod = importlib.import_module("..base", __name__)
        return getattr(mod, name)
    if name in ("Frustratometer3",):
        from ..frustratometer import Frustratometer3
        return Frustratometer3
    if name == "SingleResidueFrustratometer":
        from .singleresidue import _get_SingleResidueFrustratometer
        return _get_SingleResidueFrustratometer()
    if name == "MutationalFrustratometer":
        from .mutational import _get_MutationalFrustratometer
        return _get_MutationalFrustratometer()
    if name == "ConfigurationalFrustratometer":
        from .configurational import _get_ConfigurationalFrustratometer
        return _get_ConfigurationalFrustratometer()
    # Fall back to awsem module for any other names (e.g., compute_* functions)
    try:
        from . import awsem as _awsem
        return getattr(_awsem, name)
    except AttributeError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
