from .coefficients import Coefficients
import torch


def __getattr__(name: str):
    if name == "analysis":
        from . import analysis
        return analysis
    if name in ("FrustratometerDataset", "collate_frustratometer"):
        from .data.dataset import FrustratometerDataset, collate_frustratometer
        if name == "FrustratometerDataset":
            return FrustratometerDataset
        return collate_frustratometer
    if name == "Frustratometer3":
        from .frustratometer import Frustratometer3
        return Frustratometer3
    if name == "MJFrustratometer":
        from .algorithm.mj import MJFrustratometer
        return MJFrustratometer
    if name == "SingleResidueFrustratometer":
        from .algorithm.singleresidue import _get_SingleResidueFrustratometer
        return _get_SingleResidueFrustratometer()
    if name == "MutationalFrustratometer":
        from .algorithm.mutational import _get_MutationalFrustratometer
        return _get_MutationalFrustratometer()
    if name == "ConfigurationalFrustratometer":
        from .algorithm.configurational import _get_ConfigurationalFrustratometer
        return _get_ConfigurationalFrustratometer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def Frustratometer(mode, coefficients=None, energy_function="awsem", output_dir=None,
                   dtype=torch.float32, **coeff_kwargs):
    """Build the model for ``mode``: one mode, ``"all"`` or a list of modes.

    Returns ``Frustratometer3``, the model the CLI uses, for the AWSEM energy function and
    ``MJFrustratometer`` for ``energy_function="mj"``. Given ``coefficients`` are copied with ``mode`` set.
    """
    import dataclasses
    if coefficients is None:
        coefficients = Coefficients.from_dict({"mode": mode, **coeff_kwargs})
    else:
        coefficients = dataclasses.replace(coefficients, mode=mode)

    if energy_function == "mj":
        from .algorithm.mj import MJFrustratometer
        return MJFrustratometer(coefficients, output_dir=output_dir, dtype=dtype)

    from .frustratometer import Frustratometer3
    return Frustratometer3(coefficients, output_dir=output_dir, dtype=dtype)
