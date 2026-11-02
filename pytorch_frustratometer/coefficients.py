from __future__ import annotations

import ast
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Literal, Optional, Union

import torch
import yaml

@dataclass
class Coefficients:
    water_epsilon: float = 1.0

    ABC: torch.FloatTensor = torch.FloatTensor([
        [0.483, 0.703, -0.186], #an, bn, cn
        [0.444, 0.235,  0.321], #ap, bp, cp
        [0.841, 0.893, -0.734]  #ah, bh, ch 
    ])

    #Water
    k_water: float = 1.0
    #water_kappa_sigma: float = 5.0  # sharpness of sigma_wat switching function / Water/protein environment classification (density-based)
    water_kappa: float = 5.0 
    water_kappa_sigma: float = 7.0

    threshold: float = 2.6
    n_wells: int = 2

    #Wells
    well_rmin: torch.FloatTensor = torch.FloatTensor([4.5, 6.5])
    well_rmax: torch.FloatTensor = torch.FloatTensor([6.5, 9.5])
    well_flag: torch.IntTensor = torch.ones((2,))
   
    #Burial
    k_burial: float = 1.0
    burial_kappa: float = 4.0
    burial_ro_min: torch.FloatTensor = torch.FloatTensor([0.0, 3.0, 6.0])
    burial_ro_max: torch.FloatTensor = torch.FloatTensor([3.0, 6.0, 9.0])


    #Tertiary Frustratometer
    tert_frust_cutoff: float = 9.5
    tert_frust_ndecoys: int = 100_000
    tert_frust_exclude_native: bool = False  # deprecated: use tert_frust_decoy_distribution instead
    tert_frust_decoy_distribution: str = "sequence"  # "sequence" | "sequence_exclude_native" | "pseudocount" | "pseudocount_exclude_native"
    tert_frust_output_freq: int = 1
    mode: Union[list[Literal['singleresidue', 'mutational', 'configurational', 'smooth', 'all']], Literal['singleresidue', 'mutational', 'configurational', 'smooth', 'all']] = "all"
    run_dmfs: bool = False

    altloc: Literal['occupancy', 'first', 'all'] = 'first'

    use_electrostatics: bool = True  # set False to skip all electrostatic terms

    #DebyeHuckel]
    k_PlusPlus: float = 4.15
    k_MinusMinus: float = 4.15
    k_PlusMinus: float = 4.15
    k_screening: float = 1.0  # from config
    screening_length: float = 10.0 
    debye_huckel_min_sep: int = 1

    max_len: Optional[int] = None

    # Feature selection — intermediates are only computed when needed:
    #   _charge_intermediates skipped if both charge flags are False
    #   _density_intermediates skipped if both density flags are False
    use_charge_node:  bool = True
    use_charge_edge:  bool = True 
    use_density_node: bool = True
    use_density_edge: bool = True

    # Classification boundaries — always 0=highly, 1=neutral, 2=minimally.
    # Ascending row: standard searchsorted.
    # Descending row (smooth [2]): inverted scale, use classify_frustration().
    frustration_thresholds: torch.FloatTensor = torch.FloatTensor([
        [-1.0,  0.55],  # [0] single-residue
        [-1.0,  0.78],  # [1] pair (mutational / configurational / mj)
        [ 0.8, -1.2],   # [2] smooth (descending = inverted scale)
    ])

    # Distance / graph
    min_seq_distance: int = 3

    # Electrostatics
    electrostatic_epsilon: float = 80.0           # Dielectric constant
    electrostatic_kappa: float = 0.1              # Inverse Debye length

    # Density switching function
    rho_base: float = 0.25
    eta: float = 5.0                # Sharpness

    # Burial-specific density params (defaults match water-contact well — no-op)
    # Set burial_rmin < well_rmin[0] to count tighter contacts for the burial energy term only.
    burial_rmin: float = 4.5        # lower cutoff for burial density (default = well_rmin[0])
    burial_rmax: float = 6.5        # upper cutoff for burial density (default = well_rmax[0])
    # The residue density (burial and water terms) counts same-chain pairs with |i-j| > burial_min_sep
    # and pairs in different chains at any separation.
    burial_min_sep: int = 1

    # GLY coordinate: True = virtual H-Beta (matches FrustratometeR C++ reference, has known
    # x-coordinate bug in PDBToCoordinates.py); False = CA (original behaviour, matches AWSEM.py)
    gly_hbeta: bool = False

    # GLY pair-mode decoy correction: when True, decoy pair distances are corrected for
    # GLY↔non-GLY swaps (native-GLY decoys use vCB; native-non-GLY GLY decoys use CA).
    # Default False = original behaviour (all decoys evaluated at native CA/CB distance).
    gly_pair_correction: bool = False

    # Feature-calculator aliases (used by _density_edge_features)
    density_threshold: float = 2.6  # same as threshold
    r_max: float = 6.5              # same as well_rmax[0]
    use_anisotropy: bool = True    # was typo'd as use_anistropy in feature_calculator

    # ------------------------------------------------------------------
    def __post_init__(self):
        _valid_altloc = ('occupancy', 'first', 'all')
        if self.altloc not in _valid_altloc:
            raise ValueError(
                f"altloc={self.altloc!r} is not valid. "
                f"Choose one of: {_valid_altloc}"
            )

        # The residue density is summed over the contacts, so the contact cutoff must cover its reach.
        # Beyond the reach a pair adds less than 5e-5 to the density.
        density_reach = max(float(self.well_rmax[0]), float(self.burial_rmax)) + 5.0 / float(self.eta)
        if self.tert_frust_cutoff < density_reach:
            raise ValueError(
                f"tert_frust_cutoff={self.tert_frust_cutoff} is shorter than the residue density's reach, "
                f"{density_reach:g} A (max(well_rmax[0], burial_rmax) + 5 / eta); residue pairs beyond the "
                f"cutoff would be missing from the density."
            )

        _valid_modes = {'singleresidue', 'mutational', 'configurational', 'smooth', 'all'}
        modes = self.mode if isinstance(self.mode, list) else [self.mode]
        invalid = [m for m in modes if m not in _valid_modes]
        if invalid:
            raise ValueError(
                f"mode contains invalid values: {invalid}. "
                f"Choose from: {sorted(_valid_modes)}"
            )

    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "Coefficients":
        """
        Build a Coefficients from a plain dict, converting list values to the
        appropriate torch tensor type for fields that expect tensors.

        Example YAML input::

            tert_frust_ndecoys: 500
            well_rmin: [4.5, 6.5]
            mode: [mutational, configurational]
        """
        # Determine the expected type for each field from the default value
        defaults = {f.name: f.default for f in fields(cls)}
        kwargs = {}
        for key, val in d.items():
            default = defaults.get(key)
            if isinstance(default, torch.FloatTensor) or (
                isinstance(default, torch.Tensor) and default.dtype == torch.float32
            ):
                kwargs[key] = torch.FloatTensor(val)
            elif isinstance(default, torch.IntTensor) or (
                isinstance(default, torch.Tensor) and default.dtype == torch.int32
            ):
                kwargs[key] = torch.IntTensor(val)
            elif isinstance(default, torch.Tensor):
                kwargs[key] = torch.tensor(val, dtype=default.dtype)
            else:
                kwargs[key] = val
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Coefficients":
        """Load from a YAML file."""
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (tensors converted to lists)."""
        def _convert(v):
            return v.tolist() if isinstance(v, torch.Tensor) else v
        return {f.name: _convert(getattr(self, f.name)) for f in fields(self)}

    def to_yaml(self, path: str | Path) -> None:
        """Save current values to a YAML file (tensors serialised as lists)."""
        with open(path, "w") as fh:
            yaml.dump(self.to_dict(), fh, default_flow_style=False)

    @staticmethod
    def _parse_cli_value(raw: str):
        """
        Parse a raw string from ``--set key=value`` into the appropriate Python
        type using :func:`ast.literal_eval`, falling back to plain string.
        """
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw