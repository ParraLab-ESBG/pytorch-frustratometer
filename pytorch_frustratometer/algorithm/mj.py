"""Miyazawa-Jernigan contact potential frustratometer.

Energy per contact: E_ij = xi(mu_i, mu_j) * Delta_ij
where xi is the MJ matrix (Table 3, Miyazawa & Jernigan 1996, J Mol Biol 256:623-644)
and Delta_ij = 1 for edges in the contact graph.

Only mutational frustration is meaningful — MJ has no structural/backbone term,
so configurational decoys (which shuffle backbone conformations) would yield the
same energy as native and produce trivially zero FI.
"""
from __future__ import annotations

import torch

from ..base import FrustratometerBase
from ..coefficients import Coefficients


# ---------------------------------------------------------------------------
# MJ matrix  (Table 3, Miyazawa & Jernigan 1996, J Mol Biol 256:623-644)
# Upper triangle + diagonal in MJ residue order:
#   C  M  F  I  L  V  W  Y  A  G  T  S  N  Q  D  E  H  R  K  P
# Values are in units of kT at room temperature (negative = attractive).
# Verify against the original paper before use in publications.
# ---------------------------------------------------------------------------

_MJ_ORDER = ['C','M','F','I','L','V','W','Y','A','G','T','S','N','Q','D','E','H','R','K','P']

# fmt: off
_MJ_UPPER = [
    # C
    -5.44, -4.99, -5.80, -5.50, -5.83, -4.96, -4.95, -4.16,
    -3.57, -3.16, -3.11, -2.86, -2.59, -2.85, -2.41, -2.27,
    -3.60, -2.57, -1.95, -3.07,
    # M
    -5.46, -6.56, -6.02, -6.41, -5.32, -5.55, -5.73, -4.62,
    -3.80, -4.20, -3.49, -3.82, -3.65, -3.52, -2.99, -4.48,
    -3.59, -2.77, -3.78,
    # F
    -5.13, -6.00, -6.06, -5.32, -5.35, -5.23, -4.28, -3.69,
    -3.95, -3.70, -3.33, -3.36, -3.33, -3.29, -4.41, -3.83,
    -3.16, -3.90,
    # I
    -5.00, -5.61, -4.92, -5.35, -5.10, -4.00, -3.56, -3.75,
    -3.46, -3.25, -3.30, -3.23, -2.74, -4.12, -3.46, -2.98,
    -3.56,
    # L
    -4.91, -5.30, -5.52, -4.91, -3.97, -3.71, -3.85, -3.50,
    -3.22, -3.40, -3.26, -3.03, -4.24, -3.59, -3.00, -3.64,
    # V
    -4.04, -4.53, -4.55, -3.46, -3.10, -3.24, -2.99, -2.87,
    -2.94, -2.79, -2.45, -3.63, -3.01, -2.50, -3.19,
    # W
    -5.06, -4.66, -3.41, -3.22, -3.37, -3.19, -2.99, -2.99,
    -3.10, -2.69, -4.08, -3.36, -2.67, -3.73,
    # Y
    -3.98, -3.31, -3.11, -3.20, -2.99, -3.08, -3.05, -2.78,
    -2.54, -3.63, -3.04, -2.60, -3.06,
    # A
    -2.72, -2.49, -2.43, -2.51, -2.39, -2.22, -2.09, -2.44,
    -2.29, -1.73, -1.88, -2.67,
    # G
    -2.05, -2.11, -1.83, -1.94, -1.90, -1.86, -1.79, -2.16,
    -1.94, -1.63, -1.96,
    # T
    -2.35, -2.26, -2.11, -2.08, -2.03, -2.14, -2.78, -2.57,
    -1.67, -2.27,
    # S
    -2.16, -2.10, -2.00, -1.96, -1.78, -2.57, -2.35, -1.57,
    -2.13,
    # N
    -2.18, -1.90, -1.93, -1.58, -2.40, -2.22, -1.53, -2.08,
    # Q
    -1.93, -1.89, -1.63, -2.36, -2.13, -1.42, -1.96,
    # D
    -1.79, -1.52, -2.15, -1.99, -1.04, -1.76,
    # E
    -1.44, -2.10, -1.95, -0.74, -1.55,
    # H
    -3.30, -3.34, -2.42, -3.17,
    # R
    -3.09, -2.07, -2.79,
    # K
    -1.28, -2.03,
    # P
    -2.72,
]
# fmt: on


def _build_mj_matrix() -> torch.Tensor:
    """Build the symmetric 20×20 MJ matrix from the upper-triangle values."""
    mat = torch.zeros(20, 20)
    k = 0
    for i in range(20):
        for j in range(i, 20):
            mat[i, j] = mat[j, i] = _MJ_UPPER[k]
            k += 1
    return mat


# Our alphabetical AA ordering (matches pytorch_frustratometer conventions)
_OUR_ORDER = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

# Permutation: our_idx -> mj_idx
_MJ_PERM = torch.tensor([_MJ_ORDER.index(aa) for aa in _OUR_ORDER], dtype=torch.long)


def build_mj_matrix_our_order() -> torch.Tensor:
    """Return the 20×20 MJ matrix indexed by our alphabetical AA ordering."""
    mj = _build_mj_matrix()
    return mj[_MJ_PERM][:, _MJ_PERM]


# ---------------------------------------------------------------------------
# MJ frustratometer
# ---------------------------------------------------------------------------

class MJFrustratometer(FrustratometerBase):
    """Frustratometer using Miyazawa-Jernigan contact potentials (mutational only).

    Replaces the AWSEM water/burial energy with a simple 20×20 residue-pair
    lookup: E_ij = xi(mu_i, mu_j).  Decoy statistics are exact over all amino
    acid pairs, weighted by the decoy distribution (same as AWSEM mutational).
    """

    def __init__(self, coefficients: Coefficients, output_dir=None,
                 dtype: torch.dtype = torch.float32):
        import dataclasses
        coefficients = dataclasses.replace(coefficients, mode="mutational")
        super().__init__(coefficients, output_dir=output_dir, dtype=dtype)
        self.register_buffer("mj_matrix", build_mj_matrix_our_order().to(dtype))

    # ------------------------------------------------------------------
    # Skip background electrostatic pre-computation — not used by MJ
    # ------------------------------------------------------------------

    def build_mutational_edges(self, batch):
        return None, None

    # ------------------------------------------------------------------
    # Energy overrides
    # ------------------------------------------------------------------

    def compute_mutational_ixn(self, batch, res_type_native):
        aa_src = res_type_native[0]  # [B, E]
        aa_dst = res_type_native[1]  # [B, E]
        return self.mj_matrix[aa_src, aa_dst]  # [B, E]

    def compute_mutational_decoy_energies(self, batch, B, E, decoy_aa):
        return self.mj_matrix[decoy_aa[0], decoy_aa[1]]  # [B, E, D]

    def forward(self, batch):
        from .mutational import MutationalMixin
        # Call _forward_single directly to bypass MRO: self._forward_single would
        # resolve to SingleResidueMixin._forward_single (earlier in MRO than Mutational).
        result = MutationalMixin._forward_single(self, batch)
        return {k.replace("mutational", "mj", 1): v for k, v in result.items()}

    def _check_r_threshold(self, batch, result, r_min: float = 0.99, acc_min: float = 0.9):
        mode_cfg = [
            ("mj", "mj", "mutational_frust_true",
             "mutational_frust_mask", self.coefficients.frustration_thresholds[1]),
        ]
        self._run_mode_check(batch, result, mode_cfg, r_min, acc_min)

    # ------------------------------------------------------------------
    # TODO: with EnergyFunction protocol (not wired), MJFrustratometer
    # would not override these methods at all.  Instead it would pass
    # MJEnergy(self.mj_matrix) as the energy_fn into the base engine:
    #
    #   # in __init__:
    #   from .energy.mj import MJEnergy
    #   self.energy_fn = MJEnergy(self.mj_matrix)
    #
    # The base MutationalMixin engine then calls:
    #   E_decoy = self.energy_fn.score_decoys(batch, {"decoy_aa": decoy_aa})
    #
    # compute_native_ixn and compute_mutational_decoy_energies are removed.
    # ------------------------------------------------------------------
