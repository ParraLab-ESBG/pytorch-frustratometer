from __future__ import annotations

from typing import Optional, NamedTuple

import torch


class DensityIntermediates(NamedTuple):
    rho:               Optional[torch.Tensor]  # [B, L, L] or None (sparse path)
    residue_densities: torch.Tensor            # [B, L]  per-residue summed density


def _unpack_edges(edge_index: torch.Tensor):
    """Unpack [B, 2, E] edge_index into flat batch/src/dst arrays, ignoring -1 padding."""
    B, _, E = edge_index.shape
    device = edge_index.device
    b   = torch.arange(B, device=device).unsqueeze(1).expand(B, E)  # [B, E]
    src = edge_index[:, 0, :]   # [B, E]
    dst = edge_index[:, 1, :]   # [B, E]
    valid = (src >= 0) & (dst >= 0)
    return b, src, dst, valid   # all [B, E]


def _density_intermediates(
    distances: torch.Tensor,                    # [B, L, L]
    seq_mask: torch.Tensor,                     # [B, L]
    residue_index: torch.Tensor,                # [B, L] true PDB residue numbers
    coords,
    aatype_int,
    cfg,
    *,
    chain_id: torch.Tensor | None = None,       # [B, L]
    dtype: torch.dtype = torch.float32,
    rmin_override: float | None = None,
    rmax_override: float | None = None,
    burial_min_sep: int,                        # same-chain pairs count when |i-j| > burial_min_sep
    edge_index: torch.Tensor | None = None,     # [B, 2, E] — enables sparse path
    return_rho: bool = False,                   # True when rho [B,L,L] is needed downstream
) -> DensityIntermediates:
    """
    Rho matrix + residue densities.

    Sparse path (edge_index provided, return_rho=False):
        Computes theta only at the E valid edge positions and scatter_adds to
        residue_densities [B, L] without ever allocating [B, L, L].

    Dense path (edge_index=None or return_rho=True):
        Retains the full [B, L, L] rho matrix — required for anisotropy node
        features and density edge features.
    """
    B, L, _ = distances.shape
    device = distances.device

    eta  = float(cfg.eta)
    rmin = float(rmin_override) if rmin_override is not None else float(cfg.well_rmin[0])
    rmax = float(rmax_override) if rmax_override is not None else float(cfg.well_rmax[0])

    # ── Sparse path ──────────────────────────────────────────────────────────
    if edge_index is not None and not return_rho:
        b, src, dst, valid = _unpack_edges(edge_index)  # [B, E] each

        d_e = distances[b, src.clamp(0), dst.clamp(0)].to(dtype)  # [B, E]

        # Sequence separation at tensor positions (not PDB residue numbers).
        sep_ok = (src - dst).abs() > burial_min_sep
        if chain_id is not None:
            cross = chain_id[b, src.clamp(0)] != chain_id[b, dst.clamp(0)]
            sep_ok = sep_ok | cross

        src_valid = seq_mask.bool()[b, src.clamp(0)]
        dst_valid = seq_mask.bool()[b, dst.clamp(0)]
        edge_ok   = valid & sep_ok & src_valid & dst_valid  # [B, E]

        t_min_e = torch.tanh(eta * (d_e - rmin))
        t_max_e = torch.tanh(eta * (rmax - d_e))
        theta_e = 0.25 * (1.0 + t_min_e) * (1.0 + t_max_e) * edge_ok.to(dtype)  # [B, E]

        # Scatter: each edge contributes to both src and dst residue density
        v_theta_sum = torch.zeros(B, L, device=device, dtype=dtype)
        v_theta_sum.view(-1).scatter_add_(0, (b * L + src.clamp(0)).view(-1), theta_e.view(-1))
        v_theta_sum.view(-1).scatter_add_(0, (b * L + dst.clamp(0)).view(-1), theta_e.view(-1))

        return DensityIntermediates(rho=None, residue_densities=v_theta_sum)

    # ── Dense path ────────────────────────────────────────────────────────────
    t_min = torch.tanh(eta * (distances - rmin))
    t_max = torch.tanh(eta * (rmax - distances))
    v_theta = 0.25 * (1.0 + t_min) * (1.0 + t_max)
    del t_min, t_max  # free 2×[B,L,L] before allocating the mask tensors

    # Compute sep as [L,L] int32 (not [B,L,L] int64) — 8× smaller, no batch dim needed
    idx = torch.arange(L, device=device, dtype=torch.int32)
    sep_2d = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()  # [L, L] int32
    exclude = (sep_2d <= burial_min_sep).unsqueeze(0)       # [1, L, L] bool; expands over B
    del sep_2d
    if chain_id is not None:
        same_chain = chain_id.unsqueeze(2) == chain_id.unsqueeze(1)
        exclude = exclude & same_chain

    neighbor_cutoff = getattr(cfg, "tert_frust_cutoff", None)
    within_neighbor = (distances <= neighbor_cutoff) if neighbor_cutoff is not None \
        else torch.ones(B, L, L, dtype=torch.bool, device=device)

    valid_mask = seq_mask.bool()
    pair_mask  = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
    pair_mask &= within_neighbor
    mask = ~exclude & pair_mask
    mask &= ~torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)

    v_theta_masked = v_theta * mask
    v_theta_sum    = v_theta_masked.sum(dim=-1)

    return DensityIntermediates(rho=v_theta_masked, residue_densities=v_theta_sum)


def compute_residue_densities(
    atom_positions: torch.Tensor,               # [B, L, 6, 3]
    aatype_int: torch.Tensor,                   # [B, L]
    seq_mask: torch.Tensor,                     # [B, L]
    hetatm: torch.Tensor,                       # [B, L]
    residue_index: torch.Tensor,                # [B, L]
    distance: torch.Tensor,                     # [B, L, L]
    cfg,
    *,
    chain_index: torch.Tensor | None = None,    # [B, L]
    edge_index: torch.Tensor | None = None,     # [B, 2, E] — enables sparse path
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Physics path: burial and water densities from structural inputs.

    Returns:
        residue_densities: [B, L]   water-contact density (for sigma_wat)
        burial_densities:  [B, L]   burial-specific density (may equal residue_densities)
        distance:          [B, L, L]
    """
    edge_mask = seq_mask.bool() & ~hetatm.bool()
    backbone  = atom_positions[..., :5, :]
    di = _density_intermediates(distance, edge_mask, residue_index, backbone, aatype_int, cfg,
                                chain_id=chain_index, dtype=dtype,
                                burial_min_sep=cfg.burial_min_sep,
                                edge_index=edge_index, return_rho=False)
    water_density = di.residue_densities
    return water_density, water_density, distance
