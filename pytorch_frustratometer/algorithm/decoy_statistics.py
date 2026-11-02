"""Amino-acid decoy weights and exact (decoy-free) decoy statistics."""
from __future__ import annotations

import torch

N_AA = 20


def decoy_aa_frequencies(
    aa_type: torch.Tensor,   # [B, L]
    seq_mask: torch.Tensor,  # [B, L]
    distribution: str = "sequence",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:           # [B, 20]
    """Per-protein amino-acid probabilities used to draw or weight decoys.

    Only standard residues (aa < 20) inside ``seq_mask`` are counted. Distributions
    starting with ``"pseudocount"`` add one count to each of the 20 amino acids, so
    every amino acid has a nonzero probability.
    """
    valid = seq_mask.bool() & (aa_type < N_AA)
    one_hot = torch.nn.functional.one_hot(aa_type.clamp(0, N_AA - 1).long(), N_AA)
    counts = (one_hot.to(dtype) * valid.unsqueeze(-1)).sum(dim=-2)
    pseudo = 1.0 if distribution.startswith("pseudocount") else 0.0
    total = counts.sum(dim=-1, keepdim=True) + N_AA * pseudo
    return (counts + pseudo) / total.clamp(min=1.0)


def decoy_aa_weights(
    aa_type: torch.Tensor,   # [B, L]
    seq_mask: torch.Tensor,  # [B, L]
    native: torch.Tensor,    # [B, N]
    distribution: str = "sequence",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:           # [B, N, 20]
    """Decoy amino-acid weights at positions whose native amino acids are ``native``.

    Every position carries the ``decoy_aa_frequencies`` of its protein. For
    ``*_exclude_native`` distributions the native amino acid is removed and the rest
    renormalized. When the native carries all the probability (a single-letter
    sequence) the position keeps the frequencies, and a native that is not a standard
    amino acid (aa 20) removes nothing.
    """
    freq = decoy_aa_frequencies(aa_type, seq_mask, distribution, dtype)
    full = freq.unsqueeze(-2).expand(*native.shape, N_AA)
    if not distribution.endswith("_exclude_native"):
        return full
    is_native = native.unsqueeze(-1) == torch.arange(N_AA, device=native.device)
    excluded = full.masked_fill(is_native, 0.0)
    norm = excluded.sum(dim=-1, keepdim=True)
    return torch.where(norm > 0, excluded / norm.clamp(min=torch.finfo(norm.dtype).tiny), full)


def decoy_statistics(
    energies: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: int | tuple[int, ...] = -1,
) -> torch.Tensor:  # energies.shape without dim, then 2
    """Weighted mean and standard deviation of decoy energies over ``dim``.

    As ``np.average``: ``weights`` broadcasts against ``energies``, and None weights
    every decoy equally. The standard deviation is the population value (divides by
    the total weight), so with the decoy weights of every amino acid it is the
    infinite-decoy limit of sampling.
    """
    if weights is None:
        mean = energies.mean(dim=dim, keepdim=True)
        var = ((energies - mean) ** 2).mean(dim=dim, keepdim=True)
    else:
        weights = torch.broadcast_to(weights.to(energies.dtype), energies.shape)
        total = weights.sum(dim=dim, keepdim=True)
        mean = (weights * energies).sum(dim=dim, keepdim=True) / total
        var = (weights * (energies - mean) ** 2).sum(dim=dim, keepdim=True) / total
    return torch.stack([mean.squeeze(dim), var.sqrt().squeeze(dim)], dim=-1)
