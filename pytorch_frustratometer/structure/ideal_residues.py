"""
Ideal residue atom positions for decoy coordinate generation.

Builds a [20, 14, 3] canonical coordinate tensor (CA at origin, C along +x,
N in the +xy plane) at the reference chi angles stored in residue_constants.py.
`place_decoy_atoms` aligns these canonical coords to any N/CA/C backbone.

Usage
-----
    from .ideal_residues import place_decoy_atoms

    # aa_type: [B, E, n_decoys]  N/CA/C: [B, E, 3]
    coords = place_decoy_atoms(aa_type, N=batch_N, CA=batch_CA, C=batch_C)
    # -> [B, E, n_decoys, 14, 3]
"""
from __future__ import annotations
import functools

import torch
import torch.nn.functional as F

from .residue_constants import (
    chi_angles_atoms,
    rigid_group_atom_positions,
    restype_name_to_atom14_names,
    restype_1to3_natural,
)

# ── AA index ordering (matches restype_1to3_natural insertion order) ──────────
_AA1_ORDER: list[str] = list(restype_1to3_natural.keys())  # 20 single-letter
AA_TO_IDX:  dict[str, int] = {aa: i for i, aa in enumerate(_AA1_ORDER)}
N_AA    = 20
N_ATOM  = 14   # atom14 layout


# ── canonical coordinate table ────────────────────────────────────────────────

@functools.lru_cache(maxsize=4)
def canonical_coords(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return [20, 14, 3] ideal atom positions in the canonical backbone frame.

    Canonical frame: CA at origin, C along +x, N in +xy plane (y > 0).
    Atom axis follows the atom14 layout in restype_name_to_atom14_names.
    Side-chain atoms are placed at the reference chi angles stored in
    rigid_group_atom_positions (approximately the most common rotamer).
    """
    coords = torch.zeros(N_AA, N_ATOM, 3, dtype=dtype)

    for aa1, res3 in restype_1to3_natural.items():
        aa_idx    = AA_TO_IDX[aa1]
        atom14    = restype_name_to_atom14_names[res3]
        world_pos = _resolve_rigid_groups(res3, dtype)

        for slot, atom_name in enumerate(atom14):
            if atom_name and atom_name in world_pos:
                coords[aa_idx, slot] = world_pos[atom_name]

    return coords


def _resolve_rigid_groups(res3: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """
    Resolve all atoms in rigid_group_atom_positions into the canonical
    backbone frame (CA at origin, C along +x, N in +xy plane).

    Groups:
      0  backbone (atoms already in canonical frame)
      3  psi      (O; frame origin at C, axis along C→N direction)
      4  chi1     (axis = CA→CB, ref = N)
      5  chi2     (axis = CB→CG, ref = CA)
      6  chi3     (axis = CG→CD, ref = CB)
      7  chi4     (axis = CD→NE, ref = CG)
    """
    raw: dict[str, tuple[int, torch.Tensor]] = {
        name: (g, torch.tensor(pos, dtype=dtype))
        for name, g, pos in rigid_group_atom_positions.get(res3, [])
    }

    world: dict[str, torch.Tensor] = {}

    # ── Group 0: backbone — positions already in canonical CA frame ──
    for name, (g, pos) in raw.items():
        if g == 0:
            world[name] = pos

    # ── Group 3: psi — O is the only atom here ───────────────────────
    # Psi frame: origin at C, x along C→"next-N" direction.
    # For an isolated residue we approximate psi=180° (extended/trans)
    # using the backbone plane (N, CA, C already known).
    for name, (g, pos) in raw.items():
        if g == 3 and 'C' in world and 'N' in world and 'CA' in world:
            R, t = _local_frame(
                origin  = world['C'],
                x_dir   = world['N'] - world['C'],   # approx psi axis
                ref_dir = world['CA'] - world['C'],
            )
            world[name] = R @ pos + t

    # ── Chi groups 4–7 ───────────────────────────────────────────────
    # chi_angles_atoms[res3] = [[a,b,c,d], ...] for chi1, chi2, chi3, chi4.
    # Convention (from residue_constants comments):
    #   • rotation axis runs along the b→c bond
    #   • local frame origin = c  (axis-end atom)
    #   • x-axis = normalize(c − b)  (points away from backbone along axis)
    #   • y-axis defined so that atom a (the reference) is in the +xy plane
    chi_defs = chi_angles_atoms.get(res3, [])
    for k, (a, b, c, d) in enumerate(chi_defs):
        group_idx = k + 4
        if b not in world or c not in world or a not in world:
            break
        R, t = _local_frame(
            origin  = world[c],
            x_dir   = world[c] - world[b],
            ref_dir = world[a] - world[c],
        )
        for name, (g, pos) in raw.items():
            if g == group_idx:
                world[name] = R @ pos + t

    return world


def _local_frame(
    origin:  torch.Tensor,   # [3] frame origin in parent coords
    x_dir:   torch.Tensor,   # [3] direction of x-axis (unnormalised)
    ref_dir: torch.Tensor,   # [3] direction that lies in +xy plane (unnormalised)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a right-handed orthonormal frame and return (R, t) such that
        parent_pos = R @ local_pos + t
    x is along x_dir; y is the Gram–Schmidt component of ref_dir ⊥ x.
    """
    x = F.normalize(x_dir.unsqueeze(0), dim=-1).squeeze(0)
    y_raw = ref_dir - (ref_dir @ x) * x
    y = F.normalize(y_raw.unsqueeze(0), dim=-1).squeeze(0)
    z = torch.linalg.cross(x, y)
    R = torch.stack([x, y, z], dim=-1)   # columns → [3, 3]
    return R, origin


# ── backbone frame alignment ──────────────────────────────────────────────────

def backbone_frame(
    N:  torch.Tensor,   # [..., 3]
    CA: torch.Tensor,   # [..., 3]
    C:  torch.Tensor,   # [..., 3]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rotation R and translation t that map canonical backbone
    coordinates (CA at origin, C along +x, N in +xy) to world coordinates.

        world_pos = (R @ canonical_pos.unsqueeze(-1)).squeeze(-1) + t

    Returns
    -------
    R : [..., 3, 3]
    t : [..., 3]   (equals CA)
    """
    x = F.normalize(C - CA, dim=-1)
    nc = N - CA
    y = F.normalize(nc - (nc * x).sum(-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)
    R = torch.stack([x, y, z], dim=-1)   # [..., 3, 3]
    return R, CA


# ── main public API ───────────────────────────────────────────────────────────

def place_decoy_atoms(
    aa_type: torch.Tensor,   # [...] integer AA indices (0–19)
    N:       torch.Tensor,   # [..., 3]
    CA:      torch.Tensor,   # [..., 3]
    C:       torch.Tensor,   # [..., 3]
) -> torch.Tensor:
    """
    Place ideal heavy atoms for decoy amino-acid types aligned to the
    provided N/CA/C backbone.

    Parameters
    ----------
    aa_type : [...] LongTensor   integer AA index (0=ALA, …)
    N, CA, C : [..., 3]          backbone atom positions (world coords)

    Returns
    -------
    [..., 14, 3]  heavy-atom coordinates in world frame, atom14 layout.
                  Padding positions (atoms absent in the AA type) are zero.
    """
    ideal = canonical_coords(dtype=N.dtype).to(N.device)   # [20, 14, 3]
    local = ideal[aa_type]                                  # [...extra, 14, 3]

    R, t = backbone_frame(N, CA, C)   # [*bb, 3, 3], [*bb, 3]

    # aa_type may have extra dims beyond N's batch shape (e.g. n_decoys).
    # Flatten (extra × atom14) together, apply the per-backbone-position
    # transform, then restore the original shape.
    bb     = N.shape[:-1]                        # batch dimensions
    n_flat = local.shape[len(bb):]               # (extra..., 14)
    local_flat = local.reshape(*bb, -1, 3)       # [*bb, extra*14, 3]

    # R: [*bb, 3, 3] → unsqueeze to [*bb, 1, 3, 3] to broadcast over flat axis
    world_flat = (R.unsqueeze(-3) @ local_flat.unsqueeze(-1)).squeeze(-1)  # [*bb, extra*14, 3]
    world_flat = world_flat + t.unsqueeze(-2)    # broadcast t over flat axis

    return world_flat.reshape(*aa_type.shape, 14, 3)
