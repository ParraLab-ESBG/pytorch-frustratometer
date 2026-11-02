import math
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

try:
    from torch_cluster import radius as torch_cluster_radius
except ImportError:
    torch_cluster_radius = None


def place_fourth_atom(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    length: float,
    angle_deg: float,
    dihedral_deg: float,
) -> torch.Tensor:
    """
    Place atom D from atoms A, B, C using CHARMM internal-coordinate convention:

        |CD|      = length
        angle     = ∠BCD
        dihedral  = (A, B, C, D)

    Works for any leading batch dimensions as long as the last dim is 3.
    """
    angle    = math.radians(angle_deg)
    dihedral = math.radians(dihedral_deg)

    bc = b - c
    bc = bc / bc.norm(dim=-1, keepdim=True)

    n = torch.cross(b - a, bc, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True)

    y = torch.cross(n, bc, dim=-1)

    return (
        c
        + length * math.cos(angle) * bc
        + length * math.sin(angle) * math.cos(dihedral) * y
        - length * math.sin(angle) * math.sin(dihedral) * n
    )


def place_missing_cb(atom_positions: torch.Tensor) -> torch.Tensor:
    """
    Place CB using CHARMM ideal internal coordinates (matches MODELLER's
    build_method='INTERNAL_COORDINATES', no energy minimization).

    Parameters
    ----------
    atom_positions : [..., num_atoms, 3]
        Atom coordinates with at least N(0), CA(1), C(2), CB(3).

    Returns
    -------
    atom_positions : same shape, with CB slot filled at ideal position.
    """
    atom_positions = atom_positions.clone()
    N  = atom_positions[..., 0, :]
    CA = atom_positions[..., 1, :]
    C  = atom_positions[..., 2, :]
    CB = place_fourth_atom(N, C, CA, length=1.5220, angle_deg=111.09, dihedral_deg=123.23)
    atom_positions[..., 3, :] = torch.nan_to_num(CB)
    return atom_positions


def get_CB_coords(atom_positions, aatype_int, gly_hbeta: bool = False):
    ca = atom_positions[:, :, 1, :]   # [B, R, 3]
    cb = atom_positions[:, :, 3, :]   # [B, R, 3]
    is_gly    = (aatype_int == 5)[..., None]
    missing_cb = (cb.norm(dim=-1, keepdim=True) < 1e-6) & ~is_gly  # CB absent → (0,0,0)
    N  = atom_positions[:, :, 0, :]
    C  = atom_positions[:, :, 2, :]
    ideal_cb = torch.nan_to_num(
        place_fourth_atom(N, C, ca, length=1.5220, angle_deg=111.09, dihedral_deg=123.23)
    )
    centroid = torch.where(missing_cb, ideal_cb, cb)
    if not gly_hbeta:
        return torch.where(is_gly, ca, centroid)
    # GLY: virtual H-Beta replicating FrustratometeR PDBToCoordinates.py (line 219).
    # xyz_H[0..2] are computed correctly with aH/bH/cH, but the Atom() constructor
    # is called as Atom(..., xyz_N[0], xyz_H[1], xyz_H[2], 'H-Beta') — passing N.x
    # instead of xyz_H[0] for the x-coordinate. Replicated here for consistency with
    # training data produced by that script.
    aH, bH, cH = -0.946747, 2.50352, -0.620388
    gly_coord = torch.stack([
        N[..., 0],                                              # x: N.x (bug: should be aH*N.x+bH*CA.x+cH*C.x)
        aH * N[..., 1] + bH * ca[..., 1] + cH * C[..., 1],  # y: correct
        aH * N[..., 2] + bH * ca[..., 2] + cH * C[..., 2],  # z: correct
    ], dim=-1)
    return torch.where(is_gly, gly_coord, centroid)


def compute_mpnn_vcb(batch, dtype=None) -> torch.Tensor:
    """
    ProteinMPNN closed-form virtual CB: CA - 0.58273431·(CA-N × C-CA)
                                           + 0.56802827·(CA-N)
                                           - 0.54067466·(C-CA)

    Derived from average ideal sp3 tetrahedral geometry across protein structures.
    Used as a fallback for unusual backbone conformations (e.g. positive-φ GLY)
    where the CHARMM36 IC formula places CB pointing into solvent.

    Returns [B, L, 3]; falls back to CA where the backbone is degenerate.
    """
    atom_pos = batch["atom_positions"]
    dt = dtype if dtype is not None else atom_pos.dtype

    N  = atom_pos[:, :, 0].to(dt)
    CA = atom_pos[:, :, 1].to(dt)
    C  = atom_pos[:, :, 2].to(dt)

    n_ca = CA - N        # N→CA
    ca_c = C  - CA       # CA→C
    cross_nc = torch.cross(n_ca, ca_c, dim=-1)

    vcb_raw = CA - 0.58273431 * cross_nc + 0.56802827 * n_ca - 0.54067466 * ca_c
    is_degen = torch.isnan(vcb_raw).any(dim=-1, keepdim=True)
    return torch.where(is_degen, CA, torch.nan_to_num(vcb_raw))


def compute_pro_vcb(batch, dtype=None) -> torch.Tensor:
    """
    Compute ideal CB positions using Proline's CHARMM36 IC dihedral.

    All residues share the same CHARMM36 CB IC (length=1.5220 Å, ∠C-CA-CB=111.09°,
    dihedral=123.23°) EXCEPT Proline, whose pyrrolidine ring constrains the
    N-Cα-Cβ geometry to dihedral ≈ 115.3° (derived from CHARMM36 top_all36_prot.rtf
    and confirmed by AlphaFold rigid_group_atom_positions: Pro CB at (-0.546, -0.611, -1.293)
    vs ALA (-0.529, -0.774, -1.205) in the local backbone frame).

    Bond length and angle kept at the standard CHARMM36 values (1.5220 Å, 111.09°);
    only the dihedral differs.  Used in _compute_gly_mutant_fi for the Pro decoy row.
    """
    atom_pos = batch["atom_positions"]
    dt = dtype if dtype is not None else atom_pos.dtype

    N  = atom_pos[:, :, 0].to(dt)
    CA = atom_pos[:, :, 1].to(dt)
    C  = atom_pos[:, :, 2].to(dt)

    vcb_raw = place_fourth_atom(N, C, CA, length=1.5220, angle_deg=111.09, dihedral_deg=115.3)
    is_degen = torch.isnan(vcb_raw).any(dim=-1, keepdim=True)
    return torch.where(is_degen, CA, torch.nan_to_num(vcb_raw))


def compute_ideal_vcb(batch, dtype=None) -> torch.Tensor:
    """
    Compute ideal virtual CB positions for all residues in a batch using CHARMM36 IC:
        IC N C *CA CB: length=1.5220 Å, ∠C-CA-CB=111.09°, dihedral(N,C,CA,CB)=123.23°

    Where the backbone is degenerate (collinear N-CA-C → NaN cross product),
    returns CA to avoid spurious long-range contacts.

    Parameters
    ----------
    batch : dict with key "atom_positions" [B, L, ≥3, 3].
    dtype : optional torch.dtype; defaults to atom_positions.dtype.

    Returns
    -------
    vcb : [B, L, 3]
    """
    atom_pos = batch["atom_positions"]
    dt = dtype if dtype is not None else atom_pos.dtype

    N  = atom_pos[:, :, 0].to(dt)
    CA = atom_pos[:, :, 1].to(dt)
    C  = atom_pos[:, :, 2].to(dt)

    vcb_raw = place_fourth_atom(N, C, CA, length=1.5220, angle_deg=111.09, dihedral_deg=123.23)
    is_degen = torch.isnan(vcb_raw).any(dim=-1, keepdim=True)
    return torch.where(is_degen, CA, torch.nan_to_num(vcb_raw))


def compute_coords_diff(residue_coords: torch.Tensor) -> torch.Tensor:
    """
    Args:  residue_coords: [B, L, 3]
    Returns:               [B, L, L, 3]
    """
    return residue_coords.unsqueeze(2) - residue_coords.unsqueeze(1)


def _contacts_via_radius_graph(coords, max_distance, seq_mask, seq_pos, min_seq_distance, chain_index):
    """
    Find contact pairs using torch_cluster.radius — O(E) memory vs O(R^2) for triu_indices.

    Returns:
        ei_b [N], ei_i [N], ei_j [N]  — batch index and residue pair (i < j) for each contact
        or None if torch_cluster is not available.
    """
    try:
        from torch_cluster import radius as tc_radius
    except ImportError:
        return None

    B, R, _ = coords.shape
    device = coords.device

    coords_flat = coords.reshape(-1, 3)                          # [B*R, 3]
    batch_vec   = torch.arange(B, device=device).repeat_interleave(R)  # [B*R]

    # tc_radius(x, y, r, batch_x, batch_y): for each y[j], find all x[i] within r.
    # edge_index[0] = i (neighbor/x side), edge_index[1] = j (query/y side).
    # Returns ALL directed pairs including both (i→j) and (j→i).
    ei_raw = tc_radius(
        coords_flat, coords_flat, max_distance,
        batch_x=batch_vec, batch_y=batch_vec,
        max_num_neighbors=512,
    )

    # Upper-triangle deduplication: keep pairs where flat neighbor < flat query
    # (same as keeping r_i < r_j within each batch since flat_i = b*R + r_i).
    keep = ei_raw[0] < ei_raw[1]
    flat_i = ei_raw[0][keep]  # neighbor side → smaller residue index
    flat_j = ei_raw[1][keep]  # query side   → larger residue index

    ei_b = flat_i // R
    ei_i = flat_i % R
    ei_j = flat_j % R

    # seq_mask filter
    m_i = seq_mask[ei_b, ei_i]
    m_j = seq_mask[ei_b, ei_j]
    pair_mask = m_i & m_j

    # Fine seq-separation filter (gap residues don't count toward separation)
    sep_fine = (seq_pos[ei_b, ei_i] - seq_pos[ei_b, ei_j]).abs()
    fine_ok  = sep_fine >= min_seq_distance
    if chain_index is not None:
        cross    = chain_index[ei_i] != chain_index[ei_j]
        fine_ok  = fine_ok | cross
    pair_mask = pair_mask & fine_ok

    return ei_b[pair_mask], ei_i[pair_mask], ei_j[pair_mask]


def _build_close_iu(B, ei_b, ei_i, ei_j, device, dtype):
    """Pack (ei_b, ei_i, ei_j) into close_iu [B, 2, E_max] with -1 padding."""
    if len(ei_b) == 0:
        return torch.full((B, 2, 1), -1, device=device, dtype=dtype)

    E_per_b = torch.zeros(B, dtype=torch.long, device=device)
    E_per_b.scatter_add_(0, ei_b, torch.ones(len(ei_b), dtype=torch.long, device=device))
    E_max = int(E_per_b.max().item())

    close_iu = torch.full((B, 2, E_max), -1, device=device, dtype=dtype)

    # Sort by batch so we can compute per-batch local offsets
    order    = ei_b.argsort(stable=True)
    b_sorted = ei_b[order]
    i_sorted = ei_i[order]
    j_sorted = ei_j[order]

    cum = torch.zeros(B + 1, dtype=torch.long, device=device)
    cum[1:] = E_per_b.cumsum(0)

    local_pos = torch.arange(len(order), device=device) - cum[b_sorted]

    close_iu[b_sorted, 0, local_pos] = i_sorted
    close_iu[b_sorted, 1, local_pos] = j_sorted
    return close_iu


def calculate_distance_matrix(
    atom_positions,
    aatype_int,
    seq_mask,
    min_seq_distance=2,
    max_distance=9.5,
    chain_index=None,
    gly_hbeta: bool = False,
):
    """
    Compute pairwise CB distances and contact edge index.

    Uses torch_cluster.radius when available to find contacts in O(E) memory
    instead of O(R^2), avoiding the large triu_indices [2, R^2/2] allocation
    that causes OOM on structures with many chains.

    Args:
        atom_positions:    [B, R, num_atoms, 3]
        aatype_int:        [B, R]
        seq_mask:          [B, R]
        min_seq_distance:  exclude pairs closer than this in sequence
        max_distance:      contact cutoff in Angstroms
        chain_index:       optional [R] chain ids for multi-chain handling

    Returns:
        distances:  [B, R, R]
        iu:         [2, num_pairs]  upper-triangular indices (None when torch_cluster path taken)
        close_iu:   [B, 2, E_max]  contact pairs (padded, -1 for invalid)
    """
    B, R, _, _ = atom_positions.shape
    device = atom_positions.device

    coords = get_CB_coords(atom_positions, aatype_int, gly_hbeta=gly_hbeta)   # [B, R, 3]
    distances = torch.cdist(coords, coords)              # [B, R, R]

    seq_pos = seq_mask.long().cumsum(dim=-1) - 1          # [B, R]  0-indexed

    # --- Memory-efficient contact finding via torch_cluster (only for large structures) ---
    rg_result = None
    if R > 300:
        rg_result = _contacts_via_radius_graph(
            coords, max_distance, seq_mask, seq_pos, min_seq_distance, chain_index
        )

    if rg_result is not None:
        ei_b, ei_i, ei_j = rg_result
        close_iu = _build_close_iu(B, ei_b, ei_i, ei_j, device, torch.long)
        # iu is not needed for this path; return None so callers that need it
        # (knn_neighbors=True in load_structure) fall back to triu_indices.
        iu = None
        return distances, iu, close_iu

    # --- Fallback: dense triu_indices (torch_cluster not installed) ---
    iu = torch.triu_indices(R, R, offset=1).to(device)  # [2, num_pairs]

    # Coarse filter on tensor indices
    iu_mask = (iu[0] - iu[1]).abs() >= min_seq_distance
    if chain_index is not None:
        different_chains = chain_index[iu[0]] != chain_index[iu[1]]
        iu_mask = iu_mask | different_chains
    iu = iu[:, iu_mask]

    mask_i = seq_mask[:, iu[0]]
    mask_j = seq_mask[:, iu[1]]
    pair_seq_mask = mask_i & mask_j

    # Fine filter: protein-sequential separation
    sep_fine = (seq_pos[:, iu[0]] - seq_pos[:, iu[1]]).abs()  # [B, num_pairs]
    fine_ok  = sep_fine >= min_seq_distance
    if chain_index is not None:
        cross = (chain_index[iu[0]] != chain_index[iu[1]]).unsqueeze(0)
        fine_ok = fine_ok | cross
    pair_seq_mask = pair_seq_mask & fine_ok

    iu_dist    = distances[:, iu[0], iu[1]]
    close_mask = (iu_dist < max_distance) & pair_seq_mask

    E_max = close_mask.sum(dim=1).max()
    idx   = close_mask.cumsum(dim=1) - 1
    idx   = idx.masked_fill(~close_mask, -1)
    iu_exp = iu.unsqueeze(0).expand(B, -1, -1)

    close_iu = torch.full((B, 2, max(int(E_max), 1)), -1, device=device, dtype=iu.dtype)
    b_idx, pair_idx = torch.where(close_mask)
    close_iu[b_idx, :, idx[b_idx, pair_idx]] = iu_exp[b_idx, :, pair_idx]

    return distances, iu, close_iu


def extract_close_edge_embeddings(full_edge_embedding, edge_index):
    B, R, _, d = full_edge_embedding.shape
    device = full_edge_embedding.device
    E_max = edge_index.shape[2]

    valid_mask = edge_index[:, 0, :] >= 0
    batch_idx_expand = torch.arange(B, device=device)[:, None].expand(-1, E_max)
    row_idx = edge_index[:, 0, :]
    col_idx = edge_index[:, 1, :]

    edge_embedding = torch.zeros(B, E_max, d, device=device, dtype=full_edge_embedding.dtype)
    edge_embedding[valid_mask] = full_edge_embedding[
        batch_idx_expand[valid_mask], row_idx[valid_mask], col_idx[valid_mask]
    ]
    return edge_embedding


def backbone_frame_axes(X: torch.Tensor) -> torch.Tensor:
    """
    Local backbone frame axes per residue.

    Uses the standard ideal-Cb formula for the third axis so it is defined
    for Gly (no real Cb) and consistent across all residue types.

    Args:  X: [B, L, >=3, 3]  atom positions with N(0), CA(1), C(2)
    Returns:  [B, L, 3, 3]    three unit-vector axes per residue
    """
    N, Ca, C = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    # Backup (ProteinMPNN formula): Ca - 0.58273431*cross(Ca-N,C-Ca) + 0.56802827*(Ca-N) - 0.54067466*(C-Ca)
    Cb = place_fourth_atom(N, C, Ca, length=1.5220, angle_deg=111.09, dihedral_deg=123.23)
    return torch.stack([
        F.normalize(Ca - N,  dim=-1),   # e1: N→CA
        F.normalize(C  - Ca, dim=-1),   # e2: CA→C
        F.normalize(Cb - Ca, dim=-1),   # e3: CA→ideal-Cb
    ], dim=-2)  # [B, L, 3, 3]


def project_node_phys_to_frame(
    node_features: torch.Tensor,  # [B, L, 10]
    X: torch.Tensor,              # [B, L, >=3, 3]
) -> torch.Tensor:                # [B, L, 10]
    """
    Make E_vec (dims 2:5) and hydro_vec (dims 5:8) rotationally invariant by
    projecting them from global coordinates onto local backbone frame axes.

    The charge node feature layout is:
        dims 0-1: q_i, hydrophobic_i          (scalars — untouched)
        dims 2-4: E_vec     [B, L, 3]         (global-frame vector — projected)
        dims 5-7: hydro_vec [B, L, 3]         (global-frame vector — projected)
        dims 8-9: residue_density, anisotropy (scalars — untouched)

    Returns node_features with the same shape; dims 2:8 replaced by their
    projections onto the three backbone frame axes.
    """
    frame = backbone_frame_axes(X).to(node_features.dtype)  # [B, L, 3, 3]
    # F.normalize inside backbone_frame_axes returns NaN for degenerate/missing atoms;
    # zero those out so bad residues contribute zero physics signal rather than poisoning the batch.
    frame = torch.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    E_vec     = node_features[..., 2:5]   # [B, L, 3]
    hydro_vec = node_features[..., 5:8]   # [B, L, 3]
    # einsum: for each residue, dot each axis with the vector → [B, L, 3]
    E_proj     = torch.einsum("blai,bli->bla", frame, E_vec)
    hydro_proj = torch.einsum("blai,bli->bla", frame, hydro_vec)
    return torch.cat([
        node_features[..., :2],   # q_i, hydrophobic_i
        E_proj,                   # frame-projected E_vec
        hydro_proj,               # frame-projected hydro_vec
        node_features[..., 8:],   # residue_density, anisotropy
    ], dim=-1)


def compute_residue_neighbor_mask(
    coords: torch.Tensor,   # [B, L, 3]
    cutoff: float
) -> torch.Tensor:
    """
    Compute residue-level neighbor mask using radius search.
    Uses torch.cdist for L <= 300 and torch_cluster for L > 300.
    Returns: [B, L, L] boolean mask
    """
    B, L, _ = coords.shape
    device = coords.device

    # torch_cluster is CPU-only; MPS must use torch.cdist regardless of L
    if L <= 300 or device.type == "mps":
        return torch.cdist(coords, coords) <= cutoff

    # For large proteins use torch_cluster to avoid O(L^2) memory
    if torch_cluster_radius is None:
        raise ImportError("torch_cluster is required for proteins with L > 300. "
                          "Install with: pip install torch-cluster")

    # Flatten [B, L, 3] -> [B*L, 3] with per-point batch assignment
    flat_coords = coords.reshape(B * L, 3)
    batch = torch.arange(B, device=device).repeat_interleave(L)

    # edge_index[0]: neighbor indices, edge_index[1]: query indices (in flat space)
    edge_index = torch_cluster_radius(
        flat_coords, flat_coords, cutoff,
        batch_x=batch, batch_y=batch,
        max_num_neighbors=512,
    )

    mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    b = edge_index[1] // L
    i = edge_index[1] % L
    j = edge_index[0] % L
    mask[b, i, j] = True
    mask = mask | mask.transpose(1, 2)
    return mask
