from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytorch_frustratometer.coefficients import Coefficients

import numpy as np
import torch

from .geometry import calculate_distance_matrix, get_CB_coords, compute_residue_neighbor_mask
from .parse import get_parsed_inputs_biotite, get_cif_parsed_inputs_biotite
from .place_ideal_atoms import place_missing_cb, place_missing_o
from ..utils.utils import apply_random_se3
from .protein import PDB_CHAIN_IDS


def _compute_sidechain_centroid(X: torch.Tensor) -> torch.Tensor:
    """Average position of non-backbone heavy atoms; used internally by load_structure."""
    sc_mask = torch.ones(X.shape[:-1], device=X.device, dtype=torch.bool)  # [..., A]
    sc_mask[..., :4] = False                     # exclude backbone: N, CA, C, CB
    sc_mask = sc_mask & (X != 0).all(dim=-1)

    masked = X * sc_mask[..., None].float()
    return masked.sum(dim=-2) / (sc_mask.sum(dim=-1, keepdim=True).float() + 1e-8)  # [..., 3]


def build_residue_lookup(residue_index, chain_index, chain_id_map=None, ins_code=None):
    """
    Build a per-chain residue lookup for multi-chain proteins.

    Args:
        residue_index: [L] LongTensor of PDB residue numbers (may be non-contiguous)
        chain_index:   [L] LongTensor of 0-indexed chain integers
        chain_id_map:  optional array/list of original chain letter strings indexed by c_int
        ins_code:      optional array of insertion code strings ('' for normal residues)

    Returns:
        dict[str → dict[str → int]]: chain_letter → {res_label → tensor_position}
        where res_label is e.g. "40" or "40A" (res_id + ins_code).
    """
    import numpy as _np
    ri_np  = residue_index.cpu().numpy()
    ci_np  = chain_index.cpu().numpy()
    ins_np     = ins_code if ins_code is not None else _np.full(len(ri_np), '', dtype='U4')
    all_labels = _np.char.add(ri_np.astype('U10'), _np.char.strip(ins_np))
    residue_lookup = {}
    for c_int in chain_index.unique().tolist():
        c_letter = chain_id_map[c_int] if chain_id_map is not None else PDB_CHAIN_IDS[c_int]
        mask = ci_np == c_int
        positions = _np.where(mask)[0]
        residue_lookup[c_letter] = dict(zip(all_labels[mask].tolist(), positions.tolist()))
    return residue_lookup


def load_structure(
    structure_file,
    cfg: Coefficients | None = None,
    rotate=True,
    multichain=False,
    knn_neighbors=False,
    dtype: torch.dtype = torch.float32,
    backbone_method="auto",
    trust_all_atoms=False,
    modeller_verbose=True,
    chain_id: str | None = None,
    write_repair_files: bool = False,
    repair_sink: dict | None = None,
):
    """
    Called inside Dataset.__getitem__. Parses PDB, applies random SE3 augmentation,
    and returns raw tensors ready for collation and GPU transfer.

    The sidechain centroid is computed here on CPU from the full 37-atom repr,
    then packed as the 6th atom. Only [L, 6, 3] is sent to the GPU — never the
    full [L, 37, 3], giving a ~6x reduction in that tensor's GPU memory footprint.

    Always returns two atom-position streams:
        atom_positions:     [L, 6, 3]  CB/O placed with Frame2Seq formula (for IPA encoder)
        atom_positions_raw: [L, 6, 3]  real PDB CB/O filled with physically correct geometry
                                        (for FrustratometeR / any real-space distance calc)

    Args:
        multichain: if True, residue_lookup is returned as dict[chain_letter → LongTensor]
                    and chain_index ([L] int tensor) is returned as the last element.
                    chain_index is also passed to calculate_distance_matrix so cross-chain
                    pairs below min_seq_distance are still included as contacts.

    Returns:
        atom_positions:     [L, 6, 3]  frame2seq CB/O  (IPA encoder input)
        atom_positions_raw: [L, 6, 3]  correct CB/O    (physics / features)
        aatype_int:         [L]
        seq_mask:           [L]
        distance:           [L, L]
        edge_index:         [2, E]
        residue_index:      [L]         true PDB residue numbers
        residue_lookup:     flat LongTensor (single-chain) or dict[str→LongTensor] (multi-chain)
        hetatm:             [L]
        chain_index:        [L] int tensor (only meaningful when multichain=True, else None)
    """
    structure_file = Path(structure_file)
    if structure_file.suffix == ".pdb":
        parser = get_parsed_inputs_biotite
    elif structure_file.suffix == ".cif":
        parser = get_cif_parsed_inputs_biotite
    else:
        raise RuntimeError(f"Invalid protein file: {structure_file}: must be pdb or cif")

    # atom_positions  = IPA-ready coords; CB/O placed with cb_mode
    # atom_positions_raw = real PDB coords; CB/O filled below with "correct" always
    atom_positions, atom_positions_raw, aatype_int, seq_mask, residue_index, hetatm, chain_index_raw, chain_id_map, auth_seq_id_np, ins_code_np = \
        parser(str(structure_file), chain_id=chain_id, round=False, dtype=dtype,
               save_raw_coordinates=True, trust_all_atoms=trust_all_atoms,
               backbone_method=backbone_method, modeller_verbose=modeller_verbose,
               altloc=getattr(cfg, 'altloc', 'first'),
               write_repair_files=write_repair_files, repair_sink=repair_sink)

    # Fill missing CB and O in raw coords with physically correct ideal geometry.
    # This is independent of cb_mode so features/distances are always accurate.
    is_gly = torch.from_numpy(aatype_int == 5)
    cb_missing = (atom_positions_raw[:, 3, :].norm(dim=-1) < 1e-6) & ~is_gly
    o_missing  = (atom_positions_raw[:, 4, :].norm(dim=-1) < 1e-6) & seq_mask.bool()
    if cb_missing.any() or o_missing.any():
        raw_correct = atom_positions_raw.clone().to(dtype)
        if cb_missing.any():
            raw_correct = place_missing_cb(raw_correct, mode="correct")
            atom_positions_raw[cb_missing, 3, :] = raw_correct[cb_missing, 3, :].to(atom_positions_raw.dtype)
        if o_missing.any():
            raw_correct = place_missing_o(raw_correct, mode="correct")
            atom_positions_raw[o_missing, 4, :] = raw_correct[o_missing, 4, :].to(atom_positions_raw.dtype)

    if rotate:
        atom_positions, random_rot = apply_random_se3(atom_positions, return_matrix=True)
        atom_positions_raw = apply_random_se3(atom_positions_raw, random_rot=random_rot)

    aatype_int    = torch.from_numpy(aatype_int).long()
    residue_index = torch.from_numpy(residue_index).long()
    chain_index   = torch.from_numpy(chain_index_raw).long()
    auth_seq_id   = torch.from_numpy(auth_seq_id_np).long() if auth_seq_id_np is not None else None

    if cfg is None:
        from pytorch_frustratometer.coefficients import Coefficients
        cfg = Coefficients()

    if cfg.max_len is not None:
        from ..utils.utils import check_size
        atom_positions     = check_size(atom_positions,     max_len=cfg.max_len)
        atom_positions_raw = check_size(atom_positions_raw, max_len=cfg.max_len)
        seq_mask           = check_size(seq_mask,           max_len=cfg.max_len)
        hetatm             = check_size(hetatm,             max_len=cfg.max_len)
        residue_index      = check_size(residue_index,      max_len=cfg.max_len)
        aatype_int         = check_size(aatype_int,         max_len=cfg.max_len).long()
        chain_index        = check_size(chain_index,        max_len=cfg.max_len).long()
        if auth_seq_id is not None:
            auth_seq_id    = check_size(auth_seq_id,        max_len=cfg.max_len).long()

    sc_centroid = _compute_sidechain_centroid(atom_positions_raw).to(dtype)  # [L, 3]
    # For the encoder (IPA): ideal CB/O placement, SE3-rotated
    atom_positions_out = torch.cat([
        atom_positions[:, :5, :].to(dtype),    # ideal N, CA, C, CB, O (regularized + SE3-rotated)
        sc_centroid.unsqueeze(1),
    ], dim=1)  # [L, 6, 3]
    # For feature computation: real PDB CB/O, SE3-rotated (same rotation as above)
    atom_positions_raw_out = torch.cat([
        atom_positions_raw[:, :5, :].to(dtype),  # raw N, CA, C, CB, O (SE3-rotated, not ideal)
        sc_centroid.unsqueeze(1),
    ], dim=1)  # [L, 6, 3]

    edge_mask = seq_mask.bool() & ~hetatm.bool()
    distance, iu, edge_index = calculate_distance_matrix(
        atom_positions_raw[None, ...],
        aatype_int[None, ...],
        edge_mask[None, ...],
        max_distance=cfg.tert_frust_cutoff,
        chain_index=chain_index if multichain else None,
        gly_hbeta=getattr(cfg, 'gly_hbeta', False),
    )
    distance   = distance[0]
    edge_index = edge_index[0]

    if knn_neighbors:
        neighbor_cutoff = cfg.tert_frust_cutoff
        coords_CB = get_CB_coords(atom_positions_raw[None, ...], aatype_int[None, ...], gly_hbeta=getattr(cfg, 'gly_hbeta', False)).to(dtype)
        within_neighbor = compute_residue_neighbor_mask(coords_CB, neighbor_cutoff)
        edge_index = within_neighbor[iu[0], iu[1]]

    # For the residue lookup table and output, prefer auth_seq_id (original PDB numbers).
    # residue_index is sequential per-chain (label_seq_id style) — used for L/distance matrix.
    out_res_index = auth_seq_id if auth_seq_id is not None else residue_index

    # Build string labels: "40" for a normal residue, "40A" for an insertion residue.
    _ins = ins_code_np if ins_code_np is not None else np.full(len(out_res_index), '', dtype='U4')
    res_labels = np.char.add(out_res_index.cpu().numpy().astype('U10'), np.char.strip(_ins))

    if multichain:
        residue_lookup = build_residue_lookup(out_res_index, chain_index,
                                              chain_id_map=chain_id_map, ins_code=_ins)
    else:
        # Dict[str → int]: maps the string label (e.g. "40" or "40A") to its tensor position.
        # Replaces the old pdb_offset tensor scheme; parse functions detect dict vs tensor.
        residue_lookup = {lbl: pos for pos, lbl in enumerate(res_labels)}
        chain_index = None

    return atom_positions_out, atom_positions_raw_out, aatype_int, seq_mask, distance, edge_index, out_res_index, residue_lookup, hetatm, chain_index, chain_id_map, res_labels
