import logging
from pathlib import Path

import numpy as np
import torch

from . import protein, residue_constants
from .place_ideal_atoms import (place_missing_cb, place_missing_o,
                                 reconstruct_missing_backbone,
                                 _reconstruct_with_modeller)

logger = logging.getLogger(__name__)


def _describe_partial_backbone(prot: protein.Protein, index: int, source: Path) -> str:
    """Describe an excluded residue using its original identifiers and atom mask."""
    chain_index = int(prot.chain_index[index])
    chain = (f"chain {prot.chain_id_map[chain_index]}" if prot.chain_id_map is not None
             else f"chain_index {chain_index}")
    internal_index = int(prot.residue_index[index])
    identity = f"{chain}, internal residue_index={internal_index}"
    if prot.auth_seq_id is not None:
        insertion = str(prot.ins_code[index]).strip() if prot.ins_code is not None else ""
        identity += f", author residue={int(prot.auth_seq_id[index])}{insertion}"
    elif prot.ins_code is not None:
        identity += f", insertion code={str(prot.ins_code[index]).strip()!r}"
    mask = prot.atom_mask[index]
    missing = [name for name, present in zip(("N", "CA", "C"), mask[:3]) if not present]
    present = [name for name, exists in zip(residue_constants.atom_types, mask) if exists]
    return (f"Excluded residue ({identity}) from {source}: "
            f"original backbone missing {', '.join(missing)}; "
            f"original present atoms=[{', '.join(present)}]")


def _insert_gap_residues(aatype_int, atom_positions_raw, atom_mask_full,
                         residue_index, chain_index, hetatm, auth_seq_id=None,
                         ins_code=None):
    """
    Detect jumps in residue_index (per chain) and insert all-zero placeholder
    rows for each missing position.  Placeholders get aatype=20 (X) and all-zero
    atom positions/masks; reconstruct_missing_backbone will fill the backbone.

    residue_index must already be sequential per-chain (label_seq_id style) so
    that gaps reflect only real missing residues, not auth_seq_id renumbering
    artefacts.  auth_seq_id, if provided, is expanded with the same repeats and
    returned so the original PDB residue numbers survive for output labelling.
    ins_code, if provided, is expanded likewise; gap slots receive ''.

    Returns the expanded arrays plus is_gap (bool array marking inserted slots).
    """
    ridx = residue_index.astype(int)
    cidx = chain_index.astype(int)

    same_chain = cidx[:-1] == cidx[1:]
    gaps = np.where(same_chain, np.maximum(np.diff(ridx) - 1, 0), 0)  # [L-1]

    if gaps.sum() == 0:
        is_gap = np.zeros(len(aatype_int), dtype=bool)
        return aatype_int, atom_positions_raw, atom_mask_full, ridx, cidx, hetatm, auth_seq_id, ins_code, is_gap

    # Each position i contributes 1 (itself) + gaps[i] slots; last contributes 1.
    repeats = np.ones(len(aatype_int), dtype=int)
    repeats[:-1] += gaps

    new_aa   = np.repeat(aatype_int, repeats)
    new_pos  = np.repeat(atom_positions_raw, repeats, axis=0)
    new_mask = np.repeat(atom_mask_full, repeats, axis=0)
    new_ridx = np.repeat(ridx, repeats)
    new_cidx = np.repeat(cidx, repeats)
    new_het  = np.repeat(hetatm, repeats)
    new_auth = np.repeat(auth_seq_id, repeats) if auth_seq_id is not None else None
    new_ins  = np.repeat(ins_code, repeats) if ins_code is not None else None

    # Per-slot offset within each repeated block: 0 = original, 1,2,…= gap slots.
    block_starts = np.repeat(np.concatenate([[0], np.cumsum(repeats)[:-1]]), repeats)
    offsets = np.arange(len(new_aa)) - block_starts

    is_gap = offsets > 0
    new_aa[is_gap]   = 20   # X = unknown residue type
    new_pos[is_gap]  = 0.0
    new_mask[is_gap] = 0.0
    new_het[is_gap]  = 0
    new_ridx        += offsets   # ridx[i] + offset gives correct index for each gap slot
    if new_ins is not None:
        new_ins[is_gap] = ''
    return new_aa, new_pos, new_mask, new_ridx, new_cidx, new_het, new_auth, new_ins, is_gap

"""Added save_raw_coordinates=False parameter to get_parsed_inputs(). When True, returns atom_positions_raw (original global-frame coordinates) alongside the IPA-framed atom_positions. This is used in io.py to compute distances and sidechain centroids in real space
before the coordinate transform.
"""


def read_pdb_string(pdb_file):
    with open(pdb_file) as f:
        pdb_string = f.read()
    return pdb_string

def get_cif_parsed_inputs(cif_file, chain_id=None, round=True, dtype=None,
                          save_raw_coordinates=False, cb_mode="frame2seq",
                          backbone_method="ideal", trust_all_atoms=False,
                          modeller_verbose=True, write_repair_files=False, repair_sink=None):
    return get_parsed_inputs(cif_file, chain_id=chain_id, parser=protein.from_cif_string,
                             save_raw_coordinates=save_raw_coordinates, cb_mode=cb_mode,
                             backbone_method=backbone_method, trust_all_atoms=trust_all_atoms,
                             modeller_verbose=modeller_verbose,
                             write_repair_files=write_repair_files, repair_sink=repair_sink)

def get_parsed_inputs_biotite(pdb_file, chain_id=None, round=True, dtype=None,
                               save_raw_coordinates=False, cb_mode="frame2seq",
                               backbone_method="ideal", trust_all_atoms=False,
                               modeller_verbose=True, altloc='first',
                               write_repair_files=False, repair_sink=None):
    def _parser(pdb_string, c_id, fn):
        return protein.from_pdb_biotite(fn, chain_id=c_id, altloc=altloc)
    return get_parsed_inputs(pdb_file, chain_id=chain_id, parser=_parser,
                             save_raw_coordinates=save_raw_coordinates, cb_mode=cb_mode,
                             backbone_method=backbone_method, trust_all_atoms=trust_all_atoms,
                             modeller_verbose=modeller_verbose,
                             write_repair_files=write_repair_files, repair_sink=repair_sink)

def get_cif_parsed_inputs_biotite(cif_file, chain_id=None, round=True, dtype=None,
                                   save_raw_coordinates=False, cb_mode="frame2seq",
                                   backbone_method="ideal", trust_all_atoms=False,
                                   modeller_verbose=True, altloc='first',
                                   write_repair_files=False, repair_sink=None):
    def _parser(pdb_string, c_id, fn):
        return protein.from_cif_biotite(fn, chain_id=c_id, altloc=altloc)
    return get_parsed_inputs(cif_file, chain_id=chain_id, parser=_parser,
                             save_raw_coordinates=save_raw_coordinates, cb_mode=cb_mode,
                             backbone_method=backbone_method, trust_all_atoms=trust_all_atoms,
                             modeller_verbose=modeller_verbose,
                             write_repair_files=write_repair_files, repair_sink=repair_sink)

def _res_label(chain_index_arr, residue_index_arr, auth_seq_id_arr, ins_code_arr,
               chain_id_map, i):
    """Return a human-readable residue label for position i in the expanded arrays."""
    ci = int(chain_index_arr[i])
    ri = int(residue_index_arr[i])
    chain_name = chain_id_map[ci] if chain_id_map is not None else str(ci)
    label = f"chain {chain_name} res {ri}"
    if auth_seq_id_arr is not None:
        auth = int(auth_seq_id_arr[i])
        ins = str(ins_code_arr[i]).strip() if ins_code_arr is not None else ""
        label += f" (auth {auth}{ins})"
    return label


def _write_repair_log(log_entries, pdb_file, sink=None):
    """Write repair log entries to <pdb_file>.repair.log, or hand the text back through *sink*
    (a dict, key "log") so the caller can write it into the output directory instead."""
    text = f"# PDB repair log for {pdb_file}\n"
    if not log_entries:
        text += "# No modifications made.\n"
    else:
        text += "".join(entry + "\n" for entry in log_entries)
    if sink is not None:
        sink["log"] = text
        return
    log_path = str(pdb_file) + ".repair.log"
    try:
        with open(log_path, "w") as f:
            f.write(text)
    except OSError as e:
        logger.warning("Could not write repair log to %s: %s", log_path, e)


def _write_repair_map(map_rows, pdb_file, sink=None):
    """Write residue index mapping table to <pdb_file>.repair_map.tsv (or into *sink*["map"]).

    Columns:
      new_idx   — 0-based position in the output (repaired) array
      orig_idx  — 0-based position in the original PDB array, blank for inserted gaps
      chain     — chain letter / id
      res_idx   — sequential residue index (label_seq_id style)
      auth_seq  — original PDB residue number (if available)
      ins_code  — insertion code (blank if none)
      status    — original | gap_inserted | excluded_partial_bb | excluded_c_zero
    """
    text = "new_idx\torig_idx\tchain\tres_idx\tauth_seq\tins_code\tstatus\n"
    text += "".join("\t".join(str(v) for v in row) + "\n" for row in map_rows)
    if sink is not None:
        sink["map"] = text
        return
    map_path = str(pdb_file) + ".repair_map.tsv"
    try:
        with open(map_path, "w") as f:
            f.write(text)
    except OSError as e:
        logger.warning("Could not write repair map to %s: %s", map_path, e)


def get_parsed_inputs(pdb_file, chain_id=None, parser=protein.from_pdb_string, round=True,
                      dtype=None, save_raw_coordinates=False, cb_mode="frame2seq",
                      backbone_method="ideal", trust_all_atoms=False,
                      modeller_verbose=True, write_repair_files=False, repair_sink=None):
    pdb_string = read_pdb_string(pdb_file)
    try:
        prot_from_pdb = parser(pdb_string, chain_id, pdb_file)
    except AssertionError:
        raise

    aatype_int    = prot_from_pdb.aatype
    atom_positions_raw = prot_from_pdb.atom_positions
    start_residue = prot_from_pdb.residue_index[0]
    residue_index = prot_from_pdb.residue_index   # sequential per-chain (label_seq_id style)
    auth_seq_id_np = getattr(prot_from_pdb, 'auth_seq_id', None)  # original PDB res numbers
    ins_code_np    = getattr(prot_from_pdb, 'ins_code', None)    # insertion codes
    chain_index   = prot_from_pdb.chain_index
    hetatm_np     = prot_from_pdb.hetatm
    chain_id_map  = getattr(prot_from_pdb, 'chain_id_map', None)

    _repair_log = []   # accumulates modification records written at end of function
    _repair_map = []   # accumulates (new_idx, orig_idx, chain, res_idx, auth_seq, ins_code, status) rows
    _excluded_keys: set[tuple[int, int]] = set()   # (chain_index, residue_index) of excluded residues

    # Record residues with partial backbone (N or CA present but C missing) from the
    # ORIGINAL structure, keyed by (chain_index, internal residue_index). MODELLER will fill in
    # these atoms, but FrustratometeR still excludes such residues from output; we must
    # do the same regardless of which reconstruction method is used.
    _orig_atom_mask = prot_from_pdb.atom_mask   # [L, 37]
    _orig_res_index = prot_from_pdb.residue_index
    _orig_chain_idx = prot_from_pdb.chain_index
    _partial_bb_keys = set()
    _missing_bb_details: dict[tuple[int, int], str] = {}
    for _i in range(len(_orig_atom_mask)):
        _has_n  = _orig_atom_mask[_i, 0] > 0
        _has_ca = _orig_atom_mask[_i, 1] > 0
        _has_c  = _orig_atom_mask[_i, 2] > 0
        _key = (int(_orig_chain_idx[_i]), int(_orig_res_index[_i]))
        if not (_has_n and _has_ca and _has_c):
            _missing_bb_details[_key] = _describe_partial_backbone(prot_from_pdb, _i, Path(pdb_file))
        if (_has_n or _has_ca) and not _has_c:
            _partial_bb_keys.add(_key)

    # When backbone_method='modeller', treat the MODELLER-completed structure as
    # the authoritative source for ALL data (atom_positions, aatype, residue_index,
    # chain_index, hetatm).  MODELLER may change the residue count (adding missing
    # residues or dropping non-standard ones), so re-deriving all data from its output
    # keeps everything consistent.
    if backbone_method == 'modeller' and not trust_all_atoms and pdb_file is not None:
        try:
            result = _reconstruct_with_modeller(pdb_file, chain_id, verbose=modeller_verbose)
            if result is None:
                raise RuntimeError("MODELLER reconstruction returned no completed structure")
            if result is not None:
                completed_pos, completed_prot = result
                orig_res_set = set(zip(_orig_chain_idx.tolist(), _orig_res_index.tolist()))
                new_res_index = completed_prot.residue_index
                new_chain_idx = completed_prot.chain_index
                new_aatype    = completed_prot.aatype

                _repair_log.append(f"MODELLER: used for backbone reconstruction (method=modeller)")
                for _i in range(len(new_res_index)):
                    _key = (int(new_chain_idx[_i]), int(new_res_index[_i]))
                    if _key not in orig_res_set:
                        _repair_log.append(
                            f"MODELLER_ADDED: chain_index={_key[0]} residue_index={_key[1]}"
                        )
                for _key in orig_res_set:
                    new_res_set = set(zip(new_chain_idx.tolist(), new_res_index.tolist()))
                    if _key not in new_res_set:
                        _repair_log.append(
                            f"MODELLER_REMOVED: chain_index={_key[0]} residue_index={_key[1]}"
                        )

                aatype_int        = new_aatype
                atom_positions_raw = completed_pos
                residue_index     = new_res_index
                chain_index       = new_chain_idx
                hetatm            = torch.from_numpy(completed_prot.hetatm)
                seq_mask          = torch.from_numpy(aatype_int != 20).bool()
                # Re-exclude residues that had partial backbone in the original PDB.
                if _partial_bb_keys:
                    for _i in range(len(seq_mask)):
                        if (int(chain_index[_i]), int(residue_index[_i])) in _partial_bb_keys:
                            seq_mask[_i] = False
                            _key = (int(chain_index[_i]), int(residue_index[_i]))
                            logger.warning("%s; path=modeller; excluded after reconstruction",
                                           _missing_bb_details[_key])
                            _excluded_keys.add(_key)
                            _repair_log.append(
                                f"EXCLUDED_PARTIAL_BB: chain_index={_key[0]} residue_index={_key[1]}"
                                f" (C missing in original PDB; excluded after MODELLER reconstruction)"
                            )
                if dtype is None:
                    dtype = torch.float32
                atom_positions = atom_positions_raw.clone().to(dtype)

                pre_cb = atom_positions[:, 3, :].norm(dim=-1) < 1e-6
                atom_positions = place_missing_cb(atom_positions, mode=cb_mode)
                post_cb = atom_positions[:, 3, :].norm(dim=-1) >= 1e-6
                cb_placed = (pre_cb & post_cb).nonzero(as_tuple=True)[0].tolist()
                for _i in cb_placed:
                    _key = (int(chain_index[_i]), int(residue_index[_i]))
                    _repair_log.append(
                        f"CB_PLACED: chain_index={_key[0]} residue_index={_key[1]}"
                    )

                pre_o = atom_positions[:, 4, :].norm(dim=-1) < 1e-6
                atom_positions = place_missing_o(atom_positions, mode=cb_mode)
                post_o = atom_positions[:, 4, :].norm(dim=-1) >= 1e-6
                o_placed = (pre_o & post_o).nonzero(as_tuple=True)[0].tolist()
                for _i in o_placed:
                    _key = (int(chain_index[_i]), int(residue_index[_i]))
                    _repair_log.append(
                        f"O_PLACED: chain_index={_key[0]} residue_index={_key[1]}"
                    )

                _cid_map = getattr(completed_prot, 'chain_id_map', None) or chain_id_map
                _auth = getattr(completed_prot, 'auth_seq_id', None)
                if pdb_file is not None and write_repair_files:
                    _write_repair_log(_repair_log, pdb_file, repair_sink)
                if save_raw_coordinates:
                    return atom_positions, atom_positions_raw, aatype_int, seq_mask, residue_index, hetatm, chain_index, _cid_map, _auth, None
                return atom_positions, aatype_int, seq_mask, residue_index, hetatm, chain_index, _cid_map, _auth, None
        except Exception:
            raise

    if trust_all_atoms:
        # Completed PDB (e.g. from MODELLER) — trust every atom as-is.
        # Skip gap insertion and backbone reconstruction; include all residues
        # that have a known AA type.
        atom_mask_full = prot_from_pdb.atom_mask
        atom_positions_raw = torch.from_numpy(atom_positions_raw).double()
        hetatm = torch.from_numpy(hetatm_np)
        seq_mask = torch.from_numpy(aatype_int != 20).bool()
        # Exclude residues whose backbone C is at the origin — these were explicitly
        # zeroed by the partial_bb_idx fix (missing C in original PDB) to mark them
        # as invalid for downstream runs.  N at origin is NOT checked here: residues
        # where ideal geometry failed to reconstruct N write N=(0,0,0) to the completed
        # PDB, but FrustratometeR includes them (MODELLER places N successfully), so
        # excluding on N-zero would over-exclude relative to the reference.
        c_zero = atom_positions_raw[:, 2, :].norm(dim=-1) < 1e-6
        if c_zero.any():
            seq_mask = seq_mask.clone()
            c_zero_excluded = (c_zero & seq_mask).nonzero(as_tuple=True)[0].tolist()
            seq_mask[c_zero & seq_mask] = False
            for _i in c_zero_excluded:
                _excluded_keys.add((int(chain_index[_i]), int(residue_index[_i])))
                _repair_log.append(
                    f"EXCLUDED_C_ZERO: chain_index={int(chain_index[_i])} residue_index={int(residue_index[_i])}"
                    f" (C at origin in completed PDB — marked missing in prior run)"
                )
    else:
        # Insert placeholder rows for residues that are entirely absent from the PDB
        # (e.g. backbone too disordered to model; sidechain-only atoms removed by
        # preprocessing).  Gap is detected as a jump > 1 in residue_index within a chain.
        aatype_int, atom_positions_raw, atom_mask_full, residue_index, chain_index, hetatm_np, auth_seq_id_np, ins_code_np, _is_gap = \
            _insert_gap_residues(aatype_int, atom_positions_raw, prot_from_pdb.atom_mask,
                                 residue_index, chain_index, hetatm_np,
                                 auth_seq_id=auth_seq_id_np, ins_code=ins_code_np)

        gap_indices = _is_gap.nonzero()[0].tolist() if _is_gap.any() else []
        for _i in gap_indices:
            _repair_log.append(
                f"GAP_INSERTED: chain_index={int(chain_index[_i])} residue_index={int(residue_index[_i])}"
                + (f" (auth {int(auth_seq_id_np[_i])})" if auth_seq_id_np is not None else "")
            )

        atom_positions_raw = torch.from_numpy(atom_positions_raw).double()
        atom_mask = torch.from_numpy(atom_mask_full[:, :3]).bool()
        hetatm = torch.from_numpy(hetatm_np)

        seq_mask = torch.zeros(len(aatype_int), dtype=torch.long)
        seq_mask[aatype_int != 20] = 1
        seq_mask = seq_mask.bool()

        # First pass: mask residues with any missing backbone atom (N, CA, C).
        missing_bb_idx = [
            i for i in range(len(atom_mask))
            if atom_mask[i][0] == 0 or atom_mask[i][1] == 0 or atom_mask[i][2] == 0
        ]
        seq_mask[missing_bb_idx] = 0
        seq_mask = seq_mask.bool()

        # Residues with backbone C specifically missing (but N or CA present) must stay
        # excluded even after reconstruction.  FrustratometeR excludes these; reconstruction
        # can provide coordinates for neighbours but the residue itself should not appear.
        # This is narrower than missing_bb_idx — residues missing only N or CA are allowed
        # to be reconstructed and included.  Gap placeholders (aatype=20) are fully absent
        # and may be included once MODELLER fills them in.
        partial_bb_idx = [
            i for i in missing_bb_idx
            if aatype_int[i] != 20
            and atom_mask[i][2] == 0  # backbone C specifically missing
            and (atom_mask[i][0] == 1 or atom_mask[i][1] == 1)  # but has N or CA
        ]

        # Second pass: reconstruct missing backbone atoms from ideal geometry so
        # gap residues (aatype=20) can be filled and included.
        seq_mask_before_recon = seq_mask.clone()
        if missing_bb_idx:
            atom_positions_raw, seq_mask = reconstruct_missing_backbone(
                atom_positions_raw, atom_mask, seq_mask, pdb_file=pdb_file,
                backbone_method=backbone_method, chain_id=chain_id,
                modeller_verbose=modeller_verbose,
            )
            forced_exclusions = set(partial_bb_idx)
            for _i in missing_bb_idx:
                if aatype_int[_i] != 20 and not seq_mask[_i] and _i not in forced_exclusions:
                    _key = (int(chain_index[_i]), int(residue_index[_i]))
                    logger.warning(
                        "%s; path=normal; method=%s; backbone reconstruction incomplete; "
                        "excluded from calculations",
                        _missing_bb_details[_key], backbone_method,
                    )
                elif aatype_int[_i] != 20 and seq_mask[_i] and not seq_mask_before_recon[_i]:
                    # Backbone was successfully reconstructed for this residue
                    n_ok  = bool(atom_mask[_i][0])
                    ca_ok = bool(atom_mask[_i][1])
                    c_ok  = bool(atom_mask[_i][2])
                    missing_atoms = [a for a, ok in zip(("N", "CA", "C"), (n_ok, ca_ok, c_ok)) if not ok]
                    _repair_log.append(
                        f"BACKBONE_RECONSTRUCTED: chain_index={int(chain_index[_i])} residue_index={int(residue_index[_i])}"
                        + (f" (auth {int(auth_seq_id_np[_i])})" if auth_seq_id_np is not None else "")
                        + f" missing={','.join(missing_atoms)} method={backbone_method}"
                    )

        # Keep partial-backbone (C-missing) residues excluded regardless of reconstruction.
        # Also zero out their reconstructed C in atom_positions_raw so the completed PDB
        # records them as missing — the trust_all_atoms zero-check will then exclude them
        # consistently on a second run.
        if partial_bb_idx:
            seq_mask[partial_bb_idx] = False
            atom_positions_raw[partial_bb_idx, 2, :] = 0.0
            for _i in partial_bb_idx:
                _key = (int(chain_index[_i]), int(residue_index[_i]))
                logger.warning("%s; path=normal; C coordinates zeroed",
                               _missing_bb_details[_key])
                _excluded_keys.add(_key)
                _repair_log.append(
                    f"EXCLUDED_PARTIAL_BB: chain_index={_key[0]} residue_index={_key[1]}"
                    + (f" (auth {int(auth_seq_id_np[_i])})" if auth_seq_id_np is not None else "")
                    + " (C missing in original PDB; C zeroed, residue excluded)"
                )

        # Gap placeholders (aatype=20) must stay excluded even if their backbone was
        # reconstructed. Reconstruction gives them useful coords for adjacent atoms but
        # they should not appear in computation or output.
        seq_mask[aatype_int == 20] = False

    if dtype is None:
        dtype = torch.float32

    atom_positions = atom_positions_raw.clone().to(dtype)

    pre_cb = atom_positions[:, 3, :].norm(dim=-1) < 1e-6
    atom_positions = place_missing_cb(atom_positions, mode=cb_mode)
    post_cb = atom_positions[:, 3, :].norm(dim=-1) >= 1e-6
    cb_placed = (pre_cb & post_cb).nonzero(as_tuple=True)[0].tolist()
    for _i in cb_placed:
        _repair_log.append(
            f"CB_PLACED: chain_index={int(chain_index[_i])} residue_index={int(residue_index[_i])}"
            + (f" (auth {int(auth_seq_id_np[_i])})" if auth_seq_id_np is not None else "")
        )

    pre_o = atom_positions[:, 4, :].norm(dim=-1) < 1e-6
    atom_positions = place_missing_o(atom_positions, mode=cb_mode)
    post_o = atom_positions[:, 4, :].norm(dim=-1) >= 1e-6
    o_placed = (pre_o & post_o).nonzero(as_tuple=True)[0].tolist()
    for _i in o_placed:
        _repair_log.append(
            f"O_PLACED: chain_index={int(chain_index[_i])} residue_index={int(residue_index[_i])}"
            + (f" (auth {int(auth_seq_id_np[_i])})" if auth_seq_id_np is not None else "")
        )

    if round:
        atom_positions = torch.from_numpy(atom_positions.numpy().round(4))

    if pdb_file is not None and write_repair_files:
        # Build the residue index mapping table.
        # _is_gap exists only in the normal (non-trust_all_atoms, non-modeller) path.
        try:
            _is_gap_arr = _is_gap  # noqa: F821 — set in else branch above
        except NameError:
            _is_gap_arr = None  # trust_all_atoms or modeller path — no gaps inserted

        _n_out = len(chain_index)
        _orig_counter = 0
        for _ni in range(_n_out):
            _ci = int(chain_index[_ni])
            _ri = int(residue_index[_ni])
            _chain_name = chain_id_map[_ci] if chain_id_map is not None else str(_ci)
            _auth = int(auth_seq_id_np[_ni]) if auth_seq_id_np is not None else ""
            _ins = str(ins_code_np[_ni]).strip() if ins_code_np is not None else ""

            _is_gap_slot = bool(_is_gap_arr[_ni]) if _is_gap_arr is not None else False
            if _is_gap_slot:
                _orig_idx_str = ""
                _status = "gap_inserted"
            else:
                _orig_idx_str = str(_orig_counter)
                _orig_counter += 1
                if (_ci, _ri) in _excluded_keys:
                    _status = "excluded"
                else:
                    _status = "original"

            _repair_map.append((_ni, _orig_idx_str, _chain_name, _ri, _auth, _ins, _status))

        _write_repair_log(_repair_log, pdb_file, repair_sink)
        _write_repair_map(_repair_map, pdb_file, repair_sink)

    if save_raw_coordinates:
        return atom_positions, atom_positions_raw, aatype_int, seq_mask, residue_index, hetatm, chain_index, chain_id_map, auth_seq_id_np, ins_code_np

    return atom_positions, aatype_int, seq_mask, residue_index, hetatm, chain_index, chain_id_map, auth_seq_id_np, ins_code_np
