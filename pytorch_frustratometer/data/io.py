"""
Multi-chain aware IO functions.

Key differences from io.py:
- `build_residue_lookup` builds dict[chain_letter → LongTensor] instead of a flat tensor,
  so chains with overlapping PDB residue numbers are handled correctly.
- `parse_singleresidue_frustration` uses the ChainRes column for chain-aware mapping.
- `parse_pair_frustration` uses ChainRes1/ChainRes2 for chain-aware mapping.

Falls back to the flat-tensor behaviour from io.py when residue_lookup is not a dict
(single-chain or legacy callers).
"""

from pathlib import Path

import numpy as np
import torch
import pandas as pd

from ..utils.utils import check_size
from ..structure.protein import PDB_CHAIN_IDS


def _make_chain_remap(file_chain_ids, residue_lookup):
    """
    Map original chain letters from a frustration file to residue_lookup keys.

    ``build_residue_lookup`` (feature_calculator.py) remaps actual chain letters
    to 0-indexed alphabet keys: the alphabetically k-th unique chain → key "A"+k.
    Frustration files use the *original* chain letters.

    If the file chains already match the lookup keys (e.g. 4Z66 with A-H), the
    identity mapping is returned.  Otherwise, both sets are sorted alphabetically
    and paired positionally.

    Returns
    -------
    dict or None
        {file_chain_letter: lookup_key} or None when no remapping is possible.
    """
    file_chains_sorted = sorted(set(str(c) for c in file_chain_ids))
    lookup_keys_sorted = sorted(residue_lookup.keys())

    if set(file_chains_sorted) == set(lookup_keys_sorted):
        return None  # keys already match — no remapping needed

    if len(file_chains_sorted) != len(lookup_keys_sorted):
        # Different chain counts — build a partial remap for those that can be matched
        remap = {}
        for fc, lk in zip(file_chains_sorted, lookup_keys_sorted):
            remap[fc] = lk
        return remap

    # Positional match: sort both alphabetically and pair them
    return {fc: lk for fc, lk in zip(file_chains_sorted, lookup_keys_sorted)}


# ---------------------------------------------------------------------------
# Residue lookup builder
# ---------------------------------------------------------------------------

def build_residue_lookup(residue_index, chain_index, chain_id_map=None, ins_code=None):
    """
    Build a per-chain residue lookup from parsed PDB data.

    Args:
        residue_index: [L] LongTensor of PDB residue numbers (may be non-contiguous)
        chain_index:   [L] int tensor (or numpy array) of 0-indexed chain integers
        chain_id_map:  optional dict mapping chain int → chain letter
        ins_code:      optional [L] array of insertion codes (str); '' for normal residues

    Returns:
        dict[str → dict[str → int]]: maps chain_letter → {res_label → sequential_index}
        where res_label is str(res_num) for normal residues, str(res_num)+ins for insertion-code ones.

    Example:
        residue_lookup["A"]["42"] == 5   means PDB residue 42 on chain A is index 5
        residue_lookup["A"]["40A"] == 3  means insertion-code residue 40A on chain A is index 3
    """
    import numpy as _np
    if not isinstance(chain_index, torch.Tensor):
        chain_index = torch.from_numpy(chain_index).long()
    if not isinstance(residue_index, torch.Tensor):
        residue_index = torch.from_numpy(residue_index).long()

    ri_np  = residue_index.cpu().numpy()
    ci_np  = chain_index.cpu().numpy()
    ins_np = ins_code if ins_code is not None else _np.full(len(ri_np), '', dtype='U4')
    all_labels = _np.char.add(ri_np.astype('U10'), _np.char.strip(ins_np))

    residue_lookup = {}
    for c_int in chain_index.unique().tolist():
        c_letter = chain_id_map[c_int] if chain_id_map is not None else PDB_CHAIN_IDS[c_int]
        mask = ci_np == c_int
        positions = _np.where(mask)[0]
        residue_lookup[c_letter] = dict(zip(all_labels[mask].tolist(), positions.tolist()))

    return residue_lookup


def _lut_valid_pos(lut):
    """Return sorted valid tensor positions from a lut (LongTensor or dict[str→int])."""
    if isinstance(lut, torch.Tensor):
        return lut[lut >= 0].sort().values
    return torch.tensor(sorted(lut.values()), dtype=torch.long)


def _lut_lookup_labels(lut, res_labels):
    """Look up tensor positions by string labels. Returns LongTensor (−1 for missing)."""
    if isinstance(lut, torch.Tensor):
        idx = torch.tensor([int(l) for l in res_labels], dtype=torch.long)
        in_b = (idx >= 0) & (idx < len(lut))
        out = torch.full((len(idx),), -1, dtype=torch.long)
        out[in_b] = lut[idx[in_b]]
        return out
    return torch.tensor([lut.get(l, -1) for l in res_labels], dtype=torch.long)


# ---------------------------------------------------------------------------
# Single-residue frustration
# ---------------------------------------------------------------------------

def parse_singleresidue_frustration(
    frust_file,
    residue_index=None,
    residue_lookup=None,    # dict[chain → dict[str→int]]  (multi-chain, new)
                            # or dict[chain → LongTensor]   (multi-chain, legacy)
                            # or dict[str→int]              (single-chain, new)
                            # or LongTensor [offset, data…] (single-chain, legacy)
                            # or None
    seq_mask=None,          # [L] bool — when provided, use positional alignment (single-chain)
    max_len=500,
    with_prediction=False,
    include_extra_fields=False,
):
    """
    Parse a singleresidue frustration file with optional multi-chain mapping.

    Args:
        frust_file:      path to *.pdb_singleresidue file
        residue_index:   [L] LongTensor of PDB residue numbers; used to determine L
        residue_lookup:  dict[chain → dict[str→int]] (multi-chain, new)
                         or dict[chain → LongTensor]  (multi-chain, legacy)
                         or dict[str→int]             (single-chain, new)
                         or LongTensor [offset, data…](single-chain, legacy)
                         or None (assume sequential indices)
        seq_mask:        [L] bool tensor — when provided, enables positional alignment
                         for the single-chain path
        max_len:         optional length cap
        with_prediction: if True prefer PredFrstIndex over FrstIndex
        include_extra_fields: if True return extra columns dict as well

    Returns:
        (out_values, out_states, out_mask[, extra]) when FrstState present
        (out_values, out_mask[, extra])              otherwise
    """
    # frust_file may be a path/buffer (the usual case) OR an already-loaded
    # DataFrame (2026-10-02, for parse_mutational_scanning_singleresidue --
    # avoids a pointless to_csv/re-parse round trip when the caller already
    # has the rows in memory, e.g. after a pivot).
    df = frust_file if isinstance(frust_file, pd.DataFrame) else pd.read_csv(frust_file, sep=" ", comment="#")

    # -------------------------------------------------------------------------
    # 1. Normalize column names
    # -------------------------------------------------------------------------
    if "FrstIndex" in df.columns:
        pass
    elif with_prediction and "PredFrstIndex" in df.columns:
        df = df.rename(columns={
            "PredFrstIndex": "FrstIndex",
            "PredFrstState": "FrstState",
        })
    elif (not with_prediction) and "TrueFrstIndex" in df.columns:
        df = df.rename(columns={
            "TrueFrstIndex": "FrstIndex",
            "TrueFrstState": "FrstState",
        })
    else:
        raise RuntimeError("Unable to parse single residue frustration file: no FrstIndex column")

    # -------------------------------------------------------------------------
    # 2. Convert values / states to tensors
    # -------------------------------------------------------------------------
    state_map = {"highly": 0, "neutral": 1, "minimally": 2}
    values = torch.from_numpy(df["FrstIndex"].values.copy()).float()

    if "FrstState" in df.columns:
        states = torch.from_numpy(df["FrstState"].map(state_map).values.copy()).long()
    else:
        states = None

    # -------------------------------------------------------------------------
    # 3a. Multi-chain dict path
    # -------------------------------------------------------------------------
    _fv = next(iter(residue_lookup.values()), None) if isinstance(residue_lookup, dict) else None
    _is_multichain = isinstance(residue_lookup, dict) and isinstance(_fv, (dict, torch.Tensor))
    _is_sc_dict    = isinstance(residue_lookup, dict) and not isinstance(_fv, (dict, torch.Tensor))

    if _is_multichain:
        # Determine total sequence length from the lookup tables
        max_seq_idx = max(
            _lut_valid_pos(lut).max().item()
            for lut in residue_lookup.values()
            if len(_lut_valid_pos(lut)) > 0
        )
        n_residues = max_seq_idx + 1

        out_values = torch.full((n_residues,), float("nan"), dtype=torch.float)
        out_mask   = torch.zeros(n_residues, dtype=torch.bool)
        out_states = torch.full((n_residues,), -1, dtype=torch.long) if states is not None else None

        extra_numeric_cols = [c for c in df.columns
                              if c not in ("FrstIndex", "FrstState", "Res", "ChainRes")
                              and pd.api.types.is_numeric_dtype(df[c])]
        extra_out = {col: torch.full((n_residues,), float("nan")) for col in extra_numeric_cols} \
                    if include_extra_fields else {}

        # If ChainRes column is absent (single-chain file), assign everything to first chain
        if "ChainRes" not in df.columns:
            first_chain = next(iter(residue_lookup))
            df = df.copy()
            df["ChainRes"] = first_chain

        # build_residue_lookup remaps original chain letters to A,B,C... alphabetically.
        # The frustration file uses the original letters.  Build a remap if needed.
        chain_remap = _make_chain_remap(df["ChainRes"].unique(), residue_lookup)

        for chain_letter, chain_df in df.groupby("ChainRes"):
            cl = str(chain_letter)  # pandas may read numeric chain IDs (e.g. "3") as int
            lookup_key = chain_remap[cl] if (chain_remap and cl in chain_remap) else cl
            if lookup_key not in residue_lookup:
                continue
            lut = residue_lookup[lookup_key]

            # Valid tensor positions for this chain, in ascending order.
            # If seq_mask is provided, restrict to positions where seq_mask=True so that
            # excluded residues (e.g. missing backbone C) are not counted as valid.
            all_mapped = _lut_valid_pos(lut)
            if seq_mask is not None:
                valid_pos = all_mapped[seq_mask[all_mapped]]
            else:
                valid_pos = all_mapped

            chain_df_sorted = chain_df.sort_values("Res")
            frust_vals = torch.from_numpy(chain_df_sorted["FrstIndex"].values.copy()).float()

            if len(valid_pos) == len(chain_df_sorted):
                # Positional alignment: reference file may renumber residues sequentially
                # after excluding broken residues (e.g. FrustratometeR drops residues with
                # missing backbone and renumbers the rest 1-N). Match row k → k-th valid
                # tensor position rather than trying to look up by residue number.
                new_idx = valid_pos
                mapped  = torch.ones(len(new_idx), dtype=torch.bool)
            else:
                # Count mismatch — fall back to label-based lookup.
                res_labels = chain_df_sorted["Res"].astype(str).tolist()
                new_idx    = _lut_lookup_labels(lut, res_labels)
                mapped     = new_idx >= 0

            out_values[new_idx[mapped]] = frust_vals[mapped]
            out_mask[new_idx[mapped]]   = True

            if out_states is not None:
                chain_states = torch.from_numpy(
                    chain_df_sorted["FrstState"].map(state_map).values.copy()
                ).long()
                out_states[new_idx[mapped]] = chain_states[mapped]

            for col in extra_out:
                col_vals_np = chain_df_sorted[col].values.astype(float)
                col_vals = torch.from_numpy(col_vals_np)
                extra_out[col][new_idx[mapped]] = col_vals[mapped].float()

        out_values = torch.nan_to_num(out_values, nan=0.0)
        extra = {col: torch.nan_to_num(t, nan=0.0) for col, t in extra_out.items()} \
                if include_extra_fields else None

        if max_len is not None:
            out_values = out_values[:max_len]
            out_mask   = out_mask[:max_len]
            if out_states is not None:
                out_states = out_states[:max_len]
            if extra is not None:
                extra = {col: t[:max_len] for col, t in extra.items()}

        if include_extra_fields:
            return out_values, out_states, out_mask, extra  # out_states may be None

        if out_states is not None:
            return out_values, out_states, out_mask
        return out_values, out_mask

    # -------------------------------------------------------------------------
    # 3b. Single-chain dict path (new format: dict[str→int])
    # -------------------------------------------------------------------------
    elif _is_sc_dict:
        # -- Positional alignment when seq_mask is provided -------------------
        if seq_mask is not None:
            valid_positions = seq_mask.bool().nonzero(as_tuple=False).squeeze(1)
            n_valid = len(valid_positions)
            n_ref   = len(df)

            if n_valid == n_ref:
                mapped_idx  = valid_positions
                values      = values[:n_ref]
                if states is not None:
                    states  = states[:n_ref]
                df_row_bool = None
            else:
                seq_mask = None  # fall back to dict lookup below

        if seq_mask is None:
            res_labels  = df["Res"].astype(str).tolist()
            mapped_idx  = torch.tensor([residue_lookup.get(l, -1) for l in res_labels], dtype=torch.long)
            df_row_bool = (mapped_idx >= 0).numpy()
            mapped_idx  = mapped_idx[mapped_idx >= 0]
            values      = values[df_row_bool]
            if states is not None:
                states = states[df_row_bool]

        # -- Allocate in tensor space -----------------------------------------
        if residue_index is not None:
            L = len(residue_index)
        else:
            all_pos = list(residue_lookup.values())
            L = max(all_pos) + 1 if all_pos else 0

        out_values = torch.full((L,), -1.0)
        out_mask   = torch.zeros(L, dtype=torch.bool)

        out_values[mapped_idx] = values
        out_mask[mapped_idx]   = True

        if states is not None:
            out_states = torch.full((L,), -1, dtype=torch.long)
            out_states[mapped_idx] = states
        else:
            out_states = None

        extra = None
        if include_extra_fields:
            extra = {}
            for col in df.columns:
                if col in ["FrstIndex", "FrstState", "Res"]:
                    continue
                col_arr = df[col].to_numpy()
                if df_row_bool is not None:
                    val = col_arr[df_row_bool]
                else:
                    val = col_arr[:len(mapped_idx)]
                try:
                    extra[col] = torch.from_numpy(val.copy())
                except TypeError:
                    extra[col] = val

        if max_len is not None:
            out_values = out_values[:max_len]
            out_mask   = out_mask[:max_len]
            if out_states is not None:
                out_states = out_states[:max_len]

        if include_extra_fields:
            return out_values, out_states, out_mask, extra

        if out_states is not None:
            return out_values, out_states, out_mask
        return out_values, out_mask

    # -------------------------------------------------------------------------
    # 3c. Single-chain flat-tensor path (legacy offset format)
    # -------------------------------------------------------------------------
    elif residue_lookup is not None:
        # residue_lookup[0] is the pdb_offset; actual mapping data starts at residue_lookup[1].
        pdb_offset    = int(residue_lookup[0])
        pdb_to_tensor = residue_lookup[1:]

        # -- Positional alignment when seq_mask is provided -------------------
        if seq_mask is not None:
            valid_positions = seq_mask.bool().nonzero(as_tuple=False).squeeze(1)  # [N_valid]
            n_valid = len(valid_positions)
            n_ref   = len(df)

            if n_valid == n_ref:
                # Positional path: row k → k-th valid tensor position
                n_rows      = n_ref
                mapped_idx  = valid_positions[:n_rows]
                values      = values[:n_rows]
                if states is not None:
                    states  = states[:n_rows]
                df_row_bool = None
            else:
                # Count mismatch — fall back to PDB-number lookup so one extra
                # residue on either side doesn't cascade into a full misalignment.
                seq_mask = None   # re-enter the legacy branch below

        if seq_mask is None:
            # Legacy path: treat Res column as PDB residue numbers directly.
            pdb_res         = torch.from_numpy(df["Res"].values.copy()).long()
            pdb_res_shifted = pdb_res + pdb_offset
            in_bounds       = (pdb_res_shifted >= 0) & (pdb_res_shifted < len(pdb_to_tensor))
            mapped_idx      = pdb_to_tensor[pdb_res_shifted.clamp(min=0, max=len(pdb_to_tensor) - 1)]
            mapped_idx[~in_bounds] = -1

            df_row_bool = (mapped_idx >= 0).numpy()
            mapped_idx  = mapped_idx[mapped_idx >= 0]
            values      = values[df_row_bool]
            if states is not None:
                states = states[df_row_bool]

        # -- Allocate in tensor space -----------------------------------------
        if residue_index is not None:
            L = len(residue_index)
        else:
            valid_targets = pdb_to_tensor[pdb_to_tensor >= 0]
            L = int(valid_targets.max().item()) + 1 if len(valid_targets) > 0 else 0

        out_values = torch.full((L,), -1.0)
        out_mask   = torch.zeros(L, dtype=torch.bool)

        out_values[mapped_idx] = values
        out_mask[mapped_idx]   = True

        if states is not None:
            out_states = torch.full((L,), -1, dtype=torch.long)
            out_states[mapped_idx] = states
        else:
            out_states = None

        # -- Extra fields (SAFE version) --------------------------------------
        extra = None
        if include_extra_fields:
            extra = {}
            for col in df.columns:
                if col in ["FrstIndex", "FrstState", "Res"]:
                    continue
                col_arr = df[col].to_numpy()
                if df_row_bool is not None:
                    val = col_arr[df_row_bool]
                else:
                    # positional path: use first n_rows
                    val = col_arr[:len(mapped_idx)]
                try:
                    extra[col] = torch.from_numpy(val.copy())
                except TypeError:
                    extra[col] = val

        # -- Truncate ---------------------------------------------------------
        if max_len is not None:
            out_values = out_values[:max_len]
            out_mask   = out_mask[:max_len]
            if out_states is not None:
                out_states = out_states[:max_len]

        if include_extra_fields:
            return out_values, out_states, out_mask, extra  # out_states may be None

        if out_states is not None:
            return out_values, out_states, out_mask
        return out_values, out_mask

    # -------------------------------------------------------------------------
    # 3d. No lookup: assume sequential indices
    # -------------------------------------------------------------------------
    else:
        out_values = values
        out_mask   = torch.ones(len(out_values), dtype=torch.bool)
        out_states = states

        if max_len is not None:
            out_values = out_values[:max_len]
            out_mask   = out_mask[:max_len]
            if out_states is not None:
                out_states = out_states[:max_len]

        if include_extra_fields:
            extra_cols = [c for c in df.columns if c not in ("FrstIndex", "FrstState")]
            extra = {col: df[col].values for col in extra_cols}
            return out_values, out_states, out_mask, extra  # out_states may be None

        if out_states is not None:
            return out_values, out_states, out_mask
        return out_values, out_mask


def parse_smooth_from_single(
    frust_file,
    edge_index=None,
    max_len=500,
    residue_index=None,
    residue_lookup=None,
    seq_mask=None,
    pair_file=None,
    use_welltype=False,
    return_mapping=False,
):
    """
    Derive smooth pair frustration from a singleresidue file.

    Smooth FI for a contact pair (i, j) is defined as:
        F_smooth(i, j) = -(FI_i + FI_j)

    Contacts can be supplied in three ways (in priority order):
      1. ``edge_index`` — [2, E] LongTensor in sequential coordinates (pre-built).
      2. ``pair_file``  — path to a mutational or configurational output file; the
                          (Res1, ChainRes1, Res2, ChainRes2) columns are used to
                          reconstruct edge_index via residue_lookup.
      3. Neither        — raises ValueError.

    Args:
        frust_file:     path to *.pdb_singleresidue file, or a pre-loaded [L]
                        FloatTensor of per-residue FI values (skips re-parsing).
        edge_index:     [2, E] LongTensor of contact pairs in sequential coords.
        max_len:        optional length cap (passed to parse_singleresidue_frustration).
        residue_index:  [L] LongTensor of PDB residue numbers.
        residue_lookup: same formats as parse_singleresidue_frustration / parse_pair_frustration.
        seq_mask:       [L] bool tensor for positional alignment.
        pair_file:      path to *.pdb_mutational or *.pdb_configurational file used
                        to derive the contact list when edge_index is not provided.
        use_welltype:   ignored — smooth frustration has no welltype; accepted for
                        API compatibility with parse_pair_frustration.
        return_mapping: ignored — accepted for API compatibility.

    Returns:
        smooth_values: [E] FloatTensor — smooth FI per edge (0 where either residue lacks data)
        edge_mask:     [E] BoolTensor  — True where both residues had valid FI
    """
    # ── 1. Parse per-residue FI ───────────────────────────────────────────────
    if isinstance(frust_file, torch.Tensor):
        sr_values = frust_file.float()
        sr_mask   = torch.ones(len(sr_values), dtype=torch.bool)
    else:
        sr_result = parse_singleresidue_frustration(
            frust_file,
            residue_index=residue_index,
            residue_lookup=residue_lookup,
            seq_mask=seq_mask,
            max_len=max_len,
        )
        # returns (values, states, mask) or (values, mask)
        sr_values = sr_result[0]
        sr_mask   = sr_result[-1]

    # ── 2. Build edge_index from pair_file if not supplied ────────────────────
    if edge_index is None:
        if pair_file is None:
            raise ValueError(
                "parse_smooth_frustration: supply either edge_index or pair_file "
                "to define which residue pairs to evaluate."
            )
        pair_df = pd.read_csv(str(pair_file), sep=" ", comment="#")
        # Resolve each (Res, Chain) label → sequential position via residue_lookup
        def _resolve(res_col, chain_col):
            positions = []
            for _, row in pair_df.iterrows():
                label = str(int(row[res_col]))
                chain = str(row[chain_col])
                pos   = -1
                if residue_lookup is not None:
                    _fv = next(iter(residue_lookup.values()), None) if isinstance(residue_lookup, dict) else None
                    _is_mc = isinstance(residue_lookup, dict) and isinstance(_fv, (dict, torch.Tensor))
                    if _is_mc:
                        lut = residue_lookup.get(chain)
                        if lut is not None:
                            pos = int(_lut_lookup_labels(lut, [label])[0])
                    else:
                        pos = int(_lut_lookup_labels(residue_lookup, [label])[0])
                elif residue_index is not None:
                    matches = (residue_index == int(label)).nonzero(as_tuple=True)[0]
                    if len(matches):
                        pos = int(matches[0])
                positions.append(pos)
            return positions

        src_pos = _resolve("Res1", "ChainRes1")
        dst_pos = _resolve("Res2", "ChainRes2")
        edge_index = torch.tensor([src_pos, dst_pos], dtype=torch.long)

    # ── 3. Compute smooth FI = -(FI_i + FI_j) ────────────────────────────────
    E   = edge_index.shape[1]
    src = edge_index[0]
    dst = edge_index[1]
    L   = len(sr_values)

    src_ok = (src >= 0) & (src < L) & sr_mask[src.clamp(0, L - 1)]
    dst_ok = (dst >= 0) & (dst < L) & sr_mask[dst.clamp(0, L - 1)]
    edge_mask = src_ok & dst_ok

    smooth_values = torch.zeros(E, dtype=torch.float)
    valid = edge_mask.nonzero(as_tuple=True)[0]
    smooth_values[valid] = -(sr_values[src[valid]] + sr_values[dst[valid]])

    return smooth_values, edge_mask


AA_LIST_20 = list("ACDEFGHIKLMNPQRSTVWY")


def parse_mutational_scanning_singleresidue(
    frust_file,
    residue_index=None,
    residue_lookup=None,
    seq_mask=None,
    max_len=500,
):
    """
    Parse a `--save-mutations` singleresidue scan file (columns: Res,
    ChainRes, AA_native, AA_mutant, NativeEnergy, MutantEnergy, FrstIndex --
    one row per (residue, candidate AA) pair, 20 rows per residue instead of
    `parse_singleresidue_frustration`'s one) into a dense
    [n_residues, 20] FI tensor + mask, one column per candidate amino acid
    (alphabetical: ACDEFGHIKLMNPQRSTVWY).

    Deliberately a SEPARATE function, not a modification of
    parse_singleresidue_frustration (2026-10-02) -- but reuses it UNMODIFIED
    for the actual alignment, exactly ONCE per domain: pivots the 20
    AA_mutant rows per residue into 20 columns (one pandas reshape, in
    memory -- parse_singleresidue_frustration now accepts an already-loaded
    DataFrame directly, no to_csv/re-parse round trip), then calls it with
    include_extra_fields=True, which already aligns an arbitrary number of
    extra numeric columns in the SAME pass as the primary FrstIndex column.
    This avoids both duplicating that function's intricate chain/positional
    alignment logic AND the cost of a separate alignment pass per candidate
    AA -- the per-candidate-AA columns land on exactly the same tensor
    positions parse_singleresidue_frustration would put the native-identity
    FI on, since alignment only depends on Res/ChainRes, identical across
    all 20 AA_mutant rows for a given residue.

    Returns:
        out_values: [n_residues, 20] float, 0.0 where no data for that AA
        out_mask:   [n_residues, 20] bool, False where no data for that AA
    """
    df = pd.read_csv(frust_file, sep=r"\s+", comment="#")
    required = {"Res", "AA_native", "AA_mutant", "FrstIndex"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Mutation-scan file missing columns {missing}: {frust_file}")

    index_cols = ["Res"] + (["ChainRes"] if "ChainRes" in df.columns else [])
    pivot = df.pivot_table(index=index_cols, columns="AA_mutant", values="FrstIndex").reset_index()
    pivot.columns.name = None

    present_aas = [aa for aa in AA_LIST_20 if aa in pivot.columns]
    if not present_aas:
        raise RuntimeError(f"No recognized AA_mutant values found in {frust_file}")
    primary_aa, extra_aas = present_aas[0], present_aas[1:]
    pivot = pivot.rename(columns={primary_aa: "FrstIndex"})

    result = parse_singleresidue_frustration(
        pivot, residue_index=residue_index, residue_lookup=residue_lookup,
        seq_mask=seq_mask, max_len=max_len, include_extra_fields=True)
    values, mask, extra = result[0], result[-2], result[-1]

    n = len(values)
    out_values = torch.zeros(n, len(AA_LIST_20))
    out_mask = torch.zeros(n, len(AA_LIST_20), dtype=torch.bool)
    out_values[:, AA_LIST_20.index(primary_aa)] = values
    out_mask[:, AA_LIST_20.index(primary_aa)] = mask
    for aa in extra_aas:
        aa_vals = extra[aa]
        # The "no lookup" path returns raw numpy for extra fields (unlike the
        # multi-chain/single-chain paths, which return tensors) -- normalize.
        if not torch.is_tensor(aa_vals):
            aa_vals = torch.as_tensor(aa_vals, dtype=torch.float32)
        out_values[:, AA_LIST_20.index(aa)] = torch.nan_to_num(aa_vals.float(), nan=0.0)
        out_mask[:, AA_LIST_20.index(aa)] = mask  # same rows -> same validity as the primary column

    return out_values, out_mask


# ---------------------------------------------------------------------------
# Pair frustration
# ---------------------------------------------------------------------------

def parse_pair_frustration(
    frust_file,
    edge_index,
    max_len=500,
    residue_index=None,
    residue_lookup=None,    # dict[chain_letter → LongTensor]  (multi-chain)
                            # or [pdb_offset, pdb_to_tensor...] (single-chain flat)
                            # or None
    seq_mask=None,          # [L] bool — when provided, enables positional alignment
    use_welltype=False,
    return_mapping=False,
    with_prediction=False,
    include_energy_fields=False,
):
    """
    Parse a pair frustration file (mutational or configurational) with optional
    multi-chain mapping.

    Args:
        frust_file:      path to *.pdb_mutational / *.pdb_configurational file
        edge_index:      [2, E] LongTensor of contact pairs in sequential coords
                         (may be None — returns full matrix)
        max_len:         optional length cap
        residue_index:   [L] LongTensor of PDB residue numbers
        residue_lookup:  dict[chain → dict[str→int]] (multi-chain, new)
                         or dict[chain → LongTensor]  (multi-chain, legacy)
                         or dict[str→int]             (single-chain, new)
                         or LongTensor [offset, data…](single-chain, legacy)
                         or None
        seq_mask:        [L] bool — when provided, enables positional/sequential detection
        use_welltype:    if True also return welltype and state tensors
        return_mapping:  if True also return debug mapping dict
        with_prediction: if True prefer PredFrstIndex
        include_energy_fields: if True extract and return NativeEnergy, DecoyEnergy,
                         SDEnergy, DensityRes1, DensityRes2 when present in file

    Returns:
        frustration_values: [E] float  (or [max_idx, max_idx] when edge_index is None)
        welltypes:          [E] long   (only if use_welltype)
        states:             [E] long   (only if use_welltype)
        edge_mask:          [E] bool   (True = has frustration data)
        energy_fields:      dict       (only if include_energy_fields and has_energy)
        mapping_dict:       dict       (only if return_mapping)
    """
    pair_frustration_raw = pd.read_csv(str(frust_file), sep=" ", comment="#")

    # -------------------------------------------------------------------------
    # Energy / density column detection
    # -------------------------------------------------------------------------
    energy_cols  = ["NativeEnergy", "DecoyEnergy", "SDEnergy"]
    density_cols = ["DensityRes1", "DensityRes2"]
    has_energy  = include_energy_fields and all(c in pair_frustration_raw.columns for c in energy_cols)
    has_density = include_energy_fields and all(c in pair_frustration_raw.columns for c in density_cols)
    extra_cols  = (energy_cols if has_energy else []) + (density_cols if has_density else [])

    # -------------------------------------------------------------------------
    # Column normalisation
    # -------------------------------------------------------------------------
    if "FrstIndex" in pair_frustration_raw.columns:
        keep = ["Res1", "Res2", "ChainRes1", "ChainRes2", "Welltype", "FrstIndex", "FrstState"] + extra_cols
        # Keep only columns that exist in the file (ChainRes* may be absent for single-chain files)
        keep = [c for c in keep if c in pair_frustration_raw.columns]
        pair_frustration_raw = pair_frustration_raw[keep]
    elif with_prediction and "PredFrstIndex" in pair_frustration_raw.columns:
        keep = ["Res1", "Res2", "ChainRes1", "ChainRes2", "Welltype", "PredFrstIndex", "PredFrstState"] + extra_cols
        keep = [c for c in keep if c in pair_frustration_raw.columns]
        pair_frustration_raw = pair_frustration_raw[keep].rename(
            columns={"PredFrstIndex": "FrstIndex", "PredFrstState": "FrstState"})
    elif not with_prediction and "TrueFrstIndex" in pair_frustration_raw.columns:
        keep = ["Res1", "Res2", "ChainRes1", "ChainRes2", "Welltype", "TrueFrstIndex", "TrueFrstState"] + extra_cols
        keep = [c for c in keep if c in pair_frustration_raw.columns]
        pair_frustration_raw = pair_frustration_raw[keep].rename(
            columns={"TrueFrstIndex": "FrstIndex", "TrueFrstState": "FrstState"})
    else:
        raise RuntimeError("Unable to parse pair frustration file: no FrstIndex column")

    pair_frustration_raw["Welltype"] = pair_frustration_raw["Welltype"].map({
        "short": 0, "long": 1, "water-mediated": 2,
    })
    pair_frustration_raw["FrstState"] = pair_frustration_raw["FrstState"].map({
        "highly": 0, "neutral": 1, "minimally": 2,
    })

    n_ref_total = len(pair_frustration_raw)   # total pairs in reference before edge filtering

    # -------------------------------------------------------------------------
    # Convert to tensors (PDB residue numbers)
    # -------------------------------------------------------------------------
    pair_rows_pdb = torch.from_numpy(pair_frustration_raw["Res1"].values.copy()).long()
    pair_cols_pdb = torch.from_numpy(pair_frustration_raw["Res2"].values.copy()).long()
    pair_vals     = torch.from_numpy(pair_frustration_raw["FrstIndex"].values.copy()).float()
    pair_well     = torch.from_numpy(pair_frustration_raw["Welltype"].values.copy()).long()
    pair_state    = torch.from_numpy(pair_frustration_raw["FrstState"].values.copy()).long()
    if has_energy:
        pair_native_e = torch.from_numpy(pair_frustration_raw["NativeEnergy"].values.copy()).float()
        pair_decoy_e  = torch.from_numpy(pair_frustration_raw["DecoyEnergy"].values.copy()).float()
        pair_sd_e     = torch.from_numpy(pair_frustration_raw["SDEnergy"].values.copy()).float()
    if has_density:
        pair_density1 = torch.from_numpy(pair_frustration_raw["DensityRes1"].values.copy()).float()
        pair_density2 = torch.from_numpy(pair_frustration_raw["DensityRes2"].values.copy()).float()

    # =========================================================================
    # Path A: multi-chain dict lookup  (FIRST in every if/elif/else chain)
    # =========================================================================
    _pfv = next(iter(residue_lookup.values()), None) if isinstance(residue_lookup, dict) else None
    _is_pair_multichain = isinstance(residue_lookup, dict) and isinstance(_pfv, (dict, torch.Tensor))
    _is_pair_sc_dict    = isinstance(residue_lookup, dict) and not isinstance(_pfv, (dict, torch.Tensor))

    if _is_pair_multichain:
        n_rows = len(pair_frustration_raw)
        seq_i_arr = np.full(n_rows, -1, dtype=np.int64)
        seq_j_arr = np.full(n_rows, -1, dtype=np.int64)

        # build_residue_lookup remaps original chain letters to A,B,C... alphabetically.
        # Frustration files use the original letters — build a remap when needed.
        _all_file_chains = (
            list(pair_frustration_raw["ChainRes1"].unique()) +
            list(pair_frustration_raw["ChainRes2"].unique())
        ) if "ChainRes1" in pair_frustration_raw.columns else []
        chain_remap_pair = _make_chain_remap(_all_file_chains, residue_lookup) if _all_file_chains else None

        # Precompute per-chain valid tensor positions (seq_mask=True), used for
        # sequential-numbering detection and positional mapping below.
        _chain_valid_pos = {}
        for _ck, _lut in residue_lookup.items():
            _all = _lut_valid_pos(_lut)
            if seq_mask is not None:
                _all = _all[seq_mask[_all]]
            _chain_valid_pos[_ck] = _all

        for (chain1, chain2), grp in pair_frustration_raw.groupby(["ChainRes1", "ChainRes2"]):
            # Normalize to string: pandas reads numeric chain IDs (e.g. "3") as int,
            # but _make_chain_remap and residue_lookup always use string keys.
            c1 = str(chain1)
            c2 = str(chain2)
            lk1 = chain_remap_pair[c1] if (chain_remap_pair and c1 in chain_remap_pair) else c1
            lk2 = chain_remap_pair[c2] if (chain_remap_pair and c2 in chain_remap_pair) else c2
            if lk1 not in residue_lookup or lk2 not in residue_lookup:
                continue

            lut1 = residue_lookup[lk1]
            lut2 = residue_lookup[lk2]

            res1 = torch.from_numpy(grp["Res1"].values.copy()).long()
            res2 = torch.from_numpy(grp["Res2"].values.copy()).long()

            vp1 = _chain_valid_pos[lk1]
            vp2 = _chain_valid_pos[lk2]
            n_valid_1 = len(vp1)
            n_valid_2 = len(vp2)
            max_res = max(res1.max().item(), res2.max().item()) if len(res1) > 0 else 0

            si = torch.full((len(res1),), -1, dtype=torch.long)
            sj = torch.full((len(res2),), -1, dtype=torch.long)

            if seq_mask is not None and max_res <= max(n_valid_1, n_valid_2):
                # Sequential numbering: FrustratometeR renumbered residues 1..N_valid per chain.
                # Map FR sequential k → valid_pos[k-1].
                ib1 = (res1 >= 1) & (res1 <= n_valid_1)
                ib2 = (res2 >= 1) & (res2 <= n_valid_2)
                in_bounds = ib1 & ib2
                si[in_bounds] = vp1[(res1[in_bounds] - 1).clamp(0, n_valid_1 - 1)]
                sj[in_bounds] = vp2[(res2[in_bounds] - 1).clamp(0, n_valid_2 - 1)]
            else:
                res1_labels = [str(r.item()) for r in res1]
                res2_labels = [str(r.item()) for r in res2]
                si = _lut_lookup_labels(lut1, res1_labels)
                sj = _lut_lookup_labels(lut2, res2_labels)

            # Use original pandas index to write back in correct row order
            seq_i_arr[grp.index] = si.numpy()
            seq_j_arr[grp.index] = sj.numpy()

        df = pair_frustration_raw.copy()
        df["_seq_i"] = seq_i_arr
        df["_seq_j"] = seq_j_arr
        df["_keep"]  = (seq_i_arr >= 0) & (seq_j_arr >= 0)

        # Apply energy field filtering before dropping rows
        if has_energy or has_density:
            _keep_mask = df["_keep"].values
            if has_energy:
                pair_native_e = torch.from_numpy(df["NativeEnergy"].values[_keep_mask]).float()
                pair_decoy_e  = torch.from_numpy(df["DecoyEnergy"].values[_keep_mask]).float()
                pair_sd_e     = torch.from_numpy(df["SDEnergy"].values[_keep_mask]).float()
            if has_density:
                pair_density1 = torch.from_numpy(df["DensityRes1"].values[_keep_mask]).float()
                pair_density2 = torch.from_numpy(df["DensityRes2"].values[_keep_mask]).float()

        df = df[df["_keep"]]

        pair_rows = torch.from_numpy(df["_seq_i"].values.copy()).long()
        pair_cols = torch.from_numpy(df["_seq_j"].values.copy()).long()
        pair_vals  = torch.from_numpy(df["FrstIndex"].values.copy()).float()
        pair_well  = torch.from_numpy(df["Welltype"].values.copy()).long()
        pair_state = torch.from_numpy(df["FrstState"].values.copy()).long()

        if max_len is not None:
            length_ok  = (pair_rows < max_len) & (pair_cols < max_len)
            pair_rows  = pair_rows[length_ok]
            pair_cols  = pair_cols[length_ok]
            pair_vals  = pair_vals[length_ok]
            pair_well  = pair_well[length_ok]
            pair_state = pair_state[length_ok]
            if has_energy:
                pair_native_e = pair_native_e[length_ok]
                pair_decoy_e  = pair_decoy_e[length_ok]
                pair_sd_e     = pair_sd_e[length_ok]
            if has_density:
                pair_density1 = pair_density1[length_ok]
                pair_density2 = pair_density2[length_ok]

    # =========================================================================
    # Path B: single-chain dict (new format: dict[str→int])
    # =========================================================================
    elif _is_pair_sc_dict:
        def _sc_dict_lookup(res_tensor):
            return torch.tensor(
                [residue_lookup.get(str(r.item()), -1) for r in res_tensor],
                dtype=torch.long,
            )

        if seq_mask is not None:
            valid_positions = seq_mask.bool().nonzero(as_tuple=False).squeeze(1)
            n_valid = len(valid_positions)
            max_res = max(pair_rows_pdb.max().item(), pair_cols_pdb.max().item())

            if max_res <= n_valid:
                in_bounds = (
                    (pair_rows_pdb >= 1) & (pair_rows_pdb <= n_valid) &
                    (pair_cols_pdb >= 1) & (pair_cols_pdb <= n_valid)
                )
                r_safe    = pair_rows_pdb.clamp(1, n_valid) - 1
                c_safe    = pair_cols_pdb.clamp(1, n_valid) - 1
                pair_rows = valid_positions[r_safe].clone()
                pair_cols = valid_positions[c_safe].clone()
                pair_rows[~in_bounds] = -1
                pair_cols[~in_bounds] = -1
            else:
                pair_rows = _sc_dict_lookup(pair_rows_pdb)
                pair_cols = _sc_dict_lookup(pair_cols_pdb)
        else:
            pair_rows = _sc_dict_lookup(pair_rows_pdb)
            pair_cols = _sc_dict_lookup(pair_cols_pdb)

        valid_mask = (pair_rows >= 0) & (pair_cols >= 0)
        pair_rows  = pair_rows[valid_mask]
        pair_cols  = pair_cols[valid_mask]
        pair_vals  = pair_vals[valid_mask]
        pair_well  = pair_well[valid_mask]
        pair_state = pair_state[valid_mask]
        if has_energy:
            pair_native_e = pair_native_e[valid_mask]
            pair_decoy_e  = pair_decoy_e[valid_mask]
            pair_sd_e     = pair_sd_e[valid_mask]
        if has_density:
            pair_density1 = pair_density1[valid_mask]
            pair_density2 = pair_density2[valid_mask]

        if max_len is not None:
            length_mask = (pair_rows < max_len) & (pair_cols < max_len)
            pair_rows  = pair_rows[length_mask]
            pair_cols  = pair_cols[length_mask]
            pair_vals  = pair_vals[length_mask]
            pair_well  = pair_well[length_mask]
            pair_state = pair_state[length_mask]
            if has_energy:
                pair_native_e = pair_native_e[length_mask]
                pair_decoy_e  = pair_decoy_e[length_mask]
                pair_sd_e     = pair_sd_e[length_mask]
            if has_density:
                pair_density1 = pair_density1[length_mask]
                pair_density2 = pair_density2[length_mask]

    # =========================================================================
    # Path C: single-chain flat tensor with legacy offset format
    # =========================================================================
    elif residue_lookup is not None:
        # residue_lookup[0] is the pdb_offset; actual data is residue_lookup[1:]
        pdb_offset_pair = int(residue_lookup[0])
        pdb_lookup      = residue_lookup[1:]
        n               = len(pdb_lookup)

        if seq_mask is not None:
            # Detect whether FrustratometeR used sequential or PDB residue numbering.
            valid_positions = seq_mask.bool().nonzero(as_tuple=False).squeeze(1)  # [N_valid]
            n_valid = len(valid_positions)
            max_res = max(pair_rows_pdb.max().item(), pair_cols_pdb.max().item())

            if max_res <= n_valid:
                # Sequential numbering: FrustratometeR renumbered residues 1..N_valid
                in_bounds = (
                    (pair_rows_pdb >= 1) & (pair_rows_pdb <= n_valid) &
                    (pair_cols_pdb >= 1) & (pair_cols_pdb <= n_valid)
                )
                r_safe    = pair_rows_pdb.clamp(1, n_valid) - 1
                c_safe    = pair_cols_pdb.clamp(1, n_valid) - 1
                pair_rows = valid_positions[r_safe].clone()
                pair_cols = valid_positions[c_safe].clone()
                pair_rows[~in_bounds] = -1
                pair_cols[~in_bounds] = -1
            else:
                # PDB-numbered: use residue_lookup
                rows_shifted = pair_rows_pdb + pdb_offset_pair
                cols_shifted = pair_cols_pdb + pdb_offset_pair
                rows_in   = (rows_shifted >= 0) & (rows_shifted < n)
                cols_in   = (cols_shifted >= 0) & (cols_shifted < n)
                rows_safe = rows_shifted.clamp(min=0, max=n - 1)
                cols_safe = cols_shifted.clamp(min=0, max=n - 1)
                pair_rows = pdb_lookup[rows_safe]
                pair_cols = pdb_lookup[cols_safe]
                pair_rows[~rows_in] = -1
                pair_cols[~cols_in] = -1

            valid_mask = (pair_rows >= 0) & (pair_cols >= 0)
            pair_rows  = pair_rows[valid_mask]
            pair_cols  = pair_cols[valid_mask]
            pair_vals  = pair_vals[valid_mask]
            pair_well  = pair_well[valid_mask]
            pair_state = pair_state[valid_mask]
            if has_energy:
                pair_native_e = pair_native_e[valid_mask]
                pair_decoy_e  = pair_decoy_e[valid_mask]
                pair_sd_e     = pair_sd_e[valid_mask]
            if has_density:
                pair_density1 = pair_density1[valid_mask]
                pair_density2 = pair_density2[valid_mask]

            if max_len is not None:
                length_mask = (pair_rows < max_len) & (pair_cols < max_len)
                pair_rows  = pair_rows[length_mask]
                pair_cols  = pair_cols[length_mask]
                pair_vals  = pair_vals[length_mask]
                pair_well  = pair_well[length_mask]
                pair_state = pair_state[length_mask]
                if has_energy:
                    pair_native_e = pair_native_e[length_mask]
                    pair_decoy_e  = pair_decoy_e[length_mask]
                    pair_sd_e     = pair_sd_e[length_mask]
                if has_density:
                    pair_density1 = pair_density1[length_mask]
                    pair_density2 = pair_density2[length_mask]

        else:
            # No seq_mask — use PDB-number lookup directly
            rows_shifted = pair_rows_pdb + pdb_offset_pair
            cols_shifted = pair_cols_pdb + pdb_offset_pair
            rows_in   = (rows_shifted >= 0) & (rows_shifted < n)
            cols_in   = (cols_shifted >= 0) & (cols_shifted < n)
            rows_safe = rows_shifted.clamp(min=0, max=n - 1)
            cols_safe = cols_shifted.clamp(min=0, max=n - 1)
            pair_rows = pdb_lookup[rows_safe]
            pair_cols = pdb_lookup[cols_safe]
            pair_rows[~rows_in] = -1
            pair_cols[~cols_in] = -1

            valid_mask = (pair_rows >= 0) & (pair_cols >= 0)
            pair_rows  = pair_rows[valid_mask]
            pair_cols  = pair_cols[valid_mask]
            pair_vals  = pair_vals[valid_mask]
            pair_well  = pair_well[valid_mask]
            pair_state = pair_state[valid_mask]
            if has_energy:
                pair_native_e = pair_native_e[valid_mask]
                pair_decoy_e  = pair_decoy_e[valid_mask]
                pair_sd_e     = pair_sd_e[valid_mask]
            if has_density:
                pair_density1 = pair_density1[valid_mask]
                pair_density2 = pair_density2[valid_mask]

            if max_len is not None:
                length_mask = (pair_rows < max_len) & (pair_cols < max_len)
                pair_rows  = pair_rows[length_mask]
                pair_cols  = pair_cols[length_mask]
                pair_vals  = pair_vals[length_mask]
                pair_well  = pair_well[length_mask]
                pair_state = pair_state[length_mask]
                if has_energy:
                    pair_native_e = pair_native_e[length_mask]
                    pair_decoy_e  = pair_decoy_e[length_mask]
                    pair_sd_e     = pair_sd_e[length_mask]
                if has_density:
                    pair_density1 = pair_density1[length_mask]
                    pair_density2 = pair_density2[length_mask]

    # =========================================================================
    # Path D: no lookup — assume sequential indices
    # =========================================================================
    else:
        pair_rows = pair_rows_pdb
        pair_cols = pair_cols_pdb

        if max_len is not None:
            length_mask = (pair_rows < max_len) & (pair_cols < max_len)
            pair_rows  = pair_rows[length_mask]
            pair_cols  = pair_cols[length_mask]
            pair_vals  = pair_vals[length_mask]
            pair_well  = pair_well[length_mask]
            pair_state = pair_state[length_mask]
            if has_energy:
                pair_native_e = pair_native_e[length_mask]
                pair_decoy_e  = pair_decoy_e[length_mask]
                pair_sd_e     = pair_sd_e[length_mask]
            if has_density:
                pair_density1 = pair_density1[length_mask]
                pair_density2 = pair_density2[length_mask]

    # -------------------------------------------------------------------------
    # Fill result matrices and index by edge_index
    # -------------------------------------------------------------------------
    if edge_index is not None:
        max_idx = int(edge_index.max().item() + 1)
    else:
        max_idx = int(max(pair_cols.max().item(), pair_rows.max().item()) + 1)

    results   = torch.zeros((max_idx, max_idx))
    welltypes = torch.full((max_idx, max_idx), -1, dtype=torch.long)
    states    = torch.full((max_idx, max_idx), -1, dtype=torch.long)
    if has_energy:
        native_e_mat = torch.zeros((max_idx, max_idx))
        decoy_e_mat  = torch.zeros((max_idx, max_idx))
        sd_e_mat     = torch.zeros((max_idx, max_idx))
    if has_density:
        density1_mat = torch.zeros((max_idx, max_idx))
        density2_mat = torch.zeros((max_idx, max_idx))

    if len(pair_rows) > 0:
        valid = (pair_rows < max_idx) & (pair_cols < max_idx)
        pr, pc = pair_rows[valid], pair_cols[valid]
        pv, pw, ps = pair_vals[valid], pair_well[valid], pair_state[valid]

        results[pr, pc]   = pv;   results[pc, pr]   = pv
        welltypes[pr, pc] = pw;   welltypes[pc, pr] = pw
        states[pr, pc]    = ps;   states[pc, pr]    = ps

        if has_energy:
            _pne = pair_native_e[valid]
            _pde = pair_decoy_e[valid]
            _pse = pair_sd_e[valid]
            native_e_mat[pr, pc] = _pne;  native_e_mat[pc, pr] = _pne
            decoy_e_mat[pr, pc]  = _pde;  decoy_e_mat[pc, pr]  = _pde
            sd_e_mat[pr, pc]     = _pse;  sd_e_mat[pc, pr]     = _pse

        if has_density:
            _pd1 = pair_density1[valid]
            _pd2 = pair_density2[valid]
            density1_mat[pr, pc] = _pd1
            density2_mat[pr, pc] = _pd2
            # symmetric: swap roles — DensityRes1 of (j,i) is DensityRes2 of (i,j)
            density1_mat[pc, pr] = _pd2
            density2_mat[pc, pr] = _pd1

    # -------------------------------------------------------------------------
    # Extract per-edge values
    # -------------------------------------------------------------------------
    if edge_index is not None:
        ei0 = edge_index[0].clamp(min=0)
        ei1 = edge_index[1].clamp(min=0)
        valid_edge = (edge_index[0] >= 0) & (edge_index[1] >= 0)

        frustration_values = results[ei0, ei1]
        welltypes_out      = welltypes[ei0, ei1]
        states_out         = states[ei0, ei1]
        edge_mask          = (welltypes_out >= 0) & valid_edge
    else:
        frustration_values = results
        welltypes_out      = welltypes
        states_out         = states
        edge_mask          = welltypes_out >= 0

    if return_mapping:
        mapping_dict = {
            "pair_rows_seq": pair_rows,
            "pair_cols_seq": pair_cols,
            "pair_vals":     pair_vals,
            "pair_well":     pair_well,
        }
        if use_welltype:
            return frustration_values, welltypes_out, states_out, edge_mask, mapping_dict
        return frustration_values, edge_mask, mapping_dict

    if has_energy and edge_index is not None:
        energy_fields = {
            "NativeEnergy":  native_e_mat[ei0, ei1],
            "DecoyEnergy":   decoy_e_mat[ei0, ei1],
            "SDEnergy":      sd_e_mat[ei0, ei1],
            "n_ref_total":   n_ref_total,
        }
        if has_density:
            energy_fields["DensityRes1"] = density1_mat[ei0, ei1]
            energy_fields["DensityRes2"] = density2_mat[ei0, ei1]
        if use_welltype:
            return frustration_values, welltypes_out, states_out, edge_mask, energy_fields
        return frustration_values, edge_mask, energy_fields

    if use_welltype:
        return frustration_values, welltypes_out, states_out, edge_mask

    return frustration_values, edge_mask


def read_mutations_data(done_dir) -> pd.DataFrame:
    """
    Read all per-residue threading files from a dompdb_v4.4_mutants MutationsData directory
    and return them as a single flat DataFrame.

    Accepts either:
      - a *.done directory (reads MutationsData/ inside it)
      - the MutationsData/ directory itself

    Returns columns: Res, ChainRes, AA, FrstIndex
    """
    done_dir = Path(done_dir)
    mut_dir = done_dir / "MutationsData" if (done_dir / "MutationsData").is_dir() else done_dir
    files = sorted(mut_dir.glob("singleresidue_Res*_threading_*.txt"))
    if not files:
        raise FileNotFoundError(f"No threading files found in {mut_dir}")
    return pd.concat(
        (pd.read_csv(f, sep=r"\s+") for f in files),
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Output writers (ported from frustrai_fold.data.io)
# ---------------------------------------------------------------------------

def _compute_welltypes(edge_index_2e, residue_density_b, distances_b):
    """
    Classify each contact edge as short / long / water-mediated.

    Args:
        edge_index_2e:     [2, E_valid]  sequential residue indices
        residue_density_b: [L]           per-residue density for one batch item
        distances_b:       [L, L]        CB-CB distance matrix for one batch item

    Returns: LongTensor [E_valid]  with values 0=short, 1=long, 2=water-mediated
    """
    water_kappa_sigma = 7.0
    threshold_rho     = 2.6
    direct_well_rmax  = 6.5

    rho1 = residue_density_b[edge_index_2e[0]].float()
    rho2 = residue_density_b[edge_index_2e[1]].float()
    dist = distances_b[edge_index_2e[0], edge_index_2e[1]].float()

    sigma_wat = 0.25 * (1 - torch.tanh(water_kappa_sigma * (rho1 - threshold_rho))) \
                     * (1 - torch.tanh(water_kappa_sigma * (rho2 - threshold_rho)))

    return torch.where(
        dist < direct_well_rmax,
        torch.zeros_like(dist, dtype=torch.long),
        torch.where(sigma_wat >= 0.5,
                    torch.full_like(dist, 2, dtype=torch.long),
                    torch.ones_like(dist, dtype=torch.long)),
    )


def write_singleresidue_frustration(
    single_frustration_index,     # [B, L] or [B, L, 1]
    single_frustration_class,     # [B, L, 3] or None  (computed from threshold when None)
    aatype_int,                   # [B, L]
    seq_mask,                     # [B, L]
    densities,                    # [B, L]
    frust_file,                   # str  base path, no extension
    atom_positions=None,          # [B, L, ≥5, 3]
    structure_type="pdb",
    true_frustration_index=None,  # [B, L]
    true_frustration_class=None,  # [B, L]
    names=None,
    losses=None,
    metrics=None,
    residue_index=None,           # [B, L] PDB residue numbers; None → 1..n
    native_energy=None,           # [B, L]
    decoy_stats=None,             # [B, L, 2]
    chain_index=None,             # [B, L] 0-based int; None → chain "A"
    chain_id_map=None,            # list[list[str]] — original chain letters per batch item
    pred=True,
    structure_files=None,
    mode_sep=".",
    backbone_method=None,
    structure_only=False,         # True: write only the structure file (B-factors = single_frustration_index),
                                  # no *_singleresidue TSV. Used by pair-only runs (mutational / configurational /
                                  # smooth) so they still get a <stem>.pdb for the .pml / Mol* scripts.
):
    """Write single-residue frustration results to a TSV and optionally a structure file."""
    import biotite.structure as struc
    import biotite.structure.io as strucio
    from ..structure.residue_constants import ID_TO_AA, standardize_aa_mapping, restype_3to1
    from ..utils.weights import classify_frustration

    state_map = {0: "highly", 1: "neutral", 2: "minimally"}

    if single_frustration_index.dim() == 3:
        single_frustration_index = single_frustration_index[..., 0]

    for i in range(single_frustration_index.size(0)):
        mask  = seq_mask[i].bool()
        n_res = mask.sum().item()
        frust = single_frustration_index[i, mask].cpu().float()
        aa    = aatype_int[i, mask].cpu()

        frust_file_i = frust_file[i] if isinstance(frust_file, list) and i < len(frust_file) else frust_file

        if isinstance(structure_type, str):
            structure_type_ = structure_type.lower()
        elif isinstance(structure_type, (list, tuple)):
            structure_type_ = structure_type[i].lower()
        else:
            structure_type_ = "cif"
        structure_type_ = structure_type_.replace(".", "").replace("_", "")
        if structure_type_ not in ("pdb", "cif", "mmtf", "bcif"):
            structure_type_ = "cif"  # default to mmCIF if unknown


        name = names[i] if (names is not None and i < len(names)) else "cif"

        res_nums = residue_index[i, mask].cpu().numpy() if residue_index is not None \
                   else np.arange(1, n_res + 1)
        _cid_map_i = chain_id_map[i] if chain_id_map is not None else None
        chain_ids = [(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                     for c in chain_index[i, mask].cpu().tolist()] \
                    if chain_index is not None else ["A"] * n_res

        if single_frustration_class is not None:
            frust_class_arr = single_frustration_class[i, mask].argmax(dim=-1).cpu().numpy()
        else:
            frust_class_arr = classify_frustration(frust, 0).numpy()

        frust_col = "FrstIndex"    if not pred else "PredFrstIndex"
        state_col = "FrstState"    if not pred else "PredFrstState"

        row = {
            "Res":          res_nums,
            "ChainRes":     chain_ids,
            "DensityRes":   densities[i, mask].cpu().numpy(),
            "AA":           [ID_TO_AA.get(int(a), 'X') for a in aa.tolist()],
            "NativeEnergy": native_energy[i, mask].cpu().float().numpy() if native_energy is not None else np.full(n_res, np.nan),
            "DecoyEnergy":  decoy_stats[i, mask, 0].cpu().float().numpy() if decoy_stats is not None else np.full(n_res, np.nan),
            "SDEnergy":     decoy_stats[i, mask, 1].cpu().float().numpy() if decoy_stats is not None else np.full(n_res, np.nan),
            frust_col:      frust.numpy(),
            # state_col:      [state_map[int(c)] for c in frust_class_arr],
        }

        if true_frustration_index is not None:
            true_frust = true_frustration_index[i, mask].cpu().float()
            true_class_arr = true_frustration_class[i, mask].cpu().numpy() \
                             if true_frustration_class is not None \
                             else classify_frustration(true_frust, 0).numpy()
            row["TrueFrstIndex"] = true_frust.numpy()
            row["TrueFrstState"] = [state_map[int(c)] for c in true_class_arr]

        frust_df = pd.DataFrame(row)
        suffix   = "singleresidue.txt" if pred else "singleresidue"
        if not structure_only:
            with open(f"{frust_file_i}.{structure_type_}{mode_sep}{suffix}", "w") as f:
                # if backbone_method is not None:
                #     print(f"# backbone_method: {backbone_method}", file=f)
                if losses is not None:
                    print(f"#FrustAItometer single index loss: {losses[0]}", file=f)
                    print(f"#FrustAItometer single class loss: {losses[1]}", file=f)
                if metrics is not None:
                    for k, v in metrics.items():
                        print(f"# {k}: {v:.6g}" if isinstance(v, float) else f"# {k}: {v}", file=f)
                frust_df.to_csv(f, sep=" ", index=False, na_rep="nan", float_format="%.3f")

        sf = structure_files[i] if structure_files is not None else None
        if atom_positions is not None:
            from ..structure.residue_constants import restype_1to3
            # atom_positions layout (this codebase): 0=N, 1=CA, 2=C, 3=CB, 4=O
            # GLY has no CB — match pdb_tidy behaviour by writing only N,CA,C,O for GLY.
            ALL_NAMES  = np.array(['N', 'CA', 'C', 'CB', 'O'])
            GLY_NAMES  = np.array(['N', 'CA', 'C', 'O'])
            GLY_IDX    = np.array([0, 1, 2, 4])   # indices into the 5-atom row
            res_names_3 = np.array([restype_1to3.get(a, 'UNK') for a in frust_df.AA.values])
            is_gly      = res_names_3 == 'GLY'
            # Select only seq_mask-valid rows from the full [L, ≥5, 3] tensor
            valid_pos = atom_positions[i][mask][:, :5].cpu().numpy()  # [n_res, 5, 3]
            frust_vals = frust_df[frust_col].values                    # [n_res]
            true_frust_vals = (true_frustration_index[i, mask].cpu().float().numpy()
                               if true_frustration_index is not None else None)
            # Build per-atom lists (variable per residue for GLY vs non-GLY)
            coord_rows  = []
            chain_rows  = []
            resid_rows  = []
            resname_rows = []
            aname_rows  = []
            bfact_rows  = []
            tfact_rows  = []
            for k in range(n_res):
                if is_gly[k]:
                    idx = GLY_IDX
                    names = GLY_NAMES
                else:
                    idx = np.arange(5)
                    names = ALL_NAMES
                n_at = len(idx)
                coord_rows.append(valid_pos[k][idx])
                chain_rows.extend([chain_ids[k]] * n_at)
                resid_rows.extend([res_nums[k]]  * n_at)
                resname_rows.extend([res_names_3[k]] * n_at)
                aname_rows.extend(names.tolist())
                bfact_rows.extend([frust_vals[k]] * n_at)
                if true_frust_vals is not None:
                    tfact_rows.extend([true_frust_vals[k]] * n_at)
            n_atoms = len(aname_rows)
            atoms   = struc.AtomArray(n_atoms)
            atoms.coord = np.concatenate(coord_rows, axis=0)
            atoms.set_annotation("chain_id",  np.array(chain_rows, dtype="U4"))
            atoms.set_annotation("res_id",    np.array(resid_rows, dtype=int))
            atoms.set_annotation("res_name",  np.array(resname_rows, dtype="U3"))
            atoms.set_annotation("hetero",    np.zeros(n_atoms, dtype=bool))
            atoms.set_annotation("atom_name", np.array(aname_rows, dtype="U4"))

                
            if pred:
                atoms.set_annotation("b_factor", np.array(bfact_rows, dtype=float))
                strucio.save_structure(f"{frust_file_i}.{name}.pred.{structure_type_}", atoms)
                if true_frust_vals is not None:
                    atoms.set_annotation("b_factor", np.array(tfact_rows, dtype=float))
                    strucio.save_structure(f"{frust_file_i}.{name}.true.{structure_type_}", atoms)
            else:
                atoms.set_annotation("b_factor", np.array(bfact_rows, dtype=float))
                struct_path = (f"{frust_file_i}.{structure_type_}" if mode_sep != "."
                               else f"{frust_file_i}.{name}.{structure_type_}")
                # biotite infers format from extension; save to a path with a
                # known extension and rename if struct_path lacks one.
                _known = ('.pdb', '.cif', '.mmtf', '.bcif')
                if any(struct_path.endswith(ext) for ext in _known):
                    strucio.save_structure(struct_path, atoms)
                else:
                    import os as _os
                    _tmp = f"{struct_path}.{structure_type_}"
                    strucio.save_structure(_tmp, atoms)
                    _os.rename(_tmp, struct_path)

        elif sf is not None:
            orig = strucio.load_structure(str(sf))
            if isinstance(orig, struc.AtomArrayStack):
                orig = orig[0]
            # Normalize to FrustratometeR pdb_tidy format:
            #   1. Convert known modified residues (standardize_aa_mapping) to ATOM records
            #      with standard residue name (e.g. HETATM MSE → ATOM MET, SE → SD).
            #   2. Remove all remaining HETATM records (ligands, cofactors, waters, ions)
            #      that are not in standardize_aa_mapping — FrustratometeR excludes them.
            for src, dst in standardize_aa_mapping.items():
                conv = orig.hetero & (orig.res_name == src)
                if conv.any():
                    orig.hetero[conv]   = False
                    orig.res_name[conv] = dst
                    if src == "MSE":
                        se = conv & (orig.atom_name == "SE")
                        orig.atom_name[se] = "SD"
            # After conversion, any remaining HETATM atoms are true non-protein records.
            if orig.hetero.any():
                orig = orig[~orig.hetero]
            # Also remove ATOM-record non-standard residues (e.g. selenocysteine SEC)
            # that don't map to the 20 standard amino acids — FrustratometeR excludes them.
            _valid_res_names = set(restype_3to1.keys()) - {"UNK"}
            _valid_mask = np.isin(orig.res_name, list(_valid_res_names))
            if not _valid_mask.all():
                orig = orig[_valid_mask]
            # Add ideal CB for non-GLY residues missing CB, matching pdb_tidy behaviour.
            # Uses CHARMM36 IC: |CA-CB|=1.522 Å, ∠C-CA-CB=111.09°, φ(N-C-CA-CB)=123.23°.
            _by_res: dict = {}
            for _idx, _a in enumerate(orig):
                _rk = (str(_a.chain_id), int(_a.res_id), str(_a.ins_code))
                if _rk not in _by_res:
                    _by_res[_rk] = {"res_name": str(_a.res_name), "atoms": {}}
                _by_res[_rk]["atoms"][str(_a.atom_name)] = _idx
            _need_cb = [
                (_rk, _rd)
                for _rk, _rd in _by_res.items()
                if _rd["res_name"] != "GLY"
                and "CB" not in _rd["atoms"]
                and all(x in _rd["atoms"] for x in ("N", "CA", "C"))
            ]
            if _need_cb:
                import torch as _torch
                from ..structure.geometry import place_fourth_atom as _pfa
                _n_new = len(_need_cb)
                _cb_arr = struc.AtomArray(_n_new)
                _cb_coords = np.empty((_n_new, 3), dtype=np.float64)
                for _ni, ((_chain_id, _res_id, _ins), _rd) in enumerate(_need_cb):
                    _N  = orig.coord[_rd["atoms"]["N"]].astype(np.float64)
                    _CA = orig.coord[_rd["atoms"]["CA"]].astype(np.float64)
                    _C  = orig.coord[_rd["atoms"]["C"]].astype(np.float64)
                    _CB = _pfa(
                        _torch.from_numpy(_N).unsqueeze(0),
                        _torch.from_numpy(_C).unsqueeze(0),
                        _torch.from_numpy(_CA).unsqueeze(0),
                        length=1.5220, angle_deg=111.09, dihedral_deg=123.23,
                    )[0].numpy()
                    _cb_coords[_ni] = _CB
                    _cb_arr.chain_id[_ni]  = _chain_id
                    _cb_arr.res_id[_ni]    = _res_id
                    _cb_arr.ins_code[_ni]  = _ins
                    _cb_arr.res_name[_ni]  = _rd["res_name"]
                    _cb_arr.atom_name[_ni] = "CB"
                    _cb_arr.element[_ni]   = "C"
                    _cb_arr.hetero[_ni]    = False
                _cb_arr.coord = _cb_coords.astype(orig.coord.dtype)
                # Copy optional annotations that orig may carry (b_factor, occupancy, etc.)
                # so that concatenation with + doesn't fail on mismatched annotation sets.
                for _ann in ("b_factor", "occupancy", "charge"):
                    if hasattr(orig, _ann):
                        _cb_arr.set_annotation(
                            _ann, np.zeros(_n_new, dtype=getattr(orig, _ann).dtype)
                        )
                orig = orig + _cb_arr
                # Re-sort: canonical atom order within each residue (N, CA, C, CB, O, …)
                _prio = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4}
                _atom_prio = np.array([_prio.get(str(a), 99) for a in orig.atom_name])
                _sort_idx  = np.lexsort((_atom_prio, orig.res_id, orig.chain_id))
                orig = orig[_sort_idx]
            # Build frust_map using orig's actual chain IDs (not internal 0-based letters).
            # This avoids remapping bugs when the domain chain (e.g. 'B') differs from the
            # internal 0-based letter ('A').
            _seen_keys: set = set()
            _orig_residue_keys: list = []
            for _a in orig:
                _k = (str(_a.chain_id), int(_a.res_id))
                if _k not in _seen_keys:
                    _seen_keys.add(_k)
                    _orig_residue_keys.append(_k)
            frust_map      = {k: float(frust.numpy()[j]) for j, k in enumerate(_orig_residue_keys) if j < n_res}
            per_atom_frust = np.array([frust_map.get((str(a.chain_id), int(a.res_id)), 0.0) for a in orig])

            if pred:
                orig.set_annotation("b_factor", per_atom_frust.astype(float))
                strucio.save_structure(f"{frust_file_i}.{name}.pred.{structure_type_}", orig)
                if true_frustration_index is not None:
                    true_frust_map = {int(res_nums[j]): float(true_frustration_index[i, mask].cpu().float().numpy()[j]) for j in range(n_res)}
                    true_per_atom  = np.array([true_frust_map.get(int(a.res_id), 0.0) for a in orig])
                    orig.set_annotation("b_factor", true_per_atom.astype(float))
                    strucio.save_structure(f"{frust_file_i}.{structure_type_}.true.{structure_type_}", orig)
            else:
                orig.set_annotation("b_factor", per_atom_frust.astype(float))
                struct_path = (f"{frust_file_i}.{structure_type_}" if mode_sep != "."
                               else f"{frust_file_i}.{name}.{structure_type_}")
                strucio.save_structure(struct_path, orig)



def write_pair_frustration(
    mode,
    pair_frustration_index,       # [B, E] or [B, E, 1]
    pair_frustration_class,       # [B, E, 3] or None
    aatype_int,                   # [B, L]
    seq_mask,                     # [B, L]
    edge_index,                   # [B, 2, E]
    densities,                    # [B, L]
    welltypes,                    # [B, E] or None
    frust_file,                   # str  base path, no extension
    true_frustration_index=None,
    true_frustration_class=None,
    names=None,
    losses=None,
    metrics=None,
    residue_index=None,
    native_energy=None,
    decoy_stats=None,
    chain_index=None,
    chain_id_map=None,            # list[list[str]] — original chain letters per batch item
    distances=None,
    pred=True,
    mode_sep=".",
    reference_frust_file=None,    # path to FrustratometeR reference file; used when true_frustration_index is None
    backbone_method=None,
    mode_name=None,               # output-file label; defaults to modes[mode].
                                  # Derived modes (e.g. "smooth") reuse a mode index for
                                  # thresholds but need their own file name.
    invert_classes=False,         # if True, use reversed thresholds (for smooth mode):
                                  #   low FI = minimally frustrated, high FI = highly frustrated
                                  #   thresholds: [-1.2, 0.8]  (Guan et al. 2024 PNAS)
):
    """Write pairwise frustration results to a space-separated TSV."""
    from ..structure.residue_constants import ID_TO_AA
    from ..utils.weights import classify_frustration, frustration_thresholds, modes

    threshold_idx  = 2 if invert_classes else 1
    state_map      = {0: "highly", 1: "neutral", 2: "minimally"}
    wt_map         = {0: "short",  1: "long",    2: "water-mediated"}
    label          = mode_name if mode_name is not None else modes[mode]

    if pair_frustration_index.dim() == 3:
        pair_frustration_index = pair_frustration_index[..., 0]

    for i in range(pair_frustration_index.shape[0]):
        frust_file_i = frust_file[i] if isinstance(frust_file, list) and i < len(frust_file) else frust_file
        name         = names[i] if (names is not None and i < len(names)) else "cif"

        valid_e = (edge_index[i] >= 0).all(dim=0)
        if welltypes is not None:
            valid_e = valid_e & (welltypes[i] >= 0)
        curr_edge_index = edge_index[i][:, valid_e]
        frust           = pair_frustration_index[i, valid_e].cpu().float()
        E_valid         = frust.shape[0]

        res1 = residue_index[i, curr_edge_index[0]].cpu().numpy() if residue_index is not None \
               else (curr_edge_index[0] + 1).cpu().numpy()
        res2 = residue_index[i, curr_edge_index[1]].cpu().numpy() if residue_index is not None \
               else (curr_edge_index[1] + 1).cpu().numpy()

        _cid_map_i = chain_id_map[i] if chain_id_map is not None else None
        ch1 = [(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
               for c in chain_index[i, curr_edge_index[0]].cpu().tolist()] \
              if chain_index is not None else ["A"] * E_valid
        ch2 = [(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
               for c in chain_index[i, curr_edge_index[1]].cpu().tolist()] \
              if chain_index is not None else ["A"] * E_valid

        if welltypes is not None:
            wt_arr = welltypes[i, valid_e].cpu()
        else:
            assert distances is not None, "Either welltypes or distances must be provided"
            wt_arr = _compute_welltypes(curr_edge_index, densities[i], distances[i])

        if pair_frustration_class is not None:
            frust_class_arr = pair_frustration_class[i, valid_e].argmax(dim=-1).cpu().numpy()
        else:
            frust_class_arr = classify_frustration(frust, threshold_idx).numpy()

        frust_col = "FrstIndex"  if not pred else "PredFrstIndex"
        state_col = "FrstState"  if not pred else "PredFrstState"

        row = {
            "Res1":         res1,
            "Res2":         res2,
            "ChainRes1":    ch1,
            "ChainRes2":    ch2,
            "DensityRes1":  densities[i, curr_edge_index[0]].cpu().numpy(),
            "DensityRes2":  densities[i, curr_edge_index[1]].cpu().numpy(),
            "AA1":          [ID_TO_AA.get(int(a), 'X') for a in aatype_int[i, curr_edge_index[0]].cpu().tolist()],
            "AA2":          [ID_TO_AA.get(int(a), 'X') for a in aatype_int[i, curr_edge_index[1]].cpu().tolist()],
            "NativeEnergy": native_energy[i, valid_e].cpu().float().numpy() if native_energy is not None else np.full(E_valid, np.nan),
            "DecoyEnergy":  (decoy_stats[i, valid_e, 0].cpu().float().numpy() if decoy_stats.dim() == 3
                            else np.full(E_valid, decoy_stats[i, 0].item())) if decoy_stats is not None else np.full(E_valid, np.nan),
            "SDEnergy":     (decoy_stats[i, valid_e, 1].cpu().float().numpy() if decoy_stats.dim() == 3
                            else np.full(E_valid, decoy_stats[i, 1].item())) if decoy_stats is not None else np.full(E_valid, np.nan),
            frust_col:      frust.numpy(),
            "Welltype":     [wt_map[int(w)] for w in wt_arr.tolist()],
            state_col:      [state_map[int(c)] for c in frust_class_arr],
        }

        if true_frustration_index is not None:
            true_frust = true_frustration_index[i, valid_e].cpu().float()
            true_class_arr = true_frustration_class[i, valid_e].cpu().numpy() \
                             if true_frustration_class is not None \
                             else classify_frustration(true_frust, threshold_idx).numpy()
            row["TrueFrstIndex"] = true_frust.numpy()
            row["TrueFrstState"] = [state_map[int(c)] for c in true_class_arr]
        elif reference_frust_file is not None:
            ref_path = reference_frust_file[i] if isinstance(reference_frust_file, list) and i < len(reference_frust_file) else reference_frust_file
            import os as _os
            if _os.path.isfile(ref_path):
                ref_df = pd.read_csv(ref_path, sep=r"\s+", comment="#")
                fi_col = ("TrueFrstIndex" if "TrueFrstIndex" in ref_df.columns
                          else "FrstIndex" if "FrstIndex" in ref_df.columns else None)
                if fi_col is not None:
                    merge_key = pd.DataFrame({"Res1": row["Res1"], "Res2": row["Res2"]})
                    if "ChainRes1" in ref_df.columns and "ChainRes1" in row:
                        merge_key["ChainRes1"] = row["ChainRes1"]
                        merge_key["ChainRes2"] = row["ChainRes2"]
                        merge_on = ["Res1", "Res2", "ChainRes1", "ChainRes2"]
                    else:
                        merge_on = ["Res1", "Res2"]
                    merged = merge_key.merge(
                        ref_df[merge_on + [fi_col]].rename(columns={fi_col: "TrueFrstIndex"}),
                        on=merge_on, how="left"
                    )
                    true_fi = merged["TrueFrstIndex"].values.astype(float)
                    row["TrueFrstIndex"] = true_fi
                    true_fi_t = torch.from_numpy(true_fi)
                    row["TrueFrstState"] = [state_map[int(c)]
                                            for c in classify_frustration(true_fi_t, threshold_idx).tolist()]

        frust_df = pd.DataFrame(row)
        suffix   = f"{label}.txt" if pred else label
        with open(f"{frust_file_i}.{name}{mode_sep}{suffix}", "w") as f:
            # if backbone_method is not None:
            #     print(f"# backbone_method: {backbone_method}", file=f)
            # Losses are indexed by `mode`, so they only describe the canonical mode —
            # skip them for derived modes that borrow another mode's index.
            if losses is not None and label == modes[mode]:
                print(f"#FrustAItometer {label} index loss: {losses[2*mode+2]}", file=f)
                print(f"#FrustAItometer {label} class loss: {losses[2*mode+3]}", file=f)
            if metrics is not None:
                for k, v in metrics.items():
                    print(f"# {k}: {v:.6g}" if isinstance(v, float) else f"# {k}: {v}", file=f)
            frust_df.to_csv(f, sep=" ", index=False, na_rep="nan", float_format="%.3f")


def write_5adens(
    mode_name,              # "mutational" or "configurational"
    pair_frustration_index, # [B, E] frustration indices
    ca_positions,           # [B, L, 3] Cα coordinates
    seq_mask,               # [B, L] valid residue mask (bool or 0/1)
    edge_index,             # [B, 2, E]
    frust_file,             # str or list[str] — base path without extension
    names=None,             # list of name strings (e.g. ["pdb"])
    residue_index=None,     # [B, L] PDB residue numbers, or None
    chain_index=None,       # [B, L] chain indices (int), or None
    chain_id_map=None,      # list[list[str]] — original chain letters per batch item
    radius=5.0,             # Å sphere radius (default matches R implementation)
    mode_sep="_",
):
    """Write {frust_file}.{name}_{mode_name}_5adens — FrustratometeR-compatible density file.

    For each Cα, counts contacts whose midpoint falls within `radius` Å and
    classifies them using the same classify_frustration logic as write_pair_frustration.
    """
    import torch
    import numpy as np
    from ..utils.weights import classify_frustration

    if pair_frustration_index.dim() == 3:
        pair_frustration_index = pair_frustration_index[..., 0]

    B = pair_frustration_index.shape[0]
    for i in range(B):
        frust_file_i = frust_file[i] if isinstance(frust_file, list) and i < len(frust_file) else frust_file
        name_i       = names[i]       if (names is not None and i < len(names)) else "pdb"
        out_path     = f"{frust_file_i}.{name_i}{mode_sep}{mode_name}_5adens"

        mask_i  = seq_mask[i].bool().cpu()                                 # [L]
        ca_full = ca_positions[i].float().cpu()                         # [L, 3]
        ca_cpu  = ca_full[mask_i]                                       # [L_valid, 3]
        L_valid = ca_cpu.shape[0]

        # Valid edges for this sample
        valid_e = (edge_index[i] >= 0).all(dim=0)                      # [E]
        ei_v    = edge_index[i][:, valid_e].cpu()                      # [2, E_valid]
        frust_v = pair_frustration_index[i, valid_e].float().cpu()     # [E_valid]

        # Midpoints of contacts; cdist gives [L_valid, E_valid] distances
        mid  = (ca_full[ei_v[0]] + ca_full[ei_v[1]]) / 2.0            # [E_valid, 3]
        in_r = torch.cdist(ca_cpu, mid) < radius                       # [L_valid, E_valid]

        # Edge classification — matches write_pair_frustration exactly
        frust_class  = classify_frustration(frust_v, 1)                # [E_valid]
        is_highly    = frust_class == 0
        is_neutral   = frust_class == 1
        is_minimally = frust_class == 2

        # Vectorised per-residue counts  [L_valid]
        total  = in_r.sum(dim=1)
        n_high = (in_r & is_highly.unsqueeze(0)).sum(dim=1)
        n_neut = (in_r & is_neutral.unsqueeze(0)).sum(dim=1)
        n_min  = (in_r & is_minimally.unsqueeze(0)).sum(dim=1)

        denom    = total.float().clamp(min=1)
        has_edge = total > 0
        rel_high = torch.where(has_edge, n_high.float() / denom, torch.zeros(L_valid))
        rel_neut = torch.where(has_edge, n_neut.float() / denom, torch.zeros(L_valid))
        rel_min  = torch.where(has_edge, n_min.float()  / denom, torch.zeros(L_valid))

        # Residue numbers and chain labels
        if residue_index is not None:
            res_nums = residue_index[i][mask_i].cpu().numpy()
        else:
            res_nums = (torch.where(mask_i)[0] + 1).numpy()

        if chain_index is not None:
            _cid_map_i = chain_id_map[i] if chain_id_map is not None else None
            chain_ids = np.array([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                                  for c in chain_index[i][mask_i].cpu().tolist()])
        else:
            chain_ids = np.full(L_valid, "A")

        total_np  = total.numpy()
        n_high_np = n_high.numpy()
        n_neut_np = n_neut.numpy()
        n_min_np  = n_min.numpy()
        rel_h_np  = rel_high.numpy()
        rel_n_np  = rel_neut.numpy()
        rel_m_np  = rel_min.numpy()

        header = ("Res ChainRes Total nHighlyFrst nNeutrallyFrst nMinimallyFrst"
                  " relHighlyFrustrated relNeutralFrustrated relMinimallyFrustrated")
        with open(out_path, "w") as f:
            f.write(header + "\n")
            for k in range(L_valid):
                f.write(
                    f"{res_nums[k]} {chain_ids[k]} {total_np[k]}"
                    f" {n_high_np[k]} {n_neut_np[k]} {n_min_np[k]}"
                    f" {rel_h_np[k]:.3f} {rel_n_np[k]:.3f} {rel_m_np[k]:.3f}\n"
                )
