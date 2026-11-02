from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..structure.load_structure import load_structure
from ..utils.utils import check_size
from .io import (
    parse_singleresidue_frustration,
    parse_pair_frustration,
)
from ..coefficients import Coefficients


def _find_done_dir(sf: Path) -> "Path | None":
    """Return the .done directory for *sf*, trying {stem}.done then {stem}_*.done."""
    d = sf.parent / f"{sf.stem}.done"
    if d.is_dir():
        return d
    for d in sorted(sf.parent.glob(f"{sf.stem}_*.done")):
        if d.is_dir():
            return d
    return None


def _find_frust_file(sf: Path, suffix: str, skip_flat: bool = False) -> "tuple[Path, list[Path]]":
    """Return (found_path, tried_paths).

    found_path exists if any candidate matched; otherwise it is the first
    candidate (is_file() will be False) so callers can still check.
    """
    tried: list[Path] = []

    flat = sf.parent / f"{sf.stem}.{suffix}"
    if not skip_flat:
        tried.append(flat)
        if flat.is_file():
            return flat, tried

    done = _find_done_dir(sf)
    if done is not None:
        # Try both the input stem (7ri3A02) and the .done folder stem (7ri3A02_A)
        done_stem = done.name[: -len(".done")]  # strip trailing ".done"
        for stem in dict.fromkeys([sf.stem, done_stem]):  # deduplicated, order preserved
            cand = done / "FrustrationData" / f"{stem}.{suffix}"
            tried.append(cand)
            if cand.is_file():
                return cand, tried
    else:
        tried.append(sf.parent / f"{sf.stem}.done" / "FrustrationData" / f"{sf.stem}.{suffix}")
        tried.append(sf.parent / f"{sf.stem}_*.done" / "FrustrationData" / f"{sf.stem}.{suffix}")

    cand = sf.parent / "FrustrationData" / f"{sf.stem}.{suffix}"
    tried.append(cand)
    if cand.is_file():
        return cand, tried

    return flat, tried


class FrustratometerDataset(Dataset):
    """This is the CATH S40 dataset with splits.
    All beta-lactamases have been removed from the trianing set and inserted into
    the test dataset.
    """
    def __init__(self, structures, coefficients, compare_to_original=False,
                 use_completed_pdb=False,
                 dtype: torch.dtype = torch.float32, backbone_method="auto",
                 chain_id=None, write_repair_files=False):
        """
        chain_id: str, list[str|None], or None.
          - None or "": load all chains for every structure.
          - Single str: apply to every structure.
          - List: positionally aligned with `structures`; discovered files
            (from directories / file-list expansion) inherit None.
        """
        if isinstance(structures, str):
            structures = [structures]
        structures = list(structures)

        # Normalise chain_id into a per-structure list aligned with `structures`.
        if chain_id is None or chain_id == () or chain_id == "":
            chain_ids = [None] * len(structures)
        elif isinstance(chain_id, str):
            chain_ids = [chain_id] * len(structures)
        else:
            chain_ids = list(chain_id)
            if len(chain_ids) == 1:
                chain_ids = chain_ids * len(structures)
            elif len(chain_ids) != len(structures):
                raise ValueError(
                    f"--chain has {len(chain_ids)} values but {len(structures)} structures were given; "
                    "provide one chain for all or one per structure."
                )

        self.coefficients = coefficients
        self.compare_to_original = compare_to_original
        self.use_completed_pdb = use_completed_pdb
        self.dtype = dtype
        self.backbone_method = backbone_method
        self.write_repair_files = write_repair_files

        # self.data stores (path, chain_id) pairs so each structure keeps its chain.
        self.data = []
        _missing_explicit = []  # level-0 paths that don't exist — collected for one clear error
        # Stack entries as (structure, chain_id, level)
        stack = list(zip(structures, chain_ids))
        while stack:
            entry = stack.pop()
            if len(entry) == 3:
                structure, c_id, level = entry
            else:
                structure, c_id = entry
                level = 0

            structure = Path(structure)

            if structure.is_file():
                if structure.suffix in ('.pdb', '.cif'):
                    self.data.append((structure, c_id))
                elif level == 0:
                    # Only treat non-PDB files as path lists when explicitly provided
                    # by the user. Files discovered via directory traversal (level > 0)
                    # are skipped — output files like .pdb_singleresidue can have
                    # millions of lines and would OOM if opened as text.
                    # Line format: "path" or "path,chain"
                    n_found = n_missing = 0
                    with structure.open() as fh:
                        for line in fh:
                            line = line.rstrip()
                            if not line:
                                continue
                            if ',' in line:
                                path_part, line_chain = line.split(',', 1)
                                p = Path(path_part.strip())
                                c = line_chain.strip() or None
                            else:
                                p = Path(line)
                                c = None
                            if not p.exists():
                                import warnings
                                warnings.warn(f"Path not found (skipping): {p}")
                                n_missing += 1
                            else:
                                stack.append((p, c, level + 1))
                                n_found += 1
                    if n_missing:
                        import warnings
                        warnings.warn(
                            f"{structure}: {n_missing} path(s) not found, {n_found} loaded."
                        )

            elif structure.is_dir():
                if level == 0:
                    # Top-level directory: never apply a chain filter to discovered files.
                    found = [
                        (f, None, level + 1) for f in sorted(structure.iterdir())
                        if f.is_file() and f.suffix in ('.pdb', '.cif')
                    ]
                    if not found:
                        import warnings
                        warnings.warn(f"Directory contains no .pdb/.cif files: {structure}")
                    stack += found
                else:
                    # Subdirectory discovered via traversal: recurse, no chain filter.
                    stack += [
                        (f, None, level + 1) for f in sorted(structure.iterdir())
                        if f.is_dir() or f.suffix in ('.pdb', '.cif')
                    ]

            elif level == 0:
                # Explicitly provided path that is neither a file nor a directory.
                _missing_explicit.append(structure)

        if _missing_explicit:
            paths = "\n  ".join(str(p) for p in _missing_explicit)
            raise FileNotFoundError(
                f"{len(_missing_explicit)} structure(s) not found:\n  {paths}"
            )

        self.max_len = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import time as _time
        original_file, chain_id = self.data[idx]
        structure_file = original_file            # may be replaced with completed PDB for loading

        trust_all_atoms = False
        if self.use_completed_pdb:
            sf = Path(original_file)
            frust_pdb, _ = _find_frust_file(sf, "pdb", skip_flat=True)
            if frust_pdb.is_file():
                structure_file = frust_pdb
                trust_all_atoms = True
            else:
                print(f"[use_completed_pdb] {sf.name} — completed PDB not found ({frust_pdb}), using original")
        elif self.backbone_method is None:
            trust_all_atoms = True

        # Always collect repair text into the sink; write_repair_files controls whether
        # those strings are also flushed to disk (see base._write_frust_outputs).
        _repair_sink = {}
        _t0 = _time.perf_counter()
        atom_positions, atom_positions_out, aatype_int, seq_mask, distance, edge_index, residue_index, residue_lookup, hetatm, chain_index, chain_id_map, res_labels = load_structure(
            structure_file,
            cfg=self.coefficients,
            rotate=False,
            multichain=True,
            dtype=self.dtype,
            trust_all_atoms=trust_all_atoms,
            backbone_method=self.backbone_method,
            modeller_verbose=len(self.data) <= 1,
            chain_id=chain_id,
            write_repair_files=self.write_repair_files,
            repair_sink=_repair_sink,
        )
        _load_time = _time.perf_counter() - _t0

        out = {
            # Always use the ORIGINAL input path for output naming so that
            # _make_done_paths writes to the correct location regardless of
            # whether use_completed_pdb redirected loading to FrustrationData/.
            "structure_file": str(original_file),
            "backbone_method": "none" if self.backbone_method is None else str(self.backbone_method),
            "atom_positions": atom_positions_out,
            "edge_index": edge_index,
            "seq_mask": seq_mask,
            "load_time_sec": torch.tensor(_load_time, dtype=torch.float32),
            "aa_type": aatype_int,
            "residue_index": residue_index,
            "res_labels": res_labels.tolist(),
            "distance": distance,
            "hetatm": hetatm,
        }
        if chain_index is not None:
            out["chain_index"] = chain_index
        if chain_id_map is not None:
            out["chain_id_map"] = chain_id_map.tolist()
        out["repair_log"] = _repair_sink.get("log", "")
        out["repair_map"] = _repair_sink.get("map", "")

        if self.compare_to_original:
            sf = Path(structure_file)
            singleresidue_file, _sr_tried = _find_frust_file(sf, "pdb_singleresidue")

            if not singleresidue_file.is_file():
                print(f"[compare_to_original] WARNING: singleresidue not found for {sf.name}; tried:", flush=True)
                for p in _sr_tried:
                    print(f"  {p}", flush=True)

            if singleresidue_file.is_file():
                sr_seq_mask = seq_mask & ~hetatm.bool()
                single_residue_frust, single_residue_class, single_residue_mask, extra_fields = parse_singleresidue_frustration(
                    str(singleresidue_file),
                    residue_lookup=residue_lookup,
                    seq_mask=sr_seq_mask,
                    max_len=None,
                    include_extra_fields=True)
                single_frust_mask = sr_seq_mask & single_residue_mask

                if single_residue_class is not None:
                    true_single_frustration_class = single_residue_class
                else:
                    single_threshold = self.coefficients.frustration_thresholds[0]
                    true_single_frustration_class = torch.searchsorted(single_threshold, single_residue_frust)
                if self.max_len is not None:
                    true_single_frustration_class = check_size(true_single_frustration_class, max_len=self.max_len)

                out.update({
                    "single_frust_true": single_residue_frust,
                    "single_frust_mask": single_frust_mask,
                    "true_single_frustration_class": true_single_frustration_class,
                    "true_single_native_energy": extra_fields["NativeEnergy"],
                    "true_single_decoy_energy": extra_fields["DecoyEnergy"],
                    "true_singe_sd_energy": extra_fields["SDEnergy"],
                })

            pair_threshold = self.coefficients.frustration_thresholds[1]

            mutational_file, _ = _find_frust_file(sf, "pdb_mutational")

            if mutational_file.is_file():
                mutational_frustration, mutational_welltype, mutational_state, mutational_frustration_mask, mutational_energy_fields = parse_pair_frustration(
                    str(mutational_file),
                    edge_index,
                    residue_lookup=residue_lookup,
                    seq_mask=seq_mask,
                    use_welltype=True,
                    max_len=self.max_len,
                    include_energy_fields=True,
                )
                true_mutational_frustration_class = torch.searchsorted(
                    pair_threshold,
                    mutational_frustration)

                out.update({
                    "mutational_frust_true": mutational_frustration,
                    "true_mutational_frustration_class": true_mutational_frustration_class,
                    "mutational_welltype": mutational_welltype,
                    "mutational_frust_mask": mutational_frustration_mask,
                    "mutational_native_energy": mutational_energy_fields["NativeEnergy"],
                    "mutational_decoy_energy": mutational_energy_fields["DecoyEnergy"],
                    "mutational_sd_energy": mutational_energy_fields["SDEnergy"],
                    "mutational_n_ref": torch.tensor([mutational_energy_fields.get("n_ref_total", 0)], dtype=torch.long),
                })

            configurational_file, _ = _find_frust_file(sf, "pdb_configurational")

            if configurational_file.is_file():
                configurational_frustration, configurational_welltype, configurational_state, configurational_frustration_mask, configurational_energy_fields = parse_pair_frustration(
                    str(configurational_file),
                    edge_index,
                    residue_lookup=residue_lookup,
                    seq_mask=seq_mask,
                    use_welltype=True,
                    max_len=self.max_len,
                    include_energy_fields=True,
                )
                true_configurational_frustration_class = torch.searchsorted(
                    pair_threshold,
                    configurational_frustration)

                out.update({
                    "configurational_frust_true": configurational_frustration,
                    "true_configurational_frustration_class": true_configurational_frustration_class,
                    "configurational_welltype": configurational_welltype,
                    "configurational_frust_mask": configurational_frustration_mask,
                    "configurational_native_energy": configurational_energy_fields["NativeEnergy"],
                    "configurational_decoy_energy": configurational_energy_fields["DecoyEnergy"],
                    "configurational_sd_energy": configurational_energy_fields["SDEnergy"],
                    "configurational_density_res1": configurational_energy_fields["DensityRes1"],
                    "configurational_density_res2": configurational_energy_fields["DensityRes2"],
                    "configurational_n_ref": torch.tensor([configurational_energy_fields.get("n_ref_total", 0)], dtype=torch.long),
                })
        return out


def pad_to_size(tensor, N):
    """
    Pad:
      (M, M)     → (N, N)
      (M, M, C)  → (N, N, C)
    """
    M = tensor.shape[0]
    diff = N - M

    if diff == 0:
        return tensor

    if tensor.dim() == 2:
        return F.pad(tensor, (0, diff, 0, diff))
    elif tensor.dim() == 3:
        return F.pad(tensor, (0, 0, 0, diff, 0, diff))
    else:
        raise RuntimeError(f"Unsupported tensor shape: {tensor.shape}")


def pad_batch(tensors):
    """Pad list of (Mi, Mi, C) → (B, N, N, C)."""
    B = len(tensors)
    N = max(t.shape[0] for t in tensors)
    device = tensors[0].device

    padded = []
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    for b, t in enumerate(tensors):
        M = t.shape[0]
        padded.append(pad_to_size(t, N))
        mask[b, :M] = True

    return torch.stack(padded)


def collate_frustratometer(batch):
    """Collate variable-length protein structures into a padded batch.

    Handles:
    - ``edge_index``: padded to max E with -1, shape [B, 2, E_max]
    - string fields (e.g. ``structure_file``): kept as plain lists
    - all other tensor fields: padded to max length with ``pad_sequence``
    """
    from torch.nn.utils.rnn import pad_sequence

    edge_idxs = []
    string_fields: dict[str, list] = {}
    list_fields: dict[str, list] = {}
    tensor_fields: dict[str, list] = {}
    

    for item in batch:
        for key, value in item.items():
            if key == "edge_index":
                edge_idxs.append(value.T)
            elif isinstance(value, str):
                string_fields.setdefault(key, []).append(value)
            elif isinstance(value, torch.Tensor):
                tensor_fields.setdefault(key, []).append(value)
            elif isinstance(value, list):
                list_fields.setdefault(key, []).append(value)
            else:
                raise TypeError(
                    f"collate: unsupported type for key={key!r}: "
                    f"{type(value).__name__}"
                )

    out = {}
    if edge_idxs:
        out["edge_index"] = pad_sequence(
            edge_idxs, batch_first=True, padding_value=-1
        ).permute(0, 2, 1)

    for key, items in string_fields.items():
        out[key] = items

    # Keep list fields as lists of lists
    for key, items in list_fields.items():
        out[key] = items

    for key, items in tensor_fields.items():
        if all(t.dim() == 0 for t in items):
            # Scalar tensors (e.g. load_time_sec): stack into a 1-D batch vector.
            out[key] = torch.stack(items)
            continue
        try:
            out[key] = pad_sequence(items, batch_first=True)
        except RuntimeError:
            if all(t.dim() >= 2 and t.shape[0] == t.shape[1] for t in items):
                out[key] = pad_batch(items)
            else:
                raise RuntimeError(
                    f"collate: cannot pad key={key!r}, shapes={[tuple(x.shape) for x in items]}"
                )

    return out
