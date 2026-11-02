#!/usr/bin/env python3
"""
Profile PyTorch Frustratometer wall time decomposed by the same cost categories
as the collaborator's benchmark figure, using the same PDB files.

  session:   import | runtime_init | gpu_warmup
  structure: reading | distance_map | awsem_model
  sequence:  frustration (per mode) [+ mutational scan overhead]

Optionally runs the original FrustratometeR (R package) on the same structures
for direct comparison (--run_orig).

Usage (run on server, from PyTorch Frustratometer/ root):
    python profile_timing.py
    python profile_timing.py --device cpu --output timing_cpu.png
    python profile_timing.py --modes singleresidue,mutational,configurational
    python profile_timing.py --pdb_dir /path/to/other/pdbs
    python profile_timing.py --repeats 5
    python profile_timing.py --run_orig --rscript /path/to/Rscript
"""

# ── measure import time BEFORE any imports ────────────────────────────────
import time as _t
_import_start = _t.perf_counter()

import sys, os, argparse, subprocess, tempfile, shutil, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_import_end = _t.perf_counter()
IMPORT_TIME = _import_end - _import_start

# ── runtime init ──────────────────────────────────────────────────────────
_init_start = _t.perf_counter()

from pytorch_frustratometer.structure.geometry import calculate_distance_matrix, get_CB_coords
from pytorch_frustratometer.structure.parse import (get_parsed_inputs, get_cif_parsed_inputs,
                                                    get_parsed_inputs_biotite,
                                                    get_cif_parsed_inputs_biotite)
from pytorch_frustratometer.structure.place_ideal_atoms import place_missing_cb, place_missing_o
from pytorch_frustratometer.utils.density import compute_residue_densities as _compute_residue_densities
from pytorch_frustratometer.structure.load_structure import _compute_sidechain_centroid as compute_sidechain_centroid
from pytorch_frustratometer.coefficients import Coefficients
from pytorch_frustratometer.algorithm.awsem import AWSEM, _register_mode_mixins
_register_mode_mixins()

_init_end = _t.perf_counter()
INIT_TIME = _init_end - _init_start

# ─────────────────────────────────────────────────────────────────────────
# Constants — match the collaborator's figure colours exactly
# ─────────────────────────────────────────────────────────────────────────
COLORS = {
    "import":                  "#3d3d3d",
    "runtime_init":            "#888888",
    "gpu_warmup":              "#cccccc",
    "reading":                 "#4878cf",
    "distance_map":            "#6acc65",
    "awsem_model":             "#e68117",
    "frustration":             "#d62728",
    "frustration_mut_scan":    "#9b59b6",
    "orig_total":              "#aec7e8",   # FrustratometeR bar (light blue)
    "orig_mut_scan":           "#c5b0d5",   # FrustratometeR mut-scan overhead
}
LABELS = {
    "import":                  "import (session)",
    "runtime_init":            "runtime init (session)",
    "gpu_warmup":              "GPU warmup (session)",
    "reading":                 "reading (structure)",
    "distance_map":            "distance map (structure)",
    "awsem_model":             "density (structure)",
    "frustration":             "frustration (per structure)",
    "frustration_mut_scan":    "mutational scan overhead (per structure)",
    "orig_total":              "FrustratometeR total",
    "orig_mut_scan":           "FrustratometeR mut-scan overhead",
}
ALL_MODES = ["configurational", "mutational", "singleresidue"]
MODE_LABELS = {"configurational": "Configurational",
               "mutational":      "Mutational",
               "singleresidue":   "Single-residue",
               "singleresidue_mut_scan": "Single-residue + mut. scan"}

DEFAULT_RSCRIPT = Path(
    "/gpfs/projects/bsc08/bsc504628/frustration/FrustrAI-Fold/r_env/bin/Rscript"
)
FRUST_R_LIB = "/gpfs/projects/bsc08/bsc504628/frustration/frustratometeR"

# Old-version binary names inside --old-version-binary-dir:
#   lmp_serial_schafer  → SeqDist 3
#   lmp_serial_hegler   → SeqDist 12
_OLD_VERSION_BINARY_NAMES = {3: "lmp_serial_schafer", 12: "lmp_serial_hegler"}

DEFAULT_PDB_DIR = (
    Path(__file__).parent / "length_test"
)

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _fmt_t(v, width=8):
    """Return a fixed-width time string; use scientific notation below 10 ms."""
    if not isinstance(v, (int, float)):
        return f"{'?':>{width}}"
    if np.isnan(v):
        return f"{'nan':>{width}}"
    return f"{v:{width}.3f}" if v >= 0.01 else f"{v:{width}.2e}"


def timeit(fn, device, reps=5):
    """Mean wall time over `reps` runs with GPU sync; one warmup discarded.
    Cache is cleared before each run to avoid cross-protein fragmentation."""
    clear_cache(device)
    fn()  # warmup
    sync(device)
    times = []
    for _ in range(reps):
        clear_cache(device)
        sync(device)
        t0 = _t.perf_counter()
        fn()
        sync(device)
        times.append(_t.perf_counter() - t0)
    return float(np.median(times))


def _to_serializable(obj):
    """Recursively convert numpy scalars / Path objects to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _pdb_first_chain(pdb_path):
    """Return the chain ID of the first ATOM record in a PDB file."""
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    return line[21]
    except OSError:
        pass
    return "A"


def _run_r_chunk(pdb_path_str, residues, rscript_str, frust_r_lib, timeout=86400, chain="A"):
    """Run mutate_res for a chunk of residues in one R subprocess.
    Returns wall time of the mutate_res loop only (excluding calculate_frustration overhead).
    """
    residues_r = ",".join(str(int(r)) for r in residues)
    r_expr = (
        f'suppressPackageStartupMessages(devtools::load_all("{frust_r_lib}",quiet=TRUE,export_all=FALSE));'
        f'pdb<-calculate_frustration(PdbFile="{pdb_path_str}",'
        f'Mode="singleresidue",SeqDist=3,Electrostatics_K=4.15,'
        f'ResultsDir=tempdir(),Graphics=FALSE);'
        f'chunk<-c({residues_r});'
        f't0<-proc.time();'
        f'for(res in chunk){{tryCatch(mutate_res(Pdb=pdb,Resno=res,Chain="{chain}"),error=function(e) NULL)}};'
        f't1<-proc.time();'
        f'cat("ELAPSED",as.numeric((t1-t0)["elapsed"]),"\\n")'
    )
    try:
        out = subprocess.check_output(
            [rscript_str, "-e", r_expr],
            stderr=subprocess.DEVNULL, text=True, timeout=timeout,
        )
        for line in out.splitlines():
            if line.startswith("ELAPSED"):
                return float(line.split()[1])
    except Exception:
        pass
    return float("nan")


def gpu_warmup(device):
    if device.type not in ("cuda", "mps"):
        return 0.0
    sync(device)
    t0 = _t.perf_counter()
    x = torch.zeros(256, 256, device=device)
    (x @ x).sum().item()
    sync(device)
    return _t.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────
# Per-protein profiling
# ─────────────────────────────────────────────────────────────────────────

def _parse_only(pdb_path, dtype=torch.float32):
    """Parse PDB/CIF + fill missing CB/O + compute sidechain centroid.
    Mirrors load_structure up to (but not including) calculate_distance_matrix.
    Returns tensors ready to pass to calculate_distance_matrix.
    """
    pdb_path = Path(pdb_path)
    parser_fn = get_cif_parsed_inputs_biotite if pdb_path.suffix.lower() == ".cif" else get_parsed_inputs_biotite
    atom_positions, atom_positions_raw, aatype_int, seq_mask, residue_index, hetatm, chain_index_raw, _chain_id_map, auth_seq_id_np, *_ = \
        parser_fn(str(pdb_path), round=False, dtype=dtype, save_raw_coordinates=True)

    is_gly     = torch.from_numpy(aatype_int == 5)
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

    aatype_int    = torch.from_numpy(aatype_int).long()
    seq_res_index = torch.from_numpy(residue_index).long()   # sequential, used internally
    chain_index   = torch.from_numpy(chain_index_raw).long()
    # Prefer auth_seq_id for output (matches PDB residue numbers); fall back to sequential
    residue_index = (torch.from_numpy(auth_seq_id_np).long()
                     if auth_seq_id_np is not None else seq_res_index)

    sc_centroid = compute_sidechain_centroid(atom_positions_raw).to(dtype)
    atom_positions_raw_out = torch.cat([
        atom_positions_raw[:, :5, :].to(dtype),
        sc_centroid.unsqueeze(1),
    ], dim=1)  # [L, 6, 3]
    atom_positions_out = torch.cat([
        atom_positions[:, :5, :].to(dtype),
        sc_centroid.unsqueeze(1),
    ], dim=1)  # [L, 6, 3]

    return atom_positions_out, atom_positions_raw_out, aatype_int, seq_mask, residue_index, hetatm, chain_index


_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _write_frust_outputs(stem, batch, awsem_models, out_dir, device):
    """Run each AWSEM model once and write space-separated CSV frustration files."""
    import pandas as pd

    frust_dir = Path(out_dir) / f"{stem}.done" / "FrustrationData"
    frust_dir.mkdir(parents=True, exist_ok=True)

    # Pull tensors off device once
    aa_cpu = batch["aa_type"][0].cpu()        # [L]
    ri_cpu = batch["residue_index"][0].cpu()  # [L]
    ci_cpu = batch["chain_index"][0].cpu()    # [L]
    ei_cpu = batch["edge_index"][0].cpu()     # [2, E]

    sr_mask    = (aa_cpu >= 0)
    edge_valid = (ei_cpu[0] >= 0) & (ei_cpu[1] >= 0)

    aa_sr = aa_cpu[sr_mask].numpy()
    ri_sr = ri_cpu[sr_mask].numpy()
    cr_sr = [_CHAIN_IDS[int(c) % len(_CHAIN_IDS)] for c in ci_cpu[sr_mask].tolist()]

    ei_v  = ei_cpu[:, edge_valid]             # [2, E_valid]
    ri_v1 = ri_cpu[ei_v[0]].numpy()
    ri_v2 = ri_cpu[ei_v[1]].numpy()
    cr_v1 = [_CHAIN_IDS[int(c) % len(_CHAIN_IDS)] for c in ci_cpu[ei_v[0]].tolist()]
    cr_v2 = [_CHAIN_IDS[int(c) % len(_CHAIN_IDS)] for c in ci_cpu[ei_v[1]].tolist()]
    aa_v1 = aa_cpu[ei_v[0]].numpy()
    aa_v2 = aa_cpu[ei_v[1]].numpy()

    with torch.no_grad():
        for mode, model in awsem_models.items():
            result = model(batch)

            if mode == "singleresidue":
                fi = result.get("singleresidue")
                if fi is None:
                    continue
                fi_vals = fi[0][sr_mask].cpu().float().numpy()
                df = pd.DataFrame({
                    "Res":       ri_sr,
                    "ChainRes":  cr_sr,
                    "AA":        aa_sr,
                    "FrstIndex": fi_vals,
                })
                out_path = frust_dir / f"{stem}.pdb_singleresidue"
                df.to_csv(out_path, sep=" ", index=False)
                print(f"  [write] {out_path}")

            elif mode in ("mutational", "configurational"):
                fi = result.get(mode)
                if fi is None:
                    continue
                fi_vals = fi[0][edge_valid].cpu().float().numpy()
                df = pd.DataFrame({
                    "Res1":      ri_v1, "Res2":      ri_v2,
                    "ChainRes1": cr_v1, "ChainRes2": cr_v2,
                    "AA1":       aa_v1, "AA2":       aa_v2,
                    "FrstIndex": fi_vals,
                })
                out_path = frust_dir / f"{stem}.pdb_{mode}"
                df.to_csv(out_path, sep=" ", index=False)
                print(f"  [write] {out_path}")


def profile_protein(pdb_path, device, coeffs, featurizer, awsem_models, reps,
                    frust_output_dir=None, run_dmfs=False):
    """Return dict of phase → seconds for one protein."""
    results = {}

    # ── reading: PDB parse + CB/O fill + sidechain centroid ──────────────
    def _read():
        _parse_only(pdb_path)
    results["reading"] = timeit(_read, device, reps=reps)

    # Parse once to get tensors for remaining phases
    atom_positions, atom_positions_raw, aatype_int, seq_mask, residue_index, hetatm, chain_index = \
        _parse_only(pdb_path)

    edge_mask          = seq_mask.bool() & ~hetatm.bool()
    atom_raw_b         = atom_positions_raw.unsqueeze(0)   # [1, L, 6, 3]
    aatype_b           = aatype_int.unsqueeze(0)           # [1, L]
    edge_mask_b        = edge_mask.unsqueeze(0)            # [1, L]

    # ── distance_map: calculate_distance_matrix only ─────────────────────
    def _distmap():
        calculate_distance_matrix(
            atom_raw_b, aatype_b, edge_mask_b,
            max_distance=float(coeffs.tert_frust_cutoff),
            chain_index=None,
        )
    results["distance_map"] = timeit(_distmap, device, reps=reps)

    # Compute once for subsequent phases
    distance, _, edge_index = calculate_distance_matrix(
        atom_raw_b, aatype_b, edge_mask_b,
        max_distance=float(coeffs.tert_frust_cutoff),
        chain_index=None,
    )

    # Move to device; build [B=1, ...] batch tensors
    def _to(t):
        return t.unsqueeze(0).to(device) if isinstance(t, torch.Tensor) else t

    atom_positions   = _to(atom_positions)
    aatype_int       = _to(aatype_b.squeeze(0))
    seq_mask         = _to(seq_mask)
    hetatm           = _to(hetatm)
    residue_index    = _to(residue_index)
    chain_index      = _to(chain_index)
    distance         = distance.to(device)
    edge_index_dev   = edge_index.to(device)  # [B, 2, E] — already batched from calculate_distance_matrix

    # ── AWSEM model (density computation, dense path — matches actual system) ─
    def _awsem():
        _compute_residue_densities(
            atom_positions, aatype_int, seq_mask, hetatm,
            residue_index, distance, coeffs, chain_index=chain_index,
            # No edge_index → dense [B,L,L] path, same as base.py / frustratometer.py
        )
    results["awsem_model"] = timeit(_awsem, device, reps=reps)

    # Compute density once for frustration phases
    rho, burial_rho, _ = _compute_residue_densities(
        atom_positions, aatype_int, seq_mask, hetatm,
        residue_index, distance, coeffs, chain_index=chain_index,
    )

    batch = {
        "aa_type":         aatype_int,
        "seq_mask":        seq_mask,
        "atom_positions":  atom_positions,
        "hetatm":          hetatm,
        "residue_index":   residue_index,
        "chain_index":     chain_index,
        "distance":        distance,
        "edge_index":      edge_index_dev,
        "residue_density": rho,
        "burial_density":  burial_rho,
    }

    # ── frustration per structure (one sequence, one structure) ──────────
    for mode, model in awsem_models.items():
        def _frust(m=model, b=batch):
            m(b)
        results[f"frustration_{mode}"] = timeit(_frust, device, reps=reps)

    # ── DMFS overhead (singleresidue + run_dmfs) ──────────────────────────
    if run_dmfs and "singleresidue" in awsem_models:
        c_mut = Coefficients(mode="singleresidue", run_dmfs=True)
        model_mut = AWSEM(c_mut).to(device)
        model_mut.eval()
        def _frust_scan(m=model_mut, b=batch):
            m(b)
        t_scan = timeit(_frust_scan, device, reps=reps)
        results["frustration_singleresidue_mut_scan"] = t_scan
        results["frustration_mut_scan"] = t_scan - results["frustration_singleresidue"]

    # ── write frustration output files if requested ───────────────────────
    if frust_output_dir is not None:
        stem = Path(pdb_path).stem
        _write_frust_outputs(stem, batch, awsem_models, frust_output_dir, device)

    return results


def _cif_to_tmp_pdb(cif_path):
    """Convert a CIF file to a temporary PDB, remapping chain IDs and residue numbers
    so the file is compatible with FrustratometeR (bio3d).

    Returns (tmp_pdb_path_str, n_chains) or raises ValueError if conversion fails.
    The caller is responsible for deleting the temp file.
    """
    import string, tempfile
    import biotite.structure as struc
    import biotite.structure.io as strucio

    atoms = strucio.load_structure(str(cif_path))
    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]

    # Keep only protein ATOM records (FrustratometeR ignores HETATM/water)
    atoms = atoms[~atoms.hetero]

    unique_chains = list(dict.fromkeys(atoms.chain_id))
    n_chains = len(unique_chains)

    # PDB chain IDs must be a single alphanumeric character (62 max)
    valid_ids = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    if n_chains > len(valid_ids):
        return None, n_chains  # caller will record as too_large

    # Remap chain IDs: use existing IDs when they're already single valid chars,
    # otherwise reassign sequentially.
    needs_remap = any(len(c) != 1 or c not in valid_ids for c in unique_chains)
    if needs_remap:
        chain_map = {old: valid_ids[i] for i, old in enumerate(unique_chains)}
        new_ids = np.array([chain_map[c] for c in atoms.chain_id], dtype=atoms.chain_id.dtype)
        atoms.chain_id[:] = new_ids
        print(f"    [cif→pdb] remapped {n_chains} chain IDs", flush=True)

    # PDB residue numbers are 4-digit integers (max 9999 in standard PDB).
    # Renumber per-chain if any residue number exceeds 9999.
    max_res = int(atoms.res_id.max()) if len(atoms) > 0 else 0
    if max_res > 9999:
        new_res_id = atoms.res_id.copy()
        for chain in np.unique(atoms.chain_id):
            mask = atoms.chain_id == chain
            ids = atoms.res_id[mask]
            _, inv = np.unique(ids, return_inverse=True)
            new_res_id[mask] = (inv + 1).astype(atoms.res_id.dtype)
        atoms.res_id = new_res_id
        print(f"    [cif→pdb] renumbered residues (max was {max_res})", flush=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp.close()
    strucio.save_structure(tmp.name, atoms)
    return tmp.name, n_chains


def profile_orig_protein(pdb_path, rscript, modes, reps, mut_scan=False,
                         workers=1, residue_numbers=None, on_partial=None,
                         existing_results=None):
    """Time original FrustratometeR on one protein via Rscript subprocess.

    Returns dict:  mode → seconds (median over reps)
    Also returns 'mut_scan' → seconds if mut_scan=True.
    When workers>1 and residue_numbers is provided, mutate_res calls are
    distributed across parallel Rscript subprocesses and wall clock is reported.

    on_partial: optional callable(results_so_far) called after each mode and
    after each mut_scan rep so partial results can be checkpointed mid-run.

    existing_results: dict of already-computed values (from a prior partial run).
    Modes whose key is already non-NaN are skipped. Mut_scan resumes from
    the rep count stored in '_orig_mut_scan_reps' list.
    """
    pdb_path = Path(pdb_path).resolve()
    results  = dict(existing_results) if existing_results else {}

    # FrustratometeR only reads PDB; convert CIF files to a temp PDB first.
    _tmp_pdb = None
    if pdb_path.suffix.lower() == ".cif":
        try:
            _tmp_pdb, n_ch = _cif_to_tmp_pdb(pdb_path)
            if _tmp_pdb is None:
                msg = f"too large ({n_ch} chains > 62 PDB limit)"
                print(f"    [cif→pdb] {msg} — skipping FrustratometeR", flush=True)
                results["orig_skipped"] = msg
                return results
            print(f"    [cif→pdb] converted {pdb_path.name} ({n_ch} chain(s)) → {_tmp_pdb}", flush=True)
            pdb_path = Path(_tmp_pdb)
        except Exception as conv_exc:
            print(f"    [cif→pdb] conversion failed: {conv_exc} — skipping FrustratometeR", flush=True)
            return results  # nothing to run

    try:
        _profile_orig_protein_body(pdb_path, rscript, modes, reps, mut_scan,
                                   workers, residue_numbers, on_partial, results)
    finally:
        if _tmp_pdb is not None:
            import os as _os
            try:
                _os.unlink(_tmp_pdb)
            except OSError:
                pass

    return results


def _profile_orig_protein_body(pdb_path, rscript, modes, reps, mut_scan,
                                workers, residue_numbers, on_partial, results):
    """Inner body of profile_orig_protein — operates on a guaranteed PDB path."""
    def _is_nan(v):
        return isinstance(v, float) and v != v

    r_mode_map = {
        "singleresidue":   "singleresidue",
        "mutational":      "mutational",
        "configurational": "configurational",
    }

    for mode in modes:
        r_mode = r_mode_map.get(mode)
        if r_mode is None:
            continue

        key = f"orig_{mode}"
        if key in results and not _is_nan(results[key]):
            print(f"    [orig {mode}] already done ({results[key]:.1f}s), skipping", flush=True)
            continue

        r_expr = (
            f'devtools::load_all("{FRUST_R_LIB}");'
            f't0<-proc.time();'
            f'calculate_frustration(PdbFile="{pdb_path}",'
            f'Mode="{r_mode}",SeqDist=3,Electrostatics_K=4.15,'
            f'ResultsDir=tempdir(),Graphics=FALSE);'
            f't1<-proc.time();'
            f'cat("ELAPSED",as.numeric((t1-t0)["elapsed"]),"\\n")'
        )

        times = []
        for _ in range(reps):
            try:
                out = subprocess.check_output(
                    [str(rscript), "-e", r_expr],
                    stderr=subprocess.DEVNULL, text=True, timeout=43200,
                )
                for line in out.splitlines():
                    if line.startswith("ELAPSED"):
                        times.append(float(line.split()[1]))
                        break
            except Exception as exc:
                print(f"    [orig {mode}] error: {exc}", flush=True)
        results[key] = float(np.median(times)) if times else float("nan")
        if on_partial is not None:
            on_partial(dict(results))

    # Checkpoint all mode timings before starting the long mut_scan
    if mut_scan and on_partial is not None:
        on_partial(dict(results))

    # Mutational scanning: singleresidue + mutate_res for every residue
    if mut_scan and "singleresidue" in modes:
        t_sr = results.get("orig_singleresidue", float("nan"))

        if workers > 1 and residue_numbers:
            # Resume from previously completed reps if available
            times_wall = list(results.get("_orig_mut_scan_reps", []))
            reps_done  = len(times_wall)
            reps_left  = reps - reps_done

            n_workers = min(workers, len(residue_numbers))
            chunks = [residue_numbers[i::n_workers] for i in range(n_workers)]
            chunks = [c for c in chunks if c]

            if reps_left <= 0:
                print(f"    [orig mut_scan] all {reps} reps already done, skipping", flush=True)
            else:
                _chain = _pdb_first_chain(pdb_path)
                print(f"    [orig mut_scan] {len(residue_numbers)} residues across {len(chunks)} workers"
                      + (f" (resuming from rep {reps_done+1}/{reps})" if reps_done else "")
                      + f" chain={_chain!r}",
                      flush=True)

                for rep in range(reps_done, reps):
                    t0 = _t.perf_counter()
                    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                        futures = [pool.submit(_run_r_chunk, str(pdb_path), chunk,
                                               str(rscript), FRUST_R_LIB, chain=_chain)
                                   for chunk in chunks]
                        chunk_times = [f.result() for f in futures]
                    wall = _t.perf_counter() - t0
                    all_nan = all(_is_nan(ct) for ct in chunk_times)
                    print(f"      rep {rep+1}/{reps}  wall={wall:.1f}s  "
                          f"chunks={[f'{ct:.1f}s' if not _is_nan(ct) else 'timeout' for ct in chunk_times]}",
                          flush=True)
                    if all_nan:
                        print(f"      [mut_scan] all chunks timed out/failed — aborting remaining reps",
                              flush=True)
                        results["orig_singleresidue_mut_scan"] = float("nan")
                        results["orig_mut_scan"]               = float("nan")
                        results["orig_mut_scan_workers"]       = len(chunks)
                        if on_partial is not None:
                            on_partial(dict(results))
                        break
                    times_wall.append(wall)
                    # Checkpoint after every rep — each one takes ~30 min
                    t_so_far = float(np.median(times_wall))
                    results["_orig_mut_scan_reps"]         = times_wall[:]
                    results["orig_singleresidue_mut_scan"] = t_so_far
                    results["orig_mut_scan"] = (t_so_far - t_sr
                                                if not _is_nan(t_so_far) and not _is_nan(t_sr)
                                                else float("nan"))
                    results["orig_mut_scan_workers"] = len(chunks)
                    if on_partial is not None:
                        on_partial(dict(results))

        else:
            # Sequential path — resume from completed reps
            times_scan = list(results.get("_orig_mut_scan_reps", []))
            reps_done  = len(times_scan)
            reps_left  = reps - reps_done

            if reps_left <= 0:
                print(f"    [orig mut_scan] all {reps} reps already done, skipping", flush=True)
            else:
                if reps_done:
                    print(f"    [orig mut_scan] resuming from rep {reps_done+1}/{reps}", flush=True)
                _chain = _pdb_first_chain(pdb_path)
                r_expr_scan = (
                    f'devtools::load_all("{FRUST_R_LIB}");'
                    f't0<-proc.time();'
                    f'pdb<-calculate_frustration(PdbFile="{pdb_path}",'
                    f'Mode="singleresidue",SeqDist=3,Electrostatics_K=4.15,'
                    f'ResultsDir=tempdir(),Graphics=FALSE);'
                    f'residues<-unique(pdb$atom$resno);'
                    f'for(res in residues){{tryCatch(mutate_res(Pdb=pdb,Resno=res,Chain="{_chain}"),'
                    f'error=function(e) NULL)}};'
                    f't1<-proc.time();'
                    f'cat("ELAPSED",as.numeric((t1-t0)["elapsed"]),"\\n")'
                )
                for _ in range(reps_left):
                    try:
                        out = subprocess.check_output(
                            [str(rscript), "-e", r_expr_scan],
                            stderr=subprocess.DEVNULL, text=True, timeout=86400,
                        )
                        for line in out.splitlines():
                            if line.startswith("ELAPSED"):
                                times_scan.append(float(line.split()[1]))
                                break
                    except Exception as exc:
                        print(f"    [orig mut_scan] error: {exc}", flush=True)
                    if times_scan:
                        t_so_far = float(np.median(times_scan))
                        results["_orig_mut_scan_reps"]         = times_scan[:]
                        results["orig_singleresidue_mut_scan"] = t_so_far
                        results["orig_mut_scan"] = (t_so_far - t_sr
                                                    if not _is_nan(t_so_far) and not _is_nan(t_sr)
                                                    else float("nan"))
                        if on_partial is not None:
                            on_partial(dict(results))


def _make_batch(base_batch, B):
    """Repeat a B=1 batch dict into B genuinely independent copies.

    expand() with stride=0 lets the GPU process one copy regardless of B, giving
    unrealistically flat timing.  repeat() + contiguous() forces B real copies so
    memory bandwidth and compute both scale correctly with batch size.
    """
    return {k: (v.repeat(B, *[1] * (v.dim() - 1)).contiguous()
                if isinstance(v, torch.Tensor) else v)
            for k, v in base_batch.items()}


def _is_oom(exc):
    return "out of memory" in str(exc).lower() or "alloc" in str(exc).lower()


def profile_batch_sweep(pdb_path, device, coeffs, featurizer, awsem_models,
                         reps, batch_sizes):
    """
    Sweep over batch sizes for one protein.

    Returns dict with keys:
      "density"    → {B: s/seq}   dense compute_residue_densities
      "<mode>"     → {B: s/seq}   frustration scoring (density pre-computed)
    OOM records nan and stops sweeping that phase at larger sizes.
    """
    atom_positions, atom_positions_raw, aatype_int, seq_mask, residue_index, hetatm, chain_index = \
        _parse_only(pdb_path)

    edge_mask   = seq_mask.bool() & ~hetatm.bool()
    atom_raw_b  = atom_positions_raw.unsqueeze(0)
    aatype_b    = aatype_int.unsqueeze(0)
    edge_mask_b = edge_mask.unsqueeze(0)

    distance, _, edge_index = calculate_distance_matrix(
        atom_raw_b, aatype_b, edge_mask_b,
        max_distance=float(coeffs.tert_frust_cutoff),
        chain_index=None,
    )

    def _to1(t):
        return t.unsqueeze(0).to(device) if isinstance(t, torch.Tensor) else t

    # Base tensors without density (density is swept separately)
    base = {
        "aa_type":        _to1(aatype_b.squeeze(0)),
        "seq_mask":       _to1(seq_mask),
        "atom_positions": _to1(atom_positions),
        "hetatm":         _to1(hetatm),
        "residue_index":  _to1(residue_index),
        "chain_index":    _to1(chain_index),
        "distance":       distance.to(device),
        "edge_index":     edge_index.to(device),
    }

    # Pre-compute density at B=1 (for frustration timing — density not double-counted)
    with torch.no_grad():
        rho_1, burial_1, _ = _compute_residue_densities(
            base["atom_positions"], base["aa_type"], base["seq_mask"], base["hetatm"],
            base["residue_index"], base["distance"], coeffs, chain_index=base["chain_index"],
        )
    base_with_density = {**base, "residue_density": rho_1, "burial_density": burial_1}

    density_results = {}
    frust_results   = {mode: {} for mode in awsem_models}
    oom_density     = False
    oom_mode        = {mode: False for mode in awsem_models}

    for B in batch_sizes:
        # ── density sweep ────────────────────────────────────────────────
        batched_nd = _make_batch(base, B)
        if oom_density:
            density_results[B] = float("nan")
        else:
            try:
                def _dens(b=batched_nd):
                    _compute_residue_densities(
                        b["atom_positions"], b["aa_type"], b["seq_mask"], b["hetatm"],
                        b["residue_index"], b["distance"], coeffs, chain_index=b["chain_index"],
                    )
                density_results[B] = timeit(_dens, device, reps=reps) / B
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if _is_oom(exc):
                    print(f"    OOM at B={B} for density", flush=True)
                    density_results[B] = float("nan")
                    oom_density = True
                    clear_cache(device)
                else:
                    raise

        # ── frustration sweep (uses pre-computed B=1 density, repeated) ─
        batched = _make_batch(base_with_density, B)
        for mode, model in awsem_models.items():
            if oom_mode[mode]:
                frust_results[mode][B] = float("nan")
                continue
            try:
                def _frust(m=model, b=batched):
                    m(b)
                frust_results[mode][B] = timeit(_frust, device, reps=reps) / B
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if _is_oom(exc):
                    print(f"    OOM at B={B} for {mode}", flush=True)
                    frust_results[mode][B] = float("nan")
                    oom_mode[mode] = True
                    clear_cache(device)
                else:
                    raise

    return {"density": density_results, **frust_results}


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────

def plot_results(proteins, all_results, session_times, modes, output_path, device_label,
                 orig_results=None):
    """Stacked bar chart — one column per protein, one panel per mode.

    When orig_results is provided, a second (FrustratometeR) bar is drawn next
    to each PyTorch Frustratometer bar for direct comparison.
    """
    lengths = [p["length"] for p in proteins]
    labels  = [str(L) for L in lengths]
    has_orig = orig_results is not None

    n_modes = len(modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5), sharey=True)
    if n_modes == 1:
        axes = [axes]

    fig.suptitle(
        f"Cold-start wall time decomposed by cost recurrence\n"
        f"(per session, per structure, per sequence)  [{device_label}]",
        fontsize=11,
    )

    pt_keys = ["import", "runtime_init", "gpu_warmup",
               "reading", "distance_map", "awsem_model", "frustration",
               "frustration_mut_scan"]

    n = len(proteins)
    if has_orig:
        width  = 0.40
        x_pt   = np.arange(n) - width / 2
        x_orig = np.arange(n) + width / 2
    else:
        width = 0.65
        x_pt  = np.arange(n)

    for ax, mode in zip(axes, modes):
        # ── PyTorch Frustratometer stacked bars ───────────────────────────
        bottoms = np.zeros(n)
        for key in pt_keys:
            if key in session_times:
                vals = np.full(n, session_times.get(key, 0.0))
            elif key == "frustration":
                vals = np.array([all_results[p["name"]].get(f"frustration_{mode}", 0.0)
                                 for p in proteins])
            elif key == "frustration_mut_scan":
                if mode != "singleresidue":
                    continue
                vals = np.array([all_results[p["name"]].get("frustration_mut_scan", 0.0)
                                 for p in proteins])
            else:
                vals = np.array([all_results[p["name"]].get(key, 0.0) for p in proteins])

            if key == "gpu_warmup" and session_times.get("gpu_warmup", 0) == 0:
                continue
            ax.bar(x_pt, vals, width, bottom=bottoms,
                   color=COLORS[key], label=LABELS[key])
            bottoms += vals

        # ── FrustratometeR comparison bars ───────────────────────────────
        if has_orig:
            orig_sr  = np.array([orig_results.get(p["name"], {}).get(f"orig_{mode}", float("nan"))
                                  for p in proteins])
            orig_ovhd = np.array([orig_results.get(p["name"], {}).get("orig_mut_scan", 0.0)
                                   for p in proteins]) if mode == "singleresidue" else np.zeros(n)
            orig_ovhd = np.where(np.isnan(orig_ovhd), 0.0, orig_ovhd)
            valid = ~np.isnan(orig_sr)
            if valid.any():
                ax.bar(x_orig[valid], orig_sr[valid], width,
                       color=COLORS["orig_total"], label=LABELS["orig_total"],
                       edgecolor="black", linewidth=0.5)
                if mode == "singleresidue" and orig_ovhd[valid].any():
                    ax.bar(x_orig[valid], orig_ovhd[valid], width,
                           bottom=orig_sr[valid],
                           color=COLORS["orig_mut_scan"], label=LABELS["orig_mut_scan"],
                           edgecolor="black", linewidth=0.5)

        title = MODE_LABELS.get(mode, mode)
        if mode == "singleresidue" and has_orig:
            title += "\n(purple = mut-scan overhead)"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Protein length (residues)", fontsize=9)
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0)

        if has_orig:
            ax.text(0.02, 0.98, "Left = PyTorch Frustratometer\nRight = FrustratometeR",
                    transform=ax.transAxes, fontsize=7, va="top", color="#444444")

    axes[0].set_ylabel("Wall time (s)", fontsize=9)

    legend_keys = [k for k in pt_keys
                   if not (k == "gpu_warmup" and session_times.get("gpu_warmup", 0) == 0)]
    if has_orig:
        legend_keys += ["orig_total", "orig_mut_scan"]
    handles = [mpatches.Patch(color=COLORS[k], label=LABELS[k]) for k in legend_keys]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0), frameon=False)

    plt.tight_layout(rect=[0, 0.12, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def _plot_sweep_panel(ax, proteins, sweep_results, key, title, all_bs, colors):
    """Draw one batch-sweep panel (one key = one mode or density)."""
    for p, color in zip(proteins, colors):
        phase = sweep_results.get(p["name"], {}).get(key, {})
        if not phase:
            continue
        bs   = sorted(phase.keys())
        vals = [phase[b] for b in bs]

        valid_bs   = [b for b, v in zip(bs, vals) if not np.isnan(v)]
        valid_vals = [v for v in vals if not np.isnan(v)]
        oom_bs     = [b for b, v in zip(bs, vals) if np.isnan(v)]

        if valid_bs:
            ax.plot(valid_bs, valid_vals, "o-", color=color, label=f"L={p['length']}")
        if oom_bs and valid_bs:
            ax.axvline(valid_bs[-1], color=color, linestyle=":", alpha=0.5, linewidth=1)
            ax.plot(valid_bs[-1], valid_vals[-1], "x", color=color,
                    markersize=10, markeredgewidth=2,
                    label=f"L={p['length']} OOM@B={oom_bs[0]}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(all_bs)
    ax.set_xticklabels([str(b) for b in all_bs], fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Batch size", fontsize=9)
    ax.set_ylabel("Time per sequence (s/seq)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)


def plot_batch_sweep(proteins, sweep_results, modes, output_path, device_label):
    """
    Line chart: time/seq vs batch size, one line per protein.

    Panels: density + one per mode.
    X axis is batch size (log2 scale).  OOM boundary marked with dotted line + x.
    """
    panel_keys   = ["density"] + modes
    panel_titles = ["Density (dense path)"] + [MODE_LABELS.get(m, m) for m in modes]

    n_panels = len(panel_keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"Time per sequence vs batch size  [{device_label}]", fontsize=11)

    cmap    = plt.get_cmap("tab10")
    colors  = [cmap(i) for i in range(len(proteins))]
    all_bs  = sorted({b for p in proteins
                      for d in sweep_results.get(p["name"], {}).values()
                      for b in d})

    for ax, key, title in zip(axes, panel_keys, panel_titles):
        _plot_sweep_panel(ax, proteins, sweep_results, key, title, all_bs, colors)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_multi_hardware(proteins, hw_groups, hw_session, hw_sweep,
                        modes, output_path, orig_results=None):
    """Line plot: wall time vs protein length, one line per hardware config.

    Each hw_group provides PyTorch timings; orig_results provides FrustratometeR
    as a single comparison series (hardware-agnostic — always CPU serial).
    """
    lengths = [p["length"] for p in proteins]
    n_modes = len(modes)

    hw_labels = sorted(hw_groups.keys())
    cmap      = plt.get_cmap("tab10")
    hw_colors = {hw: cmap(i) for i, hw in enumerate(hw_labels)}

    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5), sharey=False)
    if n_modes == 1:
        axes = [axes]

    fig.suptitle("Wall time vs protein length — hardware comparison", fontsize=11)

    for ax, mode in zip(axes, modes):
        # ── PyTorch lines, one per hardware ──────────────────────────────
        for hw in hw_labels:
            results = hw_groups[hw]
            session = hw_session.get(hw, {})
            vals = []
            for p in proteins:
                r = results.get(p["name"], {})
                s = session
                total = (s.get("import", 0) + s.get("runtime_init", 0) +
                         s.get("gpu_warmup", 0) +
                         r.get("reading", 0) + r.get("distance_map", 0) +
                         r.get("awsem_model", 0) +
                         r.get(f"frustration_{mode}", 0))
                if mode == "singleresidue":
                    total += r.get("frustration_mut_scan", 0)
                vals.append(total if total > 0 else float("nan"))
            valid = [(l, v) for l, v in zip(lengths, vals) if not np.isnan(v)]
            if valid:
                ls, vs = zip(*valid)
                ax.plot(ls, vs, "o-", color=hw_colors[hw], label=f"PyTorch ({hw})", linewidth=1.5)

        # ── FrustratometeR serial line ────────────────────────────────────
        if orig_results:
            orig_key = f"orig_{mode}"
            mut_key  = "orig_mut_scan"
            orig_vals = []
            for p in proteins:
                r = orig_results.get(p["name"], {})
                v = r.get(orig_key, float("nan"))
                if mode == "singleresidue" and not np.isnan(r.get(mut_key, float("nan"))):
                    v += r.get(mut_key, 0)
                orig_vals.append(v)
            valid_orig = [(l, v) for l, v in zip(lengths, orig_vals) if not np.isnan(v)]
            if valid_orig:
                ls, vs = zip(*valid_orig)
                ax.plot(ls, vs, "s--", color="crimson", label="FrustratometeR (112 CPU)", linewidth=1.5)

        title = MODE_LABELS.get(mode, mode)
        if mode == "singleresidue":
            title += "\n(incl. mut. scan)"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Protein length (residues)", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Wall time (s)", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", default=DEFAULT_PDB_DIR,
                        help="Directory with structure files (.pdb or .cif)")
    parser.add_argument("--pdb_list", default=None,
                        help="Text file listing filenames (one per line); "
                             "defaults to pdb_files.txt; use cif_files.txt for CIF timing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modes", default="all",
                        help="Comma-separated modes or 'all'")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch_sizes", default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes to sweep (e.g. '1,2,4,8,16,32,64')")
    parser.add_argument("--output", default="timing_pytorch.png")
    parser.add_argument("--run_orig", action="store_true",
                        help="Also time original FrustratometeR (R) for comparison")
    parser.add_argument("--rscript", default=str(DEFAULT_RSCRIPT),
                        help="Path to Rscript binary")
    parser.add_argument("--orig_mut_scan", action="store_true",
                        help="Also time FrustratometeR mutational scan (slow; one call per residue)")
    parser.add_argument("--orig_workers", type=int, default=1,
                        help="Number of parallel Rscript workers for FrustratometeR mut scan")
    parser.add_argument("--protein_index", type=int, default=None,
                        help="Run only this protein (0-based index); for SLURM array jobs")
    parser.add_argument("--save_json", default=None,
                        help="Save results to this JSON file (for later --combine)")
    parser.add_argument("--combine", default=None,
                        help="Glob pattern for JSON files to combine into plots (e.g. 'logs/timing_results_*.json')")
    parser.add_argument("--hardware_label", default=None,
                        help="Hardware label for this run (e.g. 'MN5 A100', 'MN5 CPU', 'MacBook M2')")
    parser.add_argument("--rerun_orig", action="store_true",
                        help="Only (re)run FrustratometeR timing on an existing --save_json file; "
                             "skips all PyTorch profiling")
    parser.add_argument("--force", action="store_true",
                        help="With --rerun_orig: rerun even proteins that already have orig_results")
    parser.add_argument("--write_frustration", action="store_true", default=True,
                        help="Write frustration CSV files after profiling each protein (default: on)")
    parser.add_argument("--no_write_frustration", action="store_false", dest="write_frustration",
                        help="Disable frustration file writing")
    parser.add_argument("--dmfs", action="store_true", default=True,
                        help="Time DMFS overhead (singleresidue + full mutation table). Default: on.")
    parser.add_argument("--no_dmfs", action="store_false", dest="dmfs",
                        help="Disable DMFS overhead timing.")
    parser.add_argument("--frust_output_dir", default=None,
                        help="Output directory for frustration files when --write_frustration is set "
                             "(default: same directory as --output, with _frust suffix)")
    args = parser.parse_args()

    # ── combine mode: load per-protein JSONs and plot ─────────────────────
    if args.combine:
        import glob as _glob
        json_files = sorted(_glob.glob(args.combine))
        if not json_files:
            sys.exit(f"No JSON files matched: {args.combine}")

        # Group JSONs by hardware_label; orig_results taken from whichever has them
        from collections import defaultdict
        hw_groups   = defaultdict(dict)   # hw_label -> {protein_name: results}
        hw_sweep    = defaultdict(dict)
        hw_session  = {}
        orig_results_map = {}
        all_proteins_set = {}

        for jf in json_files:
            data  = json.loads(Path(jf).read_text())
            p     = data["protein"]
            hw    = data.get("hardware_label", "unknown")
            name  = p["name"]
            all_proteins_set[name] = p
            hw_groups[hw][name]  = data["results"]
            hw_sweep[hw][name]   = data.get("sweep_results", {})
            if hw not in hw_session:
                hw_session[hw] = data.get("session_times", {})
            if data.get("orig_results"):
                orig_results_map[name] = data["orig_results"]

        proteins = sorted(all_proteins_set.values(), key=lambda p: p["length"])
        modes    = ALL_MODES if args.modes == "all" else args.modes.split(",")
        orig_results = orig_results_map if orig_results_map else None

        # One plot per hardware group, all on the same figure
        plot_multi_hardware(proteins, hw_groups, hw_session, hw_sweep,
                            modes, args.output, orig_results=orig_results)
        return

    # ── rerun_orig mode: update orig_results in an existing JSON ─────────────
    if args.rerun_orig:
        if not args.save_json:
            sys.exit("--rerun_orig requires --save_json to specify the JSON file to update")
        save_path = Path(args.save_json)
        if not save_path.exists():
            sys.exit(f"JSON file not found: {save_path}")

        data  = json.loads(save_path.read_text())
        modes = ALL_MODES if args.modes == "all" else args.modes.split(",")

        # Normalize to a list of protein dicts
        p_raw    = data["protein"]
        proteins = p_raw if isinstance(p_raw, list) else [p_raw]

        # Resolve PDB paths: prefer stored path, fall back to args.pdb_dir
        pdb_dir = Path(args.pdb_dir).resolve() if args.pdb_dir else None
        for p in proteins:
            stored = Path(p.get("path", ""))
            if stored.exists():
                p["path"] = str(stored)
            elif pdb_dir is not None:
                candidate = pdb_dir / p["name"]
                if candidate.exists():
                    p["path"] = str(candidate)

        # Normalize orig_results to {protein_name: {orig_key: value}}
        raw_orig = data.get("orig_results") or {}
        if isinstance(raw_orig, dict) and raw_orig:
            first_key = next(iter(raw_orig))
            if first_key.startswith("orig_"):
                # Flat single-protein format
                existing_orig = {proteins[0]["name"]: raw_orig}
            else:
                existing_orig = dict(raw_orig)
        else:
            existing_orig = {}

        rscript = Path(args.rscript)
        if not rscript.exists():
            sys.exit(f"Rscript not found at {rscript}")

        print(f"--rerun_orig: {save_path}")
        print(f"Modes:    {modes}")
        print(f"Proteins: {[p['name'] for p in proteins]}", flush=True)

        def _not_nan(v):
            return not (isinstance(v, float) and v != v)

        # All keys that must be present and non-NaN for a protein to be considered done
        _mode_keys = [f"orig_{m}" for m in modes
                      if m in ("singleresidue", "mutational", "configurational")]
        _expected_keys = _mode_keys + (["orig_singleresidue_mut_scan"] if args.orig_mut_scan else [])

        any_ran = False
        for p in proteins:
            name     = p["name"]
            existing = existing_orig.get(name) or {}
            # Done when intentionally skipped, or every expected key is present and non-NaN
            already_done = bool(existing) and (
                "orig_skipped" in existing
                or all(k in existing and _not_nan(existing[k]) for k in _expected_keys)
            )
            if already_done and not args.force:
                print(f"  {name}: all values present, skipping (use --force to rerun)")
                continue
            # Report which keys are missing or NaN so the user knows what will be redone
            if existing and not already_done:
                bad = [k for k in _expected_keys
                       if k not in existing or not _not_nan(existing[k])]
                print(f"  {name}: NaN/missing keys will be redone: {bad}")

            pdb_path = Path(p.get("path", ""))
            if not pdb_path.exists():
                print(f"  {name}: PDB not found at {pdb_path}, skipping")
                continue

            print(f"\n── {name}  (L={p['length']}) ──", flush=True)

            res_nums = None
            if args.orig_mut_scan and args.orig_workers > 1:
                _, _, _, seq_mask, residue_index, hetatm, _ = _parse_only(pdb_path)
                edge_mask = seq_mask.bool() & ~hetatm.bool()
                res_nums  = residue_index[edge_mask].tolist()

            def _save_partial(partial_r, _name=name):
                existing_orig[_name] = partial_r
                _orig_out = existing_orig.get(proteins[0]["name"]) if len(proteins) == 1 else existing_orig
                data["orig_results"] = _to_serializable(_orig_out)
                save_path.write_text(json.dumps(data, indent=2))

            try:
                r = profile_orig_protein(
                    pdb_path, rscript, modes, args.repeats,
                    mut_scan=args.orig_mut_scan,
                    workers=args.orig_workers,
                    residue_numbers=res_nums,
                    on_partial=_save_partial,
                    existing_results=None if args.force else existing,
                )
                existing_orig[name] = r
                for k, v in r.items():
                    if not k.startswith("_"):
                        print(f"    {k:<40} {_fmt_t(v)}s")
            except Exception as exc:
                print(f"    [warn] failed for {name}: {exc} — skipping")
                existing_orig[name] = existing_orig.get(name, existing)  # keep partial if any

            any_ran = True

            # Final save for this protein
            if len(proteins) == 1:
                orig_out = existing_orig.get(proteins[0]["name"])
            else:
                orig_out = existing_orig
            data["orig_results"] = _to_serializable(orig_out)
            save_path.write_text(json.dumps(data, indent=2))
            print(f"  Saved: {save_path}", flush=True)

        all_done = all(
            bool(existing_orig.get(p["name"]))
            and any(_not_nan(v) for v in existing_orig[p["name"]].values())
            for p in proteins
        )
        if all_done and not data.get("complete"):
            data["complete"] = True
            if len(proteins) == 1:
                orig_out = existing_orig.get(proteins[0]["name"])
            else:
                orig_out = existing_orig
            data["orig_results"] = _to_serializable(orig_out)
            save_path.write_text(json.dumps(data, indent=2))
            print(f"All orig done — marked complete in {save_path}")

        if not any_ran:
            print("Nothing to run — all proteins already have orig_results.")
        return

    device = torch.device(args.device)
    modes  = ALL_MODES if args.modes == "all" else args.modes.split(",")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    pdb_dir = Path(args.pdb_dir).resolve()
    list_file = Path(args.pdb_list) if args.pdb_list else pdb_dir / "pdb_files.txt"
    pdb_names = [l.strip() for l in list_file.read_text().splitlines() if l.strip()]

    def _length(name):
        try:
            return int(name.split("_")[0])
        except ValueError:
            return 0

    proteins = sorted(
        [{"name": n, "path": pdb_dir / n, "length": _length(n)}
         for n in pdb_names if (pdb_dir / n).exists()],
        key=lambda p: p["length"],
    )

    if not proteins:
        sys.exit(f"No PDB files found in {pdb_dir}")

    if args.protein_index is not None:
        if args.protein_index >= len(proteins):
            sys.exit(f"--protein_index {args.protein_index} out of range (0–{len(proteins)-1})")
        proteins = [proteins[args.protein_index]]
        print(f"Array mode: running protein index {args.protein_index}: {proteins[0]['name']}")

    print(f"Device:       {device}")
    print(f"Modes:        {modes}")
    print(f"Proteins:     {[p['name'] for p in proteins]}")
    print(f"Import time:  {IMPORT_TIME:.3f}s")
    print(f"Init time:    {INIT_TIME:.3f}s")

    warmup_t = gpu_warmup(device)
    print(f"GPU warmup:   {warmup_t:.3f}s")

    # Instantiate models
    coeffs = Coefficients()
    featurizer = None  # density computed via _compute_residue_densities(coeffs)

    awsem_models = {}
    for mode in modes:
        c = Coefficients(mode=mode)
        m = AWSEM(c).to(device)
        m.eval()
        awsem_models[mode] = m

    session_times = {
        "import":       IMPORT_TIME,
        "runtime_init": INIT_TIME,
        "gpu_warmup":   warmup_t,
    }

    # ── JSON checkpoint helper ────────────────────────────────────────────────
    _save_path = None
    _hw_label  = None
    if args.save_json:
        _save_path = Path(args.save_json)
        _save_path.parent.mkdir(parents=True, exist_ok=True)
        _hw_label = args.hardware_label or (
            f"CUDA ({torch.cuda.get_device_name(0)})" if device.type == "cuda"
            else str(device).upper()
        )

    def _checkpoint(complete: bool = False,
                    sweep_results_=None, orig_results_=None):
        if _save_path is None:
            return
        _pr = proteins[0] if len(proteins) == 1 else proteins
        _ar = (all_results[proteins[0]["name"]]
               if len(proteins) == 1 and proteins[0]["name"] in all_results
               else all_results)
        _sr = sweep_results_ or {}
        _sr_out = (_sr[proteins[0]["name"]]
                   if len(proteins) == 1 and proteins[0]["name"] in _sr else _sr)
        _or = orig_results_
        _or_out = (_or.get(proteins[0]["name"]) if (_or and len(proteins) == 1) else _or)
        payload = {
            "complete":       complete,
            "protein":        _to_serializable(_pr),
            "session_times":  _to_serializable(session_times),
            "results":        _to_serializable(_ar),
            "sweep_results":  _to_serializable(_sr_out),
            "orig_results":   _to_serializable(_or_out),
            "hardware_label": _hw_label,
        }
        _save_path.write_text(json.dumps(payload, indent=2))

    frust_out = None
    if args.write_frustration:
        frust_out = Path(args.frust_output_dir) if args.frust_output_dir else \
            Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_frust")
        frust_out.mkdir(parents=True, exist_ok=True)
        print(f"Frustration output: {frust_out}")

    all_results = {}
    for p in proteins:
        print(f"\n── {p['name']}  (L={p['length']}) ──", flush=True)
        r = profile_protein(p["path"], device, coeffs, featurizer, awsem_models,
                            args.repeats, frust_output_dir=frust_out, run_dmfs=args.dmfs)
        all_results[p["name"]] = r
        for k, v in r.items():
            print(f"  {k:<30} {_fmt_t(v)}s")
        _checkpoint()

    # Batch-size sweep
    sweep_results = {}
    print(f"\n── Batch-size sweep {batch_sizes} ──", flush=True)
    for p in proteins:
        print(f"  {p['name']}  (L={p['length']})", flush=True)
        try:
            r = profile_batch_sweep(p["path"], device, coeffs, featurizer,
                                    awsem_models, args.repeats, batch_sizes)
            sweep_results[p["name"]] = r
            for phase in ["density"] + modes:
                row = "  ".join(f"B={b}: {_fmt_t(r.get(phase, {}).get(b, float('nan')))}s"
                                for b in batch_sizes)
                print(f"    {phase:<20} {row}")
        except Exception as exc:
            print(f"    error: {exc}")
            sweep_results[p["name"]] = {m: {} for m in modes}
        _checkpoint(sweep_results_=sweep_results)

    # Summary table
    print("\n── Summary (seconds) ──")
    all_keys = (["import", "runtime_init", "gpu_warmup", "reading", "distance_map", "awsem_model"]
                + [f"frustration_{m}" for m in modes]
                + ["frustration_singleresidue_mut_scan", "frustration_mut_scan"])
    header = f"{'Phase':<32}" + "".join(f"  {p['length']:>6}" for p in proteins)
    print(header)
    for key in all_keys:
        row = []
        for p in proteins:
            if key in session_times:
                row.append(session_times[key])
            else:
                row.append(all_results[p["name"]].get(key, 0.0))
        print(f"{key:<32}" + "".join(f"  {_fmt_t(v)}" for v in row))

    print(f"\n── Batch sweep summary (s/seq) — batch sizes {batch_sizes} ──")
    hdr = f"  {'protein':<24}" + "".join(f"  B={b:>4}" for b in batch_sizes)
    for phase in ["density"] + modes:
        print(f"\n  {phase}")
        print(hdr)
        for p in proteins:
            row = [sweep_results[p["name"]].get(phase, {}).get(b, float("nan"))
                   for b in batch_sizes]
            print(f"  {p['name']:<24}" + "".join(f"  {_fmt_t(v)}" for v in row))

    # ── Optional: FrustratometeR comparison ───────────────────────────────
    orig_results = None
    if args.run_orig:
        rscript = Path(args.rscript)
        if not rscript.exists():
            print(f"[warn] Rscript not found at {rscript} — skipping FrustratometeR timing")
        else:
            print(f"\n── FrustratometeR timing (Rscript: {rscript}, workers={args.orig_workers}) ──",
                  flush=True)
            orig_results = {}
            for p in proteins:
                print(f"  {p['name']}  (L={p['length']})", flush=True)
                # Get residue numbers from parsed structure for parallel chunking
                res_nums = None
                if args.orig_mut_scan and args.orig_workers > 1:
                    _, _, _, seq_mask, residue_index, hetatm, _ = _parse_only(p["path"])
                    edge_mask = seq_mask.bool() & ~hetatm.bool()
                    res_nums  = residue_index[edge_mask].tolist()
                try:
                    def _save_partial_orig(partial_r, _p=p):
                        orig_results[_p["name"]] = partial_r
                        _checkpoint(sweep_results_=sweep_results, orig_results_=orig_results)

                    r = profile_orig_protein(
                        p["path"], rscript, modes, args.repeats,
                        mut_scan=args.orig_mut_scan,
                        workers=args.orig_workers,
                        residue_numbers=res_nums,
                        on_partial=_save_partial_orig,
                    )
                    orig_results[p["name"]] = r
                    for k, v in r.items():
                        print(f"    {k:<40} {_fmt_t(v)}s")
                except Exception as exc:
                    print(f"    [warn] FrustratometeR failed for {p['name']}: {exc} — skipping")
                    orig_results[p["name"]] = {}
                _checkpoint(sweep_results_=sweep_results, orig_results_=orig_results)

            print("\n── FrustratometeR summary (seconds) ──")
            orig_keys = [f"orig_{m}" for m in modes]
            if args.orig_mut_scan:
                orig_keys += ["orig_singleresidue_mut_scan", "orig_mut_scan"]
            header2 = f"{'Phase':<42}" + "".join(f"  {p['length']:>6}" for p in proteins)
            print(header2)
            for key in orig_keys:
                row = [orig_results[p["name"]].get(key, float("nan")) for p in proteins]
                print(f"{key:<42}" + "".join(f"  {_fmt_t(v)}" for v in row))

    # ── Save per-protein JSON (for array job combine) ─────────────────────
    _checkpoint(complete=True, sweep_results_=sweep_results, orig_results_=orig_results)
    if _save_path is not None:
        print(f"Saved JSON: {_save_path}")

    device_label = (f"CUDA ({torch.cuda.get_device_name(0)})"
                    if device.type == "cuda" else str(device).upper())
    plot_results(proteins, all_results, session_times, modes, args.output, device_label,
                 orig_results=orig_results)

    sweep_output = args.output.replace(".png", "_batch_sweep.png")
    plot_batch_sweep(proteins, sweep_results, modes, sweep_output, device_label)


if __name__ == "__main__":
    main()
