import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*elements were guessed from atom name.*", module="biotite")
# Suppress numpy non-writable array warning from pandas .values
warnings.filterwarnings("ignore", message=".*non-writable.*", category=UserWarning)

import logging
import os
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)
# Suppress Lightning package-level tips (litlogger, litmodels, tensorboard)
os.environ.setdefault("LIGHTNING_DISABLE_TIPS", "1")
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)


from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import pandas as pd
import lightning.pytorch as pl
import torch.nn.functional as F

from .utils.density import compute_residue_densities as _compute_residue_densities
from .utils.utils import check_size
from .coefficients import Coefficients
from .algorithm.awsem import AWSEM
from .base import _FromPDBMixin
from .structure.protein import PDB_CHAIN_IDS
from .data.io import (
    parse_singleresidue_frustration,
    parse_pair_frustration,
)


def _pearson_r2(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan"), 0
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    return float(r ** 2), int(ok.sum())


def _merge_rank_csv_files(debug_dir: Path, stem: str):
    """Merge <stem>_rankN.csv files written by each DDP rank into <stem>.csv."""
    import csv
    rank_files = sorted(debug_dir.glob(f"{stem}_rank*.csv"))
    if not rank_files:
        return
    merged_path = debug_dir / f"{stem}.csv"
    with open(merged_path, "w", newline="") as out_fh:
        writer = None
        for rf in rank_files:
            with open(rf, newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                if writer is None:
                    writer = csv.DictWriter(out_fh, fieldnames=reader.fieldnames,
                                            extrasaction="ignore")
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
            rf.unlink()
    print(f"[Rank 0] Merged {len(rank_files)} rank files → {merged_path}", flush=True)


def _find_mutations_data_dir(sf: Path):
    """Return the MutationsData/ directory for a structure file, or None."""
    # Case 1: sf is inside <stem>.done/FrustrationData/
    if sf.parent.parent.name.endswith(".done"):
        cand = sf.parent.parent / "MutationsData"
        if cand.is_dir():
            return cand
    # Case 2: flat PDB — look for <stem>_*.done/MutationsData/ sibling
    for done in sf.parent.glob(f"{sf.stem}_*.done"):
        cand = done / "MutationsData"
        if cand.is_dir():
            return cand
    return None


def _find_structure_file(frust_base):
    """Structure written next to the frustration files: <stem>.pdb for PDB input, <stem>.cif for mmCIF
    input (the extension follows the input format). Returns a Path or None."""
    base = str(frust_base)
    for ext in ("pdb", "cif", "bcif", "mmtf"):
        cand = Path(f"{base}.{ext}")
        if cand.exists():
            return cand
    return None


def _find_result_file(frust_base, suffix):
    """Result file <stem>.<ext>_<suffix> (e.g. <stem>.cif_mutational): the middle part follows the input
    format (pdb / cif / ...), so look for each. Returns a Path or None."""
    for ext in ("pdb", "cif", "bcif", "mmtf"):
        cand = Path(f"{frust_base}.{ext}_{suffix}")
        if cand.exists():
            return cand
    return None


class Frustratometer3(_FromPDBMixin, pl.LightningModule):
    def __init__(
        self,
        coefficients,
        output_dir=None,
        dtype: torch.dtype = torch.float32,
        draw_figure: bool = False,
        write_pymol: bool = False,
        write_mvsj: bool = False,
        render_structure: bool = False,
        image_formats: tuple = ("png",),
        verbose: bool = False,
        write_pdb: bool = True,
        use_in_progress: bool = False,
        save_bulk_stats: bool = False,
        scratch_root: str | None = None,
        final_output_dir: str | None = None,
    ):
        super().__init__()

        self.coefficients = coefficients
        self._output_dir = output_dir
        self._r_failures: list = []
        self._r_results:  list = []          # buffer — flushed to all_results.csv periodically
        self._bulk_stats: list = []          # buffer — flushed to bulk_stats_rankN.csv when save_bulk_stats=True
        self._residue_samples: dict = {}     # buffer — flushed to residue_samples_{mode}.csv periodically
        self._csv_headers_written: set = set()   # tracks which CSV files have been initialised
        self.save_bulk_stats = save_bulk_stats
        self._flush_interval = 50            # flush every N batches
        self._batch_count = 0
        self._test_ok = 0
        self._test_fail_r = 0
        self._test_fail_acc = 0
        self._mode_stats: dict = {}   # mode -> {"r_sum": float, "r_n": int, "acc_sum": float, "acc_n": int, "n_fail_r": int}

        self.draw_figure      = draw_figure
        self.write_pymol      = write_pymol
        self.write_mvsj       = write_mvsj
        self.render_structure = render_structure
        self._write_pdb       = write_pdb
        self.image_formats    = list(image_formats) if image_formats else ["png"]
        self.verbose          = verbose

        self._use_in_progress = use_in_progress

        # Async scratch→GPFS copy.  Passed as constructor args so save_hyperparameters()
        # preserves them across DDP-spawned worker processes (post-construction assignment
        # on the main-process object is not seen by Lightning-reconstructed workers).
        self._scratch_root:     str | None = scratch_root
        self._final_output_dir: str | None = final_output_dir
        self._copy_executor = None
        self._copy_futures: list = []

        # Shard-mode output: set externally before predict_step when --shard-output is used.
        self._shard_writer = None

        self.save_hyperparameters(ignore=["dtype"])

        self._dtype = dtype
        self.awsem = AWSEM(self.coefficients, dtype=dtype)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # MPS doesn't support float64 — downcast transparently so the model
        # works on Apple Silicon without callers needing to know.
        if device.type == "mps":
            batch = {k: v.to(torch.float32)
                     if isinstance(v, torch.Tensor) and v.dtype == torch.float64
                     else v
                     for k, v in batch.items()}
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def forward(self, batch):
        B = batch["aa_type"].shape[0]
        if B == 1:
            return self._forward_single(batch)
        if self.awsem.use_c_rand:
            # c_rand uses a global srand(1) — shared across all B structures in one
            # call gives the wrong random stream for every structure past the first.
            # Process one at a time so each gets its own clean srand(1) reset.
            results = [self._forward_single(self._slice_batch(batch, i)) for i in range(B)]
            return self._merge_results(results)
        # Fast path (no c_rand): true batched forward, no per-structure loop needed.
        return self._forward_single(batch)

    @staticmethod
    def _slice_batch(batch, i):
        """Extract the i-th structure from a collated batch as a B=1 dict."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v[i:i+1]
            elif isinstance(v, (list, tuple)):
                out[k] = [v[i]]
            else:
                out[k] = v
        return out

    @staticmethod
    def _merge_results(results):
        """Concatenate a list of B=1 result dicts into a single dict."""
        merged = {}
        for key in results[0]:
            vals = [r[key] for r in results]
            if isinstance(vals[0], torch.Tensor):
                merged[key] = torch.cat(vals, dim=0)
            else:
                merged[key] = vals
        return merged

    def _forward_single(self, batch):
        with torch.no_grad():
            batch["residue_density"], batch["burial_density"], _ = _compute_residue_densities(
                batch["atom_positions"],
                batch["aa_type"],
                batch["seq_mask"],
                batch["hetatm"],
                batch["residue_index"],
                batch["distance"],
                self.coefficients,
                chain_index=batch.get("chain_index"),
                edge_index=batch.get("edge_index"),
                dtype=self._dtype,
            )
            frustration = self.awsem(batch)
        return frustration

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            rank = getattr(self, "global_rank", 0)
            ws = getattr(self.trainer, "world_size", 1)
            self._rank_log(f"test_step called: rank={rank}/{ws}")
        result = self(batch)
        self._check_r_threshold(batch, result)
        self._write_outputs(batch, result, output_dir=self._output_dir)
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    def _drain_copy_queue(self):
        """Wait for all pending scratch→GPFS copy futures to complete."""
        if not self._copy_futures:
            self._rank_log(f"_drain_copy_queue: no pending futures (executor={'ready' if self._copy_executor else 'None'})")
            return
        from concurrent.futures import as_completed
        pending = len(self._copy_futures)
        self._rank_log(f"_drain_copy_queue: waiting for {pending} copies to {self._final_output_dir!r}")
        print(f"[I/O] Waiting for {pending} pending file copies to GPFS ...", flush=True)
        errors = []
        for fut in as_completed(self._copy_futures):
            try:
                fut.result()
            except Exception as e:
                errors.append(str(e))
        self._copy_futures.clear()
        if errors:
            self._rank_log(f"_drain_copy_queue: {len(errors)} copy errors: {errors[:3]}")
            print(f"[I/O] {len(errors)} copy errors (first 3): {errors[:3]}", flush=True)
        else:
            self._rank_log(f"_drain_copy_queue: all {pending} copies done")
            print("[I/O] All copies complete.", flush=True)

    def _setup_copy_executor(self):
        """Create the scratch→GPFS copy executor for this rank's process.

        Called from on_predict_start / on_test_start so that each DDP rank
        (whether torchrun or spawn) creates its own ThreadPoolExecutor rather
        than relying on an attribute set before trainer.predict().

        Disabled when world_size > 1: DDP ranks write files concurrently and
        the async copy queue causes races when multiple ranks finish the same
        batch window simultaneously.  In multi-GPU mode files are written
        directly to the final output dir instead.
        """
        world_size = (getattr(self.trainer, "world_size", 1)
                      if hasattr(self, "trainer") and self.trainer is not None else 1)
        if world_size > 1:
            return

        from concurrent.futures import ThreadPoolExecutor as _TPE
        scratch = os.environ.get("FRUST_SCRATCH_ROOT") or self._scratch_root
        final   = os.environ.get("FRUST_FINAL_OUTPUT_DIR") or self._final_output_dir
        if scratch and final and self._copy_executor is None:
            self._scratch_root     = scratch
            self._final_output_dir = final
            self._copy_executor    = _TPE(max_workers=16)

    def _rank_log(self, msg: str):
        """Write a diagnostic line to a per-rank log file in scratch so non-rank-0 output isn't suppressed."""
        rank = getattr(self, "global_rank", 0)
        log_dir = Path(self._scratch_root) if self._scratch_root else Path(".")
        try:
            with open(log_dir / f"rank_{rank}_io.log", "a") as f:
                f.write(msg + "\n")
                f.flush()
        except Exception:
            pass
        # Also print to stderr (not suppressed by Lightning on spawn workers)
        import sys
        print(f"[Rank {rank}] {msg}", file=sys.stderr, flush=True)

    def _setup_rank_io(self):
        import pytorch_frustratometer.base as _base
        _base._use_in_progress = self._use_in_progress
        self._setup_copy_executor()
        rank = getattr(self, "global_rank", 0)
        world_size = getattr(self.trainer, "world_size", 1) if hasattr(self, "trainer") and self.trainer is not None else 1
        copy_ready = self._copy_executor is not None
        msg = (f"I/O setup: rank={rank}/{world_size} use_in_progress={self._use_in_progress} "
               f"copy_executor={'ready' if copy_ready else 'MISSING'} "
               f"scratch={self._scratch_root!r} "
               f"final={self._final_output_dir!r}")
        self._rank_log(msg)

    def on_predict_start(self):
        self._setup_rank_io()
        super().on_predict_start()

    def on_predict_end(self):
        self._drain_copy_queue()
        super().on_predict_end()

    def on_test_start(self):
        self._setup_rank_io()
        super().on_test_start()

    def on_test_end(self):
        self._drain_copy_queue()
        super().on_test_end()

    def _check_r_threshold(self, batch, result, r_min: float = 0.99, acc_min: float = 0.9):
        """Per-structure check; count ok/fail for progress bar and write debug files for failures."""
        import sys
        from .utils.utils import compute_frustration_metrics

        B = batch["aa_type"].shape[0]
        sf_raw = batch.get("structure_file")
        if isinstance(sf_raw, str):
            names_list = [sf_raw]
        elif sf_raw is not None:
            names_list = list(sf_raw)
        else:
            names_list = [f"struct_{i}" for i in range(B)]

        debug_dir = Path(self._output_dir) / "run_info" if self._output_dir else Path("run_info")

        mode_cfg = [
            ("singleresidue",   "singleresidue",   "single_frust_true",         "single_frust_mask",          self.coefficients.frustration_thresholds[0]),
            ("mutational",      "mutational",      "mutational_frust_true",      "mutational_frust_mask",      self.coefficients.frustration_thresholds[1]),
            ("configurational", "configurational", "configurational_frust_true", "configurational_frust_mask", self.coefficients.frustration_thresholds[1]),
        ]

        n_ok = n_fail_r = n_fail_acc = 0

        for i in range(B):
            name = names_list[i] if i < len(names_list) else f"struct_{i}"

            for mode_name, pred_key, true_key, mask_key, threshold in mode_cfg:
                pred = result.get(pred_key)
                true = batch.get(true_key)
                if pred is None or true is None:
                    continue
                mask = batch.get(mask_key, batch["seq_mask"])

                m = compute_frustration_metrics(
                    pred[i:i+1], true[i:i+1], mask[i:i+1], threshold
                )
                r   = m.get("pearson_r",     float("nan"))
                acc = m.get("class_accuracy", float("nan"))
                if r != r and acc != acc:
                    continue

                r_fail   = (r == r)   and (r   < r_min)
                acc_fail = (acc == acc) and (acc < acc_min)

                # --- collect residue-level subsample (max 30 per structure/mode) ---
                import random
                _AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
                p_vec = pred[i:i+1]
                t_vec = true[i:i+1]
                mk    = batch.get(mask_key, batch["seq_mask"])[i:i+1].bool()
                p_valid  = p_vec[mk].cpu().tolist()
                t_valid  = t_vec[mk].cpu().tolist()
                aa_type_i = batch.get("aa_type")
                if (aa_type_i is not None
                        and mk[0].shape == aa_type_i[i].shape):
                    aa_valid = aa_type_i[i][mk[0]].cpu().tolist()
                else:
                    aa_valid = [None] * len(p_valid)
                if p_valid:
                    n_samp = min(30, len(p_valid))
                    idx_s  = random.sample(range(len(p_valid)), n_samp)
                    pairs  = [(p_valid[k], t_valid[k],
                               _AA_LETTERS[aa_valid[k]] if aa_valid[k] is not None else "")
                              for k in idx_s]
                    self._residue_samples.setdefault(mode_name, []).extend(pairs)

                # --- record all per-protein-mode metrics (pass AND fail) ---
                row_base = {
                    "structure": name, "mode": mode_name,
                    "pearson_r": r, "class_accuracy": acc,
                    "std_ratio": m.get("std_ratio", float("nan")),
                    "pred_std":  m.get("pred_std",  float("nan")),
                    "true_std":  m.get("true_std",  float("nan")),
                    "mse":       m.get("mse",        float("nan")),
                    "mae":       m.get("mae",        float("nan")),
                    "spearman_r":       m.get("spearman_r",        float("nan")),
                    "cosine_similarity": m.get("cosine_similarity", float("nan")),
                    "n":                 m.get("n", 0),
                }
                self._r_results.append(row_base)

                if self.save_bulk_stats:
                    frust_mask = mask[i:i+1].bool()
                    pred_m = pred[i:i+1][frust_mask].float().cpu()
                    true_m = true[i:i+1][frust_mask].float().cpu()
                    if len(pred_m) > 0:
                        diff = pred_m - true_m
                        bias = diff.mean().item()
                        rmse = diff.pow(2).mean().sqrt().item()
                        fi_sq_mean = (true_m ** 2).mean().item()
                    else:
                        bias = rmse = fi_sq_mean = float("nan")
                    if mode_name in ("mutational", "configurational"):
                        n_pt = int(batch["edge_index"].shape[1]) if "edge_index" in batch else m.get("n", 0)
                        n_ref_raw = batch.get(f"{mode_name}_n_ref")
                        if n_ref_raw is not None:
                            n_ref = int(n_ref_raw[i, 0]) if n_ref_raw.dim() > 1 else int(n_ref_raw[i])
                        else:
                            n_ref = n_pt
                    else:
                        n_pt = int(mask[i].sum())
                        n_ref = n_pt
                    n_matched = m.get("n", 0)
                    frac_matched = n_matched / max(n_pt, n_ref, 1)
                    wt = batch.get(f"{mode_name}_welltype")
                    wt_stats: dict = {}
                    if wt is not None and len(pred_m) > 0:
                        wt_slice = wt[i:i+1][frust_mask].cpu()
                        diff_abs = diff.abs()
                        for wt_idx, wt_name in enumerate(["short", "long", "water_mediated"]):
                            wt_mask = (wt_slice == wt_idx)
                            if wt_mask.sum() > 0:
                                wt_stats[f"mae_{wt_name}"] = float(diff_abs[wt_mask].mean())
                                wt_stats[f"n_{wt_name}"] = int(wt_mask.sum())
                    sd_e = batch.get(f"{mode_name}_sd_energy")
                    sd_e_mean = float("nan")
                    if sd_e is not None and len(pred_m) > 0:
                        sd_e_slice = sd_e[i:i+1][frust_mask].float().cpu()
                        if len(sd_e_slice) > 0:
                            sd_e_mean = float(sd_e_slice.mean())
                    lt_raw = batch.get("load_time_sec")
                    if lt_raw is not None:
                        lt_val = lt_raw[i] if lt_raw.dim() > 0 else lt_raw
                        load_time_sec = float(lt_val.item())
                    else:
                        load_time_sec = float("nan")
                    self._bulk_stats.append({
                        "structure": name, "mode": mode_name,
                        "n_pairs": n_matched, "n_pt": n_pt, "n_ref": n_ref,
                        "frac_matched": frac_matched,
                        "mae": m.get("mae", float("nan")),
                        "rmse": rmse, "bias": bias,
                        "pearson_r": r,
                        "cosine_similarity": m.get("cosine_similarity", float("nan")),
                        "class_accuracy": m.get("class_accuracy", float("nan")),
                        "fi_sq_mean": fi_sq_mean,
                        "sd_e_mean": sd_e_mean,
                        "load_time_sec": load_time_sec,
                        **wt_stats,
                    })

                s = self._mode_stats.setdefault(mode_name, {"r_sum": 0.0, "r_n": 0, "acc_sum": 0.0, "acc_n": 0, "n_fail_r": 0})
                if r == r:  # not nan
                    s["r_sum"] += r; s["r_n"] += 1
                    if r < r_min: s["n_fail_r"] += 1
                if acc == acc:
                    s["acc_sum"] += acc; s["acc_n"] += 1

                if not r_fail and not acc_fail:
                    n_ok += 1
                    continue

                if r_fail:   n_fail_r   += 1
                if acc_fail: n_fail_acc += 1

                reasons = []
                if r_fail:   reasons.append(f"R={r:.4f}")
                if acc_fail: reasons.append(f"acc={acc:.4f}")

                lines = [
                    f"[FAIL({','.join(reasons)})] mode={mode_name!r}  structure={name!r}",
                    f"  pearson_r={r:.4f}  class_accuracy={acc:.4f}  n={m.get('n','?')}",
                    f"  mse={m.get('mse',float('nan')):.4f}  mae={m.get('mae',float('nan')):.4f}"
                    f"  spearman_r={m.get('spearman_r',float('nan')):.4f}",
                    f"  std_ratio={m.get('std_ratio',float('nan')):.4f}"
                    f"  pred_std={m.get('pred_std',float('nan')):.4f}"
                    f"  true_std={m.get('true_std',float('nan')):.4f}",
                    f"  pred_frac: highly={m.get('pred_frac_cls0',float('nan')):.3f}"
                    f"  neutral={m.get('pred_frac_cls1',float('nan')):.3f}"
                    f"  minimally={m.get('pred_frac_cls2',float('nan')):.3f}",
                    f"  true_frac: highly={m.get('true_frac_cls0',float('nan')):.3f}"
                    f"  neutral={m.get('true_frac_cls1',float('nan')):.3f}"
                    f"  minimally={m.get('true_frac_cls2',float('nan')):.3f}",
                ]
                debug_text = "\n".join(lines) + "\n"

                debug_file = None
                try:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    stem = Path(name).stem
                    debug_file = debug_dir / f"{stem}.{mode_name}.debug.txt"
                    debug_file.write_text(debug_text)
                except Exception as exc:
                    print(f"[FAIL write-error {exc}]\n{debug_text}", file=sys.stderr, flush=True)

                self._r_failures.append({
                    "structure": name,
                    "mode": mode_name,
                    "reason": ",".join(reasons),
                    "pearson_r": r,
                    "class_accuracy": acc,
                    "std_ratio": m.get("std_ratio", float("nan")),
                    "pred_std": m.get("pred_std", float("nan")),
                    "true_std": m.get("true_std", float("nan")),
                    "mse": m.get("mse", float("nan")),
                    "mae": m.get("mae", float("nan")),
                    "spearman_r": m.get("spearman_r", float("nan")),
                    "n": m.get("n", 0),
                    "debug_file": str(debug_file) if debug_file else "",
                })

        self._test_ok       += n_ok
        self._test_fail_r   += n_fail_r
        self._test_fail_acc += n_fail_acc
        try:
            pbar = self.trainer.progress_bar_callback.test_progress_bar
            pbar.set_postfix(
                ok=self._test_ok,
                fail_r=self._test_fail_r,
                fail_acc=self._test_fail_acc,
                refresh=True,
            )
        except Exception:
            pass

        self._batch_count += 1
        if self._batch_count % self._flush_interval == 0:
            self._flush_buffers()

    def _flush_buffers(self):
        """Append buffered results/samples to disk and clear in-memory lists."""
        import csv
        if not self._r_results and not self._residue_samples and not self._bulk_stats:
            return
        debug_dir = Path(self._output_dir) / "run_info" if self._output_dir else Path("run_info")
        debug_dir.mkdir(parents=True, exist_ok=True)

        if self._bulk_stats:
            bulk_fields = [
                "structure", "mode", "n_pairs", "n_pt", "n_ref", "frac_matched",
                "mae", "rmse", "bias", "pearson_r", "cosine_similarity", "class_accuracy",
                "fi_sq_mean", "sd_e_mean", "load_time_sec",
                "mae_short", "mae_long", "mae_water_mediated",
                "n_short", "n_long", "n_water_mediated",
            ]
            rank = getattr(self, "global_rank", 0)
            path = debug_dir / f"bulk_stats_rank{rank}.csv"
            key = f"bulk_stats_rank{rank}"
            write_header = key not in self._csv_headers_written
            with open(path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=bulk_fields,
                                        extrasaction="ignore", restval="")
                if write_header:
                    writer.writeheader()
                    self._csv_headers_written.add(key)
                writer.writerows(self._bulk_stats)
            self._bulk_stats.clear()

        if self._r_results:
            rank = getattr(self, "global_rank", 0)
            all_fields = ["structure", "mode", "pearson_r", "spearman_r", "cosine_similarity",
                          "class_accuracy", "std_ratio", "pred_std", "true_std", "mse", "mae", "n"]
            path = debug_dir / f"all_results_rank{rank}.csv"
            key = f"all_results_rank{rank}"
            write_header = key not in self._csv_headers_written
            with open(path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                    self._csv_headers_written.add(key)
                writer.writerows(self._r_results)
            self._r_results.clear()

        for mode_name, pairs in self._residue_samples.items():
            if not pairs:
                continue
            rank = getattr(self, "global_rank", 0)
            path = debug_dir / f"residue_samples_{mode_name}_rank{rank}.csv"
            key = f"samples_{mode_name}_rank{rank}"
            write_header = key not in self._csv_headers_written
            with open(path, "a", newline="") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(["pred", "true", "aa"])
                    self._csv_headers_written.add(key)
                writer.writerows(pairs)
            pairs.clear()

    def on_test_epoch_end(self):
        import csv
        debug_dir = Path(self._output_dir) / "run_info" if self._output_dir else Path("run_info")
        debug_dir.mkdir(parents=True, exist_ok=True)

        if self._r_failures:
            rank = getattr(self, "global_rank", 0)
            summary_path = debug_dir / f"failures_by_r_rank{rank}.csv"
            fields = ["structure", "mode", "reason", "pearson_r", "class_accuracy",
                      "std_ratio", "pred_std", "true_std", "mse", "mae", "spearman_r", "n", "debug_file"]
            rows = sorted(self._r_failures, key=lambda x: x["pearson_r"])
            with open(summary_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        self._flush_buffers()   # flush remaining data to disk

        # --- console R2 summary from incrementally-accumulated stats ---
        import math
        print("\n=== frustratometer3 comparison summary ===")
        if self._mode_stats:
            for mode, s in sorted(self._mode_stats.items()):
                n_r   = s["r_n"]
                mean_r   = s["r_sum"] / n_r   if n_r   else float("nan")
                mean_acc = s["acc_sum"] / s["acc_n"] if s["acc_n"] else float("nan")
                print(f"  {mode:20s}  n={n_r:5d}  mean_R={mean_r:.4f}  mean_acc={mean_acc:.4f}  n_fail_R={s['n_fail_r']}")
        else:
            print("  (no comparison data — reference files not found or no modes matched)")
        total_ok   = self._test_ok
        total_fail = self._test_fail_r + self._test_fail_acc
        print(f"  ok={total_ok}  fail_r={self._test_fail_r}  fail_acc={self._test_fail_acc}  total={total_ok+total_fail}")
        print("==========================================\n")

        # All ranks barrier, then rank 0 merges per-rank CSV files into single files.
        try:
            self.trainer.strategy.barrier()
        except Exception:
            pass
        if getattr(self, "global_rank", 0) == 0:
            _merge_rank_csv_files(debug_dir, "all_results")
            _merge_rank_csv_files(debug_dir, "failures_by_r")
            if self.save_bulk_stats:
                _merge_rank_csv_files(debug_dir, "bulk_stats")
            # Discover residue_samples modes from files left by any rank
            seen_stems: set = set()
            for rf in debug_dir.glob("residue_samples_*_rank*.csv"):
                # stem is everything before the last _rankN
                parts = rf.stem.rsplit("_rank", 1)
                if len(parts) == 2:
                    seen_stems.add(parts[0])
            for stem in seen_stems:
                _merge_rank_csv_files(debug_dir, stem)

    def _log_metrics(self, batch, result):
        """Log per-structure and aggregate metrics to Lightning / WandB."""
        sf_raw = batch.get("structure_file")
        if isinstance(sf_raw, str):
            names = [Path(sf_raw).stem]
        elif sf_raw is not None:
            names = [Path(sf).stem for sf in sf_raw]
        else:
            names = [str(i) for i in range(batch["aa_type"].shape[0])]

        # Build the 6-element losses list (same layout as all_loss[0:6] in the NN model)
        sr_l   = result.get("_singleresidue_losses")
        mut_l  = result.get("_mutational_losses")
        conf_l = result.get("_configurational_losses")
        losses = [
            sr_l[0]   if sr_l   else None,
            sr_l[1]   if sr_l   else None,
            mut_l[0]  if mut_l  else None,
            mut_l[1]  if mut_l  else None,
            conf_l[0] if conf_l else None,
            conf_l[1] if conf_l else None,
        ]

        mode_keys = [
            ("single",         "index", 0),
            ("single",         "class", 1),
            ("mutational",     "index", 2),
            ("mutational",     "class", 3),
            ("configurational","index", 4),
            ("configurational","class", 5),
        ]

        metrics_keys = {
            "singleresidue":    result.get("_singleresidue_metrics"),
            "mutational":       result.get("_mutational_metrics"),
            "configurational":  result.get("_configurational_metrics"),
        }

        log_kw = dict(on_step=False, on_epoch=True, sync_dist=True)

        # Aggregate metrics only — per-structure detail is written to CSV by _check_r_threshold.
        # Logging per-structure keys with on_epoch=True causes Lightning to accumulate
        # O(n_structures * n_metrics) state across the epoch, which degrades performance.
        for mode_name, sub, idx in mode_keys:
            if losses[idx] is not None:
                self.log(f"test/{mode_name}/{sub}", losses[idx], **log_kw)

        for mode_name, m_dict in metrics_keys.items():
            if m_dict is None:
                continue
            for metric_name, value in m_dict.items():
                if isinstance(value, float) and not (value != value):
                    self.log(f"test/{mode_name}/{metric_name}", value, **log_kw)

    def _write_outputs(self, batch, result, output_dir=None):
        import shutil as _shutil
        import sys
        from .base import _write_frust_outputs, _use_in_progress
        try:
            frust_bases, names, done_dirs, residue_index, chain_index, structure_files = \
                _write_frust_outputs(batch, result, output_dir, write_pdb=self._write_pdb)

        except Exception as exc:
            rank = getattr(self, "global_rank", None)
            sf = batch.get("structure_file", "?")
            print(f"[Rank {rank}] ERROR in _write_frust_outputs for {sf}: {exc}", file=sys.stderr, flush=True)
            raise
        # All scratch writes must complete before the async copy deletes the done_dir.
        if self.awsem.run_dmfs:
            self._write_mutation_files(batch, result, frust_bases, names,
                                       residue_index, chain_index)
        if self.draw_figure:
            self._draw_figures(result, frust_bases, structure_files)
            if self.awsem.run_dmfs and "singleresidue" in result:
                from .plots.singleresidue import draw_mutation_scanning
                draw_mutation_scanning(batch, result, frust_bases)
        if self.write_pymol:
            self._write_pymol_scripts(batch, result, frust_bases, names, output_dir)
        if self.write_mvsj:
            self._write_mvsj_scripts(batch, result, frust_bases, names, output_dir)

        if _use_in_progress:
            for d in done_dirs:
                target = d.parent / (d.name.removesuffix(".in_progress") + ".done")
                if target.exists():
                    _shutil.rmtree(target)
                d.rename(target)
        elif self._copy_executor is not None:
            # Async scratch→GPFS: copy each done_dir in background while GPU runs next batch.
            scratch_root = Path(self._scratch_root)
            final_root   = Path(self._final_output_dir)
            _rank = getattr(self, "global_rank", 0)

            def _copy_done_dir(done_dir):
                import sys as _sys
                try:
                    rel    = Path(done_dir).relative_to(scratch_root)
                    target = final_root / rel
                    target.mkdir(parents=True, exist_ok=True)
                    for src in Path(done_dir).rglob("*"):
                        if src.is_file():
                            dst = target / src.relative_to(done_dir)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            _shutil.copy2(src, dst)
                    _shutil.rmtree(done_dir)
                except Exception as _exc:
                    print(f"[Rank {_rank}] COPY ERROR {done_dir} → {final_root}: {_exc}",
                          file=_sys.stderr, flush=True)
                    raise

            if not self._copy_futures:  # first submission for this rank
                self._rank_log(f"first copy submitted: {done_dirs[0] if done_dirs else '?'} → {self._final_output_dir!r}")
            for d in done_dirs:
                fut = self._copy_executor.submit(_copy_done_dir, str(d))
                self._copy_futures.append(fut)
        else:
            import sys as _sys
            rank = getattr(self, "global_rank", 0)
            sf = batch.get("structure_file", "?")
            self._rank_log(f"WARNING: no copy path active for {sf} "
                           f"(use_in_progress={_use_in_progress}, copy_executor={self._copy_executor})")
            print(f"[Rank {rank}] WARNING: no copy path active for {sf} "
                  f"(use_in_progress={_use_in_progress}, copy_executor={self._copy_executor})",
                  file=_sys.stderr, flush=True)
            if self.draw_figure and "singleresidue" in result:
                from .plots.singleresidue import draw_mutation_scanning
                draw_mutation_scanning(batch, result, frust_bases)

    def _write_mutation_files(self, batch, result, frust_bases, names,
                               residue_index, chain_index):
        """
        Write per-residue and per-edge mutation energy tables.

        Single residue:
          {stem}.frustratometer3.singleresidue_mutations
          Columns: Res  ChainRes  AA_native  AA_mutant  NativeEnergy  MutantEnergy  FrstIndex
                   (FrstIndex = (DecoyMean - MutantEnergy) / DecoyStd)

        Mutational pair:
          {stem}.frustratometer3.mutational_mutations
          Columns: Res1  Res2  ChainRes1  ChainRes2  AA1_native  AA2_native
                   AA1_mutant  AA2_mutant  NativeEnergy  MutantEnergy
        """
        from pathlib import Path as _Path
        from .structure.residue_constants import ID_TO_AA
        from .data.io import read_mutations_data
        decoy_stats = result.get("_singleresidue_decoy_stats")  # [B, L, 2] or None
        sf_raw = batch.get("structure_file")
        GLY_IDX = 5
        _cid_map_list = batch.get("chain_id_map")

        for i, (frust_base, name) in enumerate(zip(frust_bases, names)):
            mask = batch["seq_mask"][i].bool()
            aa   = batch["aa_type"][i, mask].cpu()
            _cid_map_i = _cid_map_list[i] if _cid_map_list is not None else None

            res_nums = (residue_index[i, mask].cpu().numpy()
                        if residue_index is not None
                        else np.arange(1, mask.sum().item() + 1))
            _rl = batch.get("res_labels")
            res_labels_i = (
                [_rl[i][j] for j in mask.nonzero(as_tuple=False).squeeze(1).tolist()]
                if _rl is not None
                else [str(int(r)) for r in res_nums]
            )
            chain_ids = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                          for c in chain_index[i, mask].cpu().tolist()]
                         if chain_index is not None
                         else ["A"] * mask.sum().item())

            # --- Single-residue mutations ---
            sr_mut = result.get("_singleresidue_mutations")
            if sr_mut is not None:
                mut_e    = sr_mut["energies"][i, mask].cpu().float()  # [n_res, 20]
                nat_e_sr = result["_singleresidue_native_energy"][i, mask].cpu().float()

                # Use stats stored alongside mutation energies (with gly_water_corr applied)
                d_mean = d_std = None
                mut_stats = sr_mut.get("stats")
                if mut_stats is not None:
                    ds     = mut_stats[i, mask].cpu().float()  # [n_res, 2]
                    d_mean = ds[:, 0]
                    d_std  = ds[:, 1]
                elif decoy_stats is not None:
                    ds     = decoy_stats[i, mask].cpu().float()  # [n_res, 2]
                    d_mean = ds[:, 0]
                    d_std  = ds[:, 1]

                # Case 2: corrected FI for GLY decoy at non-GLY positions [n_res], NaN at GLY native
                gly_fi_masked = None
                gly_fi_all = sr_mut.get("_gly_mutation_fi")
                if gly_fi_all is not None:
                    gly_fi_masked = gly_fi_all[i, mask].cpu().float()

                # Case 1: corrected FI for all 20 decoys at GLY native positions [n_res, 20]
                gly_native_fi_masked = None
                gly_native_fi_all = sr_mut.get("_gly_native_fi")
                if gly_native_fi_all is not None:
                    gly_native_fi_masked = gly_native_fi_all[i, mask].cpu().float()

                n_res = mut_e.shape[0]
                rows = []
                for r in range(n_res):
                    aa_nat = ID_TO_AA.get(int(aa[r]), "X")
                    is_nat_gly = (int(aa[r]) == GLY_IDX)
                    for d in range(20):
                        if is_nat_gly and gly_native_fi_masked is not None:
                            # Case 1: native GLY → any decoy — self-consistent vCB geometry
                            fi = float(gly_native_fi_masked[r, d])
                        elif (d == GLY_IDX and not is_nat_gly
                                and gly_fi_masked is not None
                                and not torch.isnan(gly_fi_masked[r])):
                            # Case 2: native non-GLY → GLY decoy — self-consistent CA geometry
                            fi = float(gly_fi_masked[r])
                        else:
                            fi = (float((d_mean[r] - mut_e[r, d]) / d_std[r])
                                  if d_mean is not None else float("nan"))
                        rows.append({
                            "Res":          res_labels_i[r],
                            "ChainRes":     chain_ids[r],
                            "AA_native":    aa_nat,
                            "AA_mutant":    ID_TO_AA.get(d, "X"),
                            "NativeEnergy": float(nat_e_sr[r]),
                            "MutantEnergy": float(mut_e[r, d]),
                            "FrstIndex":    fi,
                        })
                out_df = pd.DataFrame(rows)

                # --- Merge TrueFrstIndex from reference MutationsData ---
                if d_mean is not None:
                    sf = (sf_raw[i] if isinstance(sf_raw, (list, tuple)) else sf_raw)
                    if sf is not None:
                        ref_dir = _find_mutations_data_dir(_Path(sf))
                        if ref_dir is not None:
                            try:
                                ref_df = read_mutations_data(ref_dir)
                                ref_sub = ref_df[["Res", "ChainRes", "AA", "FrstIndex"]].rename(
                                    columns={"AA": "AA_mutant", "FrstIndex": "TrueFrstIndex"})
                                out_df = out_df.merge(ref_sub,
                                                      on=["Res", "ChainRes", "AA_mutant"],
                                                      how="left")
                                valid = out_df["TrueFrstIndex"].notna() & out_df["FrstIndex"].notna()
                                if valid.sum() >= 2:
                                    r2, n = _pearson_r2(out_df.loc[valid, "FrstIndex"].values,
                                                        out_df.loc[valid, "TrueFrstIndex"].values)
                                    stem = _Path(frust_base).name
                                    print(f"[mutations R²] {stem}: r²={r2:.4f}  n={n}")
                            except Exception as exc:
                                print(f"[mutations R²] could not compare {frust_base}: {exc}")

                out_df.to_csv(
                    f"{frust_base}.{name}_singleresidue_mutations",
                    sep=" ", index=False)

            # --- Mutational pair mutations ---
            mut_data = result.get("_mutational_mutations")
            if mut_data is not None:
                all_e   = mut_data["energies"][i].cpu().float()  # [E, 400]
                all_aas = mut_data["aas"].cpu()                  # [2, 400]
                nat_e_mut = result["_mutational_native_energy"][i].cpu().float()  # [E]
                ei = batch["edge_index"][i].cpu()  # [2, E]
                valid_e = (ei >= 0).all(dim=0)
                ei_v  = ei[:, valid_e]
                all_e = all_e[valid_e]       # [E_valid, 400]
                nat_e_v = nat_e_mut[valid_e]

                res1 = (residue_index[i, ei_v[0]].cpu().numpy()
                        if residue_index is not None else (ei_v[0] + 1).numpy())
                res2 = (residue_index[i, ei_v[1]].cpu().numpy()
                        if residue_index is not None else (ei_v[1] + 1).numpy())
                _rl_i = batch.get("res_labels")
                rl1 = ([_rl_i[i][j] for j in ei_v[0].tolist()] if _rl_i else [str(int(r)) for r in res1])
                rl2 = ([_rl_i[i][j] for j in ei_v[1].tolist()] if _rl_i else [str(int(r)) for r in res2])
                aa_src = batch["aa_type"][i, ei_v[0]].cpu()
                aa_dst = batch["aa_type"][i, ei_v[1]].cpu()
                ch1 = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                        for c in chain_index[i, ei_v[0]].cpu().tolist()]
                       if chain_index is not None
                       else ["A"] * ei_v.shape[1])
                ch2 = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                        for c in chain_index[i, ei_v[1]].cpu().tolist()]
                       if chain_index is not None
                       else ["A"] * ei_v.shape[1])

                E_valid = all_e.shape[0]
                rows = []
                for e in range(E_valid):
                    aa1_nat = ID_TO_AA.get(int(aa_src[e]), "X")
                    aa2_nat = ID_TO_AA.get(int(aa_dst[e]), "X")
                    for d in range(400):
                        rows.append({
                            "Res1":         rl1[e],
                            "Res2":         rl2[e],
                            "ChainRes1":    ch1[e],
                            "ChainRes2":    ch2[e],
                            "AA1_native":   aa1_nat,
                            "AA2_native":   aa2_nat,
                            "AA1_mutant":   ID_TO_AA.get(int(all_aas[0, d]), "X"),
                            "AA2_mutant":   ID_TO_AA.get(int(all_aas[1, d]), "X"),
                            "NativeEnergy": float(nat_e_v[e]),
                            "MutantEnergy": float(all_e[e, d]),
                        })
                pd.DataFrame(rows).to_csv(
                    f"{frust_base}.{name}_mutational_mutations",
                    sep=" ", index=False)

            # --- Configurational pair mutations (global decoy samples) ---
            conf_data = result.get("_configurational_mutations")
            if conf_data is not None:
                decoy_e   = conf_data["energies"][i].cpu().float()   # [n_decoys]
                decoy_rt  = conf_data["res_type"][:, i, :].cpu()     # [2, n_decoys]
                nat_e_conf = result["_configurational_native_energy"][i].cpu().float()  # [E]
                conf_mean  = float(decoy_e.mean())
                conf_std   = float(decoy_e.std())
                rows = []
                for d in range(decoy_e.shape[0]):
                    rows.append({
                        "DecoyAA1":    ID_TO_AA.get(int(decoy_rt[0, d]), "X"),
                        "DecoyAA2":    ID_TO_AA.get(int(decoy_rt[1, d]), "X"),
                        "DecoyEnergy": float(decoy_e[d]),
                        "DecoyMean":   conf_mean,
                        "DecoyStd":    conf_std,
                    })
                pd.DataFrame(rows).to_csv(
                    f"{frust_base}.{name}_configurational_mutations",
                    sep=" ", index=False)

    def _draw_figures(self, result, frust_bases, structure_files):
        """Generate per-protein figures synchronously."""
        try:
            from .plots import per_protein as _pp
            from .plots import singleresidue as _sr
            make_per_protein_figure = _pp.make_per_protein_figure
            make_pairwise_figure    = _pp.make_pairwise_figure
        except ImportError as e:
            print(f"[draw_figure] could not import plot modules: {e}")
            return

        _pp._VERBOSE          = self.verbose
        _pp._RENDER_STRUCTURE = self.render_structure
        _pp._FORMATS          = self.image_formats
        _sr._VERBOSE          = self.verbose
        _sr._FORMATS          = self.image_formats

        has_sr     = "singleresidue" in result
        pair_modes = [m for m in ("mutational", "configurational", "smooth") if m in result]

        for frust_base in frust_bases:
            frust_base = Path(frust_base)
            fig_dir    = frust_base.parent.parent / "Visualizations"
            fig_dir.mkdir(parents=True, exist_ok=True)
            stem       = frust_base.name
            pdb_file   = _find_structure_file(frust_base) or Path(f"{frust_base}.pdb")

            if has_sr:
                sr_file = _find_result_file(frust_base, "singleresidue")
                if sr_file is not None:
                    try:
                        make_per_protein_figure(stem, sr_file, pdb_file, fig_dir)
                    except Exception as exc:
                        print(f"[draw_figure] make_per_protein_figure {stem}: {exc}")

            for mode_name in pair_modes:
                pair_file = _find_result_file(frust_base, mode_name)
                if pair_file is None:
                    print(f"[draw_figure] {stem}: no {mode_name} results file next to {frust_base}", flush=True)
                else:
                    try:
                        make_pairwise_figure(stem, pair_file, pdb_file, fig_dir, mode=mode_name)
                    except Exception as exc:
                        print(f"[draw_figure] make_pairwise_figure {stem} {mode_name}: {exc}")

    def _write_pymol_scripts(self, batch, result, frust_bases, names, output_dir):
        """Write one .pml file per (protein, mode) using the same script generators
        as render_structure_pymol / render_pairwise_pymol (out_png=None → no render tail).
        """
        from .plots.per_protein import _pymol_script_regression, _pymol_script_pairwise
        from .data.io import _compute_welltypes
        from .utils.weights import classify_frustration

        edge_index    = batch["edge_index"]
        residue_index = batch.get("residue_index")
        chain_index   = batch.get("chain_index")
        densities     = batch.get("residue_density")
        distances     = batch.get("distance")
        _cid_map_list = batch.get("chain_id_map")

        for i, (frust_base, name) in enumerate(zip(frust_bases, names)):
            out_dir = Path(frust_base).parent.parent / "Visualizations"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem       = Path(frust_base).name
            _cid_map_i = _cid_map_list[i] if _cid_map_list is not None else None

            # Put a copy of the PDB next to the .pml so its "load" line stays local. A copy (not a
            # symlink) survives Dropbox/rsync/zip, which drop or dangle relative symlinks.
            src_pdb = _find_structure_file(frust_base)
            if src_pdb is None:
                print(f"[write_pymol] no structure file ({stem}.pdb / .cif) found next to {frust_base} "
                      f"(was --no-pdb-write used?); {stem}.pml will not be able to load its structure",
                      flush=True)
                pdb_path = Path(f"{stem}.pdb")
            else:
                viz_pdb = out_dir / src_pdb.name
                if viz_pdb.is_symlink() and not viz_pdb.exists():
                    viz_pdb.unlink()                  # dangling link from an earlier run
                if not viz_pdb.exists():
                    import shutil as _sh
                    _sh.copy2(src_pdb, viz_pdb)
                pdb_path = Path(src_pdb.name)   # relative — works when PyMOL is run from Visualizations/

            valid_e = (edge_index[i] >= 0).all(dim=0)
            ei      = edge_index[i][:, valid_e]
            E       = ei.shape[1]

            res1 = (residue_index[i, ei[0]].cpu().numpy()
                    if residue_index is not None else (ei[0] + 1).cpu().numpy())
            res2 = (residue_index[i, ei[1]].cpu().numpy()
                    if residue_index is not None else (ei[1] + 1).cpu().numpy())
            ch1 = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                    for c in chain_index[i, ei[0]].cpu().tolist()]
                   if chain_index is not None else ["A"] * E)
            ch2 = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                    for c in chain_index[i, ei[1]].cpu().tolist()]
                   if chain_index is not None else ["A"] * E)

            wt_arr = (_compute_welltypes(ei, densities[i], distances[i])
                      if densities is not None and distances is not None
                      else torch.zeros(E, dtype=torch.long))

            for mode_name in ("mutational", "configurational", "smooth"):
                if mode_name not in result:
                    continue

                pred_fi  = result[mode_name][i, valid_e].cpu().float()
                # smooth uses the descending (inverted) threshold row; classify_frustration handles it
                pred_cls = classify_frustration(pred_fi, self.coefficients.frustration_thresholds[
                    2 if mode_name == "smooth" else 1]).numpy()

                highly_solid, highly_dashed = [], []
                min_solid,    min_dashed    = [], []
                for e in range(E):
                    cls = int(pred_cls[e])
                    if cls == 1:    # neutral — omit
                        continue
                    r1, r2 = int(res1[e]), int(res2[e])
                    c1, c2 = ch1[e], ch2[e]
                    is_water = (int(wt_arr[e].item()) == 2)
                    bucket = (highly_dashed if is_water else highly_solid) if cls == 0 \
                             else (min_dashed    if is_water else min_solid)
                    bucket.append((r1, c1, r2, c2))

                script = _pymol_script_pairwise(
                    pdb_path, out_png=None,
                    highly_solid=highly_solid, highly_dashed=highly_dashed,
                    min_solid=min_solid,       min_dashed=min_dashed,
                )
                pml_path = out_dir / f"{stem}.{name}_{mode_name}.pml"
                pml_path.write_text(script)
                if self.verbose:
                    print(f"\r  wrote {pml_path}", end="", flush=True)

            if "singleresidue" in result:
                script   = _pymol_script_regression(pdb_path, out_png=None)
                pml_path = out_dir / f"{stem}.{name}_singleresidue.pml"
                pml_path.write_text(script)
                if self.verbose:
                    print(f"\r  wrote {pml_path}", end="", flush=True)

    def _write_mvsj_scripts(self, batch, result, frust_bases, names, output_dir):
        """
        WARNING: This has not been run or tested.

        Write one .mvsj file per (protein, mode) for visualization in Mol*.

        Direct contacts appear as solid tubes; water-mediated contacts appear as
        dashed tubes. highly frustrated=red, minimally frustrated=green, neutral omitted.
        Single-residue mode is colored by the B-factor (frustration index) field.
        """
        from .plots.mvsj import write_contact_mvsj, write_singleresidue_mvsj
        from .data.io import _compute_welltypes
        from .utils.weights import classify_frustration

        edge_index    = batch["edge_index"]
        residue_index = batch.get("residue_index")
        chain_index   = batch.get("chain_index")
        densities     = batch.get("residue_density")
        distances     = batch.get("distance")
        _cid_map_list = batch.get("chain_id_map")

        for i, (frust_base, name) in enumerate(zip(frust_bases, names)):
            out_dir = Path(frust_base).parent.parent / "Visualizations"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(frust_base).name
            _cid_map_i = _cid_map_list[i] if _cid_map_list is not None else None

            valid_e = (edge_index[i] >= 0).all(dim=0)
            ei      = edge_index[i][:, valid_e]
            E       = ei.shape[1]

            res1 = (residue_index[i, ei[0]].cpu().numpy()
                    if residue_index is not None else (ei[0] + 1).cpu().numpy())
            res2 = (residue_index[i, ei[1]].cpu().numpy()
                    if residue_index is not None else (ei[1] + 1).cpu().numpy())
            ch1 = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                    for c in chain_index[i, ei[0]].cpu().tolist()]
                   if chain_index is not None else ["A"] * E)
            ch2 = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                    for c in chain_index[i, ei[1]].cpu().tolist()]
                   if chain_index is not None else ["A"] * E)

            wt_arr = (_compute_welltypes(ei, densities[i], distances[i])
                      if densities is not None and distances is not None
                      else torch.zeros(E, dtype=torch.long))

            for mode_name in ("mutational", "configurational", "smooth"):
                if mode_name not in result:
                    continue

                pred_fi  = result[mode_name][i, valid_e].cpu().float()
                # smooth uses the descending (inverted) threshold row; classify_frustration handles it
                pred_cls = classify_frustration(pred_fi, self.coefficients.frustration_thresholds[
                    2 if mode_name == "smooth" else 1]).numpy()

                contacts = []
                for e in range(E):
                    cls = int(pred_cls[e])
                    if cls == 1:   # neutral — omit
                        continue
                    contacts.append({
                        "chain1":      ch1[e],
                        "residue1":    int(res1[e]),
                        "chain2":      ch2[e],
                        "residue2":    int(res2[e]),
                        "frustration": "highly" if cls == 0 else "minimally",
                        "is_water":    (int(wt_arr[e].item()) == 2),
                    })

                _sf         = _find_structure_file(frust_base)
                struct_url  = f"../FrustrationData/{_sf.name if _sf else stem + '.pdb'}"
                struct_fmt  = {"cif": "mmcif", "bcif": "bcif"}.get(_sf.suffix[1:] if _sf else "pdb", "pdb")
                title      = f"Frustration {mode_name} — {stem}.{name}"
                mvsj_path  = out_dir / f"{stem}.{name}_{mode_name}.mvsj"
                mvsj_path.write_text(write_contact_mvsj(contacts, struct_url, struct_format=struct_fmt, title=title))
                if self.verbose:
                    print(f"\r  wrote {mvsj_path}", end="", flush=True)

            if "singleresidue" in result:
                struct_url = f"../FrustrationData/{stem}.{name}"
                title      = f"Frustration singleresidue — {stem}.{name}"
                mvsj_path  = out_dir / f"{stem}.{name}_singleresidue.mvsj"
                mvsj_path.write_text(write_singleresidue_mvsj(struct_url, title=title))
                if self.verbose:
                    print(f"\r  wrote {mvsj_path}", end="", flush=True)

    def _write_shards(self, batch, result):
        """Extract per-protein tensors from batch/result and send to shard writer."""
        from .data.io import _compute_welltypes

        sf_raw = batch.get("structure_file", None)
        if sf_raw is None:
            B = batch["aa_type"].shape[0]
            names_list = [f"struct_{i}" for i in range(B)]
        elif isinstance(sf_raw, str):
            names_list = [sf_raw]
        else:
            names_list = list(sf_raw)

        B             = batch["aa_type"].shape[0]
        edge_index    = batch.get("edge_index")         # [B, 2, E_max] padded with -1
        residue_dens  = batch.get("residue_density")    # [B, L]
        residue_idx   = batch.get("residue_index")      # [B, L] PDB res numbers
        chain_idx     = batch.get("chain_index")        # [B, L] 0-based int
        chain_id_map  = batch.get("chain_id_map")       # list[dict] per batch item
        distances     = batch.get("distance")           # [B, L, L] or similar

        for i in range(B):
            name = names_list[i] if i < len(names_list) else f"struct_{i}"
            aa   = batch["aa_type"][i]                  # [L_padded]
            mask = aa >= 0
            L    = int(mask.sum())

            _cid_map_i = chain_id_map[i] if chain_id_map is not None else None

            # Column names match the ASCII output files throughout.
            features: dict = {
                "entry": name,
                "AA":    aa[mask].cpu().to(torch.int8),  # [L] int AA index
            }

            pos = batch.get("atom_positions")
            if pos is not None:
                features["X"] = pos[i, mask].cpu()       # [L, n_atoms, 3]

            repair_logs = batch.get("repair_log")
            repair_maps = batch.get("repair_map")
            if repair_logs is not None and i < len(repair_logs):
                features["repair_log"] = repair_logs[i]
            if repair_maps is not None and i < len(repair_maps):
                features["repair_map"] = repair_maps[i]

            # ── per-residue fields (ASCII: Res / ChainRes / DensityRes) ────
            if residue_idx is not None:
                features["Res"] = residue_idx[i, mask].cpu().int()

            if chain_idx is not None:
                features["ChainRes"] = [
                    (_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                    for c in chain_idx[i, mask].cpu().tolist()
                ]

            if residue_dens is not None:
                features["DensityRes"] = residue_dens[i, mask].cpu().float()

            # ── singleresidue (ASCII: NativeEnergy / DecoyEnergy / SDEnergy / FrstIndex / FrstState) ──
            if "singleresidue" in result:
                sr = result["singleresidue"][i]
                fi_sr = sr[mask].cpu().float()
                features["FrstIndex"] = fi_sr
                features["FrstState"] = torch.searchsorted(
                    self.coefficients.frustration_thresholds[0], fi_sr
                ).to(torch.int8)

                sr_nat = result.get("_singleresidue_native_energy")
                if sr_nat is not None:
                    features["NativeEnergy"] = sr_nat[i, mask].cpu().float()

                sr_stats = result.get("_singleresidue_decoy_stats")  # [B, L, 2]
                if sr_stats is not None:
                    ds = sr_stats[i, mask].cpu().float()
                    features["DecoyEnergy"] = ds[:, 0]
                    features["SDEnergy"]    = ds[:, 1]

            # ── mutation scan (ASCII: MutantEnergy / ScanFrstIndex / ScanFrstState) ──
            # Written only when --save-mutations was used (result key present).
            # Res, ChainRes, AA_native (AA), NativeEnergy are already stored above
            # as per-residue scalars; AA_mutant is always 0-19 (implicit).
            sr_mut = result.get("_singleresidue_mutations")
            if sr_mut is not None and "singleresidue" in result:
                GLY_IDX = 5
                aa_cpu  = aa[mask].cpu()
                mut_e   = sr_mut["energies"][i, mask].cpu().float()  # [L, 20]

                # Decoy stats: prefer per-position stats (gly_water_corr applied)
                # over global singleresidue decoy stats.
                d_mean = d_std = None
                mut_stats = sr_mut.get("stats")
                if mut_stats is not None:
                    ds_m   = mut_stats[i, mask].cpu().float()
                    d_mean, d_std = ds_m[:, 0], ds_m[:, 1]
                elif "DecoyEnergy" in features and "SDEnergy" in features:
                    d_mean = features["DecoyEnergy"]
                    d_std  = features["SDEnergy"]

                if d_mean is not None:
                    # Base FI: (DecoyMean - MutantEnergy) / SDEnergy  →  [L, 20]
                    fi_scan = (d_mean.unsqueeze(1) - mut_e) / d_std.unsqueeze(1).clamp(min=1e-8)

                    # Case 2: non-GLY native → GLY decoy (CA geometry correction)
                    gly_fi_all = sr_mut.get("_gly_mutation_fi")
                    if gly_fi_all is not None:
                        gly_fi    = gly_fi_all[i, mask].cpu().float()   # [L]
                        not_gly   = aa_cpu != GLY_IDX
                        valid_col = not_gly & ~gly_fi.isnan()
                        fi_scan[valid_col, GLY_IDX] = gly_fi[valid_col]

                    # Case 1: GLY native → any decoy (vCB geometry correction)
                    gly_nat_fi_all = sr_mut.get("_gly_native_fi")
                    if gly_nat_fi_all is not None:
                        gly_nat_fi = gly_nat_fi_all[i, mask].cpu().float()  # [L, 20]
                        is_gly     = aa_cpu == GLY_IDX
                        fi_scan[is_gly] = gly_nat_fi[is_gly]

                    fi_scan = fi_scan.nan_to_num(0.0)
                    features["MutantEnergy"]  = mut_e                                # [L, 20]
                    features["ScanFrstIndex"] = fi_scan                              # [L, 20]
                    features["ScanFrstState"] = torch.searchsorted(
                        self.coefficients.frustration_thresholds[0],
                        fi_scan.reshape(-1)
                    ).reshape(fi_scan.shape[0], 20).to(torch.int8)                   # [L, 20]

            # ── pairwise modes ─────────────────────────────────────────────
            for mode in ("mutational", "configurational"):
                if mode not in result:
                    continue
                frust = result[mode][i]

                if edge_index is not None:
                    ei    = edge_index[i]                       # [2, E_max]
                    valid = (ei[0] >= 0) & (ei[1] >= 0)
                    ei_v  = ei[:, valid].cpu()

                    fi_pw = frust[valid].cpu().float()
                    features[f"{mode}_FrstIndex"]  = fi_pw
                    features[f"{mode}_FrstState"]  = torch.searchsorted(
                        self.coefficients.frustration_thresholds[1], fi_pw
                    ).to(torch.int8)
                    features[f"{mode}_edge_index"] = ei_v

                    # ASCII: AA1 / AA2
                    features[f"{mode}_AA1"] = aa[ei_v[0]].cpu().to(torch.int8)
                    features[f"{mode}_AA2"] = aa[ei_v[1]].cpu().to(torch.int8)

                    # ASCII: DensityRes1 / DensityRes2
                    if residue_dens is not None:
                        features[f"{mode}_DensityRes1"] = residue_dens[i, ei_v[0]].cpu().float()
                        features[f"{mode}_DensityRes2"] = residue_dens[i, ei_v[1]].cpu().float()

                    # ASCII: Res1 / Res2 / ChainRes1 / ChainRes2
                    if residue_idx is not None:
                        features[f"{mode}_Res1"] = residue_idx[i, ei_v[0]].cpu().int()
                        features[f"{mode}_Res2"] = residue_idx[i, ei_v[1]].cpu().int()

                    if chain_idx is not None:
                        features[f"{mode}_ChainRes1"] = [
                            (_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                            for c in chain_idx[i, ei_v[0]].cpu().tolist()
                        ]
                        features[f"{mode}_ChainRes2"] = [
                            (_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
                            for c in chain_idx[i, ei_v[1]].cpu().tolist()
                        ]

                    # ASCII: Welltype
                    if residue_dens is not None and distances is not None:
                        wt = _compute_welltypes(ei_v, residue_dens[i], distances[i])
                        features[f"{mode}_Welltype"] = wt.cpu().to(torch.int8)

                    # ASCII: NativeEnergy
                    pw_nat = result.get(f"_{mode}_native_energy")
                    if pw_nat is not None:
                        features[f"{mode}_NativeEnergy"] = pw_nat[i, valid].cpu().float()

                    # ASCII: DecoyEnergy / SDEnergy — [B,E,2] per-edge or [B,2] global
                    pw_stats = result.get(f"_{mode}_decoy_stats")
                    if pw_stats is not None:
                        if pw_stats.dim() == 3:
                            ds = pw_stats[i, valid].cpu().float()
                            features[f"{mode}_DecoyEnergy"] = ds[:, 0]
                            features[f"{mode}_SDEnergy"]    = ds[:, 1]
                        else:
                            E = int(valid.sum())
                            features[f"{mode}_DecoyEnergy"] = pw_stats[i, 0].item() * torch.ones(E)
                            features[f"{mode}_SDEnergy"]    = pw_stats[i, 1].item() * torch.ones(E)
                else:
                    features[f"{mode}_FrstIndex"] = frust.cpu().float()

            try:
                self._shard_writer.add(name, features, L)
            except Exception as exc:
                import sys
                print(f"[Shards] ERROR adding {name}: {exc}", file=sys.stderr, flush=True)
                self._shard_writer.record_failure()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        result = self(batch)
        if self._shard_writer is not None:
            self._write_shards(batch, result)
        else:
            self._write_outputs(batch, result, output_dir=self._output_dir)
        return result

    def configure_optimizers(self):
        # No training, so we return None
        return None


