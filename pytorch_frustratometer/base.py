"""Shared base class combining AWSEM energy with Lightning Module infrastructure."""
from __future__ import annotations
from pathlib import Path

# Populated by the CLI after parallel pre-creation so _make_done_paths can skip
# the per-batch mkdir (which is a slow GPFS stat even when the dir exists).
_precreated_dirs: set = set()

# When True, _make_done_paths creates {stem}.in_progress/ instead of {stem}.done/.
# _write_outputs renames to .done after each batch so partial results are visible.
_use_in_progress: bool = False


def _make_done_paths(structure_files, output_dir):
    """Build FrustratometeR-compatible output directories and return base paths.

    Returns (frust_bases, names, done_dirs) where each frust_base points to
    <done_dir>/FrustrationData/<stem> and name is "pdb", producing filenames
    like <stem>.pdb_singleresidue, <stem>.pdb_mutational, etc.

    When _use_in_progress is True, done_dir uses the .in_progress suffix;
    the caller is responsible for renaming to .done after writing.
    """
    suffix = ".in_progress" if _use_in_progress else ".done"
    frust_bases, names, done_dirs = [], [], []
    for sf in structure_files:
        p = Path(sf) if sf else Path("structure")
        parent = Path(output_dir) if output_dir else p.parent
        done_dir = parent / f"{p.stem}{suffix}"
        frust_data_dir = done_dir / "FrustrationData"
        if str(frust_data_dir) not in _precreated_dirs:
            frust_data_dir.mkdir(parents=True, exist_ok=True)
        frust_bases.append(str(frust_data_dir / p.stem))
        names.append(Path(sf).suffix[1:] if sf else "cif")
        done_dirs.append(done_dir)
    return frust_bases, names, done_dirs


def _write_frust_outputs(batch, result, output_dir, write_pdb=True):
    """Write all frustration files present in result.

    Creates <stem>.done/FrustrationData/ for each structure and writes
    <stem>.pdb_singleresidue, <stem>.pdb_mutational, <stem>.pdb_configurational,
    and (when write_pdb=True) <stem>.pdb (structure with frustration in B-factors).

    write_pdb=False skips the biotite PDB read + save, cutting ~2 GPFS file ops
    per protein — useful for comparison runs where only TSV files are needed.

    Returns (frust_bases, names, done_dirs, residue_index, chain_index,
    structure_files) so callers can write additional outputs.
    """
    from .data.io import write_singleresidue_frustration, write_pair_frustration, write_5adens

    sf_raw = batch.get("structure_file")
    batch_size = batch["aa_type"].shape[0]
    if sf_raw is None:
        structure_files = [None] * batch_size
    elif isinstance(sf_raw, str):
        structure_files = [sf_raw] * batch_size
    else:
        structure_files = list(sf_raw)
        # Guard against collation producing fewer items than the model batch size
        if len(structure_files) < batch_size:
            structure_files += [None] * (batch_size - len(structure_files))

    frust_bases, names, done_dirs = _make_done_paths(structure_files, output_dir)
    frust_file = frust_bases[0] if len(frust_bases) == 1 else frust_bases

    # Repair log / residue map from structure parsing: <stem>.<ext>.repair.log / .repair_map.tsv,
    # next to the frustration files (they used to land in the cwd / next to the input).
    for _key, _suffix in (("repair_log", "repair.log"), ("repair_map", "repair_map.tsv")):
        _texts = batch.get(_key)
        if _texts:
            for _fb, _nm, _txt in zip(frust_bases, names, _texts):
                if _txt:
                    Path(f"{_fb}.{_nm}.{_suffix}").write_text(_txt)

    residue_index = batch.get("residue_index")
    chain_index   = batch.get("chain_index")
    chain_id_map  = batch.get("chain_id_map")

    bm_raw = batch.get("backbone_method")
    backbone_method = bm_raw[0] if isinstance(bm_raw, list) and bm_raw else (bm_raw or None)

    sr_l   = result.get("_singleresidue_losses")
    mut_l  = result.get("_mutational_losses")
    conf_l = result.get("_configurational_losses")
    combined_losses = None
    if any(x is not None for x in (sr_l, mut_l, conf_l)):
        import math
        raw = [
            sr_l[0]   if sr_l   else float("nan"),
            sr_l[1]   if sr_l   else float("nan"),
            mut_l[0]  if mut_l  else float("nan"),
            mut_l[1]  if mut_l  else float("nan"),
            conf_l[0] if conf_l else float("nan"),
            conf_l[1] if conf_l else float("nan"),
        ]
        if not all(math.isnan(v) for v in raw):
            combined_losses = raw

    def get_structure_type(f):
        if f is None:
            return "cif"
        suffixes = Path(f).suffixes

        if not suffixes:
            return "cif"
        
        for suffix in suffixes:
            if suffix in (".pdb", ".cif"):
                return suffix[1:]  # strip leading dot
        
        return "cif"  # default if no recognized suffix

    if "singleresidue" in result:
        write_singleresidue_frustration(
            single_frustration_index=result["singleresidue"],
            single_frustration_class=None,
            aatype_int=batch["aa_type"],
            seq_mask=batch["seq_mask"],
            densities=batch["residue_density"],
            frust_file=frust_file,
            atom_positions=batch["atom_positions"] if write_pdb else None,
            residue_index=residue_index,
            chain_index=chain_index,
            chain_id_map=chain_id_map,
            native_energy=result.get("_singleresidue_native_energy"),
            decoy_stats=result.get("_singleresidue_decoy_stats"),
            true_frustration_index=batch.get("single_frust_true"),
            losses=combined_losses,
            metrics=result.get("_singleresidue_metrics"),
            pred=False,
            names=names,
            structure_files=structure_files if write_pdb else None,
            structure_type=[get_structure_type(f) for f in structure_files],
            mode_sep="_",
            backbone_method=backbone_method,
        )

    elif write_pdb and any(m in result for m in ("mutational", "configurational", "mj", "smooth")):
        # Pair-only run: no single-residue FI was computed, but the .pml / Mol* scripts and the
        # contact-map figures still need <stem>.pdb. Write the structure only (B-factors = 0).
        import torch
        write_singleresidue_frustration(
            single_frustration_index=torch.zeros_like(batch["seq_mask"], dtype=torch.float32),
            single_frustration_class=None,
            aatype_int=batch["aa_type"],
            seq_mask=batch["seq_mask"],
            densities=batch["residue_density"],
            frust_file=frust_file,
            atom_positions=batch["atom_positions"],
            residue_index=residue_index,
            chain_index=chain_index,
            chain_id_map=chain_id_map,
            pred=False,
            names=names,
            structure_files=structure_files,
            structure_type=[get_structure_type(f) for f in structure_files],
            mode_sep="_",
            backbone_method=backbone_method,
            structure_only=True,
        )

    for mode_name, mode_idx in [("mutational", 0), ("mj", 0), ("configurational", 1), ("smooth", 0)]:
        if mode_name not in result:
            continue
        raw_stats = result.get(f"_{mode_name}_decoy_stats")
        # dim==3: per-pair [B,E,2]; dim==2: global [B,2] (configurational) — both passed through
        decoy_stats_out = raw_stats if raw_stats is not None and raw_stats.dim() >= 2 else None
        write_pair_frustration(
            mode=mode_idx,
            mode_name=mode_name,
            pair_frustration_index=result[mode_name],
            pair_frustration_class=None,
            invert_classes=(mode_name == "smooth"),
            aatype_int=batch["aa_type"],
            seq_mask=batch["seq_mask"],
            edge_index=batch["edge_index"],
            densities=batch["residue_density"],
            welltypes=None,
            frust_file=frust_file,
            residue_index=residue_index,
            chain_index=chain_index,
            chain_id_map=chain_id_map,
            native_energy=result.get(f"_{mode_name}_native_energy"),
            decoy_stats=decoy_stats_out,
            distances=batch["distance"],
            true_frustration_index=batch.get(f"{mode_name}_frust_true"),
            losses=combined_losses,
            metrics=result.get(f"_{mode_name}_metrics"),
            pred=False,
            names=names,
            mode_sep="_",
            backbone_method=backbone_method,
        )
        if mode_name in ("mutational", "configurational"):
            ca_positions = batch["atom_positions"][:, :, 1, :]
            write_5adens(
                mode_name=mode_name,
                pair_frustration_index=result[mode_name],
                ca_positions=ca_positions,
                seq_mask=batch["seq_mask"],
                edge_index=batch["edge_index"],
                frust_file=frust_file,
                names=names,
                residue_index=residue_index,
                chain_index=chain_index,
                mode_sep="_",
            )

    return frust_bases, names, done_dirs, residue_index, chain_index, structure_files

import torch
import lightning.pytorch as pl

from .utils.density import compute_residue_densities as _compute_residue_densities
from .algorithm.awsem import AWSEM
from .utils.utils import compute_frustration_metrics
from .coefficients import Coefficients


class _FromPDBMixin:
    """Convenience mixin that adds from_pdb to any class with self.coefficients."""

    def from_pdb(self, pdb_path, **dataset_kwargs):
        """Run the model on a single PDB or mmCIF file and return the result dict."""
        import torch
        from .data.dataset import FrustratometerDataset, collate_frustratometer
        dataset = FrustratometerDataset([str(pdb_path)], self.coefficients, **dataset_kwargs)
        batch = collate_frustratometer([dataset[0]])
        with torch.no_grad():
            return self(batch)


class FrustratometerBase(_FromPDBMixin, AWSEM, pl.LightningModule):
    """
    Base for all mode-specific frustration classes.

    MRO: FrustratometerBase → AWSEM → pl.LightningModule → nn.Module
    AWSEM.__init__ calls super().__init__() which reaches pl.LightningModule and nn.Module
    cooperatively, so both are fully initialised before AWSEM loads its gamma buffers.

    Subclasses must override: forward(), _check_r_threshold().
    Optionally override _write_extra_outputs() for mutation tables or other extras.
    """

    def __init__(self, coefficients: Coefficients, output_dir=None,
                 dtype: torch.dtype = torch.float32):
        super().__init__(coefficients, dtype=dtype)   # → AWSEM.__init__ → LM → nn.Module
        self._dtype = dtype
        self._output_dir = output_dir

        # Async scratch→GPFS copy executor (mirrors Frustratometer3 interface)
        self._copy_executor = None
        self._copy_futures: list = []

        # Test-time accumulation state
        self._r_failures: list = []
        self._r_results: list = []
        self._residue_samples: dict = {}
        self._csv_headers_written: set = set()
        self._flush_interval = 50
        self._batch_count = 0
        self._test_ok = 0
        self._test_fail_r = 0
        self._test_fail_acc = 0

    # ------------------------------------------------------------------
    # Batch helpers (same as Frustratometer3)
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_batch(batch, i):
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
        merged = {}
        for key in results[0]:
            vals = [r[key] for r in results]
            if isinstance(vals[0], torch.Tensor):
                merged[key] = torch.cat(vals, dim=0)
            else:
                merged[key] = vals
        return merged

    def compute_densities(self, batch):
        batch["residue_density"], batch["burial_density"], _ = _compute_residue_densities(
            batch["atom_positions"], batch["aa_type"], batch["seq_mask"],
            batch["hetatm"], batch["residue_index"], batch["distance"],
            self.coefficients,
            chain_index=batch.get("chain_index"), edge_index=batch.get("edge_index"),
            dtype=self._dtype,
        )

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def test_step(self, batch, batch_idx):
        result = self(batch)
        self._check_r_threshold(batch, result)
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        result = self(batch)
        self._write_outputs(batch, result, output_dir=self._output_dir)
        return result

    def configure_optimizers(self):
        return None

    # ------------------------------------------------------------------
    # Mode-specific hooks — subclasses override forward and _check_r_threshold
    # ------------------------------------------------------------------

    def forward(self, batch):
        raise NotImplementedError

    def _check_r_threshold(self, batch, result, r_min: float = 0.99, acc_min: float = 0.9):
        raise NotImplementedError

    def _write_outputs(self, batch, result, output_dir=None):
        import pytorch_frustratometer.base as _base_mod
        frust_bases, names, done_dirs, residue_index, chain_index, _ = \
            _write_frust_outputs(batch, result, output_dir)
        self._write_extra_outputs(
            batch, result, frust_bases, names, done_dirs,
            residue_index, chain_index, output_dir)
        if _base_mod._use_in_progress:
            for d in done_dirs:
                d = Path(d)
                if d.name.endswith(".in_progress"):
                    target = d.parent / (d.name.removesuffix(".in_progress") + ".done")
                    try:
                        d.rename(target)
                    except OSError:
                        pass

    def _write_extra_outputs(self, batch, result, frust_bases, names, done_dirs,
                             residue_index, chain_index, output_dir):
        """Hook for subclasses to write mutation tables or other per-mode extras."""

    # ------------------------------------------------------------------
    # Shared metric checking loop (called by subclasses with mode_cfg)
    # ------------------------------------------------------------------

    def _run_mode_check(self, batch, result, mode_cfg, r_min=0.99, acc_min=0.9):
        """
        Core per-mode metric loop.  mode_cfg is a list of
        (mode_name, pred_key, true_key, mask_key, threshold) tuples.
        """
        import sys
        import random

        B = batch["aa_type"].shape[0]
        sf_raw = batch.get("structure_file")
        if isinstance(sf_raw, str):
            names_list = [sf_raw]
        elif sf_raw is not None:
            names_list = list(sf_raw)
        else:
            names_list = [f"struct_{i}" for i in range(B)]

        debug_dir = (Path(self._output_dir) / "run_info"
                     if self._output_dir else Path("run_info"))
        n_ok = n_fail_r = n_fail_acc = 0

        for i in range(B):
            name = names_list[i] if i < len(names_list) else f"struct_{i}"
            for mode_name, pred_key, true_key, mask_key, threshold in mode_cfg:
                pred = result.get(pred_key)
                true = batch.get(true_key)
                if pred is None or true is None:
                    continue
                mask = batch.get(mask_key, batch["seq_mask"])

                m = compute_frustration_metrics(pred[i:i+1], true[i:i+1], mask[i:i+1], threshold)
                r   = m.get("pearson_r",     float("nan"))
                acc = m.get("class_accuracy", float("nan"))
                if r != r and acc != acc:
                    continue

                r_fail   = (r == r)   and (r   < r_min)
                acc_fail = (acc == acc) and (acc < acc_min)

                _AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
                mk = batch.get(mask_key, batch["seq_mask"])[i:i+1].bool()
                p_valid  = pred[i:i+1][mk].cpu().tolist()
                t_valid  = true[i:i+1][mk].cpu().tolist()
                aa_type_i = batch.get("aa_type")
                if (aa_type_i is not None
                        and mk[0].shape == aa_type_i[i].shape):
                    aa_valid = aa_type_i[i][mk[0]].cpu().tolist()
                else:
                    aa_valid = [None] * len(p_valid)
                if p_valid:
                    n_samp = min(30, len(p_valid))
                    idx_s  = random.sample(range(len(p_valid)), n_samp)
                    self._residue_samples.setdefault(mode_name, []).extend(
                        [(p_valid[k], t_valid[k],
                          _AA_LETTERS[aa_valid[k]] if aa_valid[k] is not None else "")
                         for k in idx_s])

                self._r_results.append({
                    "structure": name, "mode": mode_name,
                    "pearson_r": r, "class_accuracy": acc,
                    "std_ratio": m.get("std_ratio",        float("nan")),
                    "pred_std":  m.get("pred_std",          float("nan")),
                    "true_std":  m.get("true_std",          float("nan")),
                    "mse":       m.get("mse",               float("nan")),
                    "mae":       m.get("mae",               float("nan")),
                    "spearman_r":m.get("spearman_r",        float("nan")),
                    "cosine_similarity": m.get("cosine_similarity", float("nan")),
                    "n":         m.get("n", 0),
                })

                if not r_fail and not acc_fail:
                    n_ok += 1
                    continue

                if r_fail:   n_fail_r   += 1
                if acc_fail: n_fail_acc += 1
                reasons = (["R={:.4f}".format(r)] if r_fail else []) + \
                          (["acc={:.4f}".format(acc)] if acc_fail else [])

                lines = [
                    f"[FAIL({','.join(reasons)})] mode={mode_name!r}  structure={name!r}",
                    f"  pearson_r={r:.4f}  class_accuracy={acc:.4f}  n={m.get('n','?')}",
                    f"  mse={m.get('mse',float('nan')):.4f}  mae={m.get('mae',float('nan')):.4f}"
                    f"  spearman_r={m.get('spearman_r',float('nan')):.4f}"
                    f"  cosine_sim={m.get('cosine_similarity',float('nan')):.4f}",
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
                    debug_file = debug_dir / f"{Path(name).stem}.{mode_name}.debug.txt"
                    debug_file.write_text(debug_text)
                except Exception as exc:
                    print(f"[FAIL write-error {exc}]\n{debug_text}", file=sys.stderr, flush=True)

                self._r_failures.append({
                    "structure": name, "mode": mode_name,
                    "reason": ",".join(reasons),
                    "pearson_r": r, "class_accuracy": acc,
                    "std_ratio": m.get("std_ratio",        float("nan")),
                    "pred_std":  m.get("pred_std",          float("nan")),
                    "true_std":  m.get("true_std",          float("nan")),
                    "mse":       m.get("mse",               float("nan")),
                    "mae":       m.get("mae",               float("nan")),
                    "spearman_r":m.get("spearman_r",        float("nan")),
                    "cosine_similarity": m.get("cosine_similarity", float("nan")),
                    "n":         m.get("n", 0),
                    "debug_file": str(debug_file) if debug_file else "",
                })

        self._test_ok       += n_ok
        self._test_fail_r   += n_fail_r
        self._test_fail_acc += n_fail_acc
        try:
            pbar = self.trainer.progress_bar_callback.test_progress_bar
            pbar.set_postfix(ok=self._test_ok, fail_r=self._test_fail_r,
                             fail_acc=self._test_fail_acc, refresh=True)
        except Exception:
            pass

        self._batch_count += 1
        if self._batch_count % self._flush_interval == 0:
            self._flush_buffers()

    def _flush_buffers(self):
        import csv
        if not self._r_results and not self._residue_samples:
            return
        debug_dir = (Path(self._output_dir) / "run_info"
                     if self._output_dir else Path("run_info"))
        debug_dir.mkdir(parents=True, exist_ok=True)

        if self._r_results:
            fields = ["structure", "mode", "pearson_r", "spearman_r", "cosine_similarity",
                      "class_accuracy", "std_ratio", "pred_std", "true_std", "mse", "mae", "n"]
            path = debug_dir / "all_results.csv"
            write_header = "all_results" not in self._csv_headers_written
            with open(path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                    self._csv_headers_written.add("all_results")
                writer.writerows(self._r_results)
            self._r_results.clear()

        for mode_name, pairs in self._residue_samples.items():
            if not pairs:
                continue
            path = debug_dir / f"residue_samples_{mode_name}.csv"
            key = f"samples_{mode_name}"
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
        debug_dir = (Path(self._output_dir) / "run_info"
                     if self._output_dir else Path("run_info"))
        debug_dir.mkdir(parents=True, exist_ok=True)

        if self._r_failures:
            fields = ["structure", "mode", "reason", "pearson_r", "spearman_r",
                      "cosine_similarity", "class_accuracy",
                      "std_ratio", "pred_std", "true_std", "mse", "mae", "n", "debug_file"]
            rows = sorted(self._r_failures, key=lambda x: x["pearson_r"])
            with open(debug_dir / "failures_by_r.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        self._flush_buffers()
