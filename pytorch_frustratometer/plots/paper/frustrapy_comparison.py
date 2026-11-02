"""
Compare frustrapy (Hana's implementation) against the PyTorch re-implementation
and/or the original FrustratometeR output.

frustrapy output layout:
  <frustrapy_dir>/<stem>_awsem_singleresidue.csv   — Res,ChainRes,AA,NativeEnergy,DecoySD,FrstIndex,ProteinID
  <frustrapy_dir>/<stem>_awsem_mutational.csv      — Res1,Res2,ChainRes1,ChainRes2,AA1,AA2,FrstIndex,ProteinID
  <frustrapy_dir>/<stem>_awsem_configurational.csv — same as mutational

PyTorch re-implementation layout (from pytorch_frustratometer):
  <pytorch_dir>/<stem>.frustratometer3.singleresidue   — space-sep, cols include FrstIndex TrueFrstIndex
  <pytorch_dir>/<stem>.frustratometer3.mutational      — same
  <pytorch_dir>/<stem>.frustratometer3.configurational — same

FrustratometeR reference layout (same space-sep format, no TrueFrstIndex):
  <ref_dir>/<stem>.done/FrustrationData/<stem>.pdb_singleresidue
  <ref_dir>/<stem>.done/FrustrationData/<stem>.pdb_mutational
  <ref_dir>/<stem>.done/FrustrationData/<stem>.pdb_configurational
  (also accepted flat: <ref_dir>/<stem>.pdb_singleresidue)

Comparisons run:
  --pytorch_dir  → frustrapy vs pytorch  (+ frustrapy vs FrustratometeR ref via TrueFrstIndex)
  --ref_dir      → frustrapy vs FrustratometeR directly from reference files

Usage:
  python -m frustrai_fold.plots.frustrapy_comparison \\
      --frustrapy_dir  results/frustrapy_baseline \\
      --pytorch_dir    /gpfs/projects/bsc08/bsc504628/frustration/frust_verify_results \\
      --ref_dir        /gpfs/projects/bsc08/bsc504628/frustration/dompdb_v4.4 \\
      --out            results/frustrapy_comparison
"""

import argparse
import contextlib
import sys
from pathlib import Path

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common import (
    PALETTE, LO_SR, HI_SR, LO_PAIR, HI_PAIR,
    fi_class, compute_metrics,
    draw_kde, draw_confusion, draw_scatter_hexbin,
)

MODES = ["singleresidue", "mutational", "configurational"]
MODE_LABELS = {
    "singleresidue":   "Single-residue",
    "mutational":      "Mutational",
    "configurational": "Configurational",
}
_LO  = {"singleresidue": LO_SR,   "mutational": LO_PAIR, "configurational": LO_PAIR}
_HI  = {"singleresidue": HI_SR,   "mutational": HI_PAIR, "configurational": HI_PAIR}

_XLAB_FP  = "frustrapy"
_XLAB_PT  = "PyTorch Re-implementation"
_XLAB_REF = "FrustratometeR (reference)"


_REF_SUFFIX = {
    "singleresidue":   "pdb_singleresidue",
    "mutational":      "pdb_mutational",
    "configurational": "pdb_configurational",
}


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_frustrapy(path: Path, mode: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df["FrstIndex"] = pd.to_numeric(df["FrstIndex"], errors="coerce")
    df = df.dropna(subset=["FrstIndex"])
    if mode == "singleresidue":
        need = {"Res", "ChainRes", "FrstIndex"}
    else:
        need = {"Res1", "Res2", "ChainRes1", "ChainRes2", "FrstIndex"}
    if not need.issubset(df.columns):
        return None
    return df


def _find_ref_file(stem: str, ref_dir: Path, mode: str) -> Path | None:
    suffix = _REF_SUFFIX[mode]
    candidates = [
        ref_dir / f"{stem}.done" / "FrustrationData" / f"{stem}.{suffix}",
        ref_dir / f"{stem}.{suffix}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_pytorch(path: Path, mode: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#")
    except Exception:
        return None
    for c in ("FrstIndex", "TrueFrstIndex"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "FrstIndex" not in df.columns:
        return None
    if mode == "singleresidue":
        need = {"Res", "ChainRes", "FrstIndex"}
    else:
        need = {"Res1", "Res2", "ChainRes1", "ChainRes2", "FrstIndex"}
    if not need.issubset(df.columns):
        return None
    return df.dropna(subset=["FrstIndex"])


def _res_rank_map(series: pd.Series) -> dict:
    """Map sorted unique residue numbers to 0-based position indices."""
    vals = np.sort(pd.to_numeric(series, errors="coerce").dropna().unique())
    return {v: i for i, v in enumerate(vals)}


def _merge(fp_df: pd.DataFrame, pt_df: pd.DataFrame, mode: str) -> pd.DataFrame | None:
    """Align frustrapy and pytorch frames, handling different residue numbering.

    frustrapy Res is always sequential 1..N; pytorch Res is original PDB numbering
    (may have gaps).  For singleresidue, sorting by Res puts both in protein order
    regardless of numbering so positional zip is correct.  For pair modes we first
    convert Res1/Res2 to dense rank indices (0-based position) before sorting, so
    pair (i,j) in frustrapy and pair (i,j) in pytorch refer to the same residue pair.
    """
    if mode == "singleresidue":
        for df in (fp_df, pt_df):
            df["Res"] = pd.to_numeric(df["Res"], errors="coerce")
        fp_s = fp_df.sort_values("Res").reset_index(drop=True)
        pt_s = pt_df.sort_values("Res").reset_index(drop=True)

        if len(fp_s) != len(pt_s):
            return None

        # Validate AA agreement — confirms positional alignment is correct.
        # Allow ≤5% mismatch for MSE/MET or other variant residue differences.
        if "AA" in fp_s.columns and "AA" in pt_s.columns:
            if np.mean(fp_s["AA"].values == pt_s["AA"].values) < 0.95:
                return None

    else:
        # Pair modes differ in two ways:
        #   1. Residue numbering: frustrapy sequential 1..N, pytorch PDB numbers.
        #   2. Pair coverage: frustrapy reports all N×N finite pairs; pytorch only
        #      reports pairs within the 9.5 Å contact cutoff.
        #
        # Strategy: convert each DF's Res to a dense rank (position index within
        # that DF), then inner-join on (rank1, rank2) to get the intersection of
        # pairs both tools computed.  Ranks are only comparable when both DFs cover
        # the same residues (same count), so skip proteins where residue sets differ.
        for col in ("Res1", "Res2"):
            fp_df[col] = pd.to_numeric(fp_df[col], errors="coerce")
            pt_df[col] = pd.to_numeric(pt_df[col], errors="coerce")

        fp_rank = _res_rank_map(pd.concat([fp_df["Res1"], fp_df["Res2"]]))
        pt_rank = _res_rank_map(pd.concat([pt_df["Res1"], pt_df["Res2"]]))

        if len(fp_rank) != len(pt_rank):
            return None  # different residue counts → ranks don't correspond

        fp_df = fp_df.copy()
        pt_df = pt_df.copy()
        fp_df["_r1"] = fp_df["Res1"].map(fp_rank)
        fp_df["_r2"] = fp_df["Res2"].map(fp_rank)
        pt_df["_r1"] = pt_df["Res1"].map(pt_rank)
        pt_df["_r2"] = pt_df["Res2"].map(pt_rank)

        # Inner join on (rank1, rank2): keeps only pairs reported by both tools.
        fp_key = fp_df.set_index(["_r1", "_r2"])
        pt_key = pt_df.set_index(["_r1", "_r2"])
        common = fp_key.index.intersection(pt_key.index)
        if len(common) < 2:
            return None

        fp_s = fp_key.loc[common].reset_index()
        pt_s = pt_key.loc[common].reset_index()

        # Validate AA1 agreement on joined pairs.
        if "AA1" in fp_s.columns and "AA1" in pt_s.columns:
            if np.mean(fp_s["AA1"].values == pt_s["AA1"].values) < 0.95:
                return None

    extra = ["TrueFrstIndex"] if "TrueFrstIndex" in pt_s.columns else []
    merged = pd.DataFrame({"fp_fi": fp_s["FrstIndex"].to_numpy(dtype=float),
                           "pt_fi": pt_s["FrstIndex"].to_numpy(dtype=float)})
    for col in extra:
        merged[col] = pt_s[col].to_numpy(dtype=float)

    merged = merged[np.isfinite(merged["fp_fi"]) & np.isfinite(merged["pt_fi"])]
    return merged if not merged.empty else None


# ── per-protein processing ────────────────────────────────────────────────────

def _process_stem(stem: str, frustrapy_dir: Path, mode: str,
                  pytorch_dir: Path | None = None,
                  ref_dir: Path | None = None,
                  ) -> dict | None:
    """Return a dict with arrays and metrics for one protein.

    Keys in returned dict:
      fp, pt (optional), ref (optional) — aligned numpy arrays
      row — metrics dict for DataFrame
    """
    fp_path = frustrapy_dir / f"{stem}_awsem_{mode}.csv"
    if not fp_path.exists():
        return None
    fp_df = _load_frustrapy(fp_path, mode)
    if fp_df is None:
        return None

    # --- pytorch ---
    pt_arr = ref_from_pt = fp_for_pt = None
    if pytorch_dir is not None:
        pt_path = pytorch_dir / f"{stem}.frustratometer3.{mode}"
        if pt_path.exists():
            pt_df = _load_pytorch(pt_path, mode)
            if pt_df is not None:
                merged_pt = _merge(fp_df, pt_df, mode)
                if merged_pt is not None:
                    ok = np.isfinite(merged_pt["fp_fi"]) & np.isfinite(merged_pt["pt_fi"])
                    if ok.sum() >= 2:
                        pt_arr       = merged_pt["pt_fi"][ok].to_numpy(dtype=float)
                        fp_for_pt    = merged_pt["fp_fi"][ok].to_numpy(dtype=float)
                        if "TrueFrstIndex" in merged_pt.columns:
                            ref_from_pt = merged_pt["TrueFrstIndex"][ok].to_numpy(dtype=float)

    # --- direct reference ---
    ref_arr = fp_for_ref = None
    if ref_dir is not None:
        ref_path = _find_ref_file(stem, ref_dir, mode)
        if ref_path is not None:
            ref_df = _load_pytorch(ref_path, mode)  # same space-sep format
            if ref_df is not None:
                merged_ref = _merge(fp_df, ref_df, mode)
                if merged_ref is not None:
                    ok2 = np.isfinite(merged_ref["fp_fi"]) & np.isfinite(merged_ref["pt_fi"])
                    if ok2.sum() >= 2:
                        fp_for_ref = merged_ref["fp_fi"][ok2].to_numpy(dtype=float)
                        ref_arr    = merged_ref["pt_fi"][ok2].to_numpy(dtype=float)

    # need at least one comparison
    if pt_arr is None and ref_arr is None:
        return None

    # use whichever fp alignment is available (pt takes priority for pt metrics,
    # ref for ref metrics — they may differ in length)
    fp_arr = fp_for_pt if pt_arr is not None else fp_for_ref

    def _m(a, b):
        if a is None or b is None:
            return {}
        ok = np.isfinite(a) & np.isfinite(b)
        return compute_metrics(a[ok], b[ok]) if ok.sum() >= 2 else {}

    row = {"stem": stem}
    if pt_arr is not None:
        row["n_pt"] = len(pt_arr)
        m = _m(fp_for_pt, pt_arr)
        row.update({"fp_pt_r": m.get("pearson_r"), "fp_pt_cos": m.get("cosine_similarity"),
                    "fp_pt_acc": m.get("class_accuracy")})
        if ref_from_pt is not None:
            m2 = _m(fp_for_pt, ref_from_pt)
            row.update({"fp_ref_r": m2.get("pearson_r"), "fp_ref_cos": m2.get("cosine_similarity"),
                        "fp_ref_acc": m2.get("class_accuracy")})
            m3 = _m(pt_arr, ref_from_pt)
            row.update({"pt_ref_r": m3.get("pearson_r"), "pt_ref_cos": m3.get("cosine_similarity"),
                        "pt_ref_acc": m3.get("class_accuracy")})
    if ref_arr is not None:
        row["n_ref"] = len(ref_arr)
        m = _m(fp_for_ref, ref_arr)
        # only overwrite if not already set from TrueFrstIndex
        row.setdefault("fp_ref_r",   m.get("pearson_r"))
        row.setdefault("fp_ref_cos", m.get("cosine_similarity"))
        row.setdefault("fp_ref_acc", m.get("class_accuracy"))

    return {"row": row, "fp_pt": (fp_for_pt, pt_arr),
            "fp_ref": (fp_for_ref if ref_arr is not None else fp_for_pt, ref_arr),
            "pt_ref": (pt_arr, ref_from_pt)}


# ── plotting helpers ──────────────────────────────────────────────────────────

def _kde_panel(ax, vals, color, xlabel, title, vline=None):
    from scipy.stats import gaussian_kde
    v = vals[np.isfinite(vals)]
    if len(v) < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    kde = gaussian_kde(v)
    xs  = np.linspace(v.min() - 0.05, v.max() + 0.05, 300)
    ax.fill_between(xs, kde(xs), alpha=0.25, color=color)
    ax.plot(xs, kde(xs), color=color, lw=1.5)
    med = float(np.nanmedian(v))
    ax.axvline(med, color="k", ls="--", lw=1.0)
    ax.text(0.04, 0.95, f"median={med:.4f}", transform=ax.transAxes,
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))
    if vline is not None:
        ax.axvline(vline, color="#aaa", ls=":", lw=1.0)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(title, fontsize=9)


# ── main comparison logic ────────────────────────────────────────────────────

@contextlib.contextmanager
def _tqdm_joblib(tqdm_bar):
    """Patch joblib so completed batches tick the given tqdm bar."""
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_bar.close()


def compare_mode(mode: str, frustrapy_dir: Path, out_dir: Path,
                 pytorch_dir: Path | None = None,
                 ref_dir: Path | None = None,
                 n_jobs: int = -1):
    color  = PALETTE.get(mode, "#555555")

    fp_stems = {p.stem.replace(f"_awsem_{mode}", "")
                for p in frustrapy_dir.glob(f"*_awsem_{mode}.csv")}

    # union of stems available in either pytorch or ref
    candidate_stems = set(fp_stems)
    if pytorch_dir is not None:
        pt_stems = {p.stem.replace(".frustratometer3", "")
                    for p in pytorch_dir.glob(f"*.frustratometer3.{mode}")}
        candidate_stems &= pt_stems
        print(f"  {mode}: {len(fp_stems)} frustrapy, {len(pt_stems)} pytorch, "
              f"{len(candidate_stems)} overlap", end="")
    if ref_dir is not None:
        # don't restrict stems — ref lookup is per-stem
        print(f"  (+ ref_dir supplied)", end="")
    print()

    stems = sorted(candidate_stems if candidate_stems else fp_stems)
    if not stems:
        print(f"  [skip] {mode}: no overlapping stems")
        return

    rows = []
    pools: dict[str, list] = {"fp_pt": [], "fp_ref": [], "pt_ref": []}

    with _tqdm_joblib(tqdm(total=len(stems), desc=f"{mode}", unit="prot")):
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_stem)(stem, frustrapy_dir, mode,
                                   pytorch_dir=pytorch_dir, ref_dir=ref_dir)
            for stem in stems
        )
    for res in results:
        if res is None:
            continue
        rows.append(res["row"])
        for key in pools:
            a, b = res[key]
            if a is not None and b is not None:
                ok = np.isfinite(a) & np.isfinite(b)
                pools[key].extend(list(zip(a[ok].tolist(), b[ok].tolist())))

    if not rows:
        print(f"  [skip] {mode}: no data after merging")
        return

    df_m = pd.DataFrame(rows)
    csv_path = out_dir / f"frustrapy_comparison_{mode}.csv"
    df_m.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}  ({len(df_m)} proteins)")

    def _pool_arrays(key):
        pairs = pools[key]
        if not pairs:
            return None, None
        a, b = zip(*pairs)
        return np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    arr_fp_pt,  arr_pt    = _pool_arrays("fp_pt")
    arr_fp_ref, arr_ref   = _pool_arrays("fp_ref")
    arr_pt2,    arr_ref2  = _pool_arrays("pt_ref")

    for label, a, b in [("frustrapy vs pytorch",     arr_fp_pt,  arr_pt),
                        ("frustrapy vs FrustratometeR", arr_fp_ref, arr_ref),
                        ("pytorch vs FrustratometeR",  arr_pt2,    arr_ref2)]:
        if a is not None:
            m = compute_metrics(a, b)
            print(f"  {mode} {label}: r={m['pearson_r']:.4f}  "
                  f"acc={m['class_accuracy']:.4f}  n={m['n']:,}")

    # ── figure: one row per comparison ────────────────────────────────────────
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(f"_c_{mode}", ["#f0f0f0", color])

    comparisons = []
    if arr_fp_pt  is not None: comparisons.append((arr_fp_pt,  arr_pt,   _XLAB_FP,  _XLAB_PT,  "fp_pt_r",  "fp_pt_cos"))
    if arr_fp_ref is not None: comparisons.append((arr_fp_ref, arr_ref,  _XLAB_FP,  _XLAB_REF, "fp_ref_r", "fp_ref_cos"))
    if arr_pt2    is not None: comparisons.append((arr_pt2,    arr_ref2, _XLAB_PT,  _XLAB_REF, "pt_ref_r", "pt_ref_cos"))

    if not comparisons:
        return

    fig, axes = plt.subplots(len(comparisons), 3,
                             figsize=(15, 4.5 * len(comparisons)),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.3},
                             squeeze=False)

    for row_idx, (pred_arr, true_arr, pred_lbl, true_lbl, r_col, cos_col) in enumerate(comparisons):
        ok = np.isfinite(pred_arr) & np.isfinite(true_arr)
        draw_scatter_hexbin(axes[row_idx, 0], pred_arr[ok], true_arr[ok],
                            title=f"{pred_lbl} vs {true_lbl}  (n={ok.sum():,})",
                            cmap=cmap, xlabel=true_lbl, ylabel=pred_lbl,
                            line_color=color)
        for col_idx, (metric_col, xlabel) in enumerate(
                [(r_col, "Pearson R"), (cos_col, "Cosine similarity")], start=1):
            vals = df_m[metric_col].dropna().to_numpy(dtype=float) if metric_col in df_m.columns else np.array([])
            _kde_panel(axes[row_idx, col_idx], vals, color, xlabel,
                       f"{xlabel}  (med={np.nanmedian(vals):.4f})" if len(vals) else xlabel,
                       vline=0.99 if "r_" in metric_col else None)

    fig.suptitle(f"frustrapy comparison  —  {MODE_LABELS[mode]}",
                 fontsize=13, fontweight="bold")
    out = out_dir / f"frustrapy_comparison_{mode}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frustrapy_dir", required=True,
                        help="Dir with {stem}_awsem_{mode}.csv files")
    parser.add_argument("--pytorch_dir",  default=None,
                        help="Dir with {stem}.frustratometer3.{mode} files (optional)")
    parser.add_argument("--ref_dir",      default=None,
                        help="FrustratometeR output root: looks for "
                             "<stem>.done/FrustrationData/<stem>.pdb_{mode} (optional)")
    parser.add_argument("--out",          default="results/frustrapy_comparison",
                        help="Output directory")
    parser.add_argument("--modes", nargs="+",
                        default=MODES, choices=MODES,
                        help="Modes to compare (default: all three)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Parallel workers for per-protein processing (-1=all CPUs)")
    args = parser.parse_args()

    if args.pytorch_dir is None and args.ref_dir is None:
        parser.error("at least one of --pytorch_dir or --ref_dir is required")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        print(f"\n── {MODE_LABELS[mode]} ──")
        compare_mode(mode, Path(args.frustrapy_dir), out_dir,
                     pytorch_dir=Path(args.pytorch_dir) if args.pytorch_dir else None,
                     ref_dir=Path(args.ref_dir)         if args.ref_dir     else None,
                     n_jobs=args.n_jobs)

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
