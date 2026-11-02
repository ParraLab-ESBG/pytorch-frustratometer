"""
WARNING: This has not been run or tested.

plot_self_consistency.py
Scatter plot comparing two independent FrustratometeR runs to show self-consistency.
Uses the same visual style as fig2_scatter.

Usage (on server):
    python docs/plot_self_consistency.py \\
        --dir1  /gpfs/.../dompdb_v4.4 \\
        --dir2  /gpfs/.../dompdb_v4.4_rerun \\
        --label1 "FrustratometeR Run 1" \\
        --label2 "FrustratometeR Run 2" \\
        --out   docs/self_consistency.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pytorch_frustratometer.data.io import parse_singleresidue_frustration
from pytorch_frustratometer.plots.common import (
    PALETTE, MODE_THRESHOLDS, LO_SR, HI_SR, LO_PAIR, HI_PAIR,
    draw_scatter_hexbin,
)
from pytorch_frustratometer.plots.paper.paper_figures import (
    _build_file_index,
    _savefig,
)

sns.set_theme(style="whitegrid", font_scale=1.15)

MODES = ["singleresidue", "mutational", "configurational"]
MODE_LABELS = {
    "singleresidue": "Single-residue",
    "mutational": "Mutational",
    "configurational": "Configurational",
}
SUFFIXES = {
    "singleresidue": "pdb_singleresidue",
    "mutational": "pdb_mutational",
    "configurational": "pdb_configurational",
}


def _load_singleresidue_pair(path1: Path, path2: Path):
    """Read two singleresidue files and return matched (vals1, vals2) numpy arrays."""
    vals1, states1, mask1, extra1 = parse_singleresidue_frustration(
        str(path1), residue_lookup=None, seq_mask=None, max_len=None,
        include_extra_fields=True,
    )
    vals2, states2, mask2, extra2 = parse_singleresidue_frustration(
        str(path2), residue_lookup=None, seq_mask=None, max_len=None,
        include_extra_fields=True,
    )

    v1, v2 = vals1.numpy(), vals2.numpy()
    res1, res2 = extra1.get("Res"), extra2.get("Res")

    if res1 is None or res2 is None or len(v1) == 0 or len(v2) == 0:
        n = min(len(v1), len(v2))
        return v1[:n], v2[:n]

    chain1 = extra1.get("ChainRes")
    chain2 = extra2.get("ChainRes")
    if chain1 is not None and chain2 is not None:
        keys1 = list(zip(chain1, res1))
        keys2 = list(zip(chain2, res2))
    else:
        keys1 = list(res1)
        keys2 = list(res2)

    key_to_v2 = dict(zip(keys2, v2))
    matched1, matched2 = [], []
    for k, fv1 in zip(keys1, v1):
        if k in key_to_v2:
            matched1.append(fv1)
            matched2.append(key_to_v2[k])

    if not matched1:
        n = min(len(v1), len(v2))
        return v1[:n], v2[:n]

    return np.array(matched1, dtype=np.float32), np.array(matched2, dtype=np.float32)


def _load_pair_pair(path1: Path, path2: Path):
    """Read two mutational/configurational files and return matched (vals1, vals2)."""
    df1 = pd.read_csv(str(path1), sep=" ", comment="#")
    df2 = pd.read_csv(str(path2), sep=" ", comment="#")

    fi_col = next((c for c in ("FrstIndex", "TrueFrstIndex") if c in df1.columns), None)
    if fi_col is None or fi_col not in df2.columns:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    has_chain = "ChainRes1" in df1.columns and "ChainRes1" in df2.columns
    if has_chain:
        keys1 = list(zip(df1["ChainRes1"], df1["Res1"], df1["ChainRes2"], df1["Res2"]))
        keys2 = list(zip(df2["ChainRes1"], df2["Res1"], df2["ChainRes2"], df2["Res2"]))
    else:
        keys1 = list(zip(df1["Res1"], df1["Res2"]))
        keys2 = list(zip(df2["Res1"], df2["Res2"]))

    v1 = df1[fi_col].values.astype(np.float32)
    v2 = df2[fi_col].values.astype(np.float32)

    key_to_v2 = dict(zip(keys2, v2))
    matched1, matched2 = [], []
    for k, fv1 in zip(keys1, v1):
        if k in key_to_v2:
            matched1.append(fv1)
            matched2.append(key_to_v2[k])

    return np.array(matched1, dtype=np.float32), np.array(matched2, dtype=np.float32)


def load_mode(dir1: Path, dir2: Path, mode: str):
    """Return (vals1, vals2) numpy arrays across all common proteins for this mode."""
    from concurrent.futures import ThreadPoolExecutor

    idx1 = _build_file_index(dir1, mode)
    idx2 = _build_file_index(dir2, mode)
    common = sorted(set(idx1) & set(idx2))
    print(f"  {len(common)} proteins in common", flush=True)
    if not common:
        return None, None

    def _process(stem):
        try:
            if mode == "singleresidue":
                v1, v2 = _load_singleresidue_pair(idx1[stem], idx2[stem])
            else:
                v1, v2 = _load_pair_pair(idx1[stem], idx2[stem])
            ok = np.isfinite(v1) & np.isfinite(v2)
            return v1[ok], v2[ok]
        except Exception as e:
            print(f"  warning: {stem} — {e}", flush=True)
            return None, None

    all_v1, all_v2 = [], []
    n_workers = min(128, len(common))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_process, common):
            if result[0] is not None and len(result[0]) > 0:
                all_v1.append(result[0])
                all_v2.append(result[1])

    if not all_v1:
        return None, None
    return np.concatenate(all_v1), np.concatenate(all_v2)


def _load_from_parquet(precomputed_dir: Path, mode: str):
    """Load (fi_1, fi_2) from pre-computed parquet. Returns (None, None) if unavailable."""
    path = precomputed_dir / f"fr_consistency_{mode}.parquet"
    if not path.exists():
        return None, None
    try:
        df = pd.read_parquet(path)
        v1 = df["fi_1"].to_numpy(dtype=np.float32)
        v2 = df["fi_2"].to_numpy(dtype=np.float32)
        ok = np.isfinite(v1) & np.isfinite(v2)
        print(f"  [parquet/{mode}] {ok.sum():,} rows loaded from {path}")
        return v1[ok], v2[ok]
    except Exception as exc:
        print(f"  [parquet/{mode}] read failed: {exc}")
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir1",   required=True, help="FrustratometeR run 1 directory")
    ap.add_argument("--dir2",   required=True, help="FrustratometeR run 2 directory")
    ap.add_argument("--label1", default="FrustratometeR Run 1")
    ap.add_argument("--label2", default="FrustratometeR Run 2")
    ap.add_argument("--precomputed-dir", default=None,
                    help="Directory of parquet files from precompute_cath_data.py")
    ap.add_argument("--out",    default="self_consistency.png")
    args = ap.parse_args()

    dir1, dir2 = Path(args.dir1), Path(args.dir2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    precomputed_dir = Path(args.precomputed_dir) if args.precomputed_dir else None

    _cmaps = {m: mcolors.LinearSegmentedColormap.from_list(f"_c_{m}", ["#d0d0d0", PALETTE[m]])
              for m in MODES}

    all_samples: dict[str, tuple] = {}
    for mode in MODES:
        print(f"Loading {mode}…", flush=True)
        if precomputed_dir is not None:
            v1, v2 = _load_from_parquet(precomputed_dir, mode)
            if v1 is None:
                print(f"  parquet not found, loading from files…", flush=True)
                v1, v2 = load_mode(dir1, dir2, mode)
        else:
            v1, v2 = load_mode(dir1, dir2, mode)
        all_samples[mode] = (v1, v2)

    sr_v1, _ = all_samples.get("singleresidue", (None, None))
    n_res_total = len(sr_v1) if sr_v1 is not None else 0  # residues only (not contacts)
    sr_idx1 = _build_file_index(dir1, "singleresidue")
    sr_idx2 = _build_file_index(dir2, "singleresidue")
    n_proteins = len(set(sr_idx1) & set(sr_idx2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), gridspec_kw={"wspace": 0.25})

    for col, mode in enumerate(MODES):
        ax = axes[col]
        v1, v2 = all_samples[mode]
        lo, hi = MODE_THRESHOLDS.get(mode, (LO_PAIR, HI_PAIR))

        cos_line = None
        if v1 is not None and len(v1) >= 2:
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom > 0:
                cos_line = f"Avg cos sim = {float(np.dot(v1, v2) / denom):.6f}"

        ax.set_title(MODE_LABELS[mode], fontsize=11, fontweight="bold", pad=6)
        if v1 is not None:
            draw_scatter_hexbin(ax, v1, v2, title="",
                                cmap=_cmaps[mode], xlabel="", ylabel="",
                                line_color=PALETTE[mode], extra_line=cos_line,
                                gly_mask=None, lo=lo, hi=hi)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_xlabel(args.label1, fontsize=10)
        ax.set_ylabel(args.label2 if col == 0 else "", fontsize=10)

    fig.text(0.5, 1.02, "FrustratometeR self-consistency",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.005,
             f"CATH v4.4 S40 • {n_proteins:,} proteins",
             ha="center", va="top", fontsize=10, color="#666")

    _savefig(fig, out)
    plt.close(fig)
    print(f"Saved → {out} (+ .svg)", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
