"""
Generate KDE plots of true FI distributions for Fig 1 SVG.
Reads TrueFrstIndex from pytorch_frustratometer output files in frust_verify_results.
Outputs: docs/kde_singleresidue.png, docs/kde_pairwise.png
Run from repo root: python -m frustrai_fold.plots.fig1_kde
"""

import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = "/gpfs/projects/bsc08/bsc504628/frustration/frust_verify_results"
DOCS_DIR    = os.path.join(ROOT, "docs")

LO_SR,   HI_SR   = -1.0, 0.55
LO_PAIR, HI_PAIR = -1.0, 0.78

COLOR_HF  = "#CC2222"
COLOR_NEU = "#BBBBBB"
COLOR_MF  = "#22AA44"
FONT      = "Helvetica Neue, Arial, sans-serif"


def load_true_fi(suffix, max_files=2000, seed=42):
    """Load TrueFrstIndex from space-sep *.frustratometer3.<suffix> files."""
    pattern = os.path.join(RESULTS_DIR, f"*.frustratometer3.{suffix}")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    if len(files) > max_files:
        files = random.Random(seed).sample(files, max_files)
    vals = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=r"\s+", comment="#")
            if "TrueFrstIndex" in df.columns:
                vals.append(pd.to_numeric(df["TrueFrstIndex"], errors="coerce").dropna().values)
        except Exception:
            pass
    print(f"  {suffix}: loaded {len(files)} files, {sum(len(v) for v in vals):,} values")
    return np.concatenate(vals) if vals else np.array([])


def plot_kde(ax, values, lo, hi, xlim=(-5, 5)):
    v = values[np.isfinite(values)]
    v = v[(v >= xlim[0]) & (v <= xlim[1])]
    if len(v) < 10:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    kde = gaussian_kde(v, bw_method=0.15)
    x   = np.linspace(xlim[0], xlim[1], 600)
    y   = kde(x)

    ax.fill_between(x, y, where=(x < lo),              color=COLOR_HF,  alpha=0.75, linewidth=0)
    ax.fill_between(x, y, where=(x >= lo) & (x <= hi), color=COLOR_NEU, alpha=0.75, linewidth=0)
    ax.fill_between(x, y, where=(x > hi),              color=COLOR_MF,  alpha=0.75, linewidth=0)
    ax.plot(x, y, color="#222", linewidth=1.0)

    y_top = float(y.max())
    for thresh, label in [(lo, str(lo)), (hi, str(hi))]:
        ax.axvline(thresh, color="#555", linestyle="--", linewidth=0.9, alpha=0.85)
        ax.text(thresh, y_top * 1.06, label, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#222", fontfamily=FONT,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.set_xlim(xlim)
    ax.set_ylim(0, y_top * 1.30)
    ax.set_xlabel("Frustration Index", fontsize=7.5, labelpad=2, fontfamily=FONT)
    ax.set_ylabel("Density", fontsize=7.5, labelpad=3, fontfamily=FONT)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    sr_vals   = load_true_fi("singleresidue")
    mut_vals  = load_true_fi("mutational")
    conf_vals = load_true_fi("configurational")
    pair_vals = np.concatenate([x for x in [mut_vals, conf_vals] if len(x)])

    # Single-Residue KDE
    fig, ax = plt.subplots(figsize=(4.3, 2.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_kde(ax, sr_vals, LO_SR, HI_SR)
    plt.tight_layout(pad=0.4)
    out = os.path.join(DOCS_DIR, "kde_singleresidue.svg")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")

    # Mutational + Configurational KDE
    fig, ax = plt.subplots(figsize=(4.3, 2.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_kde(ax, pair_vals, LO_PAIR, HI_PAIR)
    plt.tight_layout(pad=0.4)
    out = os.path.join(DOCS_DIR, "kde_pairwise.svg")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")

    # Mutational
    fig, ax = plt.subplots(figsize=(4.3, 2.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_kde(ax, mut_vals, LO_PAIR, HI_PAIR)
    plt.tight_layout(pad=0.4)
    out = os.path.join(DOCS_DIR, "kde_mutational.svg")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")

    # Configurational KDE
    fig, ax = plt.subplots(figsize=(4.3, 2.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plot_kde(ax, conf_vals, LO_PAIR, HI_PAIR)
    plt.tight_layout(pad=0.4)
    out = os.path.join(DOCS_DIR, "kde_configurational.svg")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
