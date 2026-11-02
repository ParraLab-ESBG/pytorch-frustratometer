"""
Plot frustration benchmark results produced by frustratometer3.py test run.

Expected inputs (written to run_info/ by on_test_epoch_end):
  all_results.csv           — one row per (protein, mode)
  residue_samples_*.csv     — sampled (pred, true) pairs per mode

Usage:
  python -m frustrai_fold.plots.frustration_results \
      --results baseline_comparison/run_info/all_results.csv \
      --samples-dir baseline_comparison/run_info \
      --out baseline_comparison/plots

Produces:
  summary_4x3.png            — combined 4-row × 3-col figure
                               rows: scatter | class-accuracy KDE | confusion matrix | cosine KDE
                               cols: singleresidue | mutational | configurational
  (plus individual PNGs for each plot type)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

from ..common import MODE_THRESHOLDS, fi_class, draw_confusion

MODES = ["singleresidue", "mutational", "configurational"]
PALETTE = {"singleresidue": "#2196F3", "mutational": "#4CAF50", "configurational": "#FF5722"}


def _draw_scatter(ax, pred, true, mode, color):
    slope, intercept, r, *_ = linregress(true, pred)
    r2 = r ** 2
    xlin = np.array([true.min(), true.max()])
    ax.hexbin(true, pred, gridsize=80, cmap="Blues", mincnt=1)
    ax.plot(xlin, slope * xlin + intercept, "r-", lw=1.5, label=f"R²={r2:.3f}")
    ax.plot(xlin, xlin, "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("True FI")
    ax.set_ylabel("Pred FI")
    ax.set_title(f"{mode}\n(n={len(pred):,})", fontsize=9)
    ax.legend(fontsize=8)


def scatter_plot(pred, true, mode, out_dir, color):
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_scatter(ax, pred, true, mode, color)
    fig.tight_layout()
    path = out_dir / f"scatter_{mode}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def _draw_acc_kde(ax, values, color, vline=0.9):
    values = values[np.isfinite(values)]
    sns.kdeplot(values, ax=ax, color=color, fill=True, alpha=0.3, linewidth=2)
    ax.axvline(float(np.nanmedian(values)), color="k", ls="--", lw=1,
               label=f"med={np.nanmedian(values):.3f}")
    if vline is not None:
        ax.axvline(vline, color="red", ls=":", lw=1, label=f"thr={vline}")
    ax.set_xlabel("Class accuracy")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def kde_plot(values, xlabel, title, out_path, color, vline=None):
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.kdeplot(values, ax=ax, color=color, fill=True, alpha=0.3, linewidth=2)
    ax.axvline(float(np.nanmedian(values)), color="k", ls="--", lw=1,
               label=f"median={np.nanmedian(values):.3f}")
    if vline is not None:
        ax.axvline(vline, color="red", ls=":", lw=1, label=f"threshold={vline}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def r_vs_n_plot(df_mode, mode, out_dir, color):
    df = df_mode.dropna(subset=["pearson_r", "n"]).copy()
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["pearson_r"] = pd.to_numeric(df["pearson_r"], errors="coerce")
    df = df.dropna(subset=["pearson_r", "n"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=df, x="n", y="pearson_r", ax=ax, alpha=0.25, s=8,
                    color=color, rasterized=True, linewidth=0)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Protein length (n residues)")
    ax.set_ylabel("Pearson R")
    ax.set_title(f"{mode}  –  R vs protein length")
    fig.tight_layout()
    path = out_dir / f"r_vs_n_{mode}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def _draw_confusion(ax, pred, true, mode, cbar=False):
    lo, hi = MODE_THRESHOLDS.get(mode, (-1.0, 0.78))
    draw_confusion(ax, pred, true, lo=lo, hi=hi, cbar=cbar)


def confusion_plot(pred, true, mode, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    _draw_confusion(ax, pred, true, mode, cbar=True)
    ax.set_title(f"{mode}  –  confusion matrix")
    fig.tight_layout()
    path = out_dir / f"confusion_{mode}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def _load_samples(samples_dir, mode):
    """Load and clean residue_samples_{mode}.csv. Returns (pred_arr, true_arr) or (None, None)."""
    path = samples_dir / f"residue_samples_{mode}.csv"
    if not path.is_file():
        return None, None
    sdf = pd.read_csv(path, low_memory=False)
    sdf = sdf[sdf["pred"] != "pred"].copy()
    sdf["pred"] = pd.to_numeric(sdf["pred"], errors="coerce")
    sdf["true"] = pd.to_numeric(sdf["true"], errors="coerce")
    sdf = sdf.dropna(subset=["pred", "true"])
    return sdf["pred"].to_numpy(dtype=float), sdf["true"].to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",     required=True, help="all_results.csv")
    parser.add_argument("--samples-dir", required=True, help="directory with residue_samples_*.csv")
    parser.add_argument("--out",         default="plots", help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(args.results)
    samples_dir = Path(args.samples_dir)

    # ── Collect data for all modes ────────────────────────────────────────────
    mode_data = {}   # mode → {pred, true, df_mode}
    for mode in MODES:
        df_mode = df_all[df_all["mode"] == mode].copy()
        pred_arr, true_arr = _load_samples(samples_dir, mode)
        mode_data[mode] = {"df": df_mode, "pred": pred_arr, "true": true_arr}

    # ── Individual plots (existing behaviour) ────────────────────────────────
    for mode in MODES:
        print(f"\n── {mode} ──")
        d = mode_data[mode]
        df_mode = d["df"]
        color = PALETTE[mode]

        if df_mode.empty:
            print(f"  no data for mode={mode}, skipping")
            continue

        if d["pred"] is not None:
            scatter_plot(d["pred"], d["true"], mode, out_dir, color)
            confusion_plot(d["pred"], d["true"], mode, out_dir)
        else:
            print(f"  residue_samples_{mode}.csv not found — skipping scatter/confusion")

        kde_plot(df_mode["pearson_r"].to_numpy(dtype=float),
                 "Pearson R", f"{mode}  –  R per protein",
                 out_dir / f"r_kde_{mode}.png", color, vline=0.99)
        kde_plot(df_mode["class_accuracy"].to_numpy(dtype=float),
                 "Class accuracy", f"{mode}  –  class accuracy per protein",
                 out_dir / f"acc_kde_{mode}.png", color, vline=0.9)
        kde_plot(df_mode["std_ratio"].to_numpy(dtype=float),
                 "std_ratio (pred/true)", f"{mode}  –  std ratio per protein",
                 out_dir / f"std_ratio_{mode}.png", color, vline=1.0)
        if "cosine_similarity" in df_mode.columns:
            kde_plot(df_mode["cosine_similarity"].to_numpy(dtype=float),
                     "Cosine similarity", f"{mode}  –  cosine similarity per protein",
                     out_dir / f"cosine_{mode}.png", color, vline=None)
        r_vs_n_plot(df_mode, mode, out_dir, color)

    # ── Combined 4×3 summary figure ───────────────────────────────────────────
    # rows: scatter | class-accuracy KDE | confusion matrix | cosine similarity KDE
    # cols: singleresidue | mutational | configurational
    print("\n── combined 4×3 figure ──")
    fig, axes = plt.subplots(4, 3, figsize=(14, 17))
    row_labels = ["Scatter (pred vs true FI)",
                  "Class accuracy (per protein)",
                  "Confusion matrix",
                  "Cosine similarity (per protein)"]

    for col, mode in enumerate(MODES):
        d = mode_data[mode]
        df_mode = d["df"]
        color = PALETTE[mode]
        pred_arr, true_arr = d["pred"], d["true"]

        # Row 0: scatter
        ax = axes[0, col]
        if pred_arr is not None:
            _draw_scatter(ax, pred_arr, true_arr, mode, color)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(mode, fontsize=9)

        # Row 1: class-accuracy KDE
        ax = axes[1, col]
        if not df_mode.empty:
            acc_vals = pd.to_numeric(df_mode["class_accuracy"], errors="coerce").dropna().to_numpy()
            _draw_acc_kde(ax, acc_vals, color, vline=0.9)
            ax.set_title(f"median={np.nanmedian(acc_vals):.3f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)

        # Row 2: confusion matrix (no per-panel cbar to save space)
        ax = axes[2, col]
        if pred_arr is not None:
            _draw_confusion(ax, pred_arr, true_arr, mode, cbar=False)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)

        # Row 3: cosine similarity KDE
        ax = axes[3, col]
        if not df_mode.empty and "cosine_similarity" in df_mode.columns:
            cos_vals = pd.to_numeric(df_mode["cosine_similarity"], errors="coerce").dropna().to_numpy()
            if len(cos_vals) > 0:
                sns.kdeplot(cos_vals, ax=ax, color=color, fill=True, alpha=0.3, linewidth=2)
                med = float(np.nanmedian(cos_vals))
                ax.axvline(med, color="k", ls="--", lw=1, label=f"med={med:.3f}")
                ax.set_xlabel("Cosine similarity")
                ax.set_ylabel("Density")
                ax.set_title(f"median={med:.3f}", fontsize=9)
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "no cosine data", ha="center", va="center",
                    transform=ax.transAxes)

    # Column headers (mode names) on top row
    for col, mode in enumerate(MODES):
        axes[0, col].set_title(
            f"{mode}\n(n={len(mode_data[mode]['pred']):,})" if mode_data[mode]["pred"] is not None
            else mode,
            fontsize=10, fontweight="bold",
        )

    # Row labels on the left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label + "\n" + axes[row, 0].get_ylabel(),
                                fontsize=9, labelpad=6)

    fig.suptitle("Frustratometer3 baseline  –  all modes", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    combined_path = out_dir / "summary_4x3.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {combined_path}")

    print(f"\nDone. Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
