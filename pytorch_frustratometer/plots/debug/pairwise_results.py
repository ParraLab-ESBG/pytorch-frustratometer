"""
Plot pairwise frustration benchmark results (mutational and configurational).

Reads *.frustratometer3.mutational and *.frustratometer3.configurational files,
which already contain both predicted and true FrstIndex / FrstState columns.

Usage:
  python -m frustrai_fold.plots.pairwise_results \\
      --results_dir baseline_comparison \\
      --out          baseline_comparison/plots/pairwise \\
      --mode         both          # mutational | configurational | both
      --sample       2ga8A00       # optional: protein to show contact maps for

Produces per-mode:
  contact_map_{mode}_{stem}.png   — contact map (upper=true class, lower=pred FI)
                                    with R² and cosine similarity box on top
  scatter_{mode}.png              — aggregated hexbin pred vs true FI
  confusion_{mode}.png            — class confusion matrix
  r_kde_{mode}.png                — Pearson R per protein
  acc_kde_{mode}.png              — class accuracy per protein
  cosine_kde_{mode}.png           — cosine similarity per protein
  welltype_{mode}.png             — R/acc/cos breakdown by well type (short vs long)
  summary_{mode}.png              — combined 2×3 figure

Plus:
  pairwise_metrics.csv            — per-protein per-mode metrics table
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from scipy.stats import linregress

from ..common import (
    CLASS_COLORS as _CLASS_COLORS, CMAP_FI, NORM_FI,
    LO_PAIR, HI_PAIR, fi_class, compute_metrics, draw_confusion,
)

PALETTE = {
    "mutational":      "#4CAF50",
    "configurational": "#FF5722",
}


def _state_to_int(state_series: pd.Series) -> np.ndarray:
    """Convert FrstState string column to 0/1/2 ints."""
    mapping = {"highly": 0, "neutral": 1, "minimally": 2}
    return state_series.str.lower().map(mapping).fillna(1).astype(int).to_numpy()


def load_pair_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#")
    except Exception:
        return None
    needed = {"Res1", "Res2", "FrstIndex", "TrueFrstIndex"}
    if not needed.issubset(df.columns):
        return None
    for col in ("Res1", "Res2", "FrstIndex", "TrueFrstIndex"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Res1", "Res2", "FrstIndex", "TrueFrstIndex"])
    return df


# ── contact map (plot_pair) ───────────────────────────────────────────────────

def _class_color(val: int) -> list:
    return _CLASS_COLORS.get(int(val), [0.0, 0.0, 0.0])


def _per_residue_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of col aggregated over both endpoints of each pair, indexed by Res."""
    both = pd.concat([
        df[["Res1", col]].rename(columns={"Res1": "Res"}),
        df[["Res2", col]].rename(columns={"Res2": "Res"}),
    ])
    return both.groupby("Res")[col].mean().sort_index()


def plot_contact_map(df: pd.DataFrame, stem: str, mode: str, out_dir: Path,
                     metrics: dict | None = None):
    """
    Contact map ported from ConfigurationalAnalysis.ipynb plot_pair:
      upper triangle (x=res2, y=res1) → true class  (red / grey / green)
      lower triangle (x=res1, y=res2) → pred FI     (continuous red–grey–green)
      ax_top  → per-residue mean pair frustration (true vs pred, with fill)
      ax_right → R², cosine similarity, class accuracy stats box
      ax_bottom → colorbar + class legend
    """
    res1_raw = df["Res1"].astype(int).to_numpy()
    res2_raw = df["Res2"].astype(int).to_numpy()
    pred_fi  = df["FrstIndex"].to_numpy(dtype=float)
    true_cls = _state_to_int(df["TrueFrstState"]) if "TrueFrstState" in df.columns \
               else fi_class(df["TrueFrstIndex"].to_numpy(dtype=float))

    # Sequential 0-based residue indices
    all_res = np.unique(np.concatenate([res1_raw, res2_raw]))
    idx     = {r: i for i, r in enumerate(all_res)}
    i_arr   = np.array([idx[r] for r in res1_raw])
    j_arr   = np.array([idx[r] for r in res2_raw])
    L       = len(all_res)

    if metrics is None:
        metrics = compute_metrics(pred_fi, df["TrueFrstIndex"].to_numpy(dtype=float))

    # Per-residue mean frustration (sequential x-axis positions)
    true_mean_by_res = _per_residue_mean(df, "TrueFrstIndex")
    pred_mean_by_res = _per_residue_mean(df, "FrstIndex")
    # Align to sequential positions (some residues may only appear in one)
    x_seq   = np.arange(L)
    true_load = np.array([true_mean_by_res.get(r, np.nan) for r in all_res])
    pred_load = np.array([pred_mean_by_res.get(r, np.nan) for r in all_res])

    # ── figure layout (matches notebook: 3-row × 2-col) ──────────────────────
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4, 0.8],
        hspace=0.1,
        wspace=0.05,
    )
    ax_map    = fig.add_subplot(gs[1, 0])
    ax_top    = fig.add_subplot(gs[0, 0], sharex=ax_map)
    ax_right  = fig.add_subplot(gs[1, 1])          # stats box (replaces MAE panel)
    ax_bottom = fig.add_subplot(gs[2, 0], sharex=ax_map)

    # ── ax_top: per-residue mean pair frustration ─────────────────────────────
    _red   = _CLASS_COLORS[0]
    _green = _CLASS_COLORS[2]

    ax_top.plot(x_seq, true_load, color="#1f77b4", linewidth=2,
                label="FrustratometeR (Ref)")
    ax_top.plot(x_seq, pred_load, color="#ff7f0e", linewidth=2,
                linestyle="--", label="Pred")

    ok = np.isfinite(true_load) & np.isfinite(pred_load)
    if ok.any():
        ax_top.fill_between(x_seq[ok], true_load[ok], pred_load[ok],
                            where=true_load[ok] >= pred_load[ok],
                            facecolor="#ff7f0e", alpha=0.25, interpolate=True)
        ax_top.fill_between(x_seq[ok], true_load[ok], pred_load[ok],
                            where=true_load[ok] <= pred_load[ok],
                            facecolor="#1f77b4", alpha=0.25, interpolate=True)

    ax_top.axhline(LO_PAIR, color=_red,   linestyle="--", linewidth=1, alpha=0.7)
    ax_top.axhline(HI_PAIR, color=_green, linestyle="--", linewidth=1, alpha=0.7)
    ax_top.text(L + 0.5, LO_PAIR, "MAX", va="top",    ha="left",
                fontsize=12, weight="bold", color=_red)
    ax_top.text(L + 0.5, HI_PAIR, "MIN", va="bottom", ha="left",
                fontsize=12, weight="bold", color=_green)

    ax_top.set_ylabel(r"Mean pair frustration $\bar{F}_i$", fontsize=14)
    ax_top.set_xticks([])
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.set_ylim(4, -4)
    ax_top.legend(loc="upper left", frameon=False, fontsize=12, ncol=2)

    # ── ax_right: R² / cosine / class accuracy stats box ─────────────────────
    ax_right.axis("off")
    r2  = metrics.get("r2",  float("nan"))
    cos = metrics.get("cosine_similarity", float("nan"))
    acc = metrics.get("class_accuracy",    float("nan"))
    n   = metrics.get("n", 0)
    stats_lines = [
        f"R²  =  {r2:.4f}",
        f"Cosine  =  {cos:.4f}",
        f"Accuracy  =  {acc:.4f}",
        f"n  =  {n:,}",
    ]
    ax_right.text(
        0.5, 0.5, "\n".join(stats_lines),
        transform=ax_right.transAxes,
        ha="center", va="center",
        fontsize=14, fontweight="bold", linespacing=2.0,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5",
                  edgecolor="#AAAAAA", linewidth=1.2),
    )

    # ── ax_map: contact map scatter ───────────────────────────────────────────
    true_colors = [_class_color(c) for c in true_cls]
    pred_rgba   = plt.cm.ScalarMappable(cmap=CMAP_FI, norm=NORM_FI).to_rgba(pred_fi)

    # upper triangle: true class  (x=j, y=i)
    ax_map.scatter(j_arr, i_arr, s=10, c=true_colors,  linewidths=0.25)
    # lower triangle: pred FI     (x=i, y=j)
    ax_map.scatter(i_arr, j_arr, s=10, c=pred_rgba,    linewidths=0.25)

    ax_map.plot([0, L - 1], [0, L - 1], color="black", linewidth=0.5)

    step   = max(1, L // 10)
    ticks  = np.arange(0, L, step)
    tlbls  = [str(all_res[t]) for t in ticks]
    ax_map.set_xticks(ticks); ax_map.set_xticklabels(tlbls, fontsize=8, rotation=45)
    ax_map.set_yticks(ticks); ax_map.set_yticklabels(tlbls, fontsize=8)
    ax_map.set_xlabel("Residue j", fontsize=12)
    ax_map.set_ylabel("Residue i", fontsize=12)
    ax_map.set_aspect("equal")
    ax_map.set_xlim(-0.5, L - 0.5)
    ax_map.set_ylim(-0.5, L - 0.5)
    ax_map.spines["top"].set_visible(False)
    ax_map.spines["right"].set_visible(False)
    ax_map.text(0, L + 3, "Predicted Index", fontsize=11, weight="bold", va="top")
    ax_map.text(L, -3, "True Class",         fontsize=11, weight="bold",
                va="top", ha="right")

    # ── ax_bottom: colorbar + class legend ────────────────────────────────────
    ax_bottom.axis("off")
    sm = matplotlib.cm.ScalarMappable(cmap=CMAP_FI, norm=NORM_FI)
    sm.set_array([])
    ax_cb = inset_axes(ax_bottom, width="30%", height="30%", loc="upper left",
                       bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                       bbox_transform=ax_bottom.transAxes, borderpad=0)
    cbar = plt.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cbar.set_label("Predicted frustration index", fontsize=12)

    legend_elements = [
        Patch(facecolor=[*_CLASS_COLORS[0], 1], label="Highly frustrated"),
        Patch(facecolor=[*_CLASS_COLORS[1], 1], label="Neutral"),
        Patch(facecolor=[*_CLASS_COLORS[2], 1], label="Minimally frustrated"),
    ]
    ax_bottom.legend(handles=legend_elements, loc="upper right",
                     fontsize=12, frameon=False, title="True class", title_fontsize=12)

    fig.suptitle(f"{stem}  –  {mode} frustration", fontsize=20, fontweight="bold")
    path = out_dir / f"contact_map_{mode}_{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ── aggregate plot helpers ────────────────────────────────────────────────────

def _draw_scatter(ax, pred, true, mode, n_total, color):
    ok = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[ok], true[ok]
    if len(p) < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    slope, intercept, r, *_ = linregress(t, p)
    r2 = r ** 2
    xlin = np.array([t.min(), t.max()])
    ax.hexbin(t, p, gridsize=80, cmap="Blues", mincnt=1)
    ax.plot(xlin, slope * xlin + intercept, "r-", lw=1.5, label=f"R²={r2:.3f}")
    ax.plot(xlin, xlin, "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("True FI")
    ax.set_ylabel("Pred FI")
    ax.set_title(f"{mode}  (n={n_total:,} pairs)", fontsize=9)
    ax.legend(fontsize=8)


def _draw_kde(ax, values, xlabel, color, vline=None):
    v = values[np.isfinite(values)]
    if len(v) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    sns.kdeplot(v, ax=ax, color=color, fill=True, alpha=0.3, linewidth=2)
    med = float(np.nanmedian(v))
    ax.axvline(med, color="k", ls="--", lw=1, label=f"med={med:.3f}")
    if vline is not None:
        ax.axvline(vline, color="red", ls=":", lw=1, label=f"thr={vline}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)


def _draw_confusion(ax, pred, true, cbar=False):
    draw_confusion(ax, pred, true, cbar=cbar)


def _welltype_breakdown(df_all: pd.DataFrame, mode: str, out_dir: Path):
    if "welltype" not in df_all.columns:
        return
    rows = []
    for wt, grp in df_all.groupby("welltype"):
        m = compute_metrics(grp["FrstIndex"].to_numpy(dtype=float),
                            grp["TrueFrstIndex"].to_numpy(dtype=float))
        rows.append({"welltype": wt, **m})
    if not rows:
        return
    wdf = pd.DataFrame(rows)
    metrics_cols = ["pearson_r", "class_accuracy", "cosine_similarity"]
    x = np.arange(len(metrics_cols))
    width = 0.6 / max(len(wdf), 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, (_, row) in enumerate(wdf.iterrows()):
        vals = [row[m] for m in metrics_cols]
        ax.bar(x + k * width - width * len(wdf) / 2,
               vals, width, label=row["welltype"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Pearson R", "Class acc", "Cosine sim"])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{mode}  –  metrics by well type")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"welltype_{mode}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


# ── main processing per mode ──────────────────────────────────────────────────

def process_mode(mode: str, results_dir: Path, out_dir: Path,
                 sample_stem: str | None) -> pd.DataFrame:
    ext = f".frustratometer3.{mode}"
    files = sorted(results_dir.glob(f"*{ext}"))
    if not files:
        print(f"  No *{ext} files found in {results_dir}")
        return pd.DataFrame()

    print(f"\n── {mode} ({len(files)} proteins) ──")
    color = PALETTE[mode]

    per_protein = []
    all_pred, all_true = [], []
    all_df_rows = []

    contact_map_plotted = False
    for f in files:
        stem = f.name.split(".frustratometer3")[0]
        df = load_pair_file(f)
        if df is None or df.empty:
            continue
        p = df["FrstIndex"].to_numpy(dtype=float)
        t = df["TrueFrstIndex"].to_numpy(dtype=float)
        m = compute_metrics(p, t)
        per_protein.append({"stem": stem, "mode": mode, **m})
        all_pred.extend(p.tolist())
        all_true.extend(t.tolist())
        if "Welltype" in df.columns:
            df = df.rename(columns={"Welltype": "welltype"})
        if "welltype" in df.columns:
            all_df_rows.append(df[["FrstIndex", "TrueFrstIndex", "welltype"]])

        want_map = (sample_stem and stem == sample_stem) or \
                   (not sample_stem and not contact_map_plotted)
        if want_map:
            plot_contact_map(df, stem, mode, out_dir, metrics=m)
            contact_map_plotted = True

    if not per_protein:
        return pd.DataFrame()

    df_metrics = pd.DataFrame(per_protein)
    all_pred = np.asarray(all_pred, dtype=float)
    all_true = np.asarray(all_true, dtype=float)

    overall = compute_metrics(all_pred, all_true)
    print(f"  Overall:  r={overall['pearson_r']:.4f}  r²={overall['r2']:.4f}"
          f"  acc={overall['class_accuracy']:.4f}"
          f"  cos={overall['cosine_similarity']:.4f}  n={overall['n']:,}")

    r_vals   = df_metrics["pearson_r"].to_numpy(dtype=float)
    acc_vals = df_metrics["class_accuracy"].to_numpy(dtype=float)
    cos_vals = df_metrics["cosine_similarity"].to_numpy(dtype=float)
    n_vals   = df_metrics["n"].to_numpy(dtype=float)

    # ── individual plots ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_scatter(ax, all_pred, all_true, mode, len(all_pred), color)
    fig.tight_layout()
    fig.savefig(out_dir / f"scatter_{mode}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_dir}/scatter_{mode}.png")

    for vals, xlabel, fname, vline in [
        (r_vals,   "Pearson R",         f"r_kde_{mode}.png",      0.99),
        (acc_vals, "Class accuracy",    f"acc_kde_{mode}.png",    0.9),
        (cos_vals, "Cosine similarity", f"cosine_kde_{mode}.png", None),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        _draw_kde(ax, vals, xlabel, color, vline=vline)
        ax.set_title(f"{mode}  –  {xlabel} per protein  (med={np.nanmedian(vals):.3f})")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_dir}/{fname}")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    _draw_confusion(ax, all_pred, all_true, cbar=True)
    ax.set_title(f"{mode}  –  confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{mode}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_dir}/confusion_{mode}.png")

    if all_df_rows:
        _welltype_breakdown(pd.concat(all_df_rows, ignore_index=True), mode, out_dir)

    # ── combined 2×3 summary ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    _draw_scatter(axes[0, 0], all_pred, all_true, mode, len(all_pred), color)
    _draw_kde(axes[0, 1], r_vals,   "Pearson R",      color, vline=0.99)
    axes[0, 1].set_title(f"R per protein  (med={np.nanmedian(r_vals):.3f})", fontsize=9)
    _draw_kde(axes[0, 2], acc_vals, "Class accuracy", color, vline=0.9)
    axes[0, 2].set_title(f"Class acc  (med={np.nanmedian(acc_vals):.3f})", fontsize=9)
    _draw_confusion(axes[1, 0], all_pred, all_true, cbar=False)
    axes[1, 0].set_title(f"{mode}  –  confusion matrix", fontsize=9)
    _draw_kde(axes[1, 1], cos_vals, "Cosine similarity", color)
    axes[1, 1].set_title(f"Cosine sim  (med={np.nanmedian(cos_vals):.3f})", fontsize=9)
    ok_n = np.isfinite(r_vals) & np.isfinite(n_vals)
    if ok_n.sum() > 1:
        axes[1, 2].scatter(n_vals[ok_n], r_vals[ok_n],
                           alpha=0.4, s=10, color=color, linewidths=0)
        axes[1, 2].axhline(0, color="k", lw=0.5, ls="--")
        axes[1, 2].set_xlabel("n contact pairs")
        axes[1, 2].set_ylabel("Pearson R")
        axes[1, 2].set_title("R vs protein size", fontsize=9)
    else:
        axes[1, 2].text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=axes[1, 2].transAxes)
    fig.suptitle(f"Frustratometer3 baseline  –  {mode}", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"summary_{mode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir}/summary_{mode}.png")

    return df_metrics


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True,
                        help="Directory with *.frustratometer3.{mode} files")
    parser.add_argument("--out",         default="plots",
                        help="Output directory")
    parser.add_argument("--mode",        default="both",
                        choices=["mutational", "configurational", "both"],
                        help="Which mode(s) to plot")
    parser.add_argument("--sample",      default=None,
                        help="Protein stem for contact map (e.g. 2ga8A00); "
                             "defaults to first protein found")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = (["mutational", "configurational"]
             if args.mode == "both" else [args.mode])

    all_metrics = []
    for mode in modes:
        df_m = process_mode(mode, results_dir, out_dir, args.sample)
        if not df_m.empty:
            all_metrics.append(df_m)

    if all_metrics:
        csv_path = out_dir / "pairwise_metrics.csv"
        pd.concat(all_metrics, ignore_index=True).to_csv(csv_path, index=False)
        print(f"\nWrote {csv_path}")

    print(f"\nDone. Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
