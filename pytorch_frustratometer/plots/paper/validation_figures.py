"""
Publication-quality validation figures comparing PyTorch frustration reimplementation
against R/C++ reference across configurational, mutational, and single-residue modes.

Generates:
  validation_scatter.png/pdf      — 3-panel scatter (one per mode), torch_rand + c_rand overlay
  validation_error_bars.png/pdf   — bar chart: mean_abs_error per mode × RNG setting
  validation_error_dist.png/pdf   — abs_error distribution per mode (violin)
  validation_distribution.png/pdf — KDE of values + signed-error + classification accuracy

Usage (on server):
    python -m frustrai_fold.plots.validation_figures \\
        --dompdb   ../dompdb_v4.4 \\
        --conf_outliers  baseline_comparison/frustratometer3/configurational_outliers.csv \\
        --mut_outliers   baseline_comparison/frustratometer3/mutational_outliers.csv \\
        --sr_outliers    baseline_comparison/frustratometer3/singleresidue_outliers.csv \\
        --out_dir        baseline_comparison/frustratometer3/plots \\
        --n 10 --decoys 1000
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..common import PALETTE, CLASS_COLORS

# ── style ─────────────────────────────────────────────────────────────────────

RC = {
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}

MODE_LABELS = {
    "configurational": "Configurational",
    "mutational":      "Mutational",
    "singleresidue":   "Single-residue",
}
MODE_LIM = {
    "configurational": (-4, 4),
    "mutational":      (-4, 4),
    "singleresidue":   (-4, 4),
}
RNG_STYLES = {
    "torch_rand": dict(color="#555555", alpha=0.25, s=4, linewidths=0),
    "c_rand":     dict(color="#E84393", alpha=0.55, s=4, linewidths=0),
}
RNG_LABELS = {
    "torch_rand": "PyTorch RNG",
    "c_rand":     "C rand() (matches R exactly)",
}

# ── data collection ───────────────────────────────────────────────────────────

def _top_proteins(outliers_csv: Path, n: int, exclude: set[str] | None = None) -> list[str]:
    df = pd.read_csv(outliers_csv)
    df = df[df["TrueFrstIndex"] != 0.0]
    if exclude:
        df = df[~df["protein_id"].isin(exclude)]
    return (df.groupby("protein_id")["abs_error"].mean()
              .nlargest(n).index.tolist())


def _load_r(dompdb_dir: Path, pid: str, mode: str) -> pd.DataFrame | None:
    suffix = {"configurational": "pdb_configurational",
               "mutational":      "pdb_mutational",
               "singleresidue":   "pdb_singleresidue"}[mode]
    path = dompdb_dir / f"{pid}.done" / "FrustrationData" / f"{pid}.{suffix}"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    key = "Res" if mode == "singleresidue" else ["Res1", "Res2"]
    cols = ([key] if isinstance(key, str) else key) + ["FrstIndex"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[cols].dropna().rename(columns={"FrstIndex": "TrueFrstIndex"})
    return df[df["TrueFrstIndex"] != 0.0]


def _run_model(pdb_path: Path, mode: str, n_decoys: int,
               use_c_rand: bool, dtype) -> pd.DataFrame:
    import torch
    from pytorch_frustratometer.coefficients import Coefficients
    from pytorch_frustratometer.data.dataset import FrustratometerDataset, collate_frustratometer
    from pytorch_frustratometer.frustratometer import Frustratometer3
    from torch.utils.data import DataLoader

    if mode == "configurational":
        from pytorch_frustratometer.algorithm.configurational import ConfigurationalFrustratometer

    coeff = Coefficients.from_dict({
        "mode": mode, "tert_frust_ndecoys": n_decoys, "min_seq_distance": 3,
    })
    dataset = FrustratometerDataset(
        [str(pdb_path)], coeff, compare_to_original=True, dtype=dtype,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=collate_frustratometer)

    if mode == "configurational":
        model = ConfigurationalFrustratometer(coeff, dtype=dtype)
        model.use_c_rand = use_c_rand
    else:
        model = Frustratometer3(coeff, dtype=dtype)
        model.awsem.use_c_rand = use_c_rand

    rows = []
    for batch in loader:
        with torch.no_grad():
            result = model(batch)

        ri = batch.get("residue_index", [None])[0]

        if mode == "singleresidue":
            fi   = result["singleresidue"][0]
            mask = batch["single_frust_mask"][0].bool()
            for l in range(fi.shape[0]):
                if not mask[l]:
                    continue
                rows.append({
                    "Res": ri[l].item() if ri is not None else l,
                    "FrstIndex": fi[l].item(),
                })
        else:
            key = "configurational" if mode == "configurational" else "mutational"
            mask_key = f"{key}_frust_mask"
            fi         = result[key][0]
            mask       = batch[mask_key][0].bool()
            edge_index = batch["edge_index"][0]
            for e in range(fi.shape[0]):
                if not mask[e]:
                    continue
                src, dst = edge_index[0, e].item(), edge_index[1, e].item()
                rows.append({
                    "Res1": ri[src].item() if ri is not None else src,
                    "Res2": ri[dst].item() if ri is not None else dst,
                    "FrstIndex": fi[e].item(),
                })
    return pd.DataFrame(rows)


def collect_data(mode: str, outliers_csv: Path, dompdb_dir: Path,
                 n: int, n_decoys: int, dtype,
                 exclude: set[str] | None = None) -> dict[str, pd.DataFrame]:
    """Run torch_rand and c_rand for a mode; return dict of merged DataFrames."""
    import torch
    proteins = _top_proteins(outliers_csv, n, exclude=exclude)
    merge_keys = ["Res"] if mode == "singleresidue" else ["Res1", "Res2"]
    all_data: dict[str, list] = {"torch_rand": [], "c_rand": []}

    for pid in proteins:
        pdb = dompdb_dir / f"{pid}.done" / "FrustrationData" / f"{pid}.pdb"
        if not pdb.exists():
            continue
        r_df = _load_r(dompdb_dir, pid, mode)
        if r_df is None or len(r_df) == 0:
            continue

        for rng_label, use_c_rand in [("torch_rand", False), ("c_rand", True)]:
            try:
                out = _run_model(pdb, mode, n_decoys, use_c_rand, dtype)
                merged = out.merge(r_df, on=merge_keys, how="inner")
                if len(merged) == 0:
                    continue
                merged["protein_id"] = pid
                all_data[rng_label].append(merged)
                print(f"  {pid} {rng_label}: n={len(merged)}", flush=True)
            except Exception as e:
                print(f"  {pid} {rng_label}: ERROR {e}", flush=True)

    # c_rand not meaningful for singleresidue (deterministic) — skip
    if mode == "singleresidue":
        all_data.pop("c_rand", None)

    return {
        k: pd.concat(v, ignore_index=True)
        for k, v in all_data.items()
        if v
    }


# ── figure 1: scatter ─────────────────────────────────────────────────────────

def fig_scatter(data: dict[str, dict[str, pd.DataFrame]], out_dir: Path):
    """3-panel scatter: one column per mode, torch_rand + c_rand overlaid."""
    modes = [m for m in ("configurational", "mutational", "singleresidue")
             if m in data and data[m]]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(modes), figsize=(3.5 * len(modes), 3.5),
                                 constrained_layout=True)
        if len(modes) == 1:
            axes = [axes]

        for ax, mode in zip(axes, modes):
            lim = MODE_LIM[mode]
            ax.plot(lim, lim, "k--", lw=0.8, zorder=0)

            for rng_label, kw in RNG_STYLES.items():
                if rng_label not in data[mode]:
                    continue
                df = data[mode][rng_label]
                ax.scatter(df["TrueFrstIndex"], df["FrstIndex"], **kw,
                           label=RNG_LABELS[rng_label], zorder=2)

                # annotate r and MAE
                r   = np.corrcoef(df["FrstIndex"], df["TrueFrstIndex"])[0, 1]
                mae = (df["FrstIndex"] - df["TrueFrstIndex"]).abs().mean()
                prefix = "PT" if rng_label == "torch_rand" else "C "
                ax.text(0.03, 0.97 if rng_label == "torch_rand" else 0.88,
                        f"{prefix}: r={r:.3f}, MAE={mae:.3f}",
                        transform=ax.transAxes, fontsize=7,
                        va="top", color=kw["color"],
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="none", alpha=0.8))

            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("R FrstIndex (reference)")
            ax.set_ylabel("PyTorch FrstIndex")
            ax.set_title(MODE_LABELS[mode])
            ax.set_aspect("equal")

        # shared legend
        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=RNG_STYLES[k]["color"],
                          markersize=5, label=RNG_LABELS[k])
                   for k in RNG_STYLES if k in {rng for m in data.values()
                                                 for rng in m}]
        axes[-1].legend(handles=handles, loc="lower right", frameon=True,
                        framealpha=0.9, edgecolor="none")

        for fmt in ("png", "pdf"):
            fig.savefig(out_dir / f"validation_scatter.{fmt}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    print(f"Saved validation_scatter.png/pdf → {out_dir}", flush=True)


# ── figure 2: error bars ──────────────────────────────────────────────────────

def fig_error_bars(data: dict[str, dict[str, pd.DataFrame]], out_dir: Path):
    """Bar chart: mean MAE per mode × RNG, with per-protein points overlaid."""
    modes = [m for m in ("configurational", "mutational", "singleresidue")
             if m in data and data[m]]
    rngs  = ["torch_rand", "c_rand"]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)

        n_modes = len(modes)
        n_rngs  = len(rngs)
        width   = 0.35
        x       = np.arange(n_modes)
        offsets = np.linspace(-(n_rngs - 1) / 2, (n_rngs - 1) / 2, n_rngs) * width

        rng_colors = {"torch_rand": "#777777", "c_rand": "#E84393"}

        for j, rng_label in enumerate(rngs):
            mae_per_mode, protein_maes = [], []
            for mode in modes:
                if rng_label not in data.get(mode, {}):
                    mae_per_mode.append(np.nan)
                    protein_maes.append([])
                    continue
                df = data[mode][rng_label]
                df["abs_err"] = (df["FrstIndex"] - df["TrueFrstIndex"]).abs()
                mae_per_mode.append(df["abs_err"].mean())
                protein_maes.append([
                    g["abs_err"].mean()
                    for _, g in df.groupby("protein_id")
                ])

            for i, (mode, mae, pts) in enumerate(zip(modes, mae_per_mode, protein_maes)):
                # Center bar when mode has only one RNG (e.g. singleresidue is deterministic)
                available_rngs = [r for r in rngs if r in data.get(mode, {})]
                xi = x[i] if len(available_rngs) == 1 else x[i] + offsets[j]
                bar_w = width * 1.5 if len(available_rngs) == 1 else width * 0.9
                if np.isnan(mae):
                    continue
                ax.bar(xi, mae, width=bar_w, color=rng_colors[rng_label], alpha=0.85,
                       label=RNG_LABELS[rng_label] if i == 0 else "_nolegend_", zorder=2)
                if pts:
                    jitter = np.random.uniform(-0.05, 0.05, len(pts))
                    ax.scatter(xi + jitter, pts, s=18, color="white",
                               edgecolors=rng_colors[rng_label],
                               linewidths=0.8, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABELS[m] for m in modes])
        ax.set_ylabel("Mean absolute error (FrstIndex)")
        ax.set_title("PyTorch vs R: error by mode and RNG")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="none")
        ax.set_ylim(bottom=0)

        for fmt in ("png", "pdf"):
            fig.savefig(out_dir / f"validation_error_bars.{fmt}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    print(f"Saved validation_error_bars.png/pdf → {out_dir}", flush=True)


# ── figure 3: error distribution ─────────────────────────────────────────────

def fig_error_dist(data: dict[str, dict[str, pd.DataFrame]], out_dir: Path):
    """Violin of abs_error per mode, torch_rand vs c_rand."""
    modes = [m for m in ("configurational", "mutational", "singleresidue")
             if m in data and data[m]]

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(modes), figsize=(3.2 * len(modes), 3.2),
                                 constrained_layout=True, sharey=False)
        if len(modes) == 1:
            axes = [axes]

        rng_colors = {"torch_rand": "#777777", "c_rand": "#E84393"}

        for ax, mode in zip(axes, modes):
            plot_data, positions, colors, labels = [], [], [], []
            for j, (rng_label, color) in enumerate(rng_colors.items()):
                if rng_label not in data.get(mode, {}):
                    continue
                df = data[mode][rng_label]
                errs = (df["FrstIndex"] - df["TrueFrstIndex"]).abs().values
                plot_data.append(errs)
                positions.append(j + 1)
                colors.append(color)
                labels.append(RNG_LABELS[rng_label])

            parts = ax.violinplot(plot_data, positions=positions,
                                  showmedians=True, showextrema=False)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(1.5)

            ax.set_xticks(positions)
            ax.set_xticklabels(["PyTorch\nRNG", "C rand()"][:len(positions)],
                               fontsize=7)
            ax.set_ylabel("|FrstIndex error|")
            ax.set_title(MODE_LABELS[mode])
            ax.set_ylim(bottom=0)

        for fmt in ("png", "pdf"):
            fig.savefig(out_dir / f"validation_error_dist.{fmt}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    print(f"Saved validation_error_dist.png/pdf → {out_dir}", flush=True)


# ── figure 4: summary heatmap ─────────────────────────────────────────────────

def fig_summary_heatmap(data: dict[str, dict[str, pd.DataFrame]], out_dir: Path):
    """Heatmap of MAE and Pearson r: rows=mode, cols=RNG setting."""
    modes = [m for m in ("configurational", "mutational", "singleresidue")
             if m in data and data[m]]
    rngs  = ["torch_rand", "c_rand"]

    mae_vals = np.full((len(modes), len(rngs)), np.nan)
    r_vals   = np.full((len(modes), len(rngs)), np.nan)

    for i, mode in enumerate(modes):
        for j, rng in enumerate(rngs):
            if rng not in data.get(mode, {}):
                continue
            df  = data[mode][rng]
            mae = (df["FrstIndex"] - df["TrueFrstIndex"]).abs().mean()
            r   = np.corrcoef(df["FrstIndex"], df["TrueFrstIndex"])[0, 1]
            mae_vals[i, j] = mae
            r_vals[i, j]   = r

    xlabels = ["PyTorch RNG", "C rand()"]
    ylabels  = [MODE_LABELS[m] for m in modes]

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4),
                                        constrained_layout=True)

        def _heatmap(ax, vals, title, fmt, cmap, vmin, vmax):
            im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=20, ha="right")
            ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, fmt.format(v), ha="center", va="center",
                                fontsize=8, color="white" if (v < vmin + (vmax-vmin)*0.4) else "black")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.8)

        _heatmap(ax1, mae_vals, "Mean Abs Error", "{:.3f}", "YlOrRd", 0, 0.15)
        _heatmap(ax2, r_vals,   "Pearson r",       "{:.3f}", "RdYlGn", 0.8, 1.0)

        for fmt in ("png", "pdf"):
            fig.savefig(out_dir / f"validation_heatmap.{fmt}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    print(f"Saved validation_heatmap.png/pdf → {out_dir}", flush=True)


# ── figure 5: distribution + signed error + classification accuracy ───────────

# Standard Frustratometer thresholds
_FRUST_THRESH  = (-1.0, 0.78)
_CLASS_NAMES   = ["Frustrated", "Neutral", "Min. Frust."]
_CLASS_PALETTE = ["#D7191C", "#AAAAAA", "#2C7BB6"]


def _classify(fi: np.ndarray) -> np.ndarray:
    cls = np.ones(len(fi), dtype=int)
    cls[fi <  _FRUST_THRESH[0]] = 0
    cls[fi >  _FRUST_THRESH[1]] = 2
    return cls


def fig_distribution(data: dict[str, dict[str, pd.DataFrame]], out_dir: Path):
    """3-row × 3-col figure per mode:
      col 1 — KDE of FrstIndex values (R, torch_rand, c_rand) — same distribution
      col 2 — KDE of signed error (PyTorch − R) — centered at 0, narrow vs full range
      col 3 — per-class classification accuracy (%) vs R reference
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        print("WARNING: scipy not available — skipping fig_distribution", flush=True)
        return

    modes = [m for m in ("configurational", "mutational", "singleresidue")
             if m in data and data[m]]
    n_modes = len(modes)
    rng_colors = {"torch_rand": "#555555", "c_rand": "#E84393"}
    R_COLOR    = "#2166AC"

    with plt.rc_context(RC):
        fig, axes = plt.subplots(n_modes, 3,
                                  figsize=(10, 2.8 * n_modes),
                                  constrained_layout=True)
        if n_modes == 1:
            axes = axes[np.newaxis, :]

        for row, mode in enumerate(modes):
            ax_kde, ax_err, ax_cls = axes[row]
            mode_data = data[mode]

            first_rng = next(iter(mode_data))
            r_vals = mode_data[first_rng]["TrueFrstIndex"].values
            r_vals = r_vals[np.isfinite(r_vals)]

            x_fi  = np.linspace(-5.5, 5.5, 400)
            x_err = np.linspace(-2.0, 2.0, 400)

            # ── col 1: value KDE ─────────────────────────────────────────────
            if len(r_vals) > 1:
                kde_r = gaussian_kde(r_vals, bw_method=0.25)
                ax_kde.fill_between(x_fi, kde_r(x_fi), alpha=0.20,
                                    color=R_COLOR, label="R reference")
                ax_kde.plot(x_fi, kde_r(x_fi), color=R_COLOR, lw=1.5)

            for rng_label, color in rng_colors.items():
                if rng_label not in mode_data:
                    continue
                pt = mode_data[rng_label]["FrstIndex"].values
                pt = pt[np.isfinite(pt)]
                if len(pt) > 1:
                    kde_pt = gaussian_kde(pt, bw_method=0.25)
                    ax_kde.plot(x_fi, kde_pt(x_fi), color=color, lw=1.2,
                                ls="--", label=RNG_LABELS[rng_label])

            for thresh in _FRUST_THRESH:
                ax_kde.axvline(thresh, color="black", lw=0.6, ls=":", alpha=0.4)
            ax_kde.set_xlim(-5.5, 5.5)
            ax_kde.set_ylim(bottom=0)
            ax_kde.set_xlabel("FrstIndex")
            ax_kde.set_ylabel("Density")
            ax_kde.set_title(MODE_LABELS[mode])
            if row == 0:
                ax_kde.legend(fontsize=7, frameon=False, loc="upper left")

            # ── col 2: signed error KDE ──────────────────────────────────────
            frust_range = float(np.ptp(r_vals)) if len(r_vals) > 1 else 8.0
            for rng_label, color in rng_colors.items():
                if rng_label not in mode_data:
                    continue
                df   = mode_data[rng_label]
                errs = (df["FrstIndex"] - df["TrueFrstIndex"]).values
                errs = errs[np.isfinite(errs)]
                mae  = float(np.abs(errs).mean())
                if len(errs) > 1:
                    kde_e = gaussian_kde(errs, bw_method=0.25)
                    ax_err.fill_between(x_err, kde_e(x_err), alpha=0.25, color=color)
                    ax_err.plot(x_err, kde_e(x_err), color=color, lw=1.2,
                                label=f"{RNG_LABELS[rng_label]}\nMAE={mae:.3f}")
                    # shade ±1 MAE band
                    ax_err.axvspan(-mae, mae, alpha=0.07, color=color, zorder=0)

            ax_err.axvline(0, color="black", lw=0.9, ls="--", alpha=0.6)
            ax_err.set_xlim(-1.5, 1.5)
            ax_err.set_ylim(bottom=0)
            ax_err.set_xlabel("PyTorch − R  (FrstIndex)")
            ax_err.set_ylabel("Density")
            if row == 0:
                ax_err.set_title("Signed error (centered at 0 ⟹ unbiased)")
                ax_err.legend(fontsize=7, frameon=False)
            # annotate dynamic range for context
            ax_err.text(0.97, 0.97,
                        f"Value range: {frust_range:.1f}",
                        transform=ax_err.transAxes, fontsize=7,
                        va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="none", alpha=0.85))

            # ── col 3: per-class classification accuracy ─────────────────────
            avail_rngs = [r for r in rng_colors if r in mode_data]
            bar_x  = np.arange(len(_CLASS_NAMES))
            width  = 0.75 / max(len(avail_rngs), 1)
            offs   = (np.arange(len(avail_rngs)) - (len(avail_rngs) - 1) / 2) * width

            for j, rng_label in enumerate(avail_rngs):
                df     = mode_data[rng_label]
                r_cls  = _classify(df["TrueFrstIndex"].values)
                pt_cls = _classify(df["FrstIndex"].values)
                accs   = []
                for cls_idx in range(3):
                    m = r_cls == cls_idx
                    accs.append(float((pt_cls[m] == cls_idx).mean() * 100) if m.any() else 0.0)
                ax_cls.bar(bar_x + offs[j], accs, width=width * 0.88,
                           color=rng_colors[rng_label], alpha=0.85,
                           label=RNG_LABELS[rng_label] if row == 0 else "_nolegend_")

            # colour the class-name labels to match the Frustratometer palette
            ax_cls.set_xticks(bar_x)
            ax_cls.set_xticklabels(_CLASS_NAMES, fontsize=7)
            for tick, color in zip(ax_cls.get_xticklabels(), _CLASS_PALETTE):
                tick.set_color(color)
            ax_cls.set_ylabel("Accuracy (%)")
            ax_cls.set_ylim(0, 108)
            ax_cls.axhline(100, color="black", lw=0.5, ls=":", alpha=0.4)
            if row == 0:
                ax_cls.set_title("Classification accuracy vs R")
                if len(avail_rngs) > 1:
                    ax_cls.legend(fontsize=7, frameon=False)

        for fmt in ("png", "pdf"):
            fig.savefig(out_dir / f"validation_distribution.{fmt}", dpi=200,
                        bbox_inches="tight")
        plt.close(fig)
    print(f"Saved validation_distribution.png/pdf → {out_dir}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb",         required=True, type=Path)
    ap.add_argument("--conf_outliers",  default=None,  type=Path)
    ap.add_argument("--mut_outliers",   default=None,  type=Path)
    ap.add_argument("--sr_outliers",    default=None,  type=Path)
    ap.add_argument("--out_dir",        default="validation_plots", type=Path)
    ap.add_argument("--n",              default=10,    type=int,
                    help="proteins per mode")
    ap.add_argument("--decoys",         default=1000,  type=int)
    ap.add_argument("--dtype",          default="float32",
                    choices=["float32", "float64"])
    # Known-bad structures to exclude from each mode
    ap.add_argument("--mut_exclude",    default="5zwnx00",
                    help="comma-separated protein IDs to exclude from mutational")
    args = ap.parse_args()

    import torch
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mut_exclude = set(args.mut_exclude.split(",")) if args.mut_exclude else None

    mode_csvs = {
        "configurational": args.conf_outliers,
        "mutational":      args.mut_outliers,
        "singleresidue":   args.sr_outliers,
    }

    all_data: dict[str, dict[str, pd.DataFrame]] = {}
    for mode, csv in mode_csvs.items():
        if csv is None or not csv.exists():
            print(f"Skipping {mode}: no outliers CSV", flush=True)
            continue
        print(f"\n{'='*55}\nCollecting {mode} data...", flush=True)
        exclude = mut_exclude if mode == "mutational" else None
        all_data[mode] = collect_data(
            mode, csv, args.dompdb, args.n, args.decoys, dtype, exclude=exclude
        )

    print("\nGenerating figures...", flush=True)
    fig_scatter(all_data, args.out_dir)
    fig_error_bars(all_data, args.out_dir)
    fig_error_dist(all_data, args.out_dir)
    fig_summary_heatmap(all_data, args.out_dir)
    fig_distribution(all_data, args.out_dir)
    print(f"\nAll figures saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
