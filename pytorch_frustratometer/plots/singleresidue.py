"""
Plot frustration benchmark results produced by frustratometer3.py test run.

Expected inputs (written to run_info/ by on_test_epoch_end):
  all_results.csv           — one row per (protein, mode)
  residue_samples_*.csv     — sampled (pred, true) pairs per mode

Usage:
  python -m frustrai_fold.plots.singleresidue \\
      --results baseline_comparison/run_info/all_results.csv \\
      --samples-dir baseline_comparison/run_info \\
      --out baseline_comparison/plots

Produces:
  summary_4x3.png            — 4-row × 3-col figure
  (plus individual PNGs per plot type)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0
import matplotlib.pyplot as plt
import seaborn as sns

from .common import PALETTE, MODE_THRESHOLDS, fi_class, draw_kde, draw_confusion, draw_scatter_hexbin

_VERBOSE = False
_FORMATS = ["png"]

def _log(path):
    if _VERBOSE:
        print(f"\r  wrote {path}", end="", flush=True)

def _savefig(fig, path: Path, dpi: int = 150) -> None:
    stem = Path(path).with_suffix("")
    try:
        for fmt in _FORMATS:
            p = stem.with_suffix(f".{fmt}")
            kw = {"bbox_inches": "tight"}
            if fmt.lower() not in ("svg", "pdf", "eps", "ps"):
                kw["dpi"] = dpi
            fig.savefig(p, **kw)
            _log(p)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)

MODES = ["singleresidue", "mutational", "configurational"]


# ── axes helpers (singleresidue-specific variants) ────────────────────────────

_XLABEL = "FrustratometeR"
_YLABEL = "PyTorch Re-implementation"


def _draw_scatter(ax, pred, true, mode, color):
    draw_scatter_hexbin(ax, pred, true,
                        title=f"{mode}\n(n={len(pred):,})", cmap="Blues",
                        xlabel=_XLABEL, ylabel=_YLABEL)


def _draw_acc_kde(ax, values, color, vline=0.9):
    draw_kde(ax, values, "Class accuracy", color, vline=vline)


def _draw_confusion(ax, pred, true, mode, cbar=False):
    lo, hi = MODE_THRESHOLDS.get(mode, (-1.0, 0.78))
    draw_confusion(ax, pred, true, lo=lo, hi=hi, cbar=cbar,
                   xlabel=_YLABEL, ylabel=_XLABEL)


# ── individual plot wrappers ──────────────────────────────────────────────────

def scatter_plot(pred, true, mode, out_dir, color):
    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_scatter(ax, pred, true, mode, color)
    fig.tight_layout()
    path = out_dir / f"scatter_{mode}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _log(path)


def confusion_plot(pred, true, mode, out_dir):
    lo, hi = MODE_THRESHOLDS.get(mode, (-1.0, 0.78))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    draw_confusion(ax, pred, true, lo=lo, hi=hi, cbar=True,
                   xlabel=_YLABEL, ylabel=_XLABEL)
    ax.set_title(f"{mode}  –  confusion matrix")
    fig.tight_layout()
    path = out_dir / f"confusion_{mode}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _log(path)


def r_vs_n_plot(df_mode, mode, out_dir, color):
    df = df_mode.dropna(subset=["pearson_r", "n"]).copy()
    df["n"]        = pd.to_numeric(df["n"],        errors="coerce")
    df["pearson_r"] = pd.to_numeric(df["pearson_r"], errors="coerce")
    df = df.dropna(subset=["pearson_r", "n"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=df, x="n", y="pearson_r", ax=ax,
                    alpha=0.25, s=8, color=color, rasterized=True, linewidth=0)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Protein length (n residues)")
    ax.set_ylabel("Pearson R")
    ax.set_title(f"{mode}  –  R vs protein length")
    fig.tight_layout()
    path = out_dir / f"r_vs_n_{mode}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _log(path)


def plot_singleresidue(df: "pd.DataFrame", stem: str, out_dir: "Path",
                       metrics: dict | None = None):
    """Frustration line plot for one protein, using plot_single_frustration from per_protein."""
    from .per_protein import plot_single_frustration, _to_int_class, MAX_RESIDUES_PER_ROW

    import math
    res_ids  = df["Res"].astype(int).to_numpy()
    pred_fi  = df["FrstIndex"].to_numpy(dtype=float)
    has_true = "TrueFrstIndex" in df.columns
    true_fi  = df["TrueFrstIndex"].to_numpy(dtype=float) if has_true else pred_fi
    pred_cls = _to_int_class(df["FrstState"].to_numpy()) if "FrstState" in df.columns \
               else np.ones(len(pred_fi), dtype=int)
    residues = np.array(["?"] * len(res_ids))

    N    = len(res_ids)
    nrows = math.ceil(N / MAX_RESIDUES_PER_ROW)
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(min(30, max(8, N * 0.04 + 2)), 3 * nrows),
                             squeeze=False)
    plot_single_frustration(
        axes[:, 0].tolist(),
        frst_true=true_fi,
        frst_pred=pred_fi,
        frst_class_pred=pred_cls,
        residues=residues,
        res_ids=res_ids,
        protein_id=stem,
    )

    if metrics and has_true:
        r   = metrics.get("pearson_r", float("nan"))
        acc = metrics.get("class_accuracy", float("nan"))
        n   = metrics.get("n", "?")
        fig.suptitle(f"{stem}   R={r:.3f}  acc={acc:.3f}  n={n}", fontsize=9)
    else:
        fig.suptitle(stem, fontsize=9)

    fig.tight_layout()
    path = Path(out_dir) / f"singleresidue_{stem}.png"
    _savefig(fig, path)


# ── data loading ──────────────────────────────────────────────────────────────

def _load_samples(samples_dir, mode):
    path = samples_dir / f"residue_samples_{mode}.csv"
    if not path.is_file():
        return None, None
    sdf = pd.read_csv(path, low_memory=False)
    sdf = sdf[sdf["pred"] != "pred"].copy()
    sdf["pred"] = pd.to_numeric(sdf["pred"], errors="coerce")
    sdf["true"] = pd.to_numeric(sdf["true"], errors="coerce")
    sdf = sdf.dropna(subset=["pred", "true"])
    return sdf["pred"].to_numpy(dtype=float), sdf["true"].to_numpy(dtype=float)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results",     required=True, help="all_results.csv")
    parser.add_argument("--samples-dir", required=True,
                        help="directory with residue_samples_*.csv")
    parser.add_argument("--out",         default="plots", help="output directory")
    args = parser.parse_args()

    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all      = pd.read_csv(args.results)
    samples_dir = Path(args.samples_dir)

    mode_data = {}
    for mode in MODES:
        df_mode = df_all[df_all["mode"] == mode].copy()
        pred_arr, true_arr = _load_samples(samples_dir, mode)
        mode_data[mode] = {"df": df_mode, "pred": pred_arr, "true": true_arr}

    # ── individual plots ──────────────────────────────────────────────────────
    for mode in MODES:
        print(f"\n── {mode} ──")
        d       = mode_data[mode]
        df_mode = d["df"]
        color   = PALETTE[mode]

        if df_mode.empty:
            print(f"  no data for mode={mode}, skipping")
            continue

        if d["pred"] is not None:
            scatter_plot(d["pred"], d["true"], mode, out_dir, color)
            confusion_plot(d["pred"], d["true"], mode, out_dir)
        else:
            print(f"  residue_samples_{mode}.csv not found — skipping scatter/confusion")

        kde_vals = [
            (df_mode["pearson_r"].to_numpy(dtype=float),
             "Pearson R", f"r_kde_{mode}.png", 0.99),
            (df_mode["class_accuracy"].to_numpy(dtype=float),
             "Class accuracy", f"acc_kde_{mode}.png", 0.9),
            (df_mode["std_ratio"].to_numpy(dtype=float),
             "std_ratio (pred/true)", f"std_ratio_{mode}.png", 1.0),
        ]
        for vals, xlabel, fname, vline in kde_vals:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            draw_kde(ax, vals, xlabel, color, vline=vline)
            ax.set_title(f"{mode}  –  {xlabel} per protein")
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=150)
            plt.close(fig)
            print(f"  wrote {out_dir}/{fname}")

        if "cosine_similarity" in df_mode.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            draw_kde(ax, df_mode["cosine_similarity"].to_numpy(dtype=float),
                     "Cosine similarity", color)
            ax.set_title(f"{mode}  –  cosine similarity per protein")
            fig.tight_layout()
            fig.savefig(out_dir / f"cosine_{mode}.png", dpi=150)
            plt.close(fig)
            print(f"  wrote {out_dir}/cosine_{mode}.png")

        r_vs_n_plot(df_mode, mode, out_dir, color)

    # ── combined 4×3 summary ──────────────────────────────────────────────────
    print("\n── combined 4×3 figure ──")
    fig, axes = plt.subplots(4, 3, figsize=(14, 17))
    row_labels = ["Scatter (FrustratometeR vs PyTorch)", "Class accuracy (per protein)",
                  "Confusion matrix", "Cosine similarity (per protein)"]

    for col, mode in enumerate(MODES):
        d        = mode_data[mode]
        df_mode  = d["df"]
        color    = PALETTE[mode]
        pred_arr, true_arr = d["pred"], d["true"]

        ax = axes[0, col]
        if pred_arr is not None:
            _draw_scatter(ax, pred_arr, true_arr, mode, color)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(mode, fontsize=9)

        ax = axes[1, col]
        if not df_mode.empty:
            acc_vals = pd.to_numeric(df_mode["class_accuracy"],
                                     errors="coerce").dropna().to_numpy()
            _draw_acc_kde(ax, acc_vals, color)
            ax.set_title(f"median={np.nanmedian(acc_vals):.3f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)

        ax = axes[2, col]
        if pred_arr is not None:
            _draw_confusion(ax, pred_arr, true_arr, mode, cbar=False)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)

        ax = axes[3, col]
        if not df_mode.empty and "cosine_similarity" in df_mode.columns:
            cos_vals = pd.to_numeric(df_mode["cosine_similarity"],
                                     errors="coerce").dropna().to_numpy()
            if len(cos_vals) > 0:
                sns.kdeplot(cos_vals, ax=ax, color=color,
                            fill=True, alpha=0.3, linewidth=2)
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

    for col, mode in enumerate(MODES):
        n = len(mode_data[mode]["pred"]) if mode_data[mode]["pred"] is not None else 0
        axes[0, col].set_title(
            f"{mode}\n(n={n:,})" if n else mode,
            fontsize=10, fontweight="bold",
        )
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label + "\n" + axes[row, 0].get_ylabel(),
                                fontsize=9, labelpad=6)

    fig.suptitle("Frustratometer3 baseline  –  all modes",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    combined_path = out_dir / "summary_4x3.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {combined_path}")
    print(f"\nDone. Plots written to {out_dir}/")


if __name__ == "__main__":
    main()


# ── mutation scanning figure ──────────────────────────────────────────────────

def _find_mutations_data_dir(sf: Path):
    """Return the MutationsData/ directory for a structure file, or None."""
    if sf.parent.parent.name.endswith(".done"):
        cand = sf.parent.parent / "MutationsData"
        if cand.is_dir():
            return cand
    for done in sf.parent.glob(f"{sf.stem}_*.done"):
        cand = done / "MutationsData"
        if cand.is_dir():
            return cand
    return None


def draw_mutation_scanning(batch, result, frust_bases):
    """Draw deep mutational scanning heatmap(s) to Visualizations/.

    Always produces (in each format listed in mutations._FORMATS):
      {stem}.pdb_singleresidue_scanning.*       — full FI heatmap (GLY/X→G grey-hatched)
    When FrustratometeR MutationsData is found next to the input structure:
      {stem}.pdb_singleresidue_scanning_error.* — |ΔFI|: non-G errors + X→G (gap-separated)
                                                   + GLY native panel; shared global scale
    """
    from ..structure.residue_constants import ID_TO_AA
    from ..data.io import read_mutations_data
    from .mutations import plot_scanning_heatmap, _AA_TO_IDX

    sr_mut = result.get("_singleresidue_mutations")
    if sr_mut is None:
        return

    decoy_stats   = result.get("_singleresidue_decoy_stats")
    seq_mask      = batch["seq_mask"]
    aa_type       = batch["aa_type"]
    residue_index = batch.get("residue_index")
    chain_index   = batch.get("chain_index")
    _cid_map_list = batch.get("chain_id_map")
    from ..structure.protein import PDB_CHAIN_IDS

    sf_raw = batch.get("structure_file")
    if isinstance(sf_raw, str):
        sf_list = [sf_raw]
    elif sf_raw is not None:
        sf_list = list(sf_raw)
    else:
        sf_list = [None] * len(frust_bases)

    for i, frust_base in enumerate(frust_bases):
        fig_dir = Path(frust_base).parent.parent / "Visualizations"
        fig_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(frust_base).name

        mask    = seq_mask[i].bool()
        n_res   = mask.sum().item()
        mut_e   = sr_mut["energies"][i, mask].cpu().float()  # [n_res, 20]
        wt_aa   = aa_type[i, mask].cpu().numpy()             # [n_res]

        res_nums = (residue_index[i, mask].cpu().numpy()
                   if residue_index is not None
                   else np.arange(1, n_res + 1))
        _cid_map_i = _cid_map_list[i] if _cid_map_list is not None else None
        ch = ([(_cid_map_i[int(c)] if _cid_map_i else PDB_CHAIN_IDS[int(c)])
               for c in chain_index[i, mask].cpu().tolist()]
              if chain_index is not None
              else ["A"] * n_res)

        from .mutations import _GLY_IDX as _GLY

        # Mirror frustratometer.py _write_mutation_files exactly:
        # prefer sr_mut["stats"] (gly_water_corr applied) over the top-level decoy_stats.
        mut_stats = sr_mut.get("stats")
        if mut_stats is not None:
            ds     = mut_stats[i, mask].cpu().float()
            d_mean = ds[:, 0]
            d_std  = ds[:, 1].clamp(min=1e-8)
            fi_mat = ((d_mean.unsqueeze(1) - mut_e) / d_std.unsqueeze(1)).numpy()
        elif decoy_stats is not None:
            ds     = decoy_stats[i, mask].cpu().float()
            d_mean = ds[:, 0]
            d_std  = ds[:, 1].clamp(min=1e-8)
            fi_mat = ((d_mean.unsqueeze(1) - mut_e) / d_std.unsqueeze(1)).numpy()
        else:
            fi_mat = -mut_e.numpy()

        # GLY corrections — read from sr_mut sub-dict (frustratometer.py path) with
        # fallback to top-level result keys (algorithm/singleresidue.py path).
        # Case 1: GLY native positions — all 20 mutant cols use Cα-based FI
        _tmp = sr_mut.get("_gly_native_fi")
        gly_native_fi_all = _tmp if _tmp is not None else result.get("_singleresidue_gly_native_fi")
        if gly_native_fi_all is not None:
            gly_nat_fi = gly_native_fi_all[i, mask].cpu().float().numpy()  # [n_res, 20]
            is_gly_nat = wt_aa == _GLY
            fi_mat[is_gly_nat] = gly_nat_fi[is_gly_nat]

        # Case 2: X→G column at non-GLY positions — single scalar Cα-based FI per residue
        _tmp = sr_mut.get("_gly_mutation_fi")
        gly_fi_all = _tmp if _tmp is not None else result.get("_singleresidue_gly_fi")
        if gly_fi_all is not None:
            gly_fi = gly_fi_all[i, mask].cpu().float().numpy()  # [n_res]
            is_non_gly = wt_aa != _GLY
            valid_xg   = is_non_gly & np.isfinite(gly_fi)
            fi_mat[valid_xg, _GLY] = gly_fi[valid_xg]

        wt_aas = [ID_TO_AA.get(int(a), 'X') for a in wt_aa]

        ref_fi_mat = None
        sf = sf_list[i] if i < len(sf_list) else None
        if sf is not None:
            ref_dir_path = _find_mutations_data_dir(Path(sf))
            if ref_dir_path is not None:
                try:
                    ref_df     = read_mutations_data(ref_dir_path)
                    ref_fi_mat = np.full((n_res, 20), np.nan, dtype=np.float32)
                    for pos in range(n_res):
                        sub = ref_df[(ref_df["Res"] == int(res_nums[pos])) &
                                     (ref_df["ChainRes"] == ch[pos])]
                        for _, row in sub.iterrows():
                            idx = _AA_TO_IDX.get(str(row["AA"]))
                            if idx is not None:
                                ref_fi_mat[pos, idx] = float(row["FrstIndex"])
                    if np.isnan(ref_fi_mat).all():
                        ref_fi_mat = None
                except Exception as exc:
                    print(f"[draw_figure] could not load reference for {stem}: {exc}")

        plot_scanning_heatmap(
            fi_mat, wt_aa, res_nums, wt_aas, stem, fig_dir,
            suffix="pdb_singleresidue_scanning",
            ref_fi_mat=ref_fi_mat,
        )
