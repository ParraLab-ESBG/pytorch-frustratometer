"""
MSFA visualizations from FrustraEvo AuxFiles/DF_Colores.

Produces:
  msfa_{funfam}.png           – true class heatmap + diff map side by side
  msfa_{funfam}_fracs.png     – per-column stacked area of class fractions (true)

Figure width scales linearly with MSA length (no cap).

Data paths
----------
Model DF_Colores (pytorch reimplementation):
  {results_dir}/{funfam}/FrustraEvo_{funfam}/AuxFiles/DF_Colores

True DF_Colores (original FrustraEvo, same dir as the input fasta):
  {workdir}/FrustraEvo_{funfam}/AuxFiles/DF_Colores

DF_Colores columns: PDBID, pos, AA, FstState, FstI
  FstState: MIN (minimally frustrated), MAX (highly frustrated), NEU, N/A (gap)

Run
---
  python -m frustrai_fold.plots.msfa
  python -m frustrai_fold.plots.msfa \\
      --funfam 3.40.710.10-FF-000005 \\
      --results_dir /path/to/results \\
      --workdir /gpfs/scratch/bsc08/bsc008092/CATH/all/seqdist3
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import biotite.sequence.io.fasta as fasta_io

RESULTS_BASE = "/gpfs/projects/bsc08/bsc504628/frustration/FrustrAI-Fold/baseline_comparison/FrustaEvo/results3"
WORKDIR_BASE = "/gpfs/scratch/bsc08/bsc008092/CATH/all/seqdist3"
OUT_BASE     = "/gpfs/projects/bsc08/bsc504628/frustration/FrustrAI-Fold/results/paper_figures2"

COLOR_HF  = "#CC2222"
COLOR_NEU = "#BBBBBB"
COLOR_MF  = "#22AA44"
COLOR_GAP = "#FFFFFF"

# FstState → integer class code: MAX=HF=0, NEU=1, MIN=MF=2, gap=3
_STATE_CODE = {"MAX": 0, "NEU": 1, "MIN": 2}

# layout: each MSA column → this many points of figure width (at 72 pt/in)
PTS_PER_COL = 2.0
# each sequence row → this many points of figure height
PTS_PER_ROW = 2.5
DPI = 150


def _fig_dims(msa_len, n_seq, n_panels=1):
    """Return (width_in, height_in) for one or more side-by-side heatmap panels."""
    panel_w = msa_len * PTS_PER_COL / 72.0
    fig_h   = max(4.0, n_seq * PTS_PER_ROW / 72.0)
    # extra width for colorbars / labels (0.6 in per panel)
    fig_w   = n_panels * (panel_w + 0.6) + 0.4
    return fig_w, fig_h


# ── data loading ──────────────────────────────────────────────────────────────

def load_msa(path):
    ff = fasta_io.FastaFile.read(path)
    return [(name, str(seq)) for name, seq in ff.items()]


def load_df_colores(path):
    """Return DataFrame with PDBID, pos (int), FstState (str), FstI (float)."""
    df = pd.read_csv(path)
    df["pos"]      = pd.to_numeric(df["pos"],  errors="coerce")
    df["FstI"]     = pd.to_numeric(df["FstI"], errors="coerce")
    df["FstState"] = df["FstState"].astype(str).str.strip()
    return df.dropna(subset=["pos"])


# ── matrix builders ───────────────────────────────────────────────────────────

def build_class_matrix(df, seq_order, msa_len):
    """(n_seq × msa_len) int8: 0=HF, 1=NEU, 2=MF, 3=gap/missing."""
    mat = np.full((len(seq_order), msa_len), 3, dtype=np.int8)
    idx = {name: i for i, name in enumerate(seq_order)}
    for _, row in df.iterrows():
        i = idx.get(row["PDBID"])
        if i is None:
            continue
        col = int(row["pos"]) - 1
        if 0 <= col < msa_len:
            mat[i, col] = _STATE_CODE.get(row["FstState"], 3)
    return mat


def build_fi_matrix(df, seq_order, msa_len):
    """(n_seq × msa_len) float of FstI, NaN for gaps."""
    mat = np.full((len(seq_order), msa_len), np.nan)
    idx = {name: i for i, name in enumerate(seq_order)}
    for _, row in df.iterrows():
        i = idx.get(row["PDBID"])
        if i is None:
            continue
        col = int(row["pos"]) - 1
        if 0 <= col < msa_len:
            mat[i, col] = row["FstI"]
    return mat


# ── combined side-by-side plot ────────────────────────────────────────────────

def _ytick_setup(ax, n, seq_names):
    if n <= 60:
        ax.set_yticks(range(n))
        ax.set_yticklabels([nm[:28] for nm in seq_names], fontsize=4)
    else:
        ax.set_yticks([])


def plot_combined(class_mat, diff_mat, seq_names, out_path, funfam,
                  source_label="true"):
    """
    Left panel  : class heatmap (HF/NEU/MF colours) from true DF_Colores.
    Right panel : diverging diff map (model FI − true FI).
    Both panels share the same y-axis and are sized proportionally to msa_len.
    If diff_mat is None, only the class panel is drawn.
    """
    from matplotlib.colors import ListedColormap

    n, msa_len = class_mat.shape
    has_diff   = diff_mat is not None
    n_panels   = 2 if has_diff else 1

    fig_w, fig_h = _fig_dims(msa_len, n, n_panels)
    # width_ratios: heatmap col + thin colorbar col for each panel
    panel_w   = msa_len * PTS_PER_COL / 72.0
    cbar_w    = 0.15
    ratios    = []
    for _ in range(n_panels):
        ratios += [panel_w, cbar_w]

    fig, axes = plt.subplots(
        1, n_panels * 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": ratios, "wspace": 0.05},
    )
    if n_panels == 1:
        axes = list(axes)

    # ── left: class heatmap ──────────────────────────────────────────────────
    ax_cls  = axes[0]
    ax_cbar = axes[1]

    cmap_cls = ListedColormap([COLOR_HF, COLOR_NEU, COLOR_MF, COLOR_GAP])
    ax_cls.imshow(class_mat, aspect="auto", cmap=cmap_cls,
                  vmin=0, vmax=3, interpolation="nearest")
    ax_cls.set_title(f"Frustration  [{source_label}]", fontsize=8)
    ax_cls.set_xlabel("MSA position", fontsize=7)
    ax_cls.set_ylabel("Sequence",     fontsize=7)
    ax_cls.tick_params(labelsize=6)
    _ytick_setup(ax_cls, n, seq_names)

    # legend in the cbar axes (no actual colorbar needed for discrete map)
    ax_cbar.axis("off")
    ax_cbar.legend(handles=[
        mpatches.Patch(color=COLOR_HF,  label="HF"),
        mpatches.Patch(color=COLOR_NEU, label="NEU"),
        mpatches.Patch(color=COLOR_MF,  label="MF"),
        mpatches.Patch(color=COLOR_GAP, label="gap", edgecolor="#ccc"),
    ], loc="center left", fontsize=6, framealpha=0.85)

    # ── right: diff map ──────────────────────────────────────────────────────
    if has_diff:
        ax_dif  = axes[2]
        ax_dcb  = axes[3]

        valid = diff_mat[~np.isnan(diff_mat)]
        vmax  = float(np.percentile(np.abs(valid), 95)) if len(valid) else 1.0
        vmax  = max(vmax, 0.1)

        cmap_dif = plt.get_cmap("RdBu_r").copy()
        cmap_dif.set_bad(color="white")
        im = ax_dif.imshow(diff_mat, aspect="auto", cmap=cmap_dif,
                           vmin=-vmax, vmax=vmax, interpolation="nearest")

        ax_dif.set_title("model FI − true FI", fontsize=8)
        ax_dif.set_xlabel("MSA position", fontsize=7)
        ax_dif.set_yticks([])
        ax_dif.tick_params(labelsize=6)

        fig.colorbar(im, cax=ax_dcb)
        ax_dcb.tick_params(labelsize=5)
        ax_dcb.yaxis.set_label_position("right")
        ax_dcb.set_ylabel("ΔFI", fontsize=6, rotation=90)

    fig.suptitle(funfam, fontsize=9, y=1.01)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── per-column fraction plot ──────────────────────────────────────────────────

def plot_class_fracs(class_mat, out_path, funfam, source_label=""):
    n, msa_len = class_mat.shape
    x   = np.arange(msa_len)
    hf  = np.zeros(msa_len)
    neu = np.zeros(msa_len)
    mf  = np.zeros(msa_len)

    for col in range(msa_len):
        col_data = class_mat[:, col]
        non_gap  = col_data[col_data < 3]
        if len(non_gap) == 0:
            continue
        hf[col]  = (non_gap == 0).sum() / n
        neu[col] = (non_gap == 1).sum() / n
        mf[col]  = (non_gap == 2).sum() / n

    # width matches the combined heatmap width (one panel)
    fig_w = msa_len * PTS_PER_COL / 72.0 + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))

    ax.fill_between(x, 0,           hf,              color=COLOR_HF,  alpha=0.85, linewidth=0)
    ax.fill_between(x, hf,          hf + neu,        color=COLOR_NEU, alpha=0.85, linewidth=0)
    ax.fill_between(x, hf + neu,    hf + neu + mf,   color=COLOR_MF,  alpha=0.85, linewidth=0)

    title = f"Per-column frustration fractions – {funfam}"
    if source_label:
        title += f"  [{source_label}]"
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, msa_len - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("MSA position",          fontsize=8)
    ax.set_ylabel("Fraction of sequences", fontsize=8)
    ax.legend(handles=[
        mpatches.Patch(color=COLOR_HF,  label="Highly frustrated"),
        mpatches.Patch(color=COLOR_NEU, label="Neutral"),
        mpatches.Patch(color=COLOR_MF,  label="Minimally frustrated"),
    ], loc="upper right", fontsize=6, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6)

    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--funfam",      default="3.40.710.10-FF-000005")
    p.add_argument("--results_dir", default=RESULTS_BASE,
                   help="Dir with {funfam}/FrustraEvo_{funfam}/AuxFiles/DF_Colores (model)")
    p.add_argument("--workdir",     default=WORKDIR_BASE,
                   help="seqdist3 root with FrustraEvo_{funfam}/AuxFiles/DF_Colores (true)")
    p.add_argument("--out_dir",     default=OUT_BASE)
    args = p.parse_args()

    fam_dir  = os.path.join(args.results_dir, args.funfam, f"FrustraEvo_{args.funfam}")
    model_dc = os.path.join(fam_dir, "AuxFiles", "DF_Colores")
    true_dc  = os.path.join(args.workdir, f"FrustraEvo_{args.funfam}", "AuxFiles", "DF_Colores")

    msa_path = os.path.join(fam_dir, "AuxFiles", "MSA_Clean.fasta")
    if not os.path.isfile(msa_path):
        msa_path = os.path.join(fam_dir, "MSA_Clean.fasta")
    if not os.path.isfile(msa_path):
        raise FileNotFoundError(f"MSA not found near {fam_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading MSA: {msa_path}")
    msa_entries = load_msa(msa_path)
    msa_len     = len(msa_entries[0][1])
    seq_order   = [name.split()[0] for name, _ in msa_entries]
    print(f"  {len(seq_order)} sequences, length {msa_len}")

    safe = args.funfam.replace(".", "_").replace("-", "_")

    if not os.path.isfile(true_dc):
        raise FileNotFoundError(f"True DF_Colores not found: {true_dc}")

    print(f"Loading true DF_Colores: {true_dc}")
    df_true   = load_df_colores(true_dc)
    class_mat = build_class_matrix(df_true, seq_order, msa_len)
    covered   = int((class_mat < 3).any(axis=1).sum())
    print(f"  {covered}/{len(seq_order)} sequences have class data")

    # diff matrix — optional (needs completed model run)
    diff_mat = None
    if os.path.isfile(model_dc):
        print(f"Loading model DF_Colores: {model_dc}")
        df_model  = load_df_colores(model_dc)
        model_mat = build_fi_matrix(df_model, seq_order, msa_len)
        true_fi   = build_fi_matrix(df_true,  seq_order, msa_len)
        diff_mat  = model_mat - true_fi
        n_paired  = int(np.any(~np.isnan(diff_mat), axis=1).sum())
        print(f"  {n_paired}/{len(seq_order)} sequences with paired diff data")
    else:
        print(f"Model DF_Colores not found — diff panel omitted: {model_dc}")

    plot_combined(class_mat, diff_mat, seq_order,
                  os.path.join(args.out_dir, f"msfa_{safe}.png"),
                  args.funfam)

    plot_class_fracs(class_mat,
                     os.path.join(args.out_dir, f"msfa_{safe}_fracs.png"),
                     args.funfam, source_label="true")


if __name__ == "__main__":
    main()
