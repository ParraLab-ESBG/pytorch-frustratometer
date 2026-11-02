"""
fig2_validation.py  --  Figure 2: beta-lactamase example + per-mode accuracy scatter

Panels:
  A) Single-residue frustration line plot for one protein, colored by class
  B) Scatter new vs original FI -- single-residue, all test proteins
  C) Scatter new vs original FI -- mutational, all test proteins (GLY mutations in orange)
  D) Scatter new vs original FI -- configurational, all test proteins

Each scatter panel annotates R² and mean cosine similarity.

Usage:
  python fig2_validation.py \\
      --results-dir /gpfs/.../frust_verify_results \\
      --protein 3ue3A03 \\
      --out fig2.pdf

  # Per-protein scatter plots (one subplot per protein):
  python fig2_validation.py \\
      --results-dir /gpfs/.../frust_verify_results \\
      --per-protein --per-protein-mode mutational \\
      --out fig2_per_protein.pdf
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Frustration thresholds & colours
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "singleresidue":             (-1.0, 0.55),
    "mutational":                (-1.0, 0.78),
    "configurational":           (-1.0, 0.78),
    "singleresidue_mutations":   (-1.0, 0.55),
}
COLOR_HF  = "#CC2222"
COLOR_NEU = "#BBBBBB"
COLOR_MF  = "#22AA44"
COLOR_GLY = "#FF8C00"   # orange — GLY-related mutations
ALPHA_HF  = 0.85
ALPHA_MF  = 0.85
ALPHA_NEU = 0.55

MODE_LABELS = {
    "singleresidue":   "Single Residue",
    "mutational":      "Mutational",
    "configurational": "Configurational",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_file(path):
    """Read a frustratometer3 output file, skipping comment lines."""
    return pd.read_csv(path, sep=r"\s+", comment="#")


def _gly_mask_from_df(df, valid_mask):
    """Return bool array (same length as valid_mask.sum()) marking GLY-related rows."""
    sub = df[valid_mask]
    gm = np.zeros(len(sub), dtype=bool)
    if "AA_native" in sub.columns:
        gm |= (sub["AA_native"].astype(str).values == "G")
    if "AA_mutant" in sub.columns:
        gm |= (sub["AA_mutant"].astype(str).values == "G")
    return gm


def _find_mode_files(results_dir, mode, protein=None):
    """
    Walk results_dir recursively and collect files whose name ends with a
    known suffix for the given mode.  If protein is given, the filename must
    also contain that string.

    Recognised suffixes (checked in order, shortest-specific first):
      .<mode>
      .pdb_<mode>
      .frustratometer3.<mode>
    """
    suffixes = [
        f".{mode}",
        f".pdb_{mode}",
        f".frustratometer3.{mode}",
    ]
    is_mutations_mode = mode.endswith("_mutations")
    found = []
    for dirpath, _dirs, filenames in os.walk(results_dir):
        for fn in sorted(filenames):
            if protein and protein not in fn:
                continue
            for suf in suffixes:
                if fn.endswith(suf):
                    remainder = fn[: len(fn) - len(suf)]
                    # When looking for a base mode, skip _mutations files;
                    # when looking for a _mutations mode, allow them.
                    if not is_mutations_mode and remainder.endswith("_mutations"):
                        continue
                    found.append(os.path.join(dirpath, fn))
                    break
    return found


def _pdb_id_from_filename(filename, mode):
    """Extract pdb_id from a filename like '8dc1A01.pdb_mutational'."""
    bn = os.path.basename(filename)
    for suf in (f".frustratometer3.{mode}", f".pdb_{mode}", f".{mode}"):
        if bn.endswith(suf):
            return bn[: -len(suf)]
    return None


def _merge_dompdb_reference(df, pdb_id, mode, dompdb_dir):
    """
    If df is missing TrueFrstIndex, try to load the FrustratometeR reference
    from {dompdb_dir}/{pdb_id}.done/FrustrationData/{pdb_id}.pdb_{mode}
    and join on Res1+Res2 (pair modes) or Res (singleresidue).
    Returns df with TrueFrstIndex added, or the original df if not found.
    """
    if "TrueFrstIndex" in df.columns or dompdb_dir is None or pdb_id is None:
        return df

    ref_path = os.path.join(dompdb_dir, f"{pdb_id}.done", "FrustrationData",
                            f"{pdb_id}.pdb_{mode}")
    if not os.path.isfile(ref_path):
        return df

    try:
        ref = pd.read_csv(ref_path, sep=r"\s+", comment="#")
        fi_col = next((c for c in ("FrstIndex", "PredFrstIndex") if c in ref.columns), None)
        if fi_col is None:
            return df

        if "mutations" in mode:
            key = ["Res", "AA_native", "AA_mutant"]
            if "ChainRes" in ref.columns and "ChainRes" in df.columns:
                key = ["Res", "ChainRes", "AA_native", "AA_mutant"]
        elif mode == "singleresidue":
            key = ["Res"]
        else:
            key = ["Res1", "Res2"]
            if "ChainRes1" in ref.columns and "ChainRes1" in df.columns:
                key = ["Res1", "Res2", "ChainRes1", "ChainRes2"]

        key = [k for k in key if k in ref.columns and k in df.columns]
        if not key:
            return df

        ref_sub = ref[key + [fi_col]].rename(columns={fi_col: "TrueFrstIndex"})
        df = df.merge(ref_sub, on=key, how="left")
    except Exception:
        pass

    return df


def load_example_protein(results_dir, protein, mode="singleresidue"):
    """
    Return a DataFrame for a single protein in the given mode.
    Searches results_dir and all subdirectories.
    """
    files = _find_mode_files(results_dir, mode, protein=protein)
    if not files:
        raise FileNotFoundError(
            f"No {mode} file for protein '{protein}' under {results_dir}"
        )
    return _read_file(files[0])


def load_all(results_dir, mode, max_files=None, seed=42, dompdb_dir=None):
    """
    Load FrstIndex and TrueFrstIndex from all files for a given mode.
    Returns (our_values, ref_values, gly_mask) as 1-D numpy arrays.
    gly_mask is True for rows where AA_native=='G' or AA_mutant=='G'.
    Searches results_dir and all subdirectories.
    If TrueFrstIndex is absent, falls back to dompdb_dir reference files.
    """
    files = _find_mode_files(results_dir, mode)
    if not files:
        raise FileNotFoundError(f"No {mode} files found under {results_dir}")
    print(f"  found {len(files)} file(s), e.g. {os.path.basename(files[0])}")

    if max_files and len(files) > max_files:
        rng = np.random.default_rng(seed)
        files = list(rng.choice(files, max_files, replace=False))

    ours, refs, gly_masks = [], [], []
    skipped_cols = 0
    for f in files:
        try:
            df = _read_file(f)
            pdb_id = _pdb_id_from_filename(f, mode)
            df = _merge_dompdb_reference(df, pdb_id, mode, dompdb_dir)
            if "FrstIndex" not in df.columns or "TrueFrstIndex" not in df.columns:
                skipped_cols += 1
                if skipped_cols == 1:
                    print(f"  WARNING: missing FrstIndex/TrueFrstIndex in {os.path.basename(f)}"
                          f" — columns: {list(df.columns)}")
                continue
            o = pd.to_numeric(df["FrstIndex"],     errors="coerce")
            r = pd.to_numeric(df["TrueFrstIndex"], errors="coerce")
            mask = o.notna() & r.notna()
            ours.append(o[mask].values)
            refs.append(r[mask].values)
            gly_masks.append(_gly_mask_from_df(df, mask))
        except Exception:
            pass

    if skipped_cols:
        print(f"  WARNING: {skipped_cols} file(s) skipped due to missing columns")
    if not ours:
        raise ValueError(
            f"No files with both FrstIndex and TrueFrstIndex columns for mode '{mode}'"
        )
    return np.concatenate(ours), np.concatenate(refs), np.concatenate(gly_masks)


def load_per_protein(results_dir, mode, max_files=None, seed=42, dompdb_dir=None):
    """
    Load per-protein FI data.
    Returns list of dicts: {name, ours, refs, gly_mask}.
    Searches results_dir and all subdirectories.
    If TrueFrstIndex is absent, falls back to dompdb_dir reference files.
    """
    files = _find_mode_files(results_dir, mode)
    if not files:
        raise FileNotFoundError(f"No {mode} files found under {results_dir}")

    if max_files and len(files) > max_files:
        rng = np.random.default_rng(seed)
        files = list(rng.choice(files, max_files, replace=False))

    proteins = []
    for f in files:
        try:
            df = _read_file(f)
            pdb_id = _pdb_id_from_filename(f, mode)
            df = _merge_dompdb_reference(df, pdb_id, mode, dompdb_dir)
            if "FrstIndex" not in df.columns or "TrueFrstIndex" not in df.columns:
                continue
            o = pd.to_numeric(df["FrstIndex"],     errors="coerce")
            r = pd.to_numeric(df["TrueFrstIndex"], errors="coerce")
            mask = o.notna() & r.notna()
            if mask.sum() == 0:
                continue
            proteins.append({
                "name":     os.path.basename(f),
                "ours":     o[mask].values,
                "refs":     r[mask].values,
                "gly_mask": _gly_mask_from_df(df, mask),
            })
        except Exception:
            pass

    return proteins


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    """Mean cosine similarity treating each (a_i, b_i) pair as a 1-D vector."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return float("nan")
    return float(np.dot(a, b) / (norm_a * norm_b))


def r2(a, b):
    if len(a) < 2:
        return float("nan")
    r, _ = pearsonr(a, b)
    return r ** 2


# ---------------------------------------------------------------------------
# Panel A: line plot
# ---------------------------------------------------------------------------

def panel_line(ax, df, mode="singleresidue"):
    lo, hi = THRESHOLDS[mode]

    if mode == "singleresidue":
        x_col, y_col = "Res", "FrstIndex"
    else:
        x_col, y_col = None, "FrstIndex"

    fi = pd.to_numeric(df[y_col], errors="coerce").values
    x  = df[x_col].values if x_col else np.arange(len(fi))

    colors = np.where(fi < lo, COLOR_HF, np.where(fi > hi, COLOR_MF, COLOR_NEU))

    for xi, yi, ci in zip(x, fi, colors):
        ax.plot([xi, xi], [0, yi], color=ci, linewidth=1.2, alpha=0.7, solid_capstyle="round")

    ax.axhline(lo, color="#555", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axhline(hi, color="#555", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axhline(0,  color="#999", linestyle=":",  linewidth=0.6)

    ax.set_xlabel("Residue", fontsize=8)
    ax.set_ylabel("Frustration Index", fontsize=8)
    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_HF,  linewidth=2, label="Highly frustrated"),
        Line2D([0], [0], color=COLOR_NEU, linewidth=2, label="Neutral"),
        Line2D([0], [0], color=COLOR_MF,  linewidth=2, label="Minimally frustrated"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="upper right",
              framealpha=0.85, edgecolor="none")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _scatter_limits(ours, refs, cap=6.0):
    return min(max(abs(refs).max(), abs(ours).max()) * 1.05, cap)


def _annotate_stats(ax, ours, refs, gly_mask=None):
    _r2  = r2(ours, refs)
    _cos = cosine_sim(ours, refs)
    ax.text(0.05, 0.93, f"R² = {_r2:.4f}",      transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="#222")
    ax.text(0.05, 0.86, f"cos sim = {_cos:.4f}", transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="#222")
    ax.text(0.05, 0.79, f"n = {len(ours):,}",    transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="#555")
    if gly_mask is not None and gly_mask.any():
        ax.text(0.05, 0.72, f"n(GLY) = {int(gly_mask.sum()):,}", transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color=COLOR_GLY)


def _finish_accuracy_ax(ax, lim):
    ax.plot([-lim, lim], [-lim, lim], color="#333", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("FrustratometeR FI", fontsize=8)
    ax.set_ylabel("PyTorch FI",        fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Panels B-D: scatter plots
# ---------------------------------------------------------------------------

def panel_scatter(ax, ours, refs, mode, gly_mask=None, max_points=50_000, seed=42):
    lo, hi = THRESHOLDS[mode]

    if gly_mask is None:
        gly_mask = np.zeros(len(ours), dtype=bool)

    # Subsample non-GLY points for rendering speed; always keep all GLY points
    non_gly_idx = np.where(~gly_mask)[0]
    gly_idx     = np.where(gly_mask)[0]

    if len(non_gly_idx) > max_points:
        rng = np.random.default_rng(seed)
        non_gly_idx = rng.choice(non_gly_idx, max_points, replace=False)

    idx    = np.concatenate([non_gly_idx, gly_idx])
    o_plot = ours[idx]
    r_plot = refs[idx]
    g_plot = gly_mask[idx]

    # Color non-GLY points by frustration class (based on our FrstIndex)
    colors = np.where(o_plot < lo, COLOR_HF, np.where(o_plot > hi, COLOR_MF, COLOR_NEU))

    ng = ~g_plot
    ax.scatter(r_plot[ng], o_plot[ng], c=colors[ng],
               s=2, alpha=0.25, linewidths=0, rasterized=True)

    if g_plot.any():
        ax.scatter(r_plot[g_plot], o_plot[g_plot], c=COLOR_GLY,
                   s=8, alpha=0.65, linewidths=0, rasterized=True,
                   label="GLY mutation", zorder=5)
        ax.legend(fontsize=6, loc="lower right", framealpha=0.85, edgecolor="none",
                  markerscale=2)

    # Diagonal reference line
    lim = max(abs(refs).max(), abs(ours).max()) * 1.05
    lim = min(lim, 6.0)
    ax.plot([-lim, lim], [-lim, lim], color="#333", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    _r2  = r2(ours, refs)
    _cos = cosine_sim(ours, refs)
    ax.text(0.05, 0.93, f"R² = {_r2:.4f}",      transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="#222")
    ax.text(0.05, 0.86, f"cos sim = {_cos:.4f}", transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="#222")
    ax.text(0.05, 0.79, f"n = {len(ours):,}",    transform=ax.transAxes, fontsize=7,
            va="top", ha="left", color="#555")
    if g_plot.any():
        ax.text(0.05, 0.72, f"n(GLY) = {int(gly_mask.sum()):,}", transform=ax.transAxes,
                fontsize=7, va="top", ha="left", color=COLOR_GLY)

    ax.set_xlabel("FrustratometeR FI", fontsize=8)
    ax.set_ylabel("PyTorch FI",        fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Per-protein scatter figures
# ---------------------------------------------------------------------------

def make_per_protein_figures(results_dir, mode, out_path, max_files=None, dompdb_dir=None):
    """
    Create one scatter subplot per protein and save them to a multi-page PDF
    (or a single PNG/SVG grid if out_path doesn't end in .pdf).
    """
    print(f"Loading per-protein {mode} data …")
    proteins = load_per_protein(results_dir, mode, max_files=max_files, dompdb_dir=dompdb_dir)
    if not proteins:
        sys.exit(f"No {mode} data found in {results_dir}")

    print(f"  {len(proteins)} proteins found")

    ncols  = 4
    nrows  = (len(proteins) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.5),
                              squeeze=False)

    for i, prot in enumerate(proteins):
        ax = axes[i // ncols][i % ncols]
        panel_scatter(ax, prot["ours"], prot["refs"], mode,
                      gly_mask=prot["gly_mask"])
        # Strip file extension suffixes for a clean title
        name = prot["name"]
        for suffix in (f".frustratometer3.{mode}", f".pdb_{mode}", f".{mode}"):
            name = name.replace(suffix, "")
        ax.set_title(name, fontsize=6, pad=3)

    for j in range(len(proteins), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.suptitle(f"{MODE_LABELS.get(mode, mode)} — per-protein scatter", fontsize=10, y=1.01)
    plt.tight_layout()

    if out_path.lower().endswith(".pdf"):
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig, dpi=200, bbox_inches="tight")
    else:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(results_dir, protein, out_path, max_files=None, dompdb_dir=None):
    fig = plt.figure(figsize=(15, 3.5))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.38,
                            left=0.05, right=0.98, top=0.88, bottom=0.18)

    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    panel_labels = ["A", "B", "C", "D", "E"]
    subtitles = [
        f"Single-residue frustration\n({protein})",
        "Single-residue accuracy\n(all test proteins)",
        "Mutational accuracy\n(all test proteins)",
        "Configurational accuracy\n(all test proteins)",
        "Mutation scanning accuracy\n(all test proteins)",
    ]

    # Panel A — line plot for the example protein
    print(f"Loading single-residue data for {protein} …")
    try:
        df_ex = load_example_protein(results_dir, protein, "singleresidue")
        panel_line(axes[0], df_ex, "singleresidue")
    except FileNotFoundError as e:
        axes[0].text(0.5, 0.5, str(e), ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=6, wrap=True)

    # Panels B-D — scatter for each mode; GLY coloring on mutational panel (C)
    modes = ["singleresidue", "mutational", "configurational"]
    for ax, mode in zip(axes[1:4], modes):
        print(f"Loading all {mode} data …")
        try:
            ours, refs, gly_mask = load_all(results_dir, mode, max_files=max_files,
                                            dompdb_dir=dompdb_dir)
            use_gly = gly_mask if mode == "mutational" else None
            panel_scatter(ax, ours, refs, mode, gly_mask=use_gly)
        except (FileNotFoundError, ValueError) as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, fontsize=6, wrap=True)

    # Panel E — mutation scanning scatter (GLY always colored)
    print("Loading all singleresidue_mutations data …")
    try:
        ours, refs, gly_mask = load_all(results_dir, "singleresidue_mutations",
                                        max_files=max_files, dompdb_dir=dompdb_dir)
        panel_scatter(axes[4], ours, refs, "singleresidue_mutations", gly_mask=gly_mask)
    except (FileNotFoundError, ValueError) as e:
        axes[4].text(0.5, 0.5, str(e), ha="center", va="center",
                     transform=axes[4].transAxes, fontsize=6, wrap=True)

    # Labels and titles
    for ax, label, subtitle in zip(axes, panel_labels, subtitles):
        ax.set_title(subtitle, fontsize=8, pad=4)
        ax.text(-0.12, 1.08, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing *.frustratometer3.* output files")
    parser.add_argument("--protein", default="3ue3A03",
                        help="Protein ID for panel A line plot (default: 3ue3A03)")
    parser.add_argument("--out", default="fig2.pdf",
                        help="Output file (pdf/png/svg). Default: fig2.pdf")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap number of files loaded per mode (for testing)")
    parser.add_argument("--per-protein", action="store_true",
                        help="Create one scatter subplot per protein instead of the main 4-panel figure")
    parser.add_argument("--per-protein-mode", default="mutational",
                        choices=["singleresidue", "mutational", "configurational"],
                        help="Mode to use for --per-protein scatter (default: mutational)")
    parser.add_argument("--dompdb-dir", default=None,
                        help="Path to FrustratometeR reference directory "
                             "(e.g. ../dompdb_v4.4). When provided, pair-mode files "
                             "without TrueFrstIndex are augmented by reading "
                             "{dompdb_dir}/{pdb_id}.done/FrustrationData/{pdb_id}.pdb_{mode}.")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f"Results directory not found: {args.results_dir}")

    dompdb_dir = args.dompdb_dir
    if dompdb_dir and not os.path.isdir(dompdb_dir):
        print(f"WARNING: --dompdb-dir {dompdb_dir!r} does not exist; ignoring.")
        dompdb_dir = None

    if args.per_protein:
        make_per_protein_figures(
            args.results_dir, args.per_protein_mode, args.out,
            max_files=args.max_files, dompdb_dir=dompdb_dir,
        )
    else:
        make_figure(args.results_dir, args.protein, args.out, args.max_files,
                    dompdb_dir=dompdb_dir)


if __name__ == "__main__":
    main()
