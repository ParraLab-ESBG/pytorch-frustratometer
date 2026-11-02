"""
Plot mutation FrstIndex benchmark results.

Compares predicted mutation FrstIndex (from *.frustratometer3.singleresidue_mutations)
against reference MutationsData produced by the original frustratometer.

Usage:
  python -m frustrai_fold.plots.mutations \\
      --pred_dir baseline_comparison/mutants \\
      --ref_dir  /path/to/dompdb_v4.4_mutants \\
      --out      baseline_comparison/mutants/plots

Ref layout (either):
  <ref_dir>/<stem>.done/MutationsData/singleresidue_Res*_threading_*.txt
  <ref_dir>/<stem>_*.done/MutationsData/singleresidue_Res*_threading_*.txt

Produces:
  summary_mutations.png, scatter_mutations.png, confusion_mutations.png,
  r_kde_mutations.png, acc_kde_mutations.png, cosine_kde_mutations.png,
  r_vs_n_mutations.png, mutations_metrics.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm

from .common import (
    PALETTE, LO_SR, HI_SR, fi_class, compute_metrics,
    draw_kde, draw_confusion, draw_scatter_hexbin,
    CMAP_FI, CLASS_COLORS,
)

_FORMATS = ["png"]   # extend to e.g. ["png", "svg"] to match singleresidue.py


def _savefig(fig, path: Path, dpi: int = 150) -> None:
    """Save fig in every format listed in _FORMATS (mirroring singleresidue._savefig)."""
    stem = Path(path).with_suffix("")
    try:
        for fmt in _FORMATS:
            p = stem.with_suffix(f".{fmt}")
            kw = {"bbox_inches": "tight"}
            if fmt.lower() not in ("svg", "pdf", "eps", "ps"):
                kw["dpi"] = dpi
            fig.savefig(p, **kw)
    finally:
        plt.close(fig)

COLOR   = PALETTE["mutations"]
_XLABEL = "FrustratometeR"
_YLABEL = "PyTorch Re-implementation"

# AA ordering matching ID_TO_AA: 0=A,1=C,2=D,3=E,4=F,5=G,6=H,7=I,8=K,9=L,
#                                 10=M,11=N,12=P,13=Q,14=R,15=S,16=T,17=V,18=W,19=Y
_AA_ORDER  = list("ACDEFGHIKLMNPQRSTVWY")
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ORDER)}
_GLY_IDX   = _AA_TO_IDX["G"]                             # 5
_NON_G_COLS = [i for i in range(20) if i != _GLY_IDX]   # 19 column indices (non-G mutants)
_NON_G_AA   = [_AA_ORDER[i] for i in _NON_G_COLS]        # 19 AA labels without G


# ── scanning heatmap ──────────────────────────────────────────────────────────

_CROP_SENTINEL = -9999  # marks separator columns inserted by crop_ranges

_AA_3LETTER = {
    'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe',
    'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu',
    'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg',
    'S': 'Ser', 'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr',
}

# Biochemical property groups and their label colors
_AA_PROPERTY = {
    'A': 'nonpolar', 'V': 'nonpolar', 'I': 'nonpolar', 'L': 'nonpolar',
    'M': 'nonpolar', 'F': 'nonpolar', 'W': 'nonpolar', 'P': 'nonpolar',
    'S': 'polar',    'T': 'polar',    'C': 'polar',    'Y': 'polar',
    'N': 'polar',    'Q': 'polar',
    'K': 'positive', 'R': 'positive', 'H': 'positive',
    'D': 'negative', 'E': 'negative',
    'G': 'nonpolar',
}
_PROP_COLOR = {
    'nonpolar': '#333333',
    'polar':    '#2E7D32',
    'positive': '#1565C0',
    'negative': '#B71C1C',
}


def _build_crop_arrays(fi_mat, wt_aa_indices, res_nums, wt_aas,
                       crop_ranges, warn_col_mask=None):
    """Concatenate crop_ranges slices with NaN separator columns between them.

    Returns (fi, wt, res, aas, warn, sep_col_indices) where sep_col_indices is
    the set of column positions (in the new matrix) that are separators.
    """
    res_arr  = np.asarray(res_nums)
    wt_arr   = np.asarray(wt_aa_indices)
    wt_aas_a = list(wt_aas)
    wm       = np.asarray(warn_col_mask) if warn_col_mask is not None else None

    fi_parts, wt_parts, res_parts, aa_parts, wm_parts = [], [], [], [], []
    sep_fi = np.full((1, fi_mat.shape[1]), np.nan, dtype=fi_mat.dtype)
    sep_indices, offset = set(), 0

    for i, (start, end) in enumerate(crop_ranges):
        if i > 0:
            sep_indices.add(offset)
            fi_parts.append(sep_fi)
            wt_parts.append(np.array([-1]))
            res_parts.append(np.array([_CROP_SENTINEL]))
            aa_parts.append([""])
            if wm is not None:
                wm_parts.append(np.array([False]))
            offset += 1
        fi_parts.append(fi_mat[start:end])
        wt_parts.append(wt_arr[start:end])
        res_parts.append(res_arr[start:end])
        aa_parts.append([wt_aas_a[j] for j in range(start, end)])
        if wm is not None:
            wm_parts.append(wm[start:end])
        offset += end - start

    new_fi  = np.concatenate(fi_parts, axis=0)
    new_wt  = np.concatenate(wt_parts)
    new_res = np.concatenate(res_parts)
    new_aas = [item for sub in aa_parts for item in sub]
    new_wm  = np.concatenate(wm_parts) if wm is not None else None
    return new_fi, new_wt, new_res, new_aas, new_wm, sep_indices


def _draw_heatmap(ax, fi_mat, wt_aa_indices, res_nums, wt_aas, title,
                  cmap=None, norm=None, vabs=None, aa_labels=None,
                  gap_rows=None, warn_col_mask=None, warn_row_indices=None,
                  warn_color="darkorange", tick_step=None,
                  hatch_nan_cols=False, active_site_res=None,
                  active_site_color="magenta", active_site_label=True,
                  active_site_label_names=None, active_site_mutant_aa=None,
                  active_site_colors=None, active_site_citations=None,
                  crop_ranges=None, show_title=True,
                  color_yticks_by_property=False,
                  res_num_remap=None,
                  active_site_label_x_offset=None,
                  validated_res=None,
                  validated_res_sources=None):
    """Render a [n_res, n_aa] FI matrix as a heatmap with WT circles.

    gap_rows               : list of int — y-axis row indices used as white-space separators;
                             covered with a white patch and omitted from yticks.
    warn_col_mask          : bool array [n_res] — x-tick labels coloured warn_color.
    warn_row_indices       : set of int — y-tick labels at these rows coloured warn_color.
    hatch_nan_cols         : if True, fully-NaN position columns are rendered in light grey
                             with diagonal hatching.
    active_site_res        : list of residue numbers to highlight with a coloured box.
    active_site_color      : colour for the active-site rectangle (default "magenta").
    active_site_label      : if True, label each active-site column above the x-axis.
    active_site_label_names: dict {res_num: label_str} — override the auto label for a
                             residue (e.g. {38: "S70"} shows "S70" instead of "S38").
    crop_ranges            : list of (start, end) column-index ranges to display; NaN
                             separator columns are inserted between sections with "…" ticks.
    """
    import copy as _copy

    # Apply column crop before any other processing
    sep_col_indices: set = set()
    if crop_ranges is not None:
        fi_mat, wt_aa_indices, res_nums, wt_aas, warn_col_mask, sep_col_indices = \
            _build_crop_arrays(fi_mat, wt_aa_indices, res_nums, wt_aas,
                               crop_ranges, warn_col_mask)

    n_res, n_aa = fi_mat.shape
    labels    = aa_labels if aa_labels is not None else _AA_ORDER[:n_aa]
    gap_set   = set(gap_rows or [])
    warn_rows = set(warn_row_indices or [])

    if norm is None:
        vabs = vabs or max(float(np.nanmax(np.abs(fi_mat))), 1e-3)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    if cmap is None:
        cmap = CMAP_FI

    if hatch_nan_cols:
        cmap_obj = _copy.copy(plt.get_cmap(cmap) if isinstance(cmap, str) else cmap)
        cmap_obj.set_bad("lightgrey", alpha=0.7)
        im = ax.imshow(np.ma.masked_invalid(fi_mat).T,
                       aspect="auto", cmap=cmap_obj, norm=norm, origin="lower")
        for c in np.where(np.all(np.isnan(fi_mat), axis=1))[0]:
            ax.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, n_aa,
                                       fill=False, hatch="////",
                                       edgecolor="dimgrey", linewidth=0, zorder=3))
    else:
        im = ax.imshow(fi_mat.T, aspect="auto", cmap=cmap, norm=norm, origin="lower")

    # White cell gridlines
    ax.set_xticks(np.arange(-0.5, n_res), minor=True)
    ax.set_yticks(np.arange(-0.5, n_aa), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, zorder=2)
    ax.tick_params(which="minor", length=0)

    # White gap patches — slightly oversize in height to suppress pixel boundary lines
    for r in gap_set:
        ax.add_patch(plt.Rectangle((-0.5, r - 0.6), n_res, 1.2,
                                   fill=True, facecolor="white",
                                   edgecolor="none", linewidth=0, zorder=4))

    for pos in range(n_res):
        wt = int(wt_aa_indices[pos])
        if 0 <= wt < n_aa and wt not in gap_set:
            ax.add_patch(plt.Rectangle(
                (pos - 0.5, wt - 0.5), 1, 1,
                linewidth=2.0, edgecolor="black",
                facecolor="none", zorder=8, clip_on=False,
            ))

    # y-axis: skip gap rows entirely (no tick mark, no label)
    non_gap = [r for r in range(n_aa) if r not in gap_set]
    ax.set_yticks(non_gap)
    y_lbls = ax.set_yticklabels([labels[r] for r in non_gap], fontsize=11,
                                 fontweight="bold")
    ax.set_ylabel("Possible mutations", fontsize=14, fontweight="bold")
    for lbl, r in zip(y_lbls, non_gap):
        if r in warn_rows:
            lbl.set_color(warn_color)
        elif color_yticks_by_property:
            aa = labels[r]
            prop = _AA_PROPERTY.get(aa, "nonpolar")
            lbl.set_color(_PROP_COLOR[prop])

    if color_yticks_by_property:
        from matplotlib.patches import Patch
        _prop_legend = [
            ("Nonpolar", "nonpolar"), ("Polar", "polar"),
            ("Positive", "positive"), ("Negative", "negative"),
        ]
        prop_handles = [Patch(facecolor=_PROP_COLOR[p], label=l, edgecolor="none")
                        for l, p in _prop_legend]
        ax.legend(handles=prop_handles, loc="upper left", fontsize=5,
                  framealpha=0.92, edgecolor="#cccccc", facecolor="white",
                  title="AA class", title_fontsize=6)

    step     = tick_step if tick_step is not None else max(1, n_res // 30)
    tick_pos = list(range(0, n_res, step))
    ax.set_xticks(tick_pos)

    _remap = res_num_remap or {}

    def _xtick_label(p):
        if p in sep_col_indices:
            return "…"
        rnum = int(res_nums[p])
        return f"{wt_aas[p]}{_remap.get(rnum, rnum)}"

    x_lbls = ax.set_xticklabels(
        [_xtick_label(p) for p in tick_pos],
        fontsize=6, rotation=90, fontweight="bold")
    ax.set_xlabel("Residue", fontsize=14, fontweight="bold")

    # Hide tick marks at separator columns (keep the "…" label)
    for tick, pos in zip(ax.xaxis.get_major_ticks(), tick_pos):
        if pos in sep_col_indices:
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)

    if warn_col_mask is not None:
        for lbl, pos in zip(x_lbls, tick_pos):
            if pos in sep_col_indices:
                lbl.set_color("#999999")
            elif warn_col_mask[pos]:
                lbl.set_color(warn_color)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw thin divider lines at separator columns
    for sep_i in sep_col_indices:
        ax.axvline(sep_i, color="#aaaaaa", lw=1.2, zorder=5)

    # Active site column highlight
    if active_site_res:
        res_to_pos = {int(r): i for i, r in enumerate(res_nums)
                      if int(r) != _CROP_SENTINEL}
        _label_names  = active_site_label_names or {}
        _mutant_aa    = active_site_mutant_aa or {}
        _site_colors  = active_site_colors or {}
        _citations    = active_site_citations or {}
        _label_x_offsets = active_site_label_x_offset or {}
        n_aa = fi_mat.shape[1]

        # All labels at the same y-level; label_y is top of the name text block.
        _LABEL_BASE = n_aa + 2.5
        _label_y: dict[int, float] = {int(s): _LABEL_BASE for s in active_site_res}

        for site in active_site_res:
            col_i = res_to_pos.get(int(site))
            if col_i is None:
                continue
            color = _site_colors.get(int(site), active_site_color)
            rect = plt.Rectangle(
                (col_i - 0.5, -0.5), 1, n_aa,
                linewidth=1.5, edgecolor=color, #3.5
                facecolor="none", zorder=6,
                clip_on=False,
            )
            ax.add_patch(rect)
            # Star marker(s): transparent fill shows underlying FI; black edge for visibility
            # active_site_mutant_aa values may be a str or list[str]
            mut_aa_val = _mutant_aa.get(int(site))
            if mut_aa_val is not None:
                mut_aa_list = [mut_aa_val] if isinstance(mut_aa_val, str) else list(mut_aa_val)
                for mut_aa in mut_aa_list:
                    mut_row = _AA_TO_IDX.get(mut_aa)
                    if mut_row is not None and mut_row not in gap_set:
                        ax.scatter(col_i, mut_row, marker="o", s=100,
                                   facecolors=color,
                                   edgecolors="black",
                                   linewidths=4.0, zorder=9,
                                   clip_on=False)
            if active_site_label:
                display_label = _label_names.get(int(site))
                if display_label is None:
                    wt_aa = wt_aas[col_i] if col_i < len(wt_aas) else ""
                    display_label = f"{wt_aa}{site}"
                citation = _citations.get(int(site))
                label_y = _label_y.get(int(site), _LABEL_BASE) - 1

                # # Vertical line from heatmap top to just below text (separate from text)
                # line_end = label_y - 3.2 if citation else label_y - 1.2
                # ax.plot([col_i, col_i], [n_aa - 0.5, line_end],
                #         color=color, lw=2.0, clip_on=False, zorder=5)

                # Mutation name (bold, site color) — hangs down from label_y
                _ha = ("right" if citation and "Stoj" in citation
                       else ("left" if citation and "Hin" in citation else "center"))
                _lx = col_i + _label_x_offsets.get(int(site), 0)
                ax.text(_lx, label_y, display_label,
                        fontsize=13, color=color, fontweight="bold",
                        ha=_ha, va="top", clip_on=False)

                # Citation (smaller, black, italic) — below mutation name
                if citation:
                    ax.text(_lx, label_y - 0.8, citation,
                            fontsize=8, color="black", ha=_ha,
                            va="top", style="italic", clip_on=False,
                            multialignment=_ha)

    # Stars-only for additionally validated positions (no box, no label)
    if validated_res:
        _vres_to_pos = {int(r): i for i, r in enumerate(res_nums)
                        if int(r) != _CROP_SENTINEL}
        n_added = 0
        for site, mut_aas in validated_res:
            col_i = _vres_to_pos.get(int(site))
            if col_i is None:
                print(f"  [validated_res] SKIP star: res {site} not in heatmap (maybe cropped)")
                continue
            wt_at_col = wt_aas[col_i] if col_i < len(wt_aas) else "?"
            mut_aa_list = [mut_aas] if isinstance(mut_aas, str) else list(mut_aas)
            for mut_aa in mut_aa_list:
                mut_row = _AA_TO_IDX.get(mut_aa)
                if mut_row is not None and mut_row not in gap_set:
                    ax.scatter(col_i, mut_row, marker="o", s=100,
                               facecolors="black", edgecolors="none",
                               linewidths=0, zorder=9, clip_on=False)
                    n_added += 1
                else:
                    print(f"  [validated_res] SKIP star: res {site} mut={mut_aa} not in AA list")
        print(f"  [validated_res] added {n_added}/{len(validated_res)} star markers for validated positions")
    if show_title:
        ax.set_title(title, fontsize=10)
    return im


def _heatmap_figure(fi_main, wt_main, res_main, aa_main,
                    fi_gly,  wt_gly,  res_gly,  aa_gly,
                    fi_xg,   wt_xg,   res_xg,   aa_xg,
                    title_main, title_gly, title_xg,
                    cmap, norm_main, norm_special=None):
    """Three-panel figure: non-GLY×non-G (main) | GLY native | X→G strip.

    GLY native and X→G share norm_special (defaults to norm_main if not given),
    so they are directly comparable on the same colour scale.
    Returns (fig, im_main).
    """
    has_main = fi_main is not None and fi_main.shape[0] > 0
    has_gly  = fi_gly  is not None and fi_gly.shape[0]  > 0
    has_xg   = fi_xg   is not None and fi_xg.shape[0]   > 0
    norm_sp  = norm_special if norm_special is not None else norm_main

    n_main = fi_main.shape[0] if has_main else 0
    n_gly  = fi_gly.shape[0]  if has_gly  else 0
    n_xg   = fi_xg.shape[0]   if has_xg   else 0

    w_main = max(6,   n_main * 0.18) if has_main else 0
    w_gly  = max(1.5, n_gly  * 0.40) if has_gly  else 0
    w_xg   = max(3.0, n_xg   * 0.18) if has_xg   else 0
    fig_h  = 3.5

    ratios = []
    if has_main: ratios.append(n_main)
    if has_gly:  ratios.append(n_gly)
    if has_xg:   ratios.append(n_xg)

    total_w = w_main + w_gly + w_xg + 0.8
    fig = plt.figure(figsize=(total_w, fig_h))
    gs  = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.35)

    all_axes, panel_idx = [], 0
    im_main = im_sp = None

    if has_main:
        ax_main = fig.add_subplot(gs[panel_idx])
        im_main = _draw_heatmap(ax_main, fi_main, wt_main, res_main, aa_main,
                                 title_main, cmap=cmap, norm=norm_main)
        all_axes.append(ax_main)
        panel_idx += 1

    special_axes = []

    if has_gly:
        ax_gly = fig.add_subplot(gs[panel_idx])
        im_sp  = _draw_heatmap(ax_gly, fi_gly, wt_gly, res_gly, aa_gly,
                                title_gly, cmap=cmap, norm=norm_sp)
        special_axes.append(ax_gly)
        all_axes.append(ax_gly)
        panel_idx += 1

    if has_xg:
        ax_xg = fig.add_subplot(gs[panel_idx])
        im_sp  = _draw_heatmap(ax_xg, fi_xg, wt_xg, res_xg, aa_xg,
                                title_xg, cmap=cmap, norm=norm_sp,
                                aa_labels=["G"])
        special_axes.append(ax_xg)
        all_axes.append(ax_xg)

    if norm_special is not None and has_main and special_axes:
        plt.colorbar(im_main, ax=[ax_main],    shrink=0.8, pad=0.02)
        plt.colorbar(im_sp,   ax=special_axes, shrink=0.8, pad=0.02)
    else:
        ref_im = im_main if im_main is not None else im_sp
        plt.colorbar(ref_im, ax=all_axes, shrink=0.8, pad=0.02)

    return fig, im_main


_COLOR_CASE1 = "#FF8C00"   # GLY native positions (Case 1)
_COLOR_CASE2 = "#9467BD"   # X→G mutations at non-GLY positions (Case 2)


def _draw_scatter(ax, fi_mat, ref_fi_mat, wt_idx, fi_norm, title, res_nums=None):
    """Scatter of PyTorch vs original FI.

    Regular points (non-GLY native, non-X→G) — grey.
    Case 2 (X→G mutations at non-GLY positions) — purple dots.
    Case 1 (GLY native positions) — each GLY position gets a unique tab10/tab20
    color; mutations plotted as the mutant AA letter so individual outliers are
    identifiable (e.g. G22 vs G207).
    """
    n_res, n_aa = fi_mat.shape
    fi_flat     = fi_mat.ravel()
    ref_flat    = ref_fi_mat.ravel()
    mutant_flat = np.tile(np.arange(n_aa), n_res)
    pos_flat    = np.repeat(np.arange(n_res), n_aa)   # residue index per flat point
    is_gly_flat = np.repeat(wt_idx == _GLY_IDX, n_aa)

    valid   = np.isfinite(fi_flat) & np.isfinite(ref_flat)
    case1   = valid & is_gly_flat                                    # GLY native (any mutant)
    case2   = valid & ~is_gly_flat & (mutant_flat == _GLY_IDX)      # X→G at non-GLY positions
    regular = valid & ~is_gly_flat & (mutant_flat != _GLY_IDX)

    all_fi  = fi_flat[valid]
    all_ref = ref_flat[valid]
    lim = min(max(float(np.abs(all_fi).max()), float(np.abs(all_ref).max())) * 1.05, 6.0)
    ax.plot([-lim, lim], [-lim, lim],
            color="#333", linewidth=0.8, linestyle="--", alpha=0.7, zorder=0)

    # Regular points: uniform grey
    if regular.any():
        ax.scatter(ref_flat[regular], fi_flat[regular], color="#aaaaaa",
                   s=1, alpha=0.25, linewidths=0, rasterized=True, zorder=1)

    # Case 2 (X→G at non-GLY) — purple dots
    if case2.any():
        ax.scatter(ref_flat[case2], fi_flat[case2], color=_COLOR_CASE2,
                   s=4, alpha=0.65, linewidths=0, rasterized=True, zorder=2,
                   label=f"X→G  (n={int(case2.sum()):,})")

    # Case 1 (GLY native) — orange dots for all; outlier positions additionally
    # labeled with mutant AA letter (IQR rule on mean |Δ| per GLY position)
    if case1.any():
        ax.scatter(ref_flat[case1], fi_flat[case1], color=_COLOR_CASE1,
                   s=4, alpha=0.65, linewidths=0, rasterized=True, zorder=3,
                   label=f"GLY native  (n={int(case1.sum()):,})")

        gly_res_indices = np.unique(pos_flat[case1])
        devs = np.array([
            np.mean(np.abs(fi_flat[case1 & (pos_flat == gp)] -
                           ref_flat[case1 & (pos_flat == gp)]))
            for gp in gly_res_indices
        ])
        if len(devs) > 1:
            q25, q75 = np.percentile(devs, [25, 75])
            bad_threshold = q75 + 1.5 * (q75 - q25)
            bad_threshold = max(bad_threshold, np.median(devs) * 2.0)
        else:
            bad_threshold = np.inf   # single GLY position — nothing to flag

        _tab10 = plt.cm.tab10
        bi = 0
        for gi, gpos in enumerate(gly_res_indices):
            if devs[gi] <= bad_threshold:
                continue
            color     = _tab10(bi % 10)
            bi       += 1
            rnum      = int(res_nums[gpos]) if res_nums is not None else int(gpos) + 1
            gpos_mask = case1 & (pos_flat == gpos)
            xs   = ref_flat[gpos_mask]
            ys   = fi_flat[gpos_mask]
            muts = mutant_flat[gpos_mask]
            for x, y, m in zip(xs, ys, muts):
                ax.text(x, y, _AA_ORDER[m], color=color, fontsize=3.5,
                        ha="center", va="center", fontweight="bold",
                        zorder=4, clip_on=True)
            ax.scatter([], [], color=color, s=15, marker="s", linewidths=0,
                       label=f"G{rnum} (outlier)")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    # Frustration cutoff lines — coloured by class, labelled at the plot edge
    _c_hf = CLASS_COLORS[0]
    _c_mf = CLASS_COLORS[2]
    for cutoff, color in ((LO_SR, _c_hf), (HI_SR, _c_mf)):
        ax.axhline(cutoff, color=color, linestyle="--", linewidth=0.7, alpha=0.8, zorder=0)
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=0.7, alpha=0.8, zorder=0)
    _kw = dict(transform=ax.transData, fontsize=5, ha="right", clip_on=True)
    ax.text(lim * 0.98, (LO_SR - lim) / 2,    "HF",      color=_c_hf, va="center", **_kw)
    ax.text(lim * 0.98, (LO_SR + HI_SR) / 2,  "Neutral", color=CLASS_COLORS[1], va="center", **_kw)
    ax.text(lim * 0.98, (HI_SR + lim) / 2,    "MF",      color=_c_mf, va="center", **_kw)

    metrics = compute_metrics(all_fi, all_ref)
    ax.text(0.05, 0.95, f"R² = {metrics['r2']:.4f}",
            transform=ax.transAxes, fontsize=6, va="top", ha="left", color="#222")
    ax.text(0.05, 0.88, f"cos sim = {metrics['cosine_similarity']:.4f}",
            transform=ax.transAxes, fontsize=6, va="top", ha="left", color="#222")
    ax.text(0.05, 0.81, f"n = {int(valid.sum()):,}",
            transform=ax.transAxes, fontsize=6, va="top", ha="left", color="#555")

    ax.legend(fontsize=5, loc="lower right", framealpha=0.85, edgecolor="none",
              markerscale=2)
    ax.set_xlabel("Original FI", fontsize=7)
    ax.set_ylabel("PyTorch FI",  fontsize=7)
    ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scanning_heatmap(fi_mat, wt_aa_indices, res_nums, wt_aas,
                           stem, out_dir, suffix="scanning", ref_fi_mat=None,
                           active_site_res=None, active_site_label_names=None,
                           active_site_mutant_aa=None, active_site_colors=None,
                           active_site_citations=None,
                           crop_ranges=None, struct_png=None):
    """Save DMS heatmaps (and optionally error heatmaps vs reference).

    FI heatmap: single full-sequence panel [n_all, 20].  GLY positions and the
    X→G row are NaN → rendered grey/hatched so the viewer sees where they fall
    in the sequence without a separate panel.

    Error heatmap: one figure with up to two panels (small gap between them).
    Left panel — non-GLY × non-G errors with X→G appended below a white gap row.
    Right panel (if GLY positions exist) — GLY native errors.
    All panels share a global vmax for direct comparison.

    Parameters
    ----------
    fi_mat        : np.ndarray [n_res, 20]   predicted FI
    wt_aa_indices : array-like [n_res]        WT AA index 0-19
    res_nums      : array-like [n_res]        residue numbers for x-axis labels
    wt_aas        : list[str]  [n_res]        WT AA single-letter for x-axis labels
    stem          : str                       protein name (title + filename stem)
    out_dir       : Path                      output directory
    suffix        : str                       filename suffix (default "scanning")
    ref_fi_mat    : np.ndarray [n_res, 20] or None — if given, also saves error heatmaps
    """
    from matplotlib.colors import Normalize

    out_dir      = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wt_idx    = np.asarray(wt_aa_indices)
    res_arr   = np.asarray(res_nums)
    wt_aas_a  = list(wt_aas)

    is_gly    = wt_idx == _GLY_IDX
    non_gly_i = np.where(~is_gly)[0]
    gly_i     = np.where(is_gly)[0]
    has_gly   = gly_i.size > 0

    def _sub(idx):
        return (fi_mat[idx], wt_idx[idx], res_arr[idx],
                [wt_aas_a[i] for i in idx])

    # GLY native positions — all 20 mutant columns
    fi_g, wt_g, res_g, aa_g = (_sub(gly_i) if has_gly
                                else (None, None, None, None))

    # Non-GLY positions (all 20 columns)
    fi_ng, wt_ng, res_ng, aa_ng = _sub(non_gly_i)

    # X→G strip: non-GLY positions × G mutant only
    fi_xg = fi_ng[:, _GLY_IDX:_GLY_IDX + 1]                  # [n_non_gly, 1]
    wt_xg = np.full(len(non_gly_i), -1, dtype=int)            # WT never G here → no markers

    # For error heatmap: non-GLY × non-G (own scale, no grey filler)
    fi_ng_main = fi_ng[:, _NON_G_COLS]                        # [n_non_gly, 19]
    wt_ng_main = np.where(wt_ng > _GLY_IDX, wt_ng - 1, wt_ng)  # remap into 19-AA space

    # ── FI heatmap — single full panel ──────────────────────────────────────
    # Show the full fi_mat with real values everywhere.  GLY native positions
    # and the X→G mutant row are flagged by orange tick labels so the viewer
    # knows those regions may carry higher residual error.
    fi_norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)

    eff_n = (sum(e - s for s, e in crop_ranges) + len(crop_ranges) - 1
             if crop_ranges else len(wt_idx))
    hm_w = max(6, eff_n * 0.18)
    # Taller figure to accommodate staggered labels + citations above heatmap
    has_citations = bool(active_site_citations)
    hm_h = 7.0 if has_citations else 6.0

    import matplotlib as mpl
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"]  = ["Times New Roman", "FreeSerif",
                                   "Nimbus Roman No9 L", "DejaVu Serif"]

    fig, ax = plt.subplots(figsize=(hm_w, hm_h), layout="constrained")

    im = _draw_heatmap(ax, fi_mat, wt_idx, res_arr, wt_aas_a,
                       stem, cmap=CMAP_FI, norm=fi_norm,
                       warn_col_mask=None,
                       warn_row_indices={_GLY_IDX},
                       tick_step=1,
                       active_site_res=active_site_res,
                       active_site_label_names=active_site_label_names,
                       active_site_mutant_aa=active_site_mutant_aa,
                       active_site_colors=active_site_colors,
                       active_site_citations=active_site_citations,
                       crop_ranges=crop_ranges,
                       show_title=False,
                       color_yticks_by_property=True)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("Frustration Index", size=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    heatmap_base = out_dir / f"{stem}_{suffix}"
    _savefig(fig, heatmap_base, dpi=300)

    # Combine with 3D structure image using Pillow (pixel-level, no resampling blur)
    if struct_png is not None:
        sp = Path(struct_png)
        if sp.exists():
            try:
                heatmap_png = heatmap_base.with_suffix(".png")
                combined_png = out_dir / f"{stem}_{suffix}_combined.png"
                _combine_pngs_pillow(sp, heatmap_png, combined_png)
                print(f"  [fig] combined → {combined_png}", flush=True)
            except Exception as exc:
                print(f"  [warn] Pillow combine failed: {exc}", flush=True)

    # ── error heatmap ────────────────────────────────────────────────────────
    if ref_fi_mat is not None:
        diff     = np.where(~np.isnan(ref_fi_mat), fi_mat - ref_fi_mat, np.nan)
        abs_diff = np.abs(diff)
        if np.isnan(abs_diff).all():
            return

        ng_err = abs_diff[non_gly_i]                          # [n_non_gly, 20]
        g_err  = abs_diff[gly_i] if has_gly else None

        ng_err_main = ng_err[:, _NON_G_COLS]                  # [n_non_gly, 19]
        xg_err      = ng_err[:, _GLY_IDX]                     # [n_non_gly]

        # Global scale across all panels for direct comparison
        all_valid   = abs_diff[np.isfinite(abs_diff)]
        global_vmax = max(float(np.nanmax(all_valid)), 1e-3) if all_valid.size else 1e-3
        global_norm = Normalize(vmin=0, vmax=global_vmax)

        # Left panel: full [n_all, 21] matrix so GLY positions appear as grey+hatched
        # columns (directing the viewer to the right panel).
        #   axis-0 (x): all residue positions; GLY rows stay NaN → grey+hatch
        #   axis-1 (y):
        #     cols  0-18 = non-G mutant errors  (_NON_G_AA order)
        #     col  19    = 0  (white gap separator — no tick)
        #     col  20    = X→G errors  (labelled "G" in orange)
        combined_err = np.full((len(wt_idx), 21), np.nan)
        combined_err[non_gly_i, :19] = ng_err_main
        combined_err[non_gly_i, 19]  = 0        # gap — white patch covers it
        combined_err[non_gly_i, 20]  = xg_err
        # GLY rows stay NaN → rendered grey + hatched by hatch_nan_cols=True

        # Remap WT indices into combined-panel row space:
        #   original AA idx < GLY_IDX  →  same idx (0-4)
        #   original AA idx == GLY_IDX →  -1  (skip marker; whole col is grey)
        #   original AA idx > GLY_IDX  →  idx-1 (5-18)
        combined_wt = np.where(
            wt_idx == _GLY_IDX, -1,
            np.where(wt_idx > _GLY_IDX, wt_idx - 1, wt_idx)
        ).astype(int)

        combined_labels = _NON_G_AA + [""] + ["G"]

        has_gly_err = g_err is not None and not np.isnan(g_err).all()
        n_gly_pos   = gly_i.size if has_gly_err else 0

        w_left  = max(6,   len(wt_idx)   * 0.18)
        w_right = max(1.5, n_gly_pos     * 0.40)
        n_panels = 1 + (1 if has_gly_err else 0)

        fig = plt.figure(figsize=(w_left + w_right + 0.6, 3.5), layout="constrained")
        gs  = fig.add_gridspec(
            1, n_panels,
            width_ratios=[len(wt_idx)] + ([n_gly_pos] if has_gly_err else []))

        ax_left = fig.add_subplot(gs[0])
        im_left = _draw_heatmap(ax_left, combined_err, combined_wt, res_arr, wt_aas_a,
                                 f"{stem} — |Δ FI| (vmax={global_vmax:.2f})",
                                 cmap="viridis", norm=global_norm,
                                 aa_labels=combined_labels, gap_rows=[19],
                                 warn_row_indices={20}, hatch_nan_cols=True,
                                 tick_step=1,
                                 warn_col_mask=(wt_idx == _GLY_IDX))
        ax_left.spines["right"].set_visible(False)

        if has_gly_err:
            ax_right = fig.add_subplot(gs[1])
            _draw_heatmap(ax_right, np.nan_to_num(g_err), wt_g, res_g, aa_g,
                          "GLY positions", cmap="viridis", norm=global_norm,
                          tick_step=1,
                          warn_col_mask=np.ones(n_gly_pos, dtype=bool))
            ax_right.spines["right"].set_visible(False)
            plt.colorbar(im_left, ax=[ax_left, ax_right],
                         label="|Δ FI|", shrink=0.8, pad=0.01)
        else:
            plt.colorbar(im_left, ax=ax_left,
                         label="|Δ FI|", shrink=0.8, pad=0.01)

        _savefig(fig, out_dir / f"{stem}_{suffix}_error")

        # ── 2×2 overview figure ──────────────────────────────────────────────
        # ((FI heatmap, scatter), (error heatmap, GLY error))
        # Size the right column to hold a square scatter at ~40% of figure height.
        fig_h_ov       = 7.5
        right_col_in   = max(3.5, fig_h_ov * 0.40)
        right_w        = max(n_gly_pos + 2, int(right_col_in * len(wt_idx) / w_fi))
        w_ov_right     = right_col_in + 0.8  # extra for colorbar
        fig_ov         = plt.figure(figsize=(w_fi + w_ov_right + 1.0, fig_h_ov),
                                    layout="constrained")
        gs_ov          = fig_ov.add_gridspec(
            2, 2,
            width_ratios=[len(wt_idx), right_w],
            height_ratios=[1, 1])

        # Top-left: FI heatmap
        ax_fi = fig_ov.add_subplot(gs_ov[0, 0])
        im_fi = _draw_heatmap(ax_fi, fi_mat, wt_idx, res_arr, wt_aas_a,
                               stem, cmap=CMAP_FI, norm=fi_norm,
                               warn_col_mask=(wt_idx == _GLY_IDX),
                               warn_row_indices={_GLY_IDX},
                               tick_step=1)

        # Top-right: scatter (colored by frustration; GLY positions orange)
        ax_sc = fig_ov.add_subplot(gs_ov[0, 1])
        _draw_scatter(ax_sc, fi_mat, ref_fi_mat, wt_idx, fi_norm,
                      "PyTorch vs Original FI")
        fig_ov.colorbar(im_fi, ax=[ax_fi, ax_sc],
                        label="FI", shrink=0.8, pad=0.01)

        # Bottom-left: error heatmap
        ax_err = fig_ov.add_subplot(gs_ov[1, 0])
        im_err = _draw_heatmap(ax_err, combined_err, combined_wt, res_arr, wt_aas_a,
                                f"|Δ FI| (vmax={global_vmax:.2f})",
                                cmap="viridis", norm=global_norm,
                                aa_labels=combined_labels, gap_rows=[19],
                                warn_row_indices={20}, hatch_nan_cols=True,
                                tick_step=1,
                                warn_col_mask=(wt_idx == _GLY_IDX))
        ax_err.spines["right"].set_visible(False)

        # Bottom-right: GLY error panel — forced square to match scatter size
        if has_gly_err:
            ax_ge = fig_ov.add_subplot(gs_ov[1, 1])
            _draw_heatmap(ax_ge, np.nan_to_num(g_err), wt_g, res_g, aa_g,
                          "GLY positions", cmap="viridis", norm=global_norm,
                          tick_step=1,
                          warn_col_mask=np.ones(n_gly_pos, dtype=bool))
            ax_ge.set_box_aspect(1)   # same square footprint as the scatter above
            ax_ge.spines["right"].set_visible(False)
            fig_ov.colorbar(im_err, ax=[ax_err, ax_ge],
                            label="|Δ FI|", shrink=0.8, pad=0.01)
        else:
            fig_ov.colorbar(im_err, ax=ax_err,
                            label="|Δ FI|", shrink=0.8, pad=0.01)

        _savefig(fig_ov, out_dir / f"{stem}_{suffix}_overview")


def _make_bfactor_fullatom_pdb(cath_pdb: Path, full_pdb: Path, sr_df):
    """Build a temp full-atom PDB with B-factors set to frustration values.

    Matches CATH domain residues to full PDB residues by Cα coordinate proximity.
    Returns (bf_pdb_path, cath_to_full_res_map).
    """
    import tempfile
    import numpy as np

    fi_col = "FrstIndex" if "FrstIndex" in sr_df.columns else "TrueFrstIndex"
    cath_fi = {int(row["Res"]): max(-4.0, min(4.0, float(row[fi_col])))
               for _, row in sr_df.iterrows()}

    def _parse_cas(pdb_path):
        cas = {}
        for line in pdb_path.read_text().splitlines():
            if line.startswith("ATOM") and len(line) > 54 and line[12:16].strip() == "CA" and line[21] == "A":
                try:
                    cas[int(line[22:26])] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    pass
        return cas

    cath_cas = _parse_cas(cath_pdb)
    full_cas  = _parse_cas(full_pdb)

    # Match each full PDB residue to its nearest CATH domain residue by Cα distance
    cath_keys   = list(cath_cas.keys())
    cath_coords = np.array(list(cath_cas.values()))
    full_to_cath: dict[int, int] = {}
    for full_res, full_coord in full_cas.items():
        dists = np.linalg.norm(cath_coords - full_coord, axis=1)
        best  = int(np.argmin(dists))
        if dists[best] < 0.5:
            full_to_cath[full_res] = cath_keys[best]

    cath_to_full = {v: k for k, v in full_to_cath.items()}

    tmp    = Path(tempfile.mkdtemp())
    bf_pdb = tmp / "struct_bf.pdb"
    lines  = []
    for line in full_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) > 60:
            # Skip any chain that isn't chain A to keep only the CATH domain chain
            if len(line) > 21 and line[21] not in ("A", " "):
                continue
            try:
                full_res = int(line[22:26])
                cath_res = full_to_cath.get(full_res)
                fi = cath_fi.get(cath_res, 0.0) if cath_res is not None else 0.0
                tail = line[66:] if len(line) > 66 else ""
                lines.append(line[:60] + f"{fi:6.2f}" + tail)
            except (ValueError, IndexError):
                lines.append(line)
        else:
            lines.append(line)
    bf_pdb.write_text("\n".join(lines) + "\n")
    return bf_pdb, cath_to_full


def _combine_pngs_pillow(left_png: Path, right_png: Path, out_path: Path,
                         gap_px: int = 60, left_pad_px: int = 80):
    """Paste two PNGs side by side at full pixel resolution using Pillow."""
    from PIL import Image
    left  = Image.open(str(left_png)).convert("RGBA")
    right = Image.open(str(right_png)).convert("RGBA")
    # Scale left to match right height (LANCZOS gives sharp downsample)
    if left.height != right.height:
        scale = right.height / left.height
        left  = left.resize((int(left.width * scale), right.height), Image.LANCZOS)
    total_w = left_pad_px + left.width + gap_px + right.width
    combined = Image.new("RGBA", (total_w, right.height), (255, 255, 255, 255))
    combined.paste(left,  (left_pad_px, 0), mask=left)
    combined.paste(right, (left_pad_px + left.width + gap_px, 0), mask=right)
    combined.convert("RGB").save(str(out_path))


def _residue_screen_pos(bf_pdb: Path, pdb_r: int, out_dir: Path, stem: str,
                        pymol_exe=None):
    """Return (cx, cy_top, cy_center) pixel coords of a residue in the rendered view.

    Renders a second PyMOL image with only that residue shown in magenta on
    white, using the same orient/turn as the main render, then locates the
    magenta pixels.  Returns None if the render fails.
    """
    marker_pml = out_dir / f"{stem}_marker.pml"
    marker_png = out_dir / f"{stem}_marker.png"
    script = (
        f"load {bf_pdb}, obj\n"
        f"remove not polymer\n"
        f"hide everything, obj\n"
        f"select sel_site, obj and resi {pdb_r}\n"
        f"color magenta, sel_site\n"
        f"show sticks, sel_site\n"
        f"select sel_heavy, sel_site and not hydrogen\n"
        f"show spheres, sel_heavy\n"
        f"set sphere_scale, 0.35, sel_heavy\n"
        f"orient obj\n"
        f"turn y, 180\n"
        f"zoom obj\n"
        f"bg_color white\n"
        f"set ray_opaque_background, 1\n"
        f"set antialias, 2\n"
        f"ray 3600, 3600\n"
        f"png {marker_png.name}, dpi=300\n"
        f"quit\n"
    )
    marker_pml.write_text(script)
    try:
        _try_run_pymol(marker_pml, marker_png, pymol_exe)
        if not marker_png.exists():
            return None
        from PIL import Image
        import numpy as np
        arr = np.array(Image.open(str(marker_png)).convert("RGB"))
        # magenta: R high, G low, B high
        is_site = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] > 150)
        if not is_site.any():
            return None
        rows, cols = np.where(is_site)
        return int(cols.mean()), int(rows.min()), int(rows.mean())
    except Exception as exc:
        print(f"  [warn] marker render failed: {exc}", flush=True)
        return None
    finally:
        for p in (marker_pml, marker_png):
            try:
                p.unlink()
            except Exception:
                pass


def _add_site_label_to_png(png_path: Path, label: str, color: str = "#CC00CC",
                            residue_pos=None, force_below: bool = False):
    """Overlay an active-site label on a rendered protein PNG using Pillow.

    residue_pos: (cx, cy_top, cy_center) pixel coordinates of the residue,
    obtained from _residue_screen_pos.  Falls back to protein-top if None.
    """
    from PIL import Image, ImageDraw
    import numpy as np

    img = Image.open(str(png_path)).convert("RGBA")
    arr = np.array(img)
    w, h = img.size

    if residue_pos is not None:
        cx, cy_top, cy_center = residue_pos
    else:
        is_bg   = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)
        protein = ~is_bg
        if protein.any():
            rows, cols = np.where(protein)
            cx     = int(cols.mean())
            cy_top = int(rows.min())
        else:
            cx, cy_top = w // 2, h // 6
        cy_center = cy_top

    draw      = ImageDraw.Draw(img)
    font_size = max(80, w // 18)
    font      = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            from PIL import ImageFont
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            pass
    if font is None:
        try:
            from PIL import ImageFont
            font = ImageFont.load_default(size=font_size)
        except Exception:
            from PIL import ImageFont
            font = ImageFont.load_default()

    # Estimate text width for x-centering
    est_bbox = draw.textbbox((0, 0), label, font=font)
    tw_est   = est_bbox[2] - est_bbox[0]
    pad      = 24
    aw       = 32   # arrowhead half-width

    # Horizontal center on residue
    lx = max(pad, min(w - tw_est - 3 * pad, cx - tw_est // 2))

    # Place label above residue if space allows, else below
    above_residue = (cy_top > 3 * pad + 120) and not force_below
    if above_residue:
        ly_text = cy_top - 2 * pad - 120   # text anchor y
    else:
        ly_text = cy_center + 140

    # Get ACTUAL text bounds at the drawing position (accounts for font metrics)
    actual = draw.textbbox((lx, ly_text), label, font=font)
    bx0, by0, bx1, by1 = (actual[0] - pad, actual[1] - pad,
                           actual[2] + pad, actual[3] + pad)

    draw.rectangle([bx0, by0, bx1, by1], fill="white", outline=color, width=12)
    draw.text((lx, ly_text), label, font=font, fill=color)


    img.convert("RGB").save(str(png_path))


_PYMOL_CANDIDATES = [
    "pymol",
    "/gpfs/projects/bsc08/bsc504628/envs/pymol/bin/pymol",
]


def _try_run_pymol(pml_path: Path, out_png: Path, pymol_exe: str | None = None) -> bool:
    """Try to render a .pml script headlessly. Returns True if out_png was created."""
    import subprocess, shutil
    candidates = ([pymol_exe] if pymol_exe else []) + _PYMOL_CANDIDATES
    for exe in candidates:
        if not exe:
            continue
        if not shutil.which(exe) and not Path(exe).exists():
            continue
        try:
            subprocess.run(
                [exe, "-cq", pml_path.name],
                cwd=str(pml_path.parent),
                timeout=180,
                capture_output=True,
            )
            if out_png.exists():
                print(f"  [PyMOL] rendered → {out_png}", flush=True)
                return True
        except Exception as exc:
            print(f"  [PyMOL] {exe} failed: {exc}", flush=True)
    print("  [PyMOL] not available — run the .pml script manually", flush=True)
    return False


def _write_pymol_singleresidue_script(
    pdb_path: Path,
    sr_df,
    out_dir: Path,
    stem: str,
    active_site_res=None,
    active_site_label_names=None,
    active_site_mutant_aa=None,
    active_site_colors=None,
    bf_pdb: Path | None = None,
    cath_to_full: dict | None = None,
) -> Path:
    """Write a standalone .pml script colouring the structure by single-residue FI.

    If bf_pdb is given (full-atom PDB with B-factors preset), it is loaded directly
    and the spectrum command uses B-factor — no alter commands needed.
    cath_to_full maps CATH domain residue numbers to the full PDB residue numbers.
    """
    label_names  = active_site_label_names or {}
    mutant_aas   = active_site_mutant_aa or {}
    site_colors  = active_site_colors or {}
    cath_to_full = cath_to_full or {}

    def _pdb_res(cath_res: int) -> int:
        return cath_to_full.get(cath_res, cath_res)

    if bf_pdb is not None:
        load_block     = f"load {bf_pdb}, obj\n"
        spectrum_block = (
            "set_color frust_max,  [0.80, 0.00, 0.00]\n"
            "set_color frust_zero, [0.60, 0.60, 0.60]\n"
            "set_color frust_min,  [0.00, 0.70, 0.00]\n"
            "spectrum b, frust_max frust_zero frust_min, obj, minimum=-4, maximum=4\n"
        )
    else:
        pdb_rel    = Path("..") / "FrustrationData" / pdb_path.name
        load_block = f"load {pdb_rel}, obj\n"
        fi_col     = "FrstIndex" if "FrstIndex" in sr_df.columns else "TrueFrstIndex"
        alters     = [f"alter (obj and resi {int(r['Res'])}), b={max(-4., min(4., float(r[fi_col]))):.4f}"
                      for _, r in sr_df.iterrows()]
        spectrum_block = (
            "alter obj, b=0.0\n"
            + "\n".join(alters) + "\n"
            "rebuild\n"
            "set_color frust_max,  [0.80, 0.00, 0.00]\n"
            "set_color frust_zero, [0.60, 0.60, 0.60]\n"
            "set_color frust_min,  [0.00, 0.70, 0.00]\n"
            "spectrum b, frust_max frust_zero frust_min, obj, minimum=-4, maximum=4\n"
        )

    site_block = ""
    if active_site_res:
        pdb_resi = [_pdb_res(r) for r in active_site_res]
        per_site = []
        for idx, (cath_r, pdb_r) in enumerate(zip(active_site_res, pdb_resi)):
            lbl = label_names.get(int(cath_r), f"S{cath_r}")
            if len(lbl) > 1 and lbl[0].isalpha() and lbl[1:].isdigit():
                lbl = f"{_AA_3LETTER.get(lbl[0].upper(), lbl[0])}{lbl[1:]}"
            color_hex = site_colors.get(int(cath_r), "magenta")
            # Convert hex (#RRGGBB) to PyMOL [r,g,b] 0-1 floats if needed
            if color_hex.startswith("#") and len(color_hex) == 7:
                r = int(color_hex[1:3], 16) / 255
                g = int(color_hex[3:5], 16) / 255
                b = int(color_hex[5:7], 16) / 255
                color_cmd = (f"set_color site_col_{idx}, [{r:.3f},{g:.3f},{b:.3f}]\n"
                             f"color site_col_{idx}, sel_site_{idx}\n")
            else:
                color_cmd = f"color {color_hex}, sel_site_{idx}\n"
            per_site.append(
                f"select sel_site_{idx}, obj and resi {pdb_r}\n"
                f"show sticks, sel_site_{idx} and not (name n or name c or name o)\n"
                # f"show sticks, sel_site_{idx}\n"
                # f"set stick_radius, 0.30, sel_site_{idx}\n"
                # f"select sel_site_{idx}_heavy, sel_site_{idx} and not hydrogen\n"
                # f"show spheres, sel_site_{idx}_heavy\n"
                # f"set sphere_scale, 0.45, sel_site_{idx}_heavy\n"
                + color_cmd
            )
        site_block = (
            "".join(per_site) +
            "orient obj\n"
            "turn y, 180\n"
            "zoom obj, complete=1\n"
        )
    else:
        site_block = "orient obj\nturn y, 180\nzoom obj\n"

    out_png = f"{stem}_singleresidue.png"
    script  = (
        f"# Run from the Visualizations directory:\n"
        f"#   pymol -cq {stem}_singleresidue.pml\n"
        f"{load_block}"
        f"remove not polymer\n"
        f"show cartoon, obj\n"
        f"hide lines, obj\n"
        f"hide nonbonded, obj\n"
        f"{spectrum_block}"
        f"{site_block}"
        f"bg_color white\n"
        f"set ray_opaque_background, 1\n"
        f"set antialias, 2\n"
        f"ray 3600, 3600\n"
        f"png {out_png}, dpi=300\n"
        f"quit\n"
    )

    pml_path = out_dir / f"{stem}_singleresidue.pml"
    pml_path.write_text(script)
    print(f"  [PyMOL] script → {pml_path}", flush=True)
    return pml_path


def draw_scanning_from_file(pred_path, out_dir=None, ref_dir=None, active_site_res=None,
                            active_site_label_names=None, active_site_mutant_aa=None,
                            active_site_colors=None, active_site_citations=None,
                            active_site_label_below=None,
                            crop_first_n=None, crop_last_n=None,
                            crop_ranges=None):
    """Generate a DMS scanning heatmap from a saved *_singleresidue_mutations file.

    The file should be space-separated with columns:
    Res, ChainRes, AA_native, AA_mutant, NativeEnergy, MutantEnergy, FrstIndex
    """
    pred_path = Path(pred_path)
    out_dir   = Path(out_dir) if out_dir else pred_path.parent

    df = pd.read_csv(pred_path, sep=r"\s+")
    if not {"Res", "AA_native", "AA_mutant", "FrstIndex"}.issubset(df.columns):
        raise ValueError(f"Missing expected columns in {pred_path}")

    df["Res"]       = pd.to_numeric(df["Res"],       errors="coerce")
    df["FrstIndex"] = pd.to_numeric(df["FrstIndex"], errors="coerce")
    df = df.dropna(subset=["Res", "FrstIndex"])

    res_order  = sorted(df["Res"].unique())
    n_res      = len(res_order)
    res_to_pos = {r: i for i, r in enumerate(res_order)}

    fi_mat        = np.full((n_res, 20), np.nan, dtype=np.float32)
    wt_aa_indices = np.full(n_res, -1,   dtype=int)
    wt_aas        = ["?"] * n_res

    for _, row in df.iterrows():
        pos     = res_to_pos[row["Res"]]
        mut_idx = _AA_TO_IDX.get(str(row["AA_mutant"]))
        if mut_idx is not None:
            fi_mat[pos, mut_idx] = float(row["FrstIndex"])
        wt = str(row["AA_native"])
        if wt in _AA_TO_IDX:
            wt_aa_indices[pos] = _AA_TO_IDX[wt]
            wt_aas[pos]        = wt

    stem = pred_path.name.split(".")[0]

    # Write PyMOL script and try to render the 3D structure
    resolved_out = Path(out_dir) if out_dir else pred_path.parent
    sr_file   = pred_path.parent / f"{stem}.pdb_singleresidue"
    pdb_file  = pred_path.parent / f"{stem}.pdb"
    struct_png = None
    if sr_file.exists() and pdb_file.exists():
        try:
            sr_df = pd.read_csv(sr_file, sep=r"\s+", comment="#")

            # Prefer a full-atom PDB (has complete side chains) if available
            full_pdb_cand = pred_path.parent / f"{stem.split('A')[0]}_fullatom.pdb"
            if not full_pdb_cand.exists():
                full_pdb_cand = pred_path.parent / f"{stem}_fullatom.pdb"

            bf_pdb      = None
            cath_to_full = None
            if full_pdb_cand.exists():
                try:
                    bf_pdb, cath_to_full = _make_bfactor_fullatom_pdb(
                        pdb_file, full_pdb_cand, sr_df)
                    print(f"  [PyMOL] using full-atom PDB: {full_pdb_cand}", flush=True)
                except Exception as exc:
                    print(f"  [warn] full-atom bfactor PDB failed: {exc}", flush=True)
                    bf_pdb = None

            pml_path = _write_pymol_singleresidue_script(
                pdb_file, sr_df, resolved_out,
                stem, active_site_res=active_site_res,
                active_site_label_names=active_site_label_names,
                active_site_mutant_aa=active_site_mutant_aa,
                active_site_colors=active_site_colors,
                bf_pdb=bf_pdb, cath_to_full=cath_to_full,
            )
            cand_png = resolved_out / f"{stem}_singleresidue.png"
            _try_run_pymol(pml_path, cand_png)
            if cand_png.exists():
                # Add label via Pillow — PyMOL ray tracer ignores label_depth_test
                if active_site_res:
                    _label_names   = active_site_label_names or {}
                    _site_colors   = active_site_colors or {}
                    _label_below   = set(active_site_label_below or [])
                    for cath_r in active_site_res:
                        lbl = _label_names.get(int(cath_r), f"S{cath_r}")
                        if len(lbl) > 1 and lbl[0].isalpha() and lbl[1:].isdigit():
                            lbl = f"{_AA_3LETTER.get(lbl[0].upper(), lbl[0])}{lbl[1:]}"
                        color_hex   = _site_colors.get(int(cath_r), "#CC00CC")
                        below_flag  = int(cath_r) in _label_below
                        # Find exact screen position via a second marker render
                        res_pos = None
                        if bf_pdb is not None and cath_to_full is not None:
                            pdb_r = cath_to_full.get(int(cath_r), int(cath_r))
                            res_pos = _residue_screen_pos(
                                bf_pdb, pdb_r, resolved_out, stem)
                        try:
                            _add_site_label_to_png(cand_png, lbl,
                                                   color=color_hex,
                                                   residue_pos=res_pos,
                                                   force_below=below_flag)
                        except Exception as exc:
                            print(f"  [warn] Pillow label failed: {exc}", flush=True)
                struct_png = cand_png
        except Exception as exc:
            print(f"  [warn] PyMOL step failed for {stem}: {exc}")

    ref_fi_mat = None
    if ref_dir is not None:
        mut_dir = find_ref_mutations_dir(stem, Path(ref_dir))
        if mut_dir is not None:
            try:
                ref_df     = read_mutations_data(mut_dir)
                ref_fi_mat = np.full((n_res, 20), np.nan, dtype=np.float32)
                for _, row in ref_df.iterrows():
                    r = int(row["Res"])
                    if r in res_to_pos:
                        idx = _AA_TO_IDX.get(str(row["AA"]))
                        if idx is not None:
                            ref_fi_mat[res_to_pos[r], idx] = float(row["FrstIndex"])
            except Exception as exc:
                print(f"  [warn] could not load reference for {stem}: {exc}")

    if crop_ranges is None:
        if crop_first_n is not None or crop_last_n is not None:
            first = min(crop_first_n or n_res, n_res)
            if crop_last_n and crop_last_n > 0 and first < n_res:
                crop_ranges = [(0, first), (max(first, n_res - crop_last_n), n_res)]
            else:
                crop_ranges = [(0, first)]

    plot_scanning_heatmap(fi_mat, wt_aa_indices, np.array(res_order), wt_aas,
                           stem, resolved_out, ref_fi_mat=ref_fi_mat,
                           active_site_res=active_site_res,
                           active_site_label_names=active_site_label_names,
                           active_site_mutant_aa=active_site_mutant_aa,
                           active_site_colors=active_site_colors,
                           active_site_citations=active_site_citations,
                           crop_ranges=crop_ranges,
                           struct_png=struct_png)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_mutations_data(mutations_dir: Path) -> pd.DataFrame | None:
    files = sorted(mutations_dir.glob("singleresidue_Res*_threading_*.txt"))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, sep=r"\s+"))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


def find_ref_mutations_dir(stem: str, ref_dir: Path) -> Path | None:
    cand = ref_dir / f"{stem}.done" / "MutationsData"
    if cand.is_dir():
        return cand
    for done in ref_dir.glob(f"{stem}_*.done"):
        cand = done / "MutationsData"
        if cand.is_dir():
            return cand
    return None


def load_pred(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=r"\s+")
        if not {"Res", "ChainRes", "AA_mutant", "FrstIndex"}.issubset(df.columns):
            return None
        df["Res"]       = pd.to_numeric(df["Res"],       errors="coerce")
        df["FrstIndex"] = pd.to_numeric(df["FrstIndex"], errors="coerce")
        return df.dropna(subset=["Res", "FrstIndex"])
    except Exception:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_dir", required=True,
                        help="Directory with *_singleresidue_mutations files")
    parser.add_argument("--ref_dir",  default=None,
                        help="Root dir with <stem>.done/MutationsData/ sub-trees (optional)")
    parser.add_argument("--out",      default="plots", help="Output directory")
    parser.add_argument("--scanning", action="store_true",
                        help="Draw DMS scanning heatmaps instead of comparison plots")
    parser.add_argument("--active-site", dest="active_site", nargs="+", type=int,
                        default=None, metavar="RES",
                        help="Residue numbers to highlight as active site in scanning "
                             "heatmaps (e.g. --active-site 38). Only used with --scanning.")
    parser.add_argument("--active-site-label", dest="active_site_labels", nargs="+",
                        default=None, metavar="RES=LABEL",
                        help="Override display label for an active-site residue "
                             "(e.g. --active-site-label 38=S70). Repeat for multiple.")
    parser.add_argument("--crop-first", dest="crop_first", type=int, default=None,
                        metavar="N",
                        help="Show only the first N residue columns in scanning heatmaps.")
    parser.add_argument("--crop-last", dest="crop_last", type=int, default=None,
                        metavar="N",
                        help="Append the last N residue columns after a '…' separator.")
    args = parser.parse_args()

    label_names = {}
    if args.active_site_labels:
        for item in args.active_site_labels:
            res_str, _, lbl = item.partition("=")
            try:
                label_names[int(res_str)] = lbl
            except ValueError:
                pass

    pred_dir = Path(args.pred_dir)
    ref_dir  = Path(args.ref_dir) if args.ref_dir else None
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = (sorted(pred_dir.glob("*.frustratometer3.singleresidue_mutations")) or
                  sorted(pred_dir.glob("*_singleresidue_mutations")))
    if not pred_files:
        print(f"No *_singleresidue_mutations files in {pred_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(pred_files)} prediction files")

    if args.scanning:
        for pf in pred_files:
            try:
                draw_scanning_from_file(pf, out_dir=out_dir, ref_dir=ref_dir,
                                        active_site_res=args.active_site,
                                        active_site_label_names=label_names or None,
                                        crop_first_n=args.crop_first,
                                        crop_last_n=args.crop_last)
            except Exception as exc:
                print(f"  [skip] {pf.name}: {exc}")
        print("Done.")
        return

    per_protein, all_pred, all_true = [], [], []

    for pf in pred_files:
        stem    = pf.name.split(".frustratometer3")[0]
        pred_df = load_pred(pf)
        if pred_df is None:
            print(f"  [skip] could not load {pf.name}")
            continue

        mut_dir = find_ref_mutations_dir(stem, ref_dir)
        if mut_dir is None:
            print(f"  [skip] no MutationsData for {stem}")
            continue

        ref_df = read_mutations_data(mut_dir)
        if ref_df is None or ref_df.empty:
            print(f"  [skip] empty MutationsData for {stem}")
            continue

        ref_df = ref_df.rename(columns={"AA": "AA_mutant", "FrstIndex": "FrstIndex_ref"})
        ref_df["Res"]           = pd.to_numeric(ref_df["Res"],           errors="coerce")
        ref_df["FrstIndex_ref"] = pd.to_numeric(ref_df["FrstIndex_ref"], errors="coerce")

        merged = pred_df.merge(
            ref_df[["Res", "ChainRes", "AA_mutant", "FrstIndex_ref"]],
            on=["Res", "ChainRes", "AA_mutant"], how="inner",
        )
        if merged.empty:
            print(f"  [skip] no overlap after merge for {stem}")
            continue

        p = merged["FrstIndex"].to_numpy(dtype=float)
        t = merged["FrstIndex_ref"].to_numpy(dtype=float)
        m = compute_metrics(p, t)
        per_protein.append({"stem": stem, **m})
        all_pred.extend(p.tolist())
        all_true.extend(t.tolist())
        print(f"  {stem:20s}  n={m['n']:5d}  r={m['pearson_r']:.4f}"
              f"  acc={m['class_accuracy']:.4f}  cos={m['cosine_similarity']:.4f}")

    if not per_protein:
        print("No data — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    df_metrics = pd.DataFrame(per_protein)
    df_metrics.to_csv(out_dir / "mutations_metrics.csv", index=False)
    print(f"\nWrote {out_dir}/mutations_metrics.csv  ({len(df_metrics)} proteins)")

    all_pred = np.asarray(all_pred, dtype=float)
    all_true = np.asarray(all_true, dtype=float)
    overall  = compute_metrics(all_pred, all_true)
    print(f"\nOverall: r={overall['pearson_r']:.4f}  acc={overall['class_accuracy']:.4f}"
          f"  cos={overall['cosine_similarity']:.4f}  n={overall['n']:,}")

    r_vals   = df_metrics["pearson_r"].to_numpy(dtype=float)
    acc_vals = df_metrics["class_accuracy"].to_numpy(dtype=float)
    cos_vals = df_metrics["cosine_similarity"].to_numpy(dtype=float)
    n_vals   = df_metrics["n"].to_numpy(dtype=float)

    # ── individual plots ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_scatter_hexbin(ax, all_pred, all_true,
                        title=f"mutations  (n={len(all_pred):,})", cmap="Purples",
                        xlabel=_XLABEL, ylabel=_YLABEL)
    fig.tight_layout()
    _savefig(fig, out_dir / "scatter_mutations")
    print(f"  wrote {out_dir}/scatter_mutations.*")

    for vals, xlabel, fname, vline in [
        (r_vals,   "Pearson R",         "r_kde_mutations",       0.99),
        (acc_vals, "Class accuracy",    "acc_kde_mutations",     0.9),
        (cos_vals, "Cosine similarity", "cosine_kde_mutations",  None),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        draw_kde(ax, vals, xlabel, COLOR, vline=vline)
        ax.set_title(f"mutations  –  {xlabel} per protein")
        fig.tight_layout()
        _savefig(fig, out_dir / fname)
        print(f"  wrote {out_dir}/{fname}.*")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ok = np.isfinite(all_pred) & np.isfinite(all_true)
    draw_confusion(ax, all_pred[ok], all_true[ok],
                   lo=LO_SR, hi=HI_SR, cmap="Purples", cbar=True,
                   xlabel=_YLABEL, ylabel=_XLABEL)
    ax.set_title("mutations  –  confusion matrix")
    fig.tight_layout()
    _savefig(fig, out_dir / "confusion_mutations")
    print(f"  wrote {out_dir}/confusion_mutations.*")

    ok_n = np.isfinite(r_vals) & np.isfinite(n_vals)
    if ok_n.sum() > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(n_vals[ok_n], r_vals[ok_n], alpha=0.4, s=10, color=COLOR, linewidths=0)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("n mutations per protein")
        ax.set_ylabel("Pearson R")
        ax.set_title("mutations  –  R vs n mutations")
        fig.tight_layout()
        _savefig(fig, out_dir / "r_vs_n_mutations")
        print(f"  wrote {out_dir}/r_vs_n_mutations.*")

    # ── combined 4×1 summary ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    draw_scatter_hexbin(axes[0], all_pred, all_true,
                        title=f"mutations  (n={len(all_pred):,})", cmap="Purples",
                        xlabel=_XLABEL, ylabel=_YLABEL)
    draw_kde(axes[1], r_vals,   "Pearson R",        COLOR, vline=0.99)
    axes[1].set_title(f"R per protein  (med={np.nanmedian(r_vals):.3f})", fontsize=9)
    draw_kde(axes[2], acc_vals, "Class accuracy",   COLOR, vline=0.9)
    axes[2].set_title(f"Class acc  (med={np.nanmedian(acc_vals):.3f})", fontsize=9)
    draw_kde(axes[3], cos_vals, "Cosine similarity", COLOR)
    axes[3].set_title(f"Cosine sim  (med={np.nanmedian(cos_vals):.3f})", fontsize=9)
    fig.suptitle("Frustratometer3 baseline  –  mutations",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir / "summary_mutations")
    print(f"  wrote {out_dir}/summary_mutations.*")
    print(f"\nDone. Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
