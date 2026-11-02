"""Shared constants, colour scheme, and metric helpers for all frustration plots."""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from ..utils.weights import classify_frustration


# ── frustration thresholds ────────────────────────────────────────────────────
MODE_THRESHOLDS = {
    "singleresidue":   (-1.0, 0.55),
    "mutations":       (-1.0, 0.55),
    "mutational":      (-1.0, 0.78),
    "configurational": (-1.0, 0.78),
}
LO_SR,   HI_SR   = -1.0, 0.55
LO_PAIR, HI_PAIR = -1.0, 0.78

# ── colour scheme (matches original frustratometer) ───────────────────────────
CLASS_COLORS = {
    0: [0.84, 0.10, 0.11],   # highly frustrated
    1: [0.73, 0.73, 0.73],   # neutral
    2: [0.00, 0.62, 0.45],   # minimally frustrated
}

CMAP_FI = LinearSegmentedColormap.from_list(
    "frustration",
    [(0.0, [0.84, 0.0, 0.0]),
     (0.5, [0.73, 0.73, 0.73]),
     (1.0, [0.0, 0.70, 0.0])],
)
NORM_FI = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)

PALETTE = {
    "singleresidue":   "#1E88E5", #"#2196F3",
    "mutations":       "#9C27B0",
    "mutational":      "#8E44AD", # "#4CAF50",
    "configurational": "#FF4081", #"#FF5722",
}


# ── metric helpers ────────────────────────────────────────────────────────────

def fi_class(fi_arr: np.ndarray, lo: float = LO_PAIR, hi: float = HI_PAIR) -> np.ndarray:
    """0=highly (fi<=lo), 1=neutral, 2=minimally (fi>hi). Matches classify_frustration."""
    thr = torch.tensor([lo, hi])
    t = torch.from_numpy(np.asarray(fi_arr, dtype=np.float32))
    return classify_frustration(t, thr).numpy()


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[ok], true[ok]
    n = int(ok.sum())
    if n < 2:
        return {"pearson_r": float("nan"), "r2": float("nan"),
                "class_accuracy": float("nan"),
                "cosine_similarity": float("nan"), "n": n}
    if p.std() == 0 or t.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(p, t)[0, 1])
    acc = float((fi_class(p) == fi_class(t)).mean())
    pt  = torch.tensor(p, dtype=torch.float64)
    tt  = torch.tensor(t, dtype=torch.float64)
    cos = float(F.cosine_similarity(pt.unsqueeze(0), tt.unsqueeze(0)).item())
    cos = float(np.clip(cos, -1.0, 1.0))
    return {"pearson_r": r, "r2": r ** 2, "class_accuracy": acc,
            "cosine_similarity": cos, "n": n}


# ── reusable axes helpers ─────────────────────────────────────────────────────

def draw_kde(ax, values: np.ndarray, xlabel: str, color: str,
             vline: float | None = None, vline_label: str | None = None):
    import seaborn as sns
    v = values[np.isfinite(values)]
    if len(v) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    sns.kdeplot(v, ax=ax, color=color, fill=True, alpha=0.3, linewidth=2)
    med = float(np.nanmedian(v))
    ax.axvline(med, color="k", ls="--", lw=1, label=f"med={med:.3f}")
    if vline is not None:
        ax.axvline(vline, color="red", ls=":", lw=1, label=vline_label or f"thr={vline}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)


def draw_confusion(ax, pred: np.ndarray, true: np.ndarray,
                   lo: float = LO_PAIR, hi: float = HI_PAIR,
                   cmap: str = "Blues", cbar: bool = False,
                   xlabel: str = "Predicted", ylabel: str = "True"):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    labels = ["highly", "neutral", "minimally"]
    ok = np.isfinite(pred) & np.isfinite(true)
    p_cls = fi_class(pred[ok], lo=lo, hi=hi)
    t_cls = fi_class(true[ok], lo=lo, hi=hi)
    cm = confusion_matrix(t_cls, p_cls, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=False, cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, ax=ax, square=True,
                cbar=cbar, cbar_kws={"label": "fraction"} if cbar else {})
    for i in range(3):
        for j in range(3):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i,j]:,}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center", fontsize=6,
                    color="white" if cm_norm[i, j] > 0.6 else "black")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def draw_scatter_hexbin(ax, pred: np.ndarray, true: np.ndarray,
                        title: str, cmap: str = "Blues",
                        xlabel: str = r"$\mathrm{FI}_{\mathrm{true}}$",
                        ylabel: str = r"$\mathrm{FI}_{\mathrm{pred}}$",
                        line_color: str = "crimson",
                        extra_line: str | None = None,
                        gly_mask: np.ndarray | None = None,
                        lo: float | None = None,
                        hi: float | None = None,
                        lo_y: float | None = None,
                        hi_y: float | None = None,
                        metric_pred: np.ndarray | None = None,
                        metric_true: np.ndarray | None = None,
                        true_n: int | None = None,
                        reversed_scale: bool = False,
                        reversed_scale_y: bool = False,
                        ref_slope: float | None = 1.0,
                        legend_loc: str = "lower right",
                        legend_alpha: float = 0.92,
                        stats_lines: list[str] | None = None):
    """Scatter of pred vs true FI with fit line, class thresholds and an R² stats box.

    stats_lines: lines shown under R² in the stats box. When given, they replace
    the default "n = …" line and extra_line.
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from scipy.stats import linregress
    ok = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[ok], true[ok]
    if len(p) < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    slope, intercept, r, *_ = linregress(t, p)
    # Lines always span the full axis range so they reach the corners of the plot
    xlin = np.array([-4.0, 4.0])
    from matplotlib.colors import Colormap
    if isinstance(cmap, Colormap):
        pt_color = cmap(1.0)
    elif isinstance(cmap, str) and cmap.startswith("#"):
        pt_color = cmap
    else:
        pt_color = "#2196F3"

    # lo_y/hi_y default to lo/hi (separate y-axis thresholds for mixed-scale plots)
    _lo_y = lo_y if lo_y is not None else lo
    _hi_y = hi_y if hi_y is not None else hi

    # Neutral zone: gray band along both axes drawn first (lowest zorder)
    if lo is not None and hi is not None:
        ax.axvspan(lo, hi, alpha=0.12, color="gray", zorder=0)
    if _lo_y is not None and _hi_y is not None:
        ax.axhspan(_lo_y, _hi_y, alpha=0.12, color="gray", zorder=0)

    ax.scatter(t, p, s=1, alpha=0.15, linewidths=0,
               color=pt_color, rasterized=True, zorder=2)
    ax.plot(xlin, slope * xlin + intercept, "-", color=line_color, lw=2.0, zorder=4)
    if ref_slope is not None:
        ax.plot(xlin, ref_slope * xlin, "k--", lw=1.2, alpha=0.6, zorder=3)

    # Threshold lines and legend entries (all rectangle patches for consistency)
    # reversed_scale (smooth mode): low FI = minimally frustrated (green), high FI = highly frustrated (red)
    # When lo_y/hi_y differ from lo/hi (mixed-scale panels), x-axis lines use lighter colors
    _has_separate_y = (_lo_y is not None and _lo_y != lo) or (_hi_y is not None and _hi_y != hi)

    def _lighten(rgb, factor=0.45):
        """Lighten an RGB list toward white by factor."""
        return [c + (1.0 - c) * factor for c in rgb]

    _legend_handles = []
    # ── lo threshold ─────────────────────────────────────────────────────────
    if lo is not None:
        _c_full = CLASS_COLORS[2] if reversed_scale else CLASS_COLORS[0]
        _c_x = _lighten(_c_full) if _has_separate_y else _c_full
        _label_lo = rf"Minimally ($\mathrm{{FI}} \leq {lo}$)" if reversed_scale else rf"Highly ($\mathrm{{FI}} \leq {lo}$)"
        if _has_separate_y:
            _label_lo += " (mut.)"
        ax.axvline(lo, color=_c_x, lw=2.0, ls="--", alpha=1.0, zorder=1)
        _legend_handles.append(mpatches.Patch(facecolor=_c_x, alpha=0.85, label=_label_lo))
    if _lo_y is not None:
        _c_full = CLASS_COLORS[2] if reversed_scale_y else CLASS_COLORS[0]
        ax.axhline(_lo_y, color=_c_full, lw=2.0, ls="--", alpha=1.0, zorder=1)
        if _has_separate_y:
            _label_lo_y = rf"Minimally ($\mathrm{{FI}} \leq {_lo_y}$)" if reversed_scale_y else rf"Highly ($\mathrm{{FI}} \leq {_lo_y}$)"
            _legend_handles.append(mpatches.Patch(facecolor=_c_full, alpha=0.85, label=_label_lo_y + " (smooth)"))
    # ── neutral band legend ──────────────────────────────────────────────────
    if lo is not None and hi is not None:
        _suffix = " (mut.)" if _has_separate_y else ""
        _legend_handles.append(
            mpatches.Patch(color="gray", alpha=0.3,
                           label=rf"Neutral (${lo} < \mathrm{{FI}} \leq {hi}$){_suffix}"))
    if _has_separate_y and _lo_y is not None and _hi_y is not None:
        _legend_handles.append(
            mpatches.Patch(color="gray", alpha=0.15,
                           label=rf"Neutral (${_lo_y} < \mathrm{{FI}} \leq {_hi_y}$) (smooth)"))
    # ── hi threshold ─────────────────────────────────────────────────────────
    if hi is not None:
        _c_full = CLASS_COLORS[0] if reversed_scale else CLASS_COLORS[2]
        _c_x = _lighten(_c_full) if _has_separate_y else _c_full
        _label_hi = rf"Highly ($\mathrm{{FI}} > {hi}$)" if reversed_scale else rf"Minimally ($\mathrm{{FI}} > {hi}$)"
        if _has_separate_y:
            _label_hi += " (mut.)"
        ax.axvline(hi, color=_c_x, lw=2.0, ls="--", alpha=1.0, zorder=1)
        _legend_handles.append(mpatches.Patch(facecolor=_c_x, alpha=0.85, label=_label_hi))
    if _hi_y is not None:
        _c_full = CLASS_COLORS[0] if reversed_scale_y else CLASS_COLORS[2]
        ax.axhline(_hi_y, color=_c_full, lw=2.0, ls="--", alpha=1.0, zorder=1)
        if _has_separate_y:
            _label_hi_y = rf"Highly ($\mathrm{{FI}} > {_hi_y}$)" if reversed_scale_y else rf"Minimally ($\mathrm{{FI}} > {_hi_y}$)"
            _legend_handles.append(mpatches.Patch(facecolor=_c_full, alpha=0.85, label=_label_hi_y + " (smooth)"))

    if _legend_handles:
        leg = ax.legend(handles=_legend_handles, fontsize=7, loc=legend_loc,
                  framealpha=legend_alpha, edgecolor="#aaa", markerscale=1.2,
                  handlelength=2.0)
        leg.get_frame().set_zorder(100)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=10)
    # Use full dataset for metrics if provided, otherwise fall back to plot data
    if metric_pred is not None and metric_true is not None:
        mok = np.isfinite(metric_pred) & np.isfinite(metric_true)
        mp, mt = metric_pred[mok], metric_true[mok]
    else:
        mp, mt = p, t
    from scipy.stats import linregress as _lr
    _, _, mr, *_ = _lr(mt, mp)
    cos = float(np.dot(mp, mt) / (np.linalg.norm(mp) * np.linalg.norm(mt) + 1e-12))
    n = len(mp)
    display_n = true_n if true_n is not None else n
    r2 = mr ** 2
    r2_str = f"{r2:.8f}" if r2 >= 0.9999 else f"{r2:.4f}"
    if stats_lines is None:
        stats_lines = [f"n = {display_n:,}"] + ([extra_line] if extra_line else [])
    label = "\n".join([f"$\\mathbf{{R^2}}$ = {r2_str}", *stats_lines])
    ax.text(0.04, 0.97, label, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.92))


def draw_hexbin(ax, pred: np.ndarray, true: np.ndarray,
                title: str = "", cmap="Blues",
                xlabel: str = r"$\mathrm{FI}_{\mathrm{true}}$",
                ylabel: str = r"$\mathrm{FI}_{\mathrm{pred}}$",
                line_color: str = "crimson",
                extra_line: str | None = None,
                gly_mask: np.ndarray | None = None,
                lo: float | None = None,
                hi: float | None = None,
                gridsize: int = 60,
                colorbar: bool = True):
    """Hexbin density plot with GLY scatter overlay, regression line, and diagonal."""
    from scipy.stats import linregress
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    ok = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[ok], true[ok]
    if len(p) < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    slope, intercept, r, *_ = linregress(t, p)

    gly_ok = None
    if gly_mask is not None and len(gly_mask) == len(pred):
        gly_ok = gly_mask[ok]

    # Neutral zone: gray band along both axes drawn first (lowest zorder)
    if lo is not None and hi is not None:
        ax.axvspan(lo, hi, alpha=0.12, color="gray", zorder=0)
        ax.axhspan(lo, hi, alpha=0.12, color="gray", zorder=0)

    # Main cloud: hexbin on non-GLY points (or all if no mask)
    main_mask = ~gly_ok if gly_ok is not None else np.ones(len(t), dtype=bool)
    hb = ax.hexbin(t[main_mask], p[main_mask], gridsize=gridsize,
                   bins="log", cmap=cmap, mincnt=1, linewidths=0.0, zorder=2)
    if colorbar:
        import matplotlib.pyplot as plt
        cb = plt.colorbar(hb, ax=ax, pad=0.02, shrink=0.85, aspect=20)
        cb.set_label(r"$\log_{10}(\mathrm{count})$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    xlin = np.array([t.min(), t.max()])
    ax.plot(xlin, slope * xlin + intercept, "-", color=line_color, lw=1.5, zorder=4)
    ax.plot(xlin, xlin, "k--", lw=0.8, alpha=0.5, zorder=4)

    _legend_handles = []
    if lo is not None:
        _c = CLASS_COLORS[0]
        ax.axvline(lo, color=_c, lw=0.9, ls="--", alpha=0.75, zorder=1)
        ax.axhline(lo, color=_c, lw=0.9, ls="--", alpha=0.75, zorder=1)
        _legend_handles.append(mlines.Line2D([], [], color=_c, lw=1.2, ls="--",
                                             label=rf"Highly frustrated ($\mathrm{{FI}} \leq {lo}$)"))
    if lo is not None and hi is not None:
        _legend_handles.append(mpatches.Patch(color="gray", alpha=0.3,
                                              label=rf"Neutral (${lo} < \mathrm{{FI}} \leq {hi}$)"))
    if hi is not None:
        _c = CLASS_COLORS[2]
        ax.axvline(hi, color=_c, lw=0.9, ls="--", alpha=0.75, zorder=1)
        ax.axhline(hi, color=_c, lw=0.9, ls="--", alpha=0.75, zorder=1)
        _legend_handles.append(mlines.Line2D([], [], color=_c, lw=1.2, ls="--",
                                             label=rf"Minimally frustrated ($\mathrm{{FI}} > {hi}$)"))

    if gly_ok is not None and gly_ok.any():
        gt, gp = t[gly_ok], p[gly_ok]
        ax.scatter(gt, gp, c="orange", s=4, alpha=0.55,
                   linewidths=0, rasterized=True, zorder=5)
        err_gly = float(
            (fi_class(gp, lo=lo or LO_PAIR, hi=hi or HI_PAIR) !=
             fi_class(gt, lo=lo or LO_PAIR, hi=hi or HI_PAIR)).mean()
        ) * 100
        _legend_handles.append(
            mpatches.Patch(color="orange",
                           label=f"GLY  (n={int(gly_ok.sum()):,})  err {err_gly:.1f}%"))

    if _legend_handles:
        leg = ax.legend(handles=_legend_handles, fontsize=6.5, loc="lower right",
                  framealpha=0.85, edgecolor="none", markerscale=1.5)
        leg.get_frame().set_zorder(10)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=9)
    label = f"R² = {r**2:.6f}"
    if extra_line:
        label += f"\n{extra_line}"
    ax.text(0.04, 0.94, label, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.85))
