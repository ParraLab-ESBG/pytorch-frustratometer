"""
Per-protein frustration figure: line plot + 3-D structure coloured by frustration.

Figure layout (one column per panel, width_ratios ~3:1[:1]):
  [frustration line plot] | [structure: regression threshold colouring]
  (ML model only)         | [structure: predicted-class colouring]

Usage
-----
# Single domain
python -m frustrai_fold.plots.per_protein \\
    --results_dir  /path/to/singleresidue_results/ \\
    --data_dir     /path/to/dompdb_v4.4/ \\
    --out          /path/to/plots/per_protein/ \\
    --domains      6nhsA00

# All beta-lactamases + globins in the test set (ML model)
python -m frustrai_fold.plots.per_protein \\
    --results_dir  /path/to/singleresidue_results/ \\
    --data_dir     /path/to/dompdb_v4.4/ \\
    --out          /path/to/plots/per_protein/ \\
    --family beta_lactamase globin top7 \\
    --is_ml
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import torch
from ..utils.weights import classify_frustration, frustration_thresholds as _FRUST_THR
matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from matplotlib import gridspec

warnings.filterwarnings("ignore", message=".*elements were guessed from atom name.*")

_VERBOSE          = False
_RENDER_STRUCTURE = False
_FORMATS          = ["png"]

def _log(path):
    if _VERBOSE:
        print(f"\r  wrote {path}", end="", flush=True)

# ── family → CATH superfamily code ───────────────────────────────────────────
FAMILY_CATH = {
    "beta_lactamase": "3.40.710.10",
    "globin":         "1.10.490.10",
    "top7":           "3.30.1710.10",
}

TIER_DESCS = {"A": "Bottom 5%", "B": "Average", "C": "Top 5%"}

# active site per family: aa type + canonical position + display label
FAMILY_ACTIVE_SITE: dict[str, dict] = {
    "beta_lactamase": {"aa": "S", "pos": 70,  "label": "Ser70*"},
    "globin":         {"aa": "H", "pos": 87,  "label": "His87*"},
    "top7":           {},   # no canonical active site
}


_ACTIVE_SITE_CSV_CACHE: dict[str, dict[str, int]] = {}  # family_key → {domain → res_id}


def _load_active_site_csv(csv_path: Path) -> dict[str, int]:
    """Load domain→res_id mapping from a find_active_site.py output CSV."""
    df = pd.read_csv(csv_path)
    return dict(zip(df["domain"].astype(str), df["res_id"].astype(int)))


def _lookup_domain_in_any_csv(domain: str,
                               csv_dir: Path) -> tuple[int, str] | None:
    """Scan all *_active_site.csv files in csv_dir for domain.

    Returns (res_id, family_key) if found, else None.
    Useful when family is not known (e.g. --domains CLI without --family).
    """
    for csv_path in sorted(Path(csv_dir).glob("*_active_site.csv")):
        family_key = csv_path.stem.replace("_active_site", "")
        if family_key not in _ACTIVE_SITE_CSV_CACHE:
            _ACTIVE_SITE_CSV_CACHE[family_key] = _load_active_site_csv(csv_path)
        if domain in _ACTIVE_SITE_CSV_CACHE[family_key]:
            return _ACTIVE_SITE_CSV_CACHE[family_key][domain], family_key
    return None


def _find_active_site_idx(
    res_ids: np.ndarray,
    residues: np.ndarray,
    aa: str,
    pos: int,
    domain: str | None = None,
    family: str | None = None,
    csv_dir: Path | None = None,
) -> int | None:
    """Return x-array index of the active-site residue.

    Priority:
      1. CSV lookup (family_active_site.csv in *csv_dir*) — structurally derived.
      2. Number-based fallback: closest residue to *pos* (prefer matching *aa*).
    """
    # 1. CSV lookup
    if domain and family and csv_dir:
        family_key = family.lower().replace(" ", "_").replace("-", "_")
        if family_key not in _ACTIVE_SITE_CSV_CACHE:
            csv_path = csv_dir / f"{family_key}_active_site.csv"
            if csv_path.exists():
                _ACTIVE_SITE_CSV_CACHE[family_key] = _load_active_site_csv(csv_path)
            else:
                _ACTIVE_SITE_CSV_CACHE[family_key] = {}
        mapping = _ACTIVE_SITE_CSV_CACHE.get(family_key, {})
        if domain in mapping:
            target_id = mapping[domain]
            hits = np.where(res_ids == target_id)[0]
            if len(hits):
                return int(hits[0])

    # 2. Number-based fallback
    aa_up = aa.upper()
    aa_mask = np.array([str(r).upper().startswith(aa_up) for r in residues])
    candidates = res_ids[aa_mask] if aa_mask.any() else res_ids
    if len(candidates) == 0:
        return None
    closest_id = int(candidates[np.argmin(np.abs(candidates.astype(float) - pos))])
    hits = np.where(res_ids == closest_id)[0]
    return int(hits[0]) if len(hits) else None

# ── frustration colour scheme ─────────────────────────────────────────────────
_FRUST_CMAP = LinearSegmentedColormap.from_list(
    "frustration",
    [(0.0, [0.8, 0.0, 0.0]),
     (0.5, [0.6, 0.6, 0.6]),
     (1.0, [0.0, 0.7, 0.0])],
)
_FRUST_NORM = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
_CLASS_CMAP = ListedColormap(["#D55E00", "#BBBBBB", "#009E73"])  # MAX / NEU / MIN

_FRUST_RGB = {
    "MAX": [0.8, 0.0, 0.0],
    "NEU": [0.6, 0.6, 0.6],
    "MIN": [0.0, 0.7, 0.0],
}


def _color_from_value(val: float):
    return _FRUST_CMAP(_FRUST_NORM(val))


def _color_from_class(cls) -> list:
    """cls: int (0=MAX,1=NEU,2=MIN) or string ('minimally'/'neutral'/'highly')."""
    if isinstance(cls, str):
        cls = cls.strip().lower()
        if cls.startswith("min"):
            return _FRUST_RGB["MIN"]
        if cls.startswith("neu") or cls.startswith("neutral"):
            return _FRUST_RGB["NEU"]
        return _FRUST_RGB["MAX"]
    if cls == 0:
        return _FRUST_RGB["MAX"]
    if cls == 2:
        return _FRUST_RGB["MIN"]
    return _FRUST_RGB["NEU"]


def _to_int_class(arr) -> np.ndarray:
    """Map FrstState strings or ints → 0=MAX, 1=NEU, 2=MIN."""
    out = np.ones(len(arr), dtype=int)
    for i, c in enumerate(arr):
        if isinstance(c, str):
            c = c.strip().lower()
            if c.startswith("min"):
                out[i] = 2
            elif c.startswith("high") or c.startswith("max"):
                out[i] = 0
        else:
            out[i] = int(c)
    return out


# ── frustration line plot (ported from ConfigurationalAnalysis.ipynb) ─────────

MAX_RESIDUES_PER_ROW = 400   # split into 2 rows only for very long proteins


def _draw_frustration_segment(
    ax,
    x: np.ndarray,
    frst_true: np.ndarray,
    frst_pred: np.ndarray | None,
    true_colors, pred_colors, class_colors,
    residues,
    res_ids,
    pred_label: str,
    mode: int = 0,
    show_legend: bool = True,
    show_ylabel: bool = True,
    show_class_strip: bool = True,
    show_strip_labels: bool = True,
    active_site_x: int | None = None,
    active_site_label: str = "",
    true_label: str = "Original FrustratometeR",
):
    """Draw one segment of the frustration line plot onto *ax*."""
    _thr = _FRUST_THR[mode]
    true_threshold = float(_thr[0])
    min_threshold  = float(_thr[1])

    ax.plot(x, frst_true, color="#1f77b4", linewidth=2)
    if frst_pred is not None:
        ax.plot(x, frst_pred, color="#ff7f0e", linewidth=2, linestyle="--")
        ax.fill_between(x, frst_true, frst_pred,
                        where=frst_true >= frst_pred,
                        facecolor="#ff7f0e", alpha=0.25, interpolate=True)
        ax.fill_between(x, frst_true, frst_pred,
                        where=frst_true <= frst_pred,
                        facecolor="#1f77b4", alpha=0.25, interpolate=True)

    ax.axhline(true_threshold, color=_FRUST_RGB["MAX"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(min_threshold,  color=_FRUST_RGB["MIN"], linestyle="--", linewidth=1, alpha=0.7)
    ax.text(x[-1] + 0.5, true_threshold, "MAX", va="top",    ha="left",
            fontsize=9, weight="bold", color=_FRUST_RGB["MAX"])
    ax.text(x[-1] + 0.5, min_threshold,  "MIN", va="bottom", ha="left",
            fontsize=9, weight="bold", color=_FRUST_RGB["MIN"])

    # annotation tracks below the plot
    height = 0.06
    gap    = 0.01
    y_base = -0.10 - height   # True strip top ~0.10 axes-fraction below tick labels
    trans  = blended_transform_factory(ax.transData, ax.transAxes)

    _lkw = dict(transform=trans, ha="left", va="center", fontsize=7,
                weight="bold", clip_on=False, zorder=5,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", alpha=0.85, ec="none"))

    for xi, c in zip(x, true_colors):
        ax.add_patch(plt.Rectangle((xi - 0.4, y_base), 0.8, height,
                                   transform=trans, color=c, clip_on=False))

    if frst_pred is not None:
        y_pred = y_base - height - gap
        for xi, c in zip(x, pred_colors):
            ax.add_patch(plt.Rectangle((xi - 0.4, y_pred), 0.8, height,
                                       transform=trans, color=c, clip_on=False))
        if show_strip_labels:
            ax.text(x[0] - 0.35, y_base + height / 2, true_label, **_lkw)
            ax.text(x[0] - 0.35, y_pred + height / 2, pred_label, **_lkw)
        y_next = y_pred
    else:
        if show_strip_labels:
            ax.text(x[0] - 0.35, y_base + height / 2, true_label, **_lkw)
        y_next = y_base

    if show_class_strip:
        y_cls = y_next - height - gap
        for xi, c in zip(x, class_colors):
            ax.add_patch(plt.Rectangle((xi - 0.4, y_cls), 0.8, height,
                                       transform=trans, color=c, clip_on=False))
        if show_strip_labels:
            ax.text(x[0] - 0.35, y_cls + height / 2, "Pred (class)", **_lkw)
        y_label_row = y_cls - gap
    else:
        y_label_row = y_next - gap

    # Rotated tick labels: "A 123" format, 90° so any residue number fits
    step = max(1, len(x) // 20)
    idx  = list(range(0, len(x), step))
    ax.set_xticks(x[idx])
    ax.set_xticklabels(
        [f"{residues[i]} {res_ids[i]}" for i in idx],
        rotation=45, ha="right", va="top", fontsize=8,
    )

    if active_site_x is not None:
        ax.axvline(active_site_x, color="#FFD700", linewidth=1.2,
                   linestyle="--", alpha=0.7, zorder=1)
        ax.plot(active_site_x, frst_true[active_site_x],
                marker="*", color="#FFD700", markersize=16,
                markeredgecolor="#8B6914", markeredgewidth=0.7,
                zorder=6, linestyle="none")
        if active_site_label:
            ax.text(active_site_x + 0.4, frst_true[active_site_x],
                    active_site_label, ha="left", va="center", fontsize=7.5,
                    color="#8B6914", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#FFF9C4",
                              ec="#FFD700", alpha=0.85, lw=0.7),
                    zorder=7)

    ax.set_ylim(4, -4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_ylabel:
        ax.set_ylabel("Frustration index", fontsize=10)

    if show_legend:
        if frst_pred is not None:
            legend_elements = [
                Line2D([0], [0], color="#1f77b4", lw=2, label=true_label),
                Line2D([0], [0], color="#ff7f0e", lw=2, linestyle="--", label=pred_label),
                Patch(facecolor="#ff7f0e", alpha=0.25, label=f"{true_label} >\n{pred_label}"),
                Patch(facecolor="#1f77b4", alpha=0.25, label=f"{pred_label} >\n{true_label}"),
            ]
            ax.legend(handles=legend_elements, loc="upper left",
                      frameon=False, fontsize=8, ncol=2)
        else:
            ax.legend(handles=[Line2D([0], [0], color="#1f77b4", lw=2, label=true_label)],
                      loc="upper left", frameon=False, fontsize=8)


def plot_single_frustration(
    axes,
    frst_true: np.ndarray,
    frst_pred: np.ndarray | None,
    frst_class_pred,
    residues,
    res_ids,
    protein_id: str,
    *,
    pred_label: str = "PyTorch Re-implementation",
    true_label: str = "Original FrustratometeR",
    show_class_strip: bool = True,
    active_site_x: int | None = None,
    active_site_label: str = "",
):
    """Draw frustration line plot onto *axes* (any number of rows)."""
    import math
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    N     = len(frst_true)
    nrows = len(axes)
    chunk = math.ceil(N / nrows)
    splits = [slice(i * chunk, min((i + 1) * chunk, N)) for i in range(nrows)]

    def _thresh(v):
        return int(classify_frustration(torch.tensor([float(v)]), 0).item())

    true_colors  = np.array([_color_from_class(_thresh(v)) for v in frst_true])
    pred_colors  = (np.array([_color_from_class(_thresh(v)) for v in frst_pred])
                    if frst_pred is not None else true_colors)
    class_colors = np.array([_color_from_class(c) for c in frst_class_pred])

    for row_idx, (ax, sl) in enumerate(zip(axes, splits)):
        x_local = np.arange(sl.stop - sl.start)
        _draw_frustration_segment(
            ax,
            x          = x_local,
            frst_true  = frst_true[sl],
            frst_pred  = frst_pred[sl] if frst_pred is not None else None,
            true_colors  = true_colors[sl],
            pred_colors  = pred_colors[sl],
            class_colors = class_colors[sl],
            residues     = residues[sl],
            res_ids      = res_ids[sl],
            pred_label   = pred_label,
            true_label   = true_label,
            mode              = 0,
            show_legend       = (row_idx == 0),
            show_ylabel       = (row_idx == 0),
            show_strip_labels = (row_idx == 0),
            show_class_strip  = show_class_strip,
            active_site_x     = (active_site_x - sl.start
                                  if active_site_x is not None
                                  and sl.start <= active_site_x < sl.stop
                                  else None),
            active_site_label = active_site_label,
        )

    if frst_pred is not None:
        norm_t = np.linalg.norm(frst_true)
        norm_p = np.linalg.norm(frst_pred)
        cos_sim = (float(np.dot(frst_true, frst_pred) / (norm_t * norm_p))
                   if norm_t > 0 and norm_p > 0 else float("nan"))
        axes[0].set_title(f"Cosine similarity: {cos_sim:.3f}", fontsize=10)


def _savefig(fig, path: Path, dpi: int = 150) -> None:
    """Save *fig* in each format listed in _FORMATS, then close it."""
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
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


# ── 3-D structure utilities (PyMOL headless) ──────────────────────────────────

def _write_bfactor_pdb(src_pdb: Path, res_ids: np.ndarray,
                        values: np.ndarray, out_pdb: Path) -> None:
    """Copy *src_pdb* to *out_pdb*, replacing B-factor column with *values*
    aligned by residue number (*res_ids*)."""
    if Path(src_pdb).suffix.lower() != ".pdb":
        # The B-factor rewrite below assumes fixed-column PDB records; mmCIF lines start with "ATOM  "
        # too and would be corrupted. The structure writer already put the values in the B-factors.
        import shutil
        shutil.copyfile(src_pdb, out_pdb)
        return
    val_map = dict(zip(res_ids.tolist(), values.tolist()))
    with open(src_pdb) as fh_in, open(out_pdb, "w") as fh_out:
        for line in fh_in:
            if line[:6] in ("ATOM  ", "HETATM"):
                try:
                    res_id = int(line[22:26])
                    bval   = val_map.get(res_id, 0.0)
                    line   = f"{line[:60]}{bval:6.2f}{line[66:]}"
                except (ValueError, IndexError):
                    pass
            fh_out.write(line)


def _pymol_orient_and_label_block(sticks_resi: str, n_resid: int | None,
                                   c_resid: int | None) -> str:
    """Return PyMOL commands to orient the active site to front and label N/C termini."""
    lines = []

    # Orient: show full protein but center on active site so it faces the camera
    if sticks_resi:
        lines += [
            "orient obj",
            "center sel_site",
            "zoom sel_site, 12",  # zoom in on active site with surrounding context
        ]
    else:
        lines += [
            #"orient obj",
            "zoom obj, complete=1, buffer=5",  # zoom in on full protein
        ]

    # N / C terminus labels
    label_style = [
        "set label_size, 20",
        "set label_color, black",
        "set label_font_id, 7",
        "set label_outline_color, white",
    ]
    lines += label_style
    # if n_resid is not None:
    #     lines.append(f'label (obj and resi {n_resid} and name CA), "N"')
    # if c_resid is not None:
    #     lines.append(f'label (obj and resi {c_resid} and name CA), "C"')

    return "\n".join(lines)


def _render_tail(out_png) -> str:
    """ray/png/quit lines; empty when out_png is None so a .pml opened in the PyMOL GUI
    keeps the session open instead of writing 'None.png' and quitting."""
    if out_png is None:
        return ""
    return (f"set ray_opaque_background, 0\nset antialias, 2\n"
            f"ray 1800, 1800\npng {out_png}, dpi=600\nquit\n")


def _pymol_script_regression(pdb_path: Path, out_png: Path,
                              sticks_resi: str = "",
                              n_resid: int | None = None,
                              c_resid: int | None = None) -> str:
    """Return a PyMOL .pml script that colours by B-factor (frustration index)."""
    sticks_block = f"""select sel_site, obj and resi {sticks_resi}
show sticks, sel_site
set stick_radius, 0.20
select sel_site_ca, sel_site and name CA
show spheres, sel_site_ca
set sphere_scale, 0.45, sel_site_ca
color yellow, sel_site_ca
""" if sticks_resi else ""
    orient_label = _pymol_orient_and_label_block(sticks_resi, n_resid, c_resid)
    return f"""\
load {pdb_path}, obj
show cartoon, obj
hide lines, obj
hide nonbonded, obj
set_color frust_max,  [0.80, 0.00, 0.00]
set_color frust_zero, [0.60, 0.60, 0.60]
set_color frust_min,  [0.00, 0.70, 0.00]
spectrum b, frust_max frust_zero frust_min, obj, minimum=-4, maximum=4
{sticks_block}{orient_label}
bg_color white
{_render_tail(out_png)}"""


def _pymol_script_class(pdb_path: Path, out_png: Path,
                         max_resi: str, neu_resi: str, min_resi: str,
                         sticks_resi: str = "",
                         n_resid: int | None = None,
                         c_resid: int | None = None) -> str:
    """Return a PyMOL .pml script that colours residues by predicted class."""
    max_r, max_g, max_b = _FRUST_RGB["MAX"]
    neu_r, neu_g, neu_b = _FRUST_RGB["NEU"]
    min_r, min_g, min_b = _FRUST_RGB["MIN"]
    sticks_block = f"""select sel_site, obj and resi {sticks_resi}
show sticks, sel_site
set stick_radius, 0.30
select sel_site_ca, sel_site and name CA
show spheres, sel_site_ca
set sphere_scale, 0.45, sel_site_ca
color yellow, sel_site_ca
""" if sticks_resi else ""
    orient_label = _pymol_orient_and_label_block(sticks_resi, n_resid, c_resid)
    return f"""\
load {pdb_path}, obj
show cartoon, obj
hide lines, obj
hide nonbonded, obj
set_color frust_max, [{max_r}, {max_g}, {max_b}]
set_color frust_neu, [{neu_r}, {neu_g}, {neu_b}]
set_color frust_min, [{min_r}, {min_g}, {min_b}]
color frust_neu, obj
select sel_max, obj and resi {max_resi or "none"}
select sel_min, obj and resi {min_resi or "none"}
color frust_max, sel_max
color frust_min, sel_min
{sticks_block}{orient_label}
bg_color white
set ray_opaque_background, 0
set antialias, 2
ray 1800, 1800
png {out_png}, dpi=600
quit
"""


def render_structure_pymol(
    pdb_file: Path,
    out_png: Path,
    res_ids: np.ndarray,
    values: np.ndarray,
    mode: str = "regression",   # 'regression' | 'class'
    pymol_exe: str = "pymol",
    sticks_residues: list[int] | None = None,
) -> bool:
    """Render a cartoon structure PNG via PyMOL headless.

    Returns True on success, False if PyMOL is unavailable or fails.
    """
    import subprocess, tempfile, shutil

    if shutil.which(pymol_exe) is None:
        print(f"    [PyMOL] executable '{pymol_exe}' not found in PATH", flush=True)
        return False

    return _run_pymol(pymol_exe, _build_pymol_script(
        pdb_file, out_png, res_ids, values, mode,
        sticks_residues=sticks_residues), out_png)


def _build_pymol_script(pdb_file, out_png, res_ids, values, mode,
                         sticks_residues=None):
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    sticks_resi = "+".join(str(r) for r in sticks_residues) if sticks_residues else ""
    n_resid = int(res_ids[0])  if len(res_ids) else None
    c_resid = int(res_ids[-1]) if len(res_ids) else None
    if mode == "regression":
        bf_pdb = tmp / "structure_bf.pdb"
        _write_bfactor_pdb(pdb_file, res_ids, values, bf_pdb)
        return tmp, _pymol_script_regression(bf_pdb, out_png,
                                              sticks_resi=sticks_resi,
                                              n_resid=n_resid, c_resid=c_resid)
    else:
        int_cls = np.asarray(values, dtype=int)
        def _resi_str(mask):
            ids = res_ids[mask]
            return "+".join(str(r) for r in ids) if len(ids) else "none"
        max_resi = _resi_str(int_cls == 0)
        neu_resi = _resi_str(int_cls == 1)
        min_resi = _resi_str(int_cls == 2)
        return tmp, _pymol_script_class(pdb_file, out_png, max_resi, neu_resi, min_resi,
                                         sticks_resi=sticks_resi,
                                         n_resid=n_resid, c_resid=c_resid)


def _run_pymol(pymol_exe: str, tmp_and_script, out_png: Path) -> bool:
    """Run a PyMOL .pml script headlessly, with automatic Xvfb fallback.

    Returns True if *out_png* was created.
    """
    import subprocess, shutil, time
    tmp, script = tmp_and_script
    try:
        pml = tmp / "render.pml"
        pml.write_text(script)

        env = os.environ.copy()
        xvfb_proc = None
        if "DISPLAY" not in env:
            if shutil.which("Xvfb") is not None:
                disp = ":99"
                xvfb_proc = subprocess.Popen(
                    ["Xvfb", disp, "-screen", "0", "1024x768x24"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                time.sleep(1)
                env["DISPLAY"] = disp
            # If no Xvfb, leave DISPLAY unset and let PyMOL run headless (-c mode)

        try:
            subprocess.run(
                [pymol_exe, "-cq", str(pml)],
                timeout=120, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        finally:
            if xvfb_proc is not None:
                xvfb_proc.terminate()

        return out_png.exists()
    except Exception as e:
        print(f"    [PyMOL exception] {e}", flush=True)
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)




def show_structure_panel(ax, png_path: Path, title: str,
                          mode: str = "regression",
                          show_legend: bool = True,
                          title_fontsize: int = 10) -> None:
    """Display a pre-rendered structure PNG in *ax*.

    Set show_legend=False in combined figures to suppress per-panel colorbars/legends.
    """
    img = plt.imread(str(png_path))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=title_fontsize)

    if not show_legend:
        return

    if mode == "regression":
        import matplotlib.cm as cm  # noqa: F401
        sm = plt.cm.ScalarMappable(cmap=_FRUST_CMAP, norm=_FRUST_NORM)
        plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04,
                     pad=0.02, label="Frustration index")
    elif mode == "class":
        legend_elements = [
            Patch(facecolor="#D55E00", label="Highly frustrated"),
            Patch(facecolor="#BBBBBB", label="Neutral"),
            Patch(facecolor="#009E73", label="Minimally frustrated"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, frameon=False,
                  loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    elif mode == "pairwise":
        legend_elements = [
            Patch(facecolor=_FRUST_RGB["MAX"], label="Highly frustrated"),
            Patch(facecolor=_FRUST_RGB["NEU"], label="Neutral"),
            Patch(facecolor=_FRUST_RGB["MIN"], label="Minimally frustrated"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, frameon=False,
                  loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))


# ── file loading ──────────────────────────────────────────────────────────────

def load_singleresidue_file(path: Path) -> pd.DataFrame:
    """Read a singleresidue output file (space-sep, # comments).

    Returns DataFrame with at minimum:
      Res, AA, TrueFrstIndex (or FrstIndex), PredFrstIndex, PredFrstState.
    """
    df = pd.read_csv(str(path), sep=r"\s+", comment="#")
    # Normalise column names.
    # Case 1 — baseline without compare_to_original: only FrstIndex present.
    #   Treat it as TrueFrstIndex (ground truth = PyTorch re-impl, no comparison).
    # Case 2 — baseline with compare_to_original: both FrstIndex (PyTorch) and
    #   TrueFrstIndex (original C++) present.  Promote FrstIndex → PredFrstIndex
    #   so the plotter can show both panels side-by-side.
    if "TrueFrstIndex" not in df.columns and "FrstIndex" in df.columns:
        df = df.rename(columns={"FrstIndex": "TrueFrstIndex",
                                 "FrstState": "TrueFrstState"})
    elif "TrueFrstIndex" in df.columns and "FrstIndex" in df.columns:
        df = df.rename(columns={"FrstIndex": "PredFrstIndex",
                                 "FrstState": "PredFrstState"})
    return df


# ── pairwise loading helpers ──────────────────────────────────────────────────

_WELLTYPE_VALS = {"short", "long", "water-mediated"}


def load_pairwise_file(path: Path) -> pd.DataFrame:
    """Load mutational or configurational pairwise frustration file."""
    df = pd.read_csv(str(path), sep=r"\s+", comment="#")
    rename = {}
    if "Res1" not in df.columns:
        if "i" in df.columns: rename["i"] = "Res1"
        if "j" in df.columns: rename["j"] = "Res2"
    if rename:
        df = df.rename(columns=rename)

    # Detect column shift caused by empty DecoyEnergy/SDEnergy fields written
    # with na_rep='' (old files).  pandas collapses consecutive spaces, shifting
    # all subsequent columns two positions left.
    # Symptom: DecoyEnergy contains welltype strings instead of numbers.
    if "DecoyEnergy" in df.columns:
        sample = df["DecoyEnergy"].dropna().astype(str)
        if sample.isin(_WELLTYPE_VALS).any():
            cols = df.columns.tolist()
            de_idx = cols.index("DecoyEnergy")
            shifted = cols[de_idx:]              # e.g. [DecoyEnergy, SDEnergy, Welltype, FrstIndex, ...]
            actual  = shifted[2:]                # the correct names for the shifted data
            # Save shifted values before any overwrite
            saved = {a: df[s].copy() for a, s in zip(actual, shifted)}
            for a, vals in saved.items():
                df[a] = vals
            df["DecoyEnergy"] = np.nan
            df["SDEnergy"]    = np.nan

    # true/pred separation (mirrors singleresidue logic)
    if "TrueFrstIndex" not in df.columns and "FrstIndex" in df.columns:
        df = df.rename(columns={"FrstIndex": "TrueFrstIndex",
                                 "FrstState": "TrueFrstState"})
    elif "TrueFrstIndex" in df.columns and "FrstIndex" in df.columns:
        df = df.rename(columns={"FrstIndex": "PredFrstIndex",
                                 "FrstState": "PredFrstState"})
    return df


def _get_frst_col(df: pd.DataFrame) -> str:
    for col in ("TrueFrstIndex", "FrstIndex", "FrustrationIndex", "NativeFrustration",
                "frustration_index", "FrstInd", "PredFrstIndex"):
        if col in df.columns:
            return col
    raise KeyError(f"No frustration index column found. Columns: {list(df.columns)}")


def _get_state_col(df: pd.DataFrame) -> str | None:
    for col in ("FrstState", "TrueFrstState", "FrustrationState", "FrustrationType",
                "frustration_state", "FrstSt", "PredFrstState"):
        if col in df.columns:
            return col
    return None


def _matrix_to_class_rgb(matrix: np.ndarray, mode: int = 1) -> np.ndarray:
    """Convert float frustration matrix to RGBA image using discrete 3-class colors."""
    N   = matrix.shape[0]
    rgb = np.ones((N, N, 4), dtype=float)   # white + opaque default
    fin = np.isfinite(matrix)
    flat = matrix.ravel().astype(np.float32)
    cls  = classify_frustration(torch.from_numpy(flat), mode).numpy().reshape(N, N)
    rgb[fin, :3] = _FRUST_RGB["NEU"]
    rgb[fin,  3] = 1.0
    rgb[fin & (cls == 0), :3] = _FRUST_RGB["MAX"]
    rgb[fin & (cls == 2), :3] = _FRUST_RGB["MIN"]
    rgb[~fin] = [1, 1, 1, 0]
    return rgb


def _pairwise_to_matrix(df: pd.DataFrame, res_ids: np.ndarray,
                         col: str | None = None) -> np.ndarray:
    """Convert pairwise DataFrame to symmetric NxN matrix (NaN where absent)."""
    id2idx   = {int(r): i for i, r in enumerate(res_ids)}
    frst_col = col if col is not None else _get_frst_col(df)
    N   = len(res_ids)
    mat = np.full((N, N), np.nan)
    for _, row in df.iterrows():
        i = id2idx.get(int(row["Res1"]))
        j = id2idx.get(int(row["Res2"]))
        if i is None or j is None:
            continue
        v = float(row[frst_col])
        mat[i, j] = v
        mat[j, i] = v
    return mat


# ── contact map ───────────────────────────────────────────────────────────────

def plot_contact_map(
    ax_top,
    ax_map,
    matrix: np.ndarray,
    res_ids: np.ndarray,
    title: str,
    matrix_pred: np.ndarray | None = None,
    pred_label: str = "PyTorch Re-implementation",
    true_label: str = "Original FrustratometeR",
    mode: int = 1,
) -> None:
    """Heatmap contact map (ax_map) with per-residue mean frustration line plot (ax_top).

    If *matrix_pred* is supplied the upper triangle shows *matrix* (true) and
    the lower triangle shows *matrix_pred*, separated by a y=x diagonal.
    """
    N = len(res_ids)
    true_threshold = float(_FRUST_THR[mode][0])
    min_threshold  = float(_FRUST_THR[mode][1])

    # ── build display matrix (split triangles when pred available) ────────────
    if matrix_pred is not None:
        display = matrix.copy()
        il, jl  = np.tril_indices(N, k=-1)   # strict lower triangle
        display[il, jl] = matrix_pred[il, jl]
    else:
        display = matrix

    # Small circles at each contact instead of a filled-square heatmap --
    # reads less "boxy" for sparse contact maps where most cells are empty.
    ii, jj = np.where(np.isfinite(display))
    vals = display[ii, jj]
    cls  = classify_frustration(torch.from_numpy(vals.astype(np.float32)), mode).numpy()
    dot_colors = np.empty((len(vals), 3))
    dot_colors[:] = _FRUST_RGB["NEU"]
    dot_colors[cls == 0] = _FRUST_RGB["MAX"]
    dot_colors[cls == 2] = _FRUST_RGB["MIN"]
    dot_size = max(4.0, min(30.0, 4000.0 / max(N, 1)))
    ax_map.scatter(jj, ii, c=dot_colors, s=dot_size, marker="o", linewidths=0)
    ax_map.set_xlim(-0.5, N - 0.5)
    ax_map.set_ylim(N - 0.5, -0.5)
    ax_map.set_aspect("equal")

    if matrix_pred is not None:
        # y=x diagonal separator
        ax_map.plot([-0.5, N - 0.5], [-0.5, N - 0.5],
                    color="black", lw=1.5, zorder=5)
        # triangle labels — horizontal, bold, inside each triangle
        ax_map.text(N * 0.98, N * 0.15, true_label,
                    ha="right", va="top", fontsize=10, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"))
        ax_map.text(N * 0.15, N * 0.98, pred_label,
                    ha="left", va="bottom", fontsize=10, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="none"))

    step = max(1, N // 20)
    ticks = list(range(0, N, step))
    tick_labels = [str(res_ids[t]) for t in ticks]
    ax_map.set_xticks(ticks)
    ax_map.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax_map.set_yticks(ticks)
    ax_map.set_yticklabels(tick_labels, fontsize=7)
    ax_map.set_xlabel("Residue", fontsize=10)
    ax_map.set_ylabel("Residue", fontsize=10)

    # ── top panel: mean frustration per residue as line plot ──────────────────
    x             = np.arange(N)
    marginal_true = np.nanmean(matrix, axis=1)
    marginal_pred = np.nanmean(matrix_pred, axis=1) if matrix_pred is not None else None

    ax_top.plot(x, marginal_true, color="#1f77b4", linewidth=2,
                label=true_label)
    if marginal_pred is not None:
        ax_top.plot(x, marginal_pred, color="#ff7f0e", linewidth=2,
                    linestyle="--", label=pred_label)
        ax_top.fill_between(x, marginal_true, marginal_pred,
                            where=marginal_true >= marginal_pred,
                            facecolor="#ff7f0e", alpha=0.25, interpolate=True)
        ax_top.fill_between(x, marginal_true, marginal_pred,
                            where=marginal_true < marginal_pred,
                            facecolor="#1f77b4", alpha=0.25, interpolate=True)
    else:
        ax_top.fill_between(x, marginal_true, 0,
                            where=marginal_true >= 0,
                            facecolor="#1f77b4", alpha=0.25, interpolate=True)
        ax_top.fill_between(x, marginal_true, 0,
                            where=marginal_true < 0,
                            facecolor="#ff7f0e", alpha=0.25, interpolate=True)

    ax_top.axhline(true_threshold, color=_FRUST_RGB["MAX"],
                   linestyle="--", linewidth=1, alpha=0.7)
    ax_top.axhline(min_threshold,  color=_FRUST_RGB["MIN"],
                   linestyle="--", linewidth=1, alpha=0.7)
    # ascending rows: MAX threshold is the lower line; smooth (descending) is the reverse
    _max_va, _min_va = ("bottom", "top") if true_threshold > min_threshold else ("top", "bottom")
    ax_top.text(N - 0.5, true_threshold, "MAX", va=_max_va, ha="right",
                fontsize=8, weight="bold", color=_FRUST_RGB["MAX"])
    ax_top.text(N - 0.5, min_threshold,  "MIN", va=_min_va, ha="right",
                fontsize=8, weight="bold", color=_FRUST_RGB["MIN"])

    # cosine similarity and R² appended to the axes title
    if matrix_pred is not None:
        mask    = np.isfinite(matrix) & np.isfinite(matrix_pred)
        tv, pv  = matrix[mask], matrix_pred[mask]
        nt, np_ = np.linalg.norm(tv), np.linalg.norm(pv)
        cos_sim = float(np.dot(tv, pv) / (nt * np_)) if nt > 0 and np_ > 0 else float("nan")
        ss_res  = np.sum((tv - pv) ** 2)
        ss_tot  = np.sum((tv - tv.mean()) ** 2)
        r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        full_title = f"{title}\nCos. sim: {cos_sim:.3f}   R²: {r2:.3f}"
    else:
        full_title = title

    ax_top.set_title(full_title, fontsize=11, weight="bold", pad=4)
    ax_top.set_xlim(-0.5, N - 0.5)
    ax_top.set_ylim(-4, 4)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.set_ylabel("Mean\nfrustration", fontsize=8)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)

    # Legend: 2-column layout matching single-residue style, no MIN/MAX entries
    _tl = ("Orig." if true_label == "Original FrustratometeR"
           else "PyTorch" if "PyTorch Frustratometer" in true_label
           else true_label)
    _pl = "Re-impl." if "Re-implementation" in pred_label else pred_label
    if marginal_pred is not None:
        legend_elements = [
            Line2D([0], [0], color="#1f77b4", lw=2, label=true_label),
            Line2D([0], [0], color="#ff7f0e", lw=2, linestyle="--", label=pred_label),
            Patch(facecolor="#ff7f0e", alpha=0.25, label=f"{_tl} > {_pl}"),
            Patch(facecolor="#1f77b4", alpha=0.25, label=f"{_pl} > {_tl}"),
        ]
        ax_top.legend(handles=legend_elements, loc="upper left",
                      frameon=False, fontsize=8, ncol=2)
    else:
        ax_top.legend(
            handles=[Line2D([0], [0], color="#1f77b4", lw=2, label=true_label)],
            loc="upper left", frameon=False, fontsize=8,
        )


# ── pairwise PyMOL (CGO cylinders between Cα) ────────────────────────────────

def _pymol_script_pairwise(
    pdb_path: Path, out_png: Path,
    highly_solid: list, highly_dashed: list,
    min_solid: list, min_dashed: list,
) -> str:
    """PyMOL .pml with CGO solid cylinders (direct contacts) and distance dashed lines
    (water-mediated contacts).  Colors match _FRUST_RGB: dark-red / dark-green.
    Neutral contacts are omitted from the 3D rendering.
    pairs are (res1, chain1, res2, chain2) 4-tuples; chain may be empty string.
    """
    max_r, max_g, max_b = _FRUST_RGB["MAX"]
    min_r, min_g, min_b = _FRUST_RGB["MIN"]

    # Build PyMOL `distance` commands for water-mediated (dashed) contacts
    dashed_lines: list[str] = []
    for pairs, obj_name, color_name, dash_r in [
        (highly_dashed, "wm_max", "frust_max", 0.10),
        (min_dashed,    "wm_min", "frust_min", 0.10),
    ]:
        for r1, c1, r2, c2 in pairs:
            dashed_lines.append(
                f"distance {obj_name}, "
                f"(obj//{c1}/{r1}/CA), (obj//{c2}/{r2}/CA)"
            )
        if pairs:
            dashed_lines += [
                f"color {color_name}, {obj_name}",
                f"hide labels, {obj_name}",
                f"set dash_radius, {dash_r}, {obj_name}",
                f"set dash_gap, 0.3, {obj_name}",
            ]
    dashed_block = "\n".join(dashed_lines) if dashed_lines else ""

    return f"""\
load {pdb_path}, obj
hide line, obj
unset dynamic_measures
show cartoon, obj
color grey, obj
bg_color white
set_color frust_max, [{max_r}, {max_g}, {max_b}]
set_color frust_min, [{min_r}, {min_g}, {min_b}]
python
from pymol.cgo import CYLINDER
from pymol import cmd

highly_solid = {highly_solid!r}
min_solid    = {min_solid!r}

# CA coords keyed by (resi_str, chain); fall back to resi only if chain unknown
ca_coords = {{}}
model = cmd.get_model('obj and name CA')
for atom in model.atom:
    ca_coords[(atom.resi, atom.chain)] = (atom.coord[0], atom.coord[1], atom.coord[2])

def _ca(res, chain):
    coord = ca_coords.get((str(res), str(chain)))
    if coord is not None:
        return coord
    coord = ca_coords.get((str(res), ''))
    if coord is not None:
        return coord
    res_str = str(res)
    for (r, c), coord in ca_coords.items():
        if r == res_str:
            return coord
    return None

MAX_COL = ({max_r}, {max_g}, {max_b})
MIN_COL = ({min_r}, {min_g}, {min_b})

cgo = []
for r1, c1, r2, c2 in highly_solid:
    p1, p2 = _ca(r1, c1), _ca(r2, c2)
    if p1 is None or p2 is None:
        continue
    cgo += [CYLINDER, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], 0.10,
            MAX_COL[0], MAX_COL[1], MAX_COL[2], MAX_COL[0], MAX_COL[1], MAX_COL[2]]
for r1, c1, r2, c2 in min_solid:
    p1, p2 = _ca(r1, c1), _ca(r2, c2)
    if p1 is None or p2 is None:
        continue
    cgo += [CYLINDER, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], 0.10,
            MIN_COL[0], MIN_COL[1], MIN_COL[2], MIN_COL[0], MIN_COL[1], MIN_COL[2]]
if cgo:
    cmd.load_cgo(cgo, 'contacts_solid')
python end
{dashed_block}
zoom obj, complete=1, buffer=5
{_render_tail(out_png)}"""


def render_pairwise_pymol(
    pdb_file: Path,
    out_png: Path,
    pairs_df: pd.DataFrame,
    pymol_exe: str = "pymol",
    *,
    frst_col: str | None = None,
    state_col: str | None = None,
    thr_idx: int = 1,
) -> bool:
    """Render structure with Cα-contact CGO/distance lines via PyMOL headless.

    Direct contacts (short/long-range) → solid CGO cylinders.
    Water-mediated contacts → dashed PyMOL distance objects.
    Colors match _FRUST_RGB: dark-red for highly, dark-green for minimally.
    Neutral contacts are omitted.  Returns True on success.
    frst_col / state_col override which DataFrame columns to use (useful for
    rendering true vs pred panels separately).
    """
    import shutil, tempfile

    if shutil.which(pymol_exe) is None:
        return False

    # Resolve state / frustration columns
    if state_col is None or state_col not in pairs_df.columns:
        state_col = _get_state_col(pairs_df)
    if frst_col is None or frst_col not in pairs_df.columns:
        try:
            frst_col = _get_frst_col(pairs_df)
        except KeyError:
            frst_col = None

    # Chain columns
    chain1_col = next((c for c in ("Chain1", "ChainRes1") if c in pairs_df.columns), None)
    chain2_col = next((c for c in ("Chain2", "ChainRes2") if c in pairs_df.columns), None)
    has_chain  = chain1_col is not None and chain2_col is not None

    # Welltype column → solid (direct) vs dashed (water-mediated)
    wt_col = next((c for c in ("Welltype", "WellType", "well_type")
                   if c in pairs_df.columns), None)

    def _dashed(row) -> bool:
        if wt_col is None:
            return False
        return "water" in str(row[wt_col]).lower()

    highly_solid:  list = []
    highly_dashed: list = []
    min_solid:     list = []
    min_dashed:    list = []

    if state_col:
        states = pairs_df[state_col].astype(str).str.strip().str.lower()
        for (_, row), st in zip(pairs_df.iterrows(), states):
            if not (st.startswith("high") or st.startswith("min")):
                continue   # skip neutral
            r1, r2 = int(row["Res1"]), int(row["Res2"])
            c1 = str(row[chain1_col]) if has_chain else ""
            c2 = str(row[chain2_col]) if has_chain else ""
            is_dsh = _dashed(row)
            if st.startswith("high"):
                (highly_dashed if is_dsh else highly_solid).append((r1, c1, r2, c2))
            else:
                (min_dashed if is_dsh else min_solid).append((r1, c1, r2, c2))
    elif frst_col:
        frst = pairs_df[frst_col].astype(float)
        for (_, row), fv in zip(pairs_df.iterrows(), frst):
            r1, r2 = int(row["Res1"]), int(row["Res2"])
            c1 = str(row[chain1_col]) if has_chain else ""
            c2 = str(row[chain2_col]) if has_chain else ""
            is_dsh = _dashed(row)
            cls = int(classify_frustration(torch.tensor([fv]), thr_idx).item())
            if cls == 0:
                (highly_dashed if is_dsh else highly_solid).append((r1, c1, r2, c2))
            elif cls == 2:
                (min_dashed if is_dsh else min_solid).append((r1, c1, r2, c2))

    tmp = Path(tempfile.mkdtemp())
    script = _pymol_script_pairwise(
        pdb_file, out_png,
        highly_solid, highly_dashed,
        min_solid, min_dashed,
    )
    return _run_pymol(pymol_exe, (tmp, script), out_png)


# ── combined per-protein figure ───────────────────────────────────────────────

def make_per_protein_figure(
    domain: str,
    frust_file: Path,
    pdb_file: Path,
    out_dir: Path,
    *,
    is_ml: bool = False,
    family: str | None = None,
    csv_dir: Path | None = None,
):
    """Generate and save the composite figure for *domain*.

    Structure panel order:
      Baseline: 1. Original Frustratometer  2. PyTorch Re-implementation (thresholded)
      ML model: 1. Original Frustratometer  2. Regression (Pred) (thresholded)
                3. Classification (Pred)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load frustration data ─────────────────────────────────────────────────
    try:
        df = load_singleresidue_file(frust_file)
    except Exception as e:
        print(f"  [SKIP] {domain}: cannot read {frust_file}: {e}")
        return

    has_pred = "PredFrstIndex" in df.columns

    true_fi  = df["TrueFrstIndex"].to_numpy(dtype=float)
    pred_fi  = df["PredFrstIndex"].to_numpy(dtype=float) if has_pred else None
    pred_cls = df["PredFrstState"].to_numpy() if "PredFrstState" in df.columns else np.ones(len(true_fi))
    residues = df["AA"].to_numpy(dtype=str) if "AA" in df.columns else np.array(["?"] * len(true_fi))
    res_ids  = df["Res"].to_numpy(dtype=int)

    pred_label = "Regression (Pred)" if is_ml else "PyTorch Re-implementation"
    true_label = "Original FrustratometeR" if has_pred else "PyTorch Frustratometer"

    # ── map pred_cls to int (0=MAX, 1=NEU, 2=MIN) ────────────────────────────
    pred_cls_int = _to_int_class(pred_cls)

    # ── active site lookup ────────────────────────────────────────────────────
    site_ids: list[int] = []
    site_x:   int | None = None
    site_lbl: str = ""

    # 1. CSV scan — works even when family is unknown (e.g. --domains without --family)
    if csv_dir:
        csv_result = _lookup_domain_in_any_csv(domain, csv_dir)
        if csv_result is not None:
            target_id, inferred_family = csv_result
            hits = np.where(res_ids == target_id)[0]
            if len(hits):
                site_x   = int(hits[0])
                site_ids = [target_id]
                site_lbl = FAMILY_ACTIVE_SITE.get(inferred_family, {}).get("label", "")
                print(f"    active site: {site_lbl}  res_id={target_id}", flush=True)

    # 2. Family-based number search fallback (no CSV or domain not in CSV)
    if not site_ids and family:
        site_cfg = FAMILY_ACTIVE_SITE.get(
            family.lower().replace(" ", "_").replace("-", "_"), {}
        )
        if site_cfg:
            sx = _find_active_site_idx(
                res_ids, residues, site_cfg["aa"], site_cfg["pos"],
                domain=domain, family=family, csv_dir=csv_dir,
            )
            if sx is not None:
                site_x   = sx
                site_lbl = site_cfg.get("label", "")
                site_ids = [int(res_ids[sx])]
                print(f"    active site: {site_lbl}  res_id={res_ids[sx]}", flush=True)

    # ── render structure panels via PyMOL ─────────────────────────────────────
    # Panel order: true (original) → pred regression → pred class (ML only)
    true_ok = pred_reg_ok = cls_ok = False
    if _RENDER_STRUCTURE:
        has_pdb      = pdb_file.exists()
        true_png     = out_dir / f"{domain}_struct_true.png"
        pred_reg_png = out_dir / f"{domain}_struct_pred_reg.png"
        cls_png      = out_dir / f"{domain}_struct_pred_cls.png"

        def _fi_cls(fi):
            return classify_frustration(
                torch.from_numpy(np.asarray(fi, dtype=np.float32)), 0).numpy()

        if has_pdb:
            true_ok = render_structure_pymol(pdb_file, true_png, res_ids, _fi_cls(true_fi),
                                             mode="class", sticks_residues=site_ids or None)
            if not true_ok:
                print(f"  [WARN] {domain}: PyMOL render failed — structure panels omitted")
            if true_ok and pred_fi is not None:
                pred_reg_ok = render_structure_pymol(pdb_file, pred_reg_png, res_ids,
                                                      _fi_cls(pred_fi), mode="class",
                                                      sticks_residues=site_ids or None)
            if is_ml and true_ok:
                cls_ok = render_structure_pymol(pdb_file, cls_png, res_ids,
                                                pred_cls_int, mode="class",
                                                sticks_residues=site_ids or None)

    # ordered list of (png, title) for structure panels — all use class colouring
    struct_panels = []
    if true_ok:
        struct_panels.append((true_png,     true_label))
    if pred_reg_ok:
        struct_panels.append((pred_reg_png, pred_label))
    if cls_ok:
        struct_panels.append((cls_png,      "Classification (Pred)"))

    # ── figure layout ─────────────────────────────────────────────────────────
    import math
    n_line_rows  = max(1, math.ceil(len(true_fi) / MAX_RESIDUES_PER_ROW))
    n_struct     = len(struct_panels)
    n_cols       = 1 + n_struct
    width_ratios = [5] + [3] * n_struct
    fig_w        = 7 + 5 * n_struct
    fig_h        = 7 * n_line_rows
    fig          = plt.figure(figsize=(fig_w, fig_h))
    gs           = gridspec.GridSpec(n_line_rows, n_cols,
                                     width_ratios=width_ratios, wspace=0.1)

    if n_line_rows == 1:
        line_axes   = fig.add_subplot(gs[0, 0])
        struct_axes = [fig.add_subplot(gs[0, i + 1]) for i in range(n_struct)]
    else:
        line_axes   = [fig.add_subplot(gs[i, 0]) for i in range(n_line_rows)]
        struct_axes = [fig.add_subplot(gs[:, i + 1]) for i in range(n_struct)]

    plot_single_frustration(
        line_axes, true_fi, pred_fi, pred_cls,
        residues, res_ids, domain, pred_label=pred_label,
        true_label=true_label,
        show_class_strip=is_ml,
        active_site_x=site_x,
        active_site_label=site_lbl,
    )

    for ax, (png, title) in zip(struct_axes, struct_panels):
        show_structure_panel(ax, png, title=title, mode="class", show_legend=False)

    # shared class legend centered under structure panels
    if struct_axes:
        positions  = [ax.get_position() for ax in struct_axes]
        legend_cx  = (positions[0].x0 + positions[-1].x1) / 2
        class_legend = [
            Patch(facecolor=_FRUST_RGB["MAX"], label="Highly frustrated"),
            Patch(facecolor=_FRUST_RGB["NEU"], label="Neutral"),
            Patch(facecolor=_FRUST_RGB["MIN"], label="Minimally frustrated"),
        ]
        fig.legend(handles=class_legend, fontsize=13, frameon=False,
                   loc="lower center", ncol=3,
                   bbox_to_anchor=(legend_cx, 0.0),
                   bbox_transform=fig.transFigure,
                   handlelength=2.5, handleheight=1.5)

    fig.suptitle(domain, fontsize=13, weight="bold")

    out_path = out_dir / f"{domain}_per_protein.png"
    _savefig(fig, out_path)


# ── pairwise contact-map figure ───────────────────────────────────────────────

def make_pairwise_figure(
    domain: str,
    frust_file: Path,
    pdb_file: Path,
    out_dir: Path,
    *,
    mode: str = "mutational",
    is_ml: bool = False,
    true_label: str | None = None,
    pred_label: str | None = None,
) -> None:
    """Generate and save a pairwise (mutational or configurational) contact-map figure.

    Panels:
      Left  — NxN heatmap coloured by frustration index + per-residue mean on top
      Right — 3D cartoon coloured by frustration class with CGO edges (PyMOL)

    true_label/pred_label override the default guess (which assumes "true" is
    the original R-based FrustratometeR whenever a pred column is present --
    wrong whenever "true" is actually PyTorch Frustratometer's own classic
    computation, e.g. compared against an unrelated ML model's prediction).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_idx = 2 if mode == "smooth" else 1   # smooth uses the inverted threshold row

    try:
        df = load_pairwise_file(frust_file)
    except Exception as e:
        print(f"  [SKIP] {domain} ({mode}): cannot read {frust_file}: {e}")
        return

    if "Res1" not in df.columns or "Res2" not in df.columns:
        print(f"  [SKIP] {domain} ({mode}): Res1/Res2 missing in {frust_file.name}")
        return

    all_ids = np.unique(
        np.concatenate([df["Res1"].to_numpy(dtype=int),
                        df["Res2"].to_numpy(dtype=int)])
    )
    has_pred   = "PredFrstIndex" in df.columns and "TrueFrstIndex" in df.columns
    true_col   = "TrueFrstIndex" if "TrueFrstIndex" in df.columns else None
    matrix     = _pairwise_to_matrix(df, all_ids, col=true_col)
    matrix_pred = (_pairwise_to_matrix(df, all_ids, col="PredFrstIndex")
                   if has_pred else None)
    if pred_label is None:
        pred_label = "Regression (Pred)" if is_ml else "PyTorch Re-implementation"

    # If the "true" matrix is all-NaN (e.g. original Frustratometer pairs don't
    # align with graph edges) but the pred matrix has data, fall back to single display.
    if true_label is None:
        true_label = "Original FrustratometeR" if has_pred else "PyTorch Frustratometer"
    if matrix_pred is not None and not np.any(np.isfinite(matrix)):
        print(f"  [WARN] {domain} ({mode}): TrueFrstIndex has no finite values "
              f"— using PredFrstIndex as sole display", flush=True)
        matrix = matrix_pred
        matrix_pred = None
        has_pred = False
        true_label = pred_label   # label correctly — this IS the pred data

    # ── render PyMOL structure panels ────────────────────────────────────────
    has_pdb = pdb_file.exists()
    has_true_col = "TrueFrstIndex" in df.columns
    has_pred_col = "PredFrstIndex" in df.columns

    has_class_col = is_ml and "ClassFrstState" in df.columns

    true_ok = pred_ok = cls_ok = False
    if _RENDER_STRUCTURE:
        true_png = out_dir / f"{domain}_{mode}_struct_true.png"
        pred_png = out_dir / f"{domain}_{mode}_struct_pred.png"
        cls_png  = out_dir / f"{domain}_{mode}_struct_cls.png"

        if has_pdb:
            if has_true_col:
                true_ok = render_pairwise_pymol(
                    pdb_file, true_png, df,
                    state_col=("TrueFrstState" if "TrueFrstState" in df.columns else None),
                    frst_col="TrueFrstIndex", thr_idx=thr_idx,
                )
                if not true_ok:
                    print(f"  [WARN] {domain}: PyMOL true pairwise render failed", flush=True)
            if has_pred_col:
                pred_state = next(
                    (c for c in ("FrstState", "PredFrstState") if c in df.columns), None
                )
                pred_ok = render_pairwise_pymol(
                    pdb_file, pred_png, df,
                    state_col=pred_state,
                    frst_col="PredFrstIndex", thr_idx=thr_idx,
                )
                if not pred_ok:
                    print(f"  [WARN] {domain}: PyMOL pred pairwise render failed", flush=True)
            if has_class_col:
                cls_ok = render_pairwise_pymol(
                    pdb_file, cls_png, df,
                    state_col="ClassFrstState",
                    frst_col=None,
                )
                if not cls_ok:
                    print(f"  [WARN] {domain}: PyMOL classification pairwise render failed", flush=True)
            # Fallback: single render when only one index column available
            if not has_true_col and not has_pred_col:
                single_png = out_dir / f"{domain}_{mode}_struct.png"
                true_ok = render_pairwise_pymol(pdb_file, single_png, df, thr_idx=thr_idx)
                if true_ok:
                    true_png = single_png

    struct_panels = []
    if true_ok:
        struct_panels.append((true_png, true_label))
    if pred_ok:
        struct_panels.append((pred_png, pred_label))
    if cls_ok:
        struct_panels.append((cls_png, "Classification (Pred)"))

    # ── figure layout ─────────────────────────────────────────────────────────
    n_struct = len(struct_panels)
    n_cols   = 1 + n_struct
    N        = len(all_ids)
    fig_w    = 8 + 5 * n_struct
    fig_h    = min(max(6.0, N * 0.06 + 2.0), 16.0)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(1, n_cols,
                            width_ratios=[3] + [2] * n_struct,
                            wspace=0.15)
    ax_map = fig.add_subplot(gs[0, 0])

    # ax_top is appended directly above ax_map with pad=0 — guaranteed zero gap
    divider = make_axes_locatable(ax_map)
    ax_top  = divider.append_axes("top", size="30%", pad=0)
    ax_top.sharex(ax_map)

    struct_axes = [fig.add_subplot(gs[0, i + 1]) for i in range(n_struct)]

    plot_contact_map(ax_top, ax_map, matrix, all_ids,
                     title=f"{domain} ({mode})",
                     matrix_pred=matrix_pred, pred_label=pred_label,
                     true_label=true_label, mode=thr_idx)

    for ax, (png, title) in zip(struct_axes, struct_panels):
        show_structure_panel(ax, png, title=title, mode="pairwise", show_legend=False,
                             title_fontsize=13)

    # Centered class legend (3 classes including neutral) just below struct panels
    if struct_axes:
        fig.canvas.draw()
        positions  = [ax.get_position() for ax in struct_axes]
        legend_cx  = (positions[0].x0 + positions[-1].x1) / 2
        legend_y   = min(p.y0 for p in positions) - 0.03
        class_legend = [
            Patch(facecolor=_FRUST_RGB["MAX"], label="Highly frustrated"),
            Patch(facecolor=_FRUST_RGB["NEU"], label="Neutral"),
            Patch(facecolor=_FRUST_RGB["MIN"], label="Minimally frustrated"),
        ]
        fig.legend(handles=class_legend, fontsize=13, frameon=False,
                   loc="upper center", ncol=3,
                   bbox_to_anchor=(legend_cx, legend_y),
                   bbox_transform=fig.transFigure,
                   handlelength=2.5, handleheight=1.5)

    out_path = out_dir / f"{domain}_{mode}.png"
    _savefig(fig, out_path)


# ── combined family figure (rows A / B / C stacked) ──────────────────────────

def make_family_figure(
    family: str,
    reps: list[dict],
    out_dir: Path,
    *,
    is_ml: bool = False,
    csv_dir: Path | None = None,
) -> None:
    """Single figure with one row per representative (A=bottom, B=median, C=top).

    Layout (matches FrustrAI-Seq Fig 3):
      col 0            : frustration line plot
      col 1            : FrustratometeR structure
      col 2            : Regression structure
      col 3 (ML only)  : Classification structure

    Column headers appear once on the top row.
    Circled tier labels (A/B/C) appear left of each line plot.
    Tier descriptions (Bottom 5% / Average / Top 5%) appear to the right.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_label = "Regression (Pred)" if is_ml else "PyTorch Re-implementation"

    # ── load data and render structures for all reps ──────────────────────────
    rows = []
    for rep in reps:
        domain    = rep["domain"]
        frust_f   = rep.get("frust_file")
        pdb_f     = rep.get("pdb_file")

        if not frust_f or not Path(frust_f).exists():
            print(f"  [SKIP] {domain}: frust file missing")
            continue
        try:
            df = load_singleresidue_file(Path(frust_f))
        except Exception as e:
            print(f"  [SKIP] {domain}: {e}")
            continue
        if "PredFrstIndex" not in df.columns:
            print(f"  [SKIP] {domain}: no PredFrstIndex")
            continue

        true_fi  = df["TrueFrstIndex"].to_numpy(dtype=float)
        pred_fi  = df["PredFrstIndex"].to_numpy(dtype=float)
        pred_cls = (df["PredFrstState"].to_numpy()
                    if "PredFrstState" in df.columns else np.ones(len(true_fi)))
        residues = (df["AA"].to_numpy(dtype=str)
                    if "AA" in df.columns else np.array(["?"] * len(true_fi)))
        res_ids  = df["Res"].to_numpy(dtype=int)
        pred_cls_int = _to_int_class(pred_cls)

        # ── active site lookup ─────────────────────────────────────────────────
        site_cfg = FAMILY_ACTIVE_SITE.get(
            family.lower().replace(" ", "_").replace("-", "_"), {}
        )
        site_ids: list[int] = []
        site_x:   int | None = None
        site_lbl: str = ""
        if site_cfg:
            sx = _find_active_site_idx(
                res_ids, residues, site_cfg["aa"], site_cfg["pos"],
                domain=domain, family=family, csv_dir=csv_dir,
            )
            if sx is not None:
                site_x   = sx
                site_lbl = site_cfg.get("label", "")
                site_ids = [int(res_ids[sx])]
                print(f"    active site: {site_lbl}  res_id={res_ids[sx]}", flush=True)

        pdb = Path(pdb_f) if pdb_f else None
        true_png     = out_dir / f"{domain}_struct_true.png"
        pred_reg_png = out_dir / f"{domain}_struct_pred_reg.png"
        cls_png      = out_dir / f"{domain}_struct_pred_cls.png"

        def _fi_cls(fi):
            return classify_frustration(
                torch.from_numpy(np.asarray(fi, dtype=np.float32)), 0).numpy()

        true_ok = pred_reg_ok = cls_ok = False
        if pdb and pdb.exists():
            true_ok = render_structure_pymol(pdb, true_png, res_ids, _fi_cls(true_fi),
                                             mode="class", sticks_residues=site_ids or None)
            if true_ok:
                pred_reg_ok = render_structure_pymol(pdb, pred_reg_png, res_ids,
                                                      _fi_cls(pred_fi), mode="class",
                                                      sticks_residues=site_ids or None)
            if is_ml and true_ok:
                cls_ok = render_structure_pymol(pdb, cls_png, res_ids,
                                                pred_cls_int, mode="class",
                                                sticks_residues=site_ids or None)

        rows.append(dict(
            rep          = rep,
            true_fi      = true_fi,  pred_fi=pred_fi,
            pred_cls     = pred_cls, pred_cls_int=pred_cls_int,
            residues     = residues, res_ids=res_ids,
            true_png     = true_png     if true_ok     else None,
            pred_reg_png = pred_reg_png if pred_reg_ok else None,
            cls_png      = cls_png      if cls_ok      else None,
            site_x       = site_x,
            site_lbl     = site_lbl,
        ))

    if not rows:
        print(f"  [SKIP] {family}: no rows to plot")
        return

    # ── determine which structure columns to show ─────────────────────────────
    struct_cols: list[tuple[str, str, str]] = []   # (row_key, col_header, mode)
    if any(r["true_png"] for r in rows):
        struct_cols.append(("true_png",     "FrustratometeR",  "regression"))
    if any(r["pred_reg_png"] for r in rows):
        struct_cols.append(("pred_reg_png", "Regression",      "regression"))
    if any(r["cls_png"] for r in rows):
        struct_cols.append(("cls_png",      "Classification",  "class"))

    n_rows   = len(rows)
    n_struct = len(struct_cols)
    n_cols   = 1 + n_struct

    fig_w = 7 + 3.0 * n_struct
    fig_h = 4.5 * n_rows
    fig   = plt.figure(figsize=(fig_w, fig_h))
    gs    = gridspec.GridSpec(
        n_rows, n_cols,
        width_ratios=[3] + [1.4] * n_struct,
        hspace=0.55, wspace=0.25,
    )

    for row_idx, rd in enumerate(rows):
        rep = rd["rep"]
        ax_line = fig.add_subplot(gs[row_idx, 0])

        # ── line plot ─────────────────────────────────────────────────────────
        title_parts = [rep["domain"]]
        if rep.get("cos_sim") is not None:
            title_parts.append(f"cos_sim={rep['cos_sim']:.3f}")
        if rep.get("hamming") is not None:
            title_parts.append(f"hamming={rep['hamming']:.3f}")
        ax_line.set_title(" | ".join(title_parts), fontsize=8, pad=3)

        def _thresh_family(v):
            return int(classify_frustration(torch.tensor([float(v)]), 0).item())

        true_colors  = [_color_from_class(_thresh_family(v)) for v in rd["true_fi"]]
        pred_colors  = [_color_from_class(_thresh_family(v)) for v in rd["pred_fi"]]
        class_colors = [_color_from_class(c) for c in rd["pred_cls"]]
        x = np.arange(len(rd["true_fi"]))

        _draw_frustration_segment(
            ax_line, x,
            rd["true_fi"], rd["pred_fi"],
            true_colors, pred_colors, class_colors,
            rd["residues"],
            rd["res_ids"],
            pred_label  = pred_label,
            mode        = 0,
            show_legend = (row_idx == 0),
            show_ylabel     = True,
            active_site_x   = rd.get("site_x"),
            active_site_label = rd.get("site_lbl", ""),
        )

        # circled tier label (A / B / C) left of line plot
        ax_line.annotate(
            rep["tier"],
            xy=(-0.08, 0.5), xycoords="axes fraction",
            fontsize=13, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", lw=1.5),
        )

        # tier description right of line plot
        ax_line.annotate(
            rep.get("tier_desc", ""),
            xy=(1.01, 0.5), xycoords="axes fraction",
            fontsize=9, ha="left", va="center", style="italic",
        )

        # ── structure panels ──────────────────────────────────────────────────
        for col_idx, (key, col_header, mode) in enumerate(struct_cols):
            ax_s = fig.add_subplot(gs[row_idx, col_idx + 1])
            png  = rd[key]

            # column header only on first row
            title = col_header if row_idx == 0 else ""

            if png and Path(png).exists():
                show_structure_panel(ax_s, png, title=title,
                                     mode=mode, show_legend=False)
            else:
                ax_s.axis("off")
                if title:
                    ax_s.set_title(title, fontsize=10)

    fig.suptitle(family.replace("_", " ").title(), fontsize=13, weight="bold", y=1.01)

    out_path = out_dir / f"{family}_representatives.png"
    _savefig(fig, out_path)
    plt.close(fig)


# ── test-set family discovery ─────────────────────────────────────────────────

def find_family_domains(data_dir: Path, families: list[str],
                        test_csv: Path | None = None,
                        domain_list: Path | None = None) -> list[str]:
    """Return CATH domain IDs matching the requested families.

    Domain pool (in priority order):
      1. domain_list — text file with one PDB path per line (e.g. domains.txt);
         domain IDs are extracted from the path stems.
      2. test.csv in data_dir — full CATH-annotated test split.

    Family filtering always uses test.csv for CATH annotations (C/A/T columns).
    If test.csv lacks C/A/T columns all domains in the pool are returned.
    """
    # ── Build pool of available domain IDs ───────────────────────────────
    pool: set[str] | None = None
    if domain_list is not None and Path(domain_list).exists():
        pool = set()
        for line in Path(domain_list).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            stem = Path(line).stem          # e.g. "3be3A00" from "/…/3be3A00.pdb"
            # strip any suffix after the 8-char CATH ID (e.g. ".frustratometer3")
            stem = stem.split(".")[0]
            pool.add(stem)
        print(f"  Domain pool: {len(pool)} domains from {domain_list}")

    # ── Load CATH annotation ──────────────────────────────────────────────
    if test_csv is None:
        test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        if pool is not None:
            print(f"  [WARN] test.csv not found at {test_csv} — "
                  f"returning all {len(pool)} pool domains (no family filter)")
            return sorted(pool)
        raise FileNotFoundError(
            f"test.csv not found at {test_csv}.\n"
            f"Pass --domain_list domains.txt or --test_csv /path/to/test.csv."
        )

    df = pd.read_csv(test_csv)
    print(f"  test.csv: {len(df)} rows, columns: {list(df.columns)}")
    if pool is not None:
        before = len(df)
        df = df[df["cath_domain"].isin(pool)]
        print(f"  After pool filter: {len(df)}/{before} rows "
              f"(pool sample: {sorted(pool)[:3]}…, "
              f"csv sample: {df['cath_domain'].tolist()[:3]})")

    target_codes = set()
    for fam in families:
        code = FAMILY_CATH.get(fam.lower().replace("-", "_").replace(" ", "_"))
        if code is None:
            print(f"  [WARN] unknown family '{fam}' — choose from: {list(FAMILY_CATH)}")
            continue
        target_codes.add(code)
    print(f"  Target CATH codes: {target_codes}")

    # build per-row CATH topology string; if columns absent skip family filter
    if not {"C", "A", "T"}.issubset(df.columns):
        print(f"  [WARN] test.csv has no C/A/T columns — returning all pool domains")
        domains = df["cath_domain"].tolist()
    else:
        parts = [df["C"].astype(str), df["A"].astype(str), df["T"].astype(str)]
        if "H" in df.columns:
            parts.append(df["H"].astype(str))
        cath_str = parts[0]
        for p in parts[1:]:
            cath_str = cath_str + "." + p
        print(f"  CATH string sample: {cath_str.tolist()[:3]}")
        matching = df[cath_str.isin(target_codes)]
        domains  = matching["cath_domain"].tolist()

    print(f"  Found {len(domains)} domains for families {families}")
    return domains


def find_family_representatives(
    data_dir: Path,
    results_csv: Path,
    families: list[str],
    *,
    bottom_pct: float = 5.0,
    top_pct: float = 5.0,
) -> list[dict]:
    """Return one representative per tier per family as a list of dicts.

    Each dict has keys: domain, family, tier, tier_desc, cos_sim, hamming.
      tier 'A' = bottom_pct%-ile  (lowest cosine similarity)
      tier 'B' = 50th percentile  (median)
      tier 'C' = (100-top_pct)%-ile  (highest cosine similarity)
    hamming is None when the column is absent from results_csv.
    """
    test_csv = data_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found in {data_dir}")
    test_df = pd.read_csv(test_csv)

    parts = [test_df["C"].astype(str), test_df["A"].astype(str), test_df["T"].astype(str)]
    if "H" in test_df.columns:
        parts.append(test_df["H"].astype(str))
    cath_str = parts[0]
    for p in parts[1:]:
        cath_str = cath_str + "." + p
    test_df = test_df.copy()
    test_df["_cath"] = cath_str

    res_df  = pd.read_csv(results_csv)
    cos_col = next(
        (c for c in ("cosine_similarity", "CosineSimilarity", "cosine_sim", "cos_sim")
         if c in res_df.columns),
        None,
    )
    if cos_col is None:
        raise KeyError(f"No cosine similarity column in {results_csv}. "
                       f"Columns: {list(res_df.columns)}")
    dom_col = next(
        (c for c in ("domain", "Domain", "cath_domain", "pdb", "name")
         if c in res_df.columns),
        None,
    )
    if dom_col is None:
        raise KeyError(f"No domain column in {results_csv}. "
                       f"Columns: {list(res_df.columns)}")
    ham_col = next(
        (c for c in ("hamming", "hamming_distance", "Hamming", "hamming_dist")
         if c in res_df.columns),
        None,
    )

    keep_cols = [dom_col, cos_col] + ([ham_col] if ham_col else [])
    merged = test_df.merge(
        res_df[keep_cols].rename(columns={dom_col: "cath_domain",
                                          cos_col:  "cos_sim",
                                          **({ham_col: "hamming"} if ham_col else {})}),
        on="cath_domain", how="inner",
    )

    tier_defs = [
        ("A", bottom_pct / 100.0,         TIER_DESCS["A"]),
        ("B", 0.50,                        TIER_DESCS["B"]),
        ("C", 1.0 - top_pct / 100.0,      TIER_DESCS["C"]),
    ]

    reps: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for fam in families:
        code = FAMILY_CATH.get(fam.lower().replace("-", "_").replace(" ", "_"))
        if code is None:
            print(f"  [WARN] unknown family '{fam}' — skipped")
            continue

        fam_df = merged[merged["_cath"] == code].dropna(subset=["cos_sim"])
        if fam_df.empty:
            print(f"  [WARN] no cosine-similarity data for family '{fam}'")
            continue

        for tier, q, tier_desc in tier_defs:
            target  = fam_df["cos_sim"].quantile(q)
            idx     = (fam_df["cos_sim"] - target).abs().idxmin()
            domain  = fam_df.loc[idx, "cath_domain"]
            cos     = float(fam_df.loc[idx, "cos_sim"])
            hamming = float(fam_df.loc[idx, "hamming"]) if ham_col else None
            key     = (domain, fam, tier)
            if key not in seen:
                reps.append(dict(domain=domain, family=fam, tier=tier,
                                 tier_desc=tier_desc, cos_sim=cos,
                                 hamming=hamming))
                seen.add(key)
            ham_str = f"  hamming={hamming:.3f}" if hamming is not None else ""
            print(f"  {fam:20s}  tier {tier} ({tier_desc:10s}): "
                  f"{domain}  cos_sim={cos:.4f}{ham_str}")

    return reps


# ── main ──────────────────────────────────────────────────────────────────────

def _find_frust_file(results_dir: Path, domain: str) -> Path | None:
    """Try common filename patterns for a domain's singleresidue output."""
    candidates = [
        results_dir / f"{domain}.frustratometer3.singleresidue",
        results_dir / f"singleresidue_results.{domain}.singleresidue.txt",
        results_dir / f"{domain}.frustratometer3_singleresidue",
        results_dir / f"{domain}_singleresidue.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_pdb_file(data_dir: Path, domain: str) -> Path | None:
    candidates = [
        data_dir / f"{domain}.frustratometer3.pdb",
        data_dir / f"{domain}.done" / "FrustrationData" / f"{domain}.pdb",
        data_dir / f"{domain}.pdb",
        data_dir / f"{domain}.cif",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_pairwise_file(results_dir: Path, domain: str, mode: str) -> Path | None:
    """Try common filename patterns for a domain's mutational/configurational output."""
    candidates = [
        results_dir / f"{domain}.frustratometer3.{mode}",
        results_dir / f"{mode}_results.{domain}.{mode}.txt",
        results_dir / f"{domain}.frustratometer3_{mode}",
        results_dir / f"{domain}_{mode}.txt",
        results_dir / f"{domain}.{mode}",
        results_dir / f"{domain}_{mode}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing frustration output files")
    parser.add_argument("--data_dir", required=True,
                        help="Data root containing *.done/FrustrationData/*.pdb")
    parser.add_argument("--out", default="plots/per_protein",
                        help="Output directory for PNGs")
    parser.add_argument("--is_ml", action="store_true",
                        help="Show predicted-class structure panel (ML model only)")
    parser.add_argument("--domains", nargs="+", default=[],
                        help="Explicit list of CATH domain IDs")
    parser.add_argument("--family", nargs="+", default=[],
                        choices=list(FAMILY_CATH.keys()),
                        help="Test-set families to include "
                             f"({', '.join(FAMILY_CATH.keys())})")
    parser.add_argument("--domain_list",
                        help="Text file with one PDB path per line (e.g. domains.txt); "
                             "used as the domain pool when --family is given, "
                             "instead of scanning test.csv for all test domains")
    parser.add_argument("--test_csv",
                        help="Path to test.csv with CATH annotations (default: "
                             "<data_dir>/test.csv)")
    parser.add_argument("--cosine_csv",
                        help="CSV with per-domain cosine similarity scores; "
                             "when given with --family, selects one representative "
                             "per tier (A=bottom, B=median, C=top) instead of all domains")
    parser.add_argument("--percentile", type=float, default=5.0,
                        help="Percentile cutoff for bottom/top tiers (default: 5)")
    parser.add_argument("--active_site_csv_dir",
                        help="Directory containing <family>_active_site.csv files "
                             "(from find_active_site.py); used for per-domain structural "
                             "active-site lookup instead of number-based search")
    parser.add_argument("--mode", nargs="+",
                        choices=["singleresidue", "mutational",
                                 "configurational", "all"],
                        default=["singleresidue"],
                        help="Frustration mode(s) to plot (default: singleresidue)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data_dir    = Path(args.data_dir)
    out_dir     = Path(args.out)
    csv_dir     = Path(args.active_site_csv_dir) if args.active_site_csv_dir else None

    modes: set[str] = set(args.mode)
    if "all" in modes:
        modes = {"singleresidue", "mutational", "configurational"}

    # ── build domain → output-subdir mapping ─────────────────────────────────
    # domain_jobs: list of (domain, per_domain_out_dir)
    domain_jobs: list[tuple[str, Path, str | None]] = []

    for d in args.domains:
        domain_jobs.append((d, out_dir, None))

    if args.family:
        if args.cosine_csv:
            # percentile representatives → combined family figure (one per family)
            reps = find_family_representatives(
                data_dir, Path(args.cosine_csv), args.family,
                bottom_pct=args.percentile,
                top_pct=args.percentile,
            )
            # enrich reps with file paths needed by make_family_figure / pairwise
            for rep in reps:
                rep["frust_file"] = _find_frust_file(results_dir, rep["domain"])
                rep["pdb_file"]   = _find_pdb_file(data_dir, rep["domain"])

            # singleresidue → one combined figure per family
            if "singleresidue" in modes:
                from itertools import groupby as _groupby
                sorted_reps = sorted(reps, key=lambda r: (r["family"], r["tier"]))
                for fam, fam_iter in _groupby(sorted_reps, key=lambda r: r["family"]):
                    fam_reps = list(fam_iter)
                    make_family_figure(fam, fam_reps,
                                       out_dir / fam, is_ml=args.is_ml,
                                       csv_dir=csv_dir)

            # pairwise modes → individual figures per domain in family/tier subdirs
            for rep in reps:
                domain_jobs.append((rep["domain"], out_dir / rep["family"] / rep["tier"],
                                    rep["family"]))
        else:
            for fam in args.family:
                for d in find_family_domains(
                    data_dir, [fam],
                    test_csv=Path(args.test_csv) if args.test_csv else None,
                    domain_list=Path(args.domain_list) if args.domain_list else None,
                ):
                    domain_jobs.append((d, out_dir / fam, fam))

    # deduplicate while preserving first-seen out_dir per domain
    seen_jobs: dict[str, tuple[Path, str | None]] = {}
    for domain, dout, fam in domain_jobs:
        if domain not in seen_jobs:
            seen_jobs[domain] = (dout, fam)
    domain_jobs = [(d, v[0], v[1]) for d, v in seen_jobs.items()]

    # singleresidue already handled by make_family_figure when cosine_csv used
    sr_handled_by_family = bool(args.family and args.cosine_csv
                                and "singleresidue" in modes)

    if not domain_jobs and not (args.family and args.cosine_csv):
        parser.error("Provide --domains and/or --family")

    print(f"Processing {len(domain_jobs)} domain(s), modes={sorted(modes)}", flush=True)
    ok, skip = 0, 0
    for domain, dout, domain_family in domain_jobs:
        print(f"[{domain}]  → {dout}", flush=True)
        pdb_file = _find_pdb_file(data_dir, domain)
        if pdb_file is None:
            print(f"  [WARN] no PDB found — structure panels will be omitted")

        domain_ok = False

        if "singleresidue" in modes and not sr_handled_by_family:
            ff = _find_frust_file(results_dir, domain)
            if ff is None:
                print(f"  [SKIP singleresidue] no file found in {results_dir}", flush=True)
            else:
                make_per_protein_figure(
                    domain, ff, pdb_file or Path("/dev/null"),
                    dout, is_ml=args.is_ml, family=domain_family,
                    csv_dir=csv_dir,
                )
                domain_ok = True

        for pmode in ("mutational", "configurational"):
            if pmode not in modes:
                continue
            ff = _find_pairwise_file(results_dir, domain, pmode)
            if ff is None:
                print(f"  [SKIP {pmode}] no file found in {results_dir}", flush=True)
                continue
            make_pairwise_figure(
                domain, ff, pdb_file or Path("/dev/null"),
                dout, mode=pmode, is_ml=args.is_ml,
            )
            domain_ok = True

        if domain_ok:
            ok += 1
        else:
            skip += 1

    print(f"\nDone: {ok} written, {skip} skipped → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
