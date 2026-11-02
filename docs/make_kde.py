"""
Generate and inline a combined 3-panel KDE figure into frustration_modes_diagram.svg.

Replaces inline_kde.py + fig1_kde data loading.  All three mode KDEs share the same
x/y axis range; only the leftmost subplot carries a y-axis label.

Usage (from docs/):
    python make_kde.py [--results-dir PATH]
"""
import argparse, glob, io, os, random, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

DOCS_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = "/gpfs/projects/bsc08/bsc504628/frustration/frust_verify_results"
MAIN_SVG    = os.path.join(DOCS_DIR, "frustration_modes_diagram.svg")

# ── target slot (translate-group coordinates, i.e. SVG y = val + 24) ─────────
# Positioned close under the mode panels (which end at translate-y ≈ 595).
SLOT_X, SLOT_Y = 15, 611       # top-left corner
SLOT_W, SLOT_H = 990, 200      # spans all three mode panels (3×310 + 2×30 gaps)

# ── frustration thresholds & colours ─────────────────────────────────────────
LO_SR,   HI_SR   = -1.0,  0.55
LO_PAIR, HI_PAIR = -1.0,  0.78
COLOR_HF  = "#CC2222"
COLOR_NEU = "#BBBBBB"
COLOR_MF  = "#22AA44"
FONT      = "Helvetica Neue, Arial, sans-serif"
MPLOT_STYLE = "*{stroke-linejoin: round; stroke-linecap: butt}"

# ── FI equation (drawn AFTER KDE block → appears on top) ─────────────────────
# Positioned 20 px below slot top, centered in the 30 px gap between SR and MUT.
_FI_TOP = SLOT_Y + 20   # box top in translate-group coords
FI_SVG = f"""\
<!-- ── FI equation (overlapping KDE, between SR and MUT) ── -->
<rect x="270" y="{_FI_TOP}" width="140" height="56" rx="5"
      fill="white" fill-opacity="0.92" stroke="#AAA" stroke-width="1"/>
<text x="340" y="{_FI_TOP + 15}" text-anchor="middle"
      font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-style="italic" fill="#333">FI =</text>
<text x="340" y="{_FI_TOP + 31}" text-anchor="middle"
      font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-style="italic" fill="#333"
      ><tspan>E</tspan><tspan font-size="8.5" dy="3">native</tspan><tspan dy="-3"> &#x2212; &#x03BC;</tspan><tspan font-size="8.5" dy="3">decoy</tspan><tspan dy="-3"></tspan></text>
<line x1="279" y1="{_FI_TOP + 35}" x2="401" y2="{_FI_TOP + 35}" stroke="#555" stroke-width="1"/>
<text x="340" y="{_FI_TOP + 48}" text-anchor="middle"
      font-family="Helvetica Neue,Arial,sans-serif" font-size="12" font-style="italic" fill="#333"
      >&#x03C3;<tspan font-size="8.5" dy="3">decoy</tspan></text>"""


# ── data loading ──────────────────────────────────────────────────────────────

def load_true_fi(results_dir, suffix, max_files=2000, seed=42):
    import pandas as pd
    pattern = os.path.join(results_dir, f"*.frustratometer3.{suffix}")
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
    if not vals:
        raise ValueError(f"No TrueFrstIndex column found in {suffix} files")
    result = np.concatenate(vals)
    print(f"  {suffix}: {len(files)} files, {len(result):,} values")
    return result


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_kde(ax, values, lo, hi, xlim=(-5, 5), ylabel=False, xlabel=False):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v >= xlim[0]) & (v <= xlim[1])]

    if len(v) < 10:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return

    kde  = gaussian_kde(v, bw_method=0.15)
    x    = np.linspace(xlim[0], xlim[1], 600)
    y    = kde(x)
    ytop = float(y.max())

    ax.fill_between(x, y, where=(x < lo),              color=COLOR_HF,  alpha=0.75, linewidth=0)
    ax.fill_between(x, y, where=(x >= lo) & (x <= hi), color=COLOR_NEU, alpha=0.75, linewidth=0)
    ax.fill_between(x, y, where=(x > hi),              color=COLOR_MF,  alpha=0.75, linewidth=0)
    ax.plot(x, y, color="#222", linewidth=0.8)

    for thresh, label in [(lo, str(lo)), (hi, str(hi))]:
        ax.axvline(thresh, color="#555", linestyle="--", linewidth=0.8, alpha=0.85)
        ax.text(thresh, ytop * 1.05, label, ha="center", va="bottom",
                fontsize=6, fontweight="bold", color="#222", fontfamily=FONT,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

    ax.set_xlim(xlim)
    ax.set_ylim(0, ytop * 1.35)
    ax.tick_params(axis="both", labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylabel:
        ax.set_ylabel("Density", fontsize=7, labelpad=3, fontfamily=FONT)
    if xlabel:
        ax.set_xlabel("Frustration Index", fontsize=7, labelpad=2, fontfamily=FONT)


def make_figure(sr_vals, mut_vals, conf_vals):
    """Return in-memory SVG string of the combined 3-panel KDE figure."""
    fig_w = SLOT_W / 72   # SVG uses 72 pt/inch
    fig_h = SLOT_H / 72

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
    )
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("white")

    # left=20/SLOT_W shifts ax1 right by 20pt so y-axis tick labels (which render at
    # ~18pt left of the spine) clear SVG x=0 (the viewport boundary).  With left=0
    # the labels land at SVG x≈-3 and are clipped by the viewport, not a clipPath.
    # 20pt is 6% of the 310pt SR panel — visually minor.
    fig.subplots_adjust(
        left   = 20 / SLOT_W,
        right  = 1,
        bottom = 42 / SLOT_H,
        top    = 1 - 10 / SLOT_H,
        wspace = 30 / 310,
    )

    ax1.tick_params(axis="y", direction="in", pad=2)

    # Remove y-axis spine and ticks on MUT and CONF (shared y with SR)
    for ax in (ax2, ax3):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)

    plot_kde(ax1, sr_vals,   LO_SR,   HI_SR,   ylabel=True,  xlabel=False)
    plot_kde(ax2, mut_vals,  LO_PAIR, HI_PAIR, ylabel=False, xlabel=True)
    plot_kde(ax3, conf_vals, LO_PAIR, HI_PAIR, ylabel=False, xlabel=False)

    buf = io.BytesIO()
    fig.savefig(buf, format="svg", facecolor="white")
    plt.close(fig)
    return buf.getvalue().decode("utf-8")


# ── SVG utilities ─────────────────────────────────────────────────────────────

def extract_inner(svg_text):
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_text)
    vw, vh = float(m.group(1)), float(m.group(2))
    body = re.sub(r'<\?xml[^>]*\?>', '', svg_text)
    body = re.sub(r'<!DOCTYPE[^>]*>', '', body)
    body = re.sub(r'<svg[^>]*>', '', body, count=1)
    body = re.sub(r'</svg\s*>', '', body)
    body = re.sub(r'<metadata>.*?</metadata>', '', body, flags=re.DOTALL)
    body = re.sub(r'<defs>\s*<style[^>]*>[^<]*</style>\s*</defs>', '', body)
    return vw, vh, body.strip()


def prefix_ids(content, prefix):
    content = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}_{m.group(1)}"', content)
    content = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}_{m.group(1)})', content)
    content = re.sub(r'xlink:href="#([^"]+)"', lambda m: f'xlink:href="#{prefix}_{m.group(1)}"', content)
    content = re.sub(r'(?<!xlink:)href="#([^"]+)"', lambda m: f'href="#{prefix}_{m.group(1)}"', content)
    return content


def meet_transform(vw, vh):
    """Scale to fill SLOT with meet (no clipping); figure is generated at exact size so scale≈1."""
    sx, sy = SLOT_W / vw, SLOT_H / vh
    scale = min(sx, sy)
    tx = SLOT_X + (SLOT_W - vw * scale) / 2
    ty = SLOT_Y + (SLOT_H - vh * scale) / 2
    return f"translate({tx:.3f},{ty:.3f}) scale({scale:.6f})"


def build_kde_block(kde_svg):
    vw, vh, inner = extract_inner(kde_svg)
    inner = prefix_ids(inner, "kde")

    # Matplotlib adds an internal clip-path covering the full figure [x=0..vw, y=0..vh].
    # With left=0 the y-axis tick labels render at figure x<0, so that clip hides them.
    # Extend any such rect leftward by 60pt so the labels come through.
    def _extend_clip(m):
        rx, ry, rw, rh = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
        if abs(rx) < 2 and abs(ry) < 2 and abs(rw - vw) < 3 and abs(rh - vh) < 3:
            return f'<rect x="-60" y="{ry}" width="{rw + 60:.3f}" height="{rh}"/>'
        return m.group(0)
    inner = re.sub(
        r'<rect x="([\d.-]+)" y="([\d.-]+)" width="([\d.]+)" height="([\d.]+)"/>',
        _extend_clip, inner
    )

    tf = meet_transform(vw, vh)
    clip_id = "kde_combined_clip"
    # Extend clip 50px left so y-axis tick labels on ax1 aren't cut off;
    # the SVG viewport itself clips anything beyond x=0.
    clip = (f'<clipPath id="{clip_id}">'
            f'<rect x="{SLOT_X - 50}" y="{SLOT_Y}" width="{SLOT_W + 50}" height="{SLOT_H}"/>'
            f'</clipPath>')
    return (
        f'<!-- kde_combined inlined -->\n'
        f'<defs>{clip}</defs>\n'
        f'<g clip-path="url(#{clip_id})">'
        f'<g transform="{tf}">\n{inner}\n</g></g>'
    )


def patch_svg(svg, kde_block):
    """Strip old KDE/FI content, update heights, rebuild in correct z-order."""

    # Ensure xmlns:xlink
    if 'xmlns:xlink' not in svg:
        svg = svg.replace(
            '<svg xmlns="http://www.w3.org/2000/svg"',
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"', 1)

    # Ensure matplotlib stroke style
    if MPLOT_STYLE not in svg:
        svg = svg.replace('<defs>', f'<defs>\n  <style type="text/css">{MPLOT_STYLE}</style>', 1)

    # Update SVG canvas and background rect height
    svg_h = SLOT_Y + 24 + SLOT_H + 20   # translate offset(24) + slot + bottom margin
    svg = re.sub(
        r'(width="1020" height=")[0-9]+(" viewBox="0 0 1020 )[0-9]+(")',
        lambda m: f'{m.group(1)}{svg_h}{m.group(2)}{svg_h}{m.group(3)}', svg)
    svg = re.sub(
        r'(<rect width="1020" height=")[0-9]+(")',
        lambda m: f'{m.group(1)}{svg_h}{m.group(2)}', svg)

    # Update "E" section label to sit just above the slot
    e_label_y = SLOT_Y - 3
    svg = re.sub(
        r'(<text x="12" y=")[0-9]+("(?:[^>]*)fill="#1A1A2E">E</text>)',
        lambda m: f'{m.group(1)}{e_label_y}{m.group(2)}',
        svg)

    # Find the E label end position and cut everything after it to the tail
    tail = '\n\n</g>\n</svg>'
    e_label_end = svg.find('fill="#1A1A2E">E</text>') + len('fill="#1A1A2E">E</text>')
    tail_pos = svg.rfind(tail)

    if e_label_end < 0 or tail_pos < 0:
        raise RuntimeError("Could not locate KDE section anchors in SVG")

    # Rebuild: KDE block (background z-order) then FI equation (foreground z-order)
    svg = svg[:e_label_end] + f'\n{kde_block}\n{FI_SVG}' + svg[tail_pos:]
    return svg


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Directory with *.frustratometer3.* result files")
    args = parser.parse_args()

    print(f"Loading data from {args.results_dir} …")
    try:
        sr_vals   = load_true_fi(args.results_dir, "singleresidue")
        mut_vals  = load_true_fi(args.results_dir, "mutational")
        conf_vals = load_true_fi(args.results_dir, "configurational")
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error loading data: {e}\nRun with --results-dir pointing to the frustration results.")

    print("Generating combined KDE figure …")
    kde_svg = make_figure(sr_vals, mut_vals, conf_vals)

    out_kde = os.path.join(DOCS_DIR, "kde_combined.svg")
    with open(out_kde, "w") as f:
        f.write(kde_svg)
    print(f"Saved {out_kde}")

    kde_block = build_kde_block(kde_svg)

    with open(MAIN_SVG) as f:
        svg = f.read()

    svg = patch_svg(svg, kde_block)

    with open(MAIN_SVG, "w") as f:
        f.write(svg)
    print(f"Updated {MAIN_SVG}")


if __name__ == "__main__":
    main()
