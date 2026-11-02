"""
Inline the three per-mode KDE SVGs into frustration_modes_diagram.svg,
replacing <image href="kde_*.svg"> tags with actual SVG content.

Run from docs/: python inline_kde.py
"""

import re

MAIN_SVG  = "frustration_modes_diagram.svg"
SR_SVG    = "kde_singleresidue.svg"
MUT_SVG   = "kde_mutational.svg"
CONF_SVG  = "kde_configurational.svg"
OUT_SVG   = "frustration_modes_diagram.svg"

# ── target slots (px) — one slot per mode panel ───────────────────────────────
SR_SLOT   = dict(x=15,  y=663, w=310, h=165)
MUT_SLOT  = dict(x=355, y=663, w=310, h=165)
CONF_SLOT = dict(x=695, y=663, w=310, h=165)

MPLOT_STYLE = "*{stroke-linejoin: round; stroke-linecap: butt}"


def extract_inner(svg_text):
    """Return (viewbox_w, viewbox_h, inner_content_str)."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_text)
    vw, vh = float(m.group(1)), float(m.group(2))
    body = re.sub(r'<\?xml[^>]*\?>', '', svg_text)
    body = re.sub(r'<!DOCTYPE[^>]*>', '', body)
    body = re.sub(r'<svg[^>]*>', '', body, count=1)
    body = re.sub(r'</svg\s*>', '', body)
    body = re.sub(r'<metadata>.*?</metadata>', '', body, flags=re.DOTALL)
    body = re.sub(r'<defs>\s*<style[^>]*>[^<]*</style>\s*</defs>', '', body)
    return vw, vh, body.strip()


def fill_transform(vw, vh, slot):
    """Scale to fill full slot width, bottom-anchor content so x-axis labels are visible."""
    scale = slot['w'] / vw          # width-fills slot exactly
    content_h = vh * scale
    tx = slot['x']                  # no horizontal offset — fills edge-to-edge
    ty = slot['y'] + slot['h'] - content_h   # bottom-anchor: clips empty top margin only
    return f"translate({tx:.3f},{ty:.3f}) scale({scale:.6f})"


def prefix_ids(content, prefix):
    """Prefix all id= attributes and url(#...) / href="#..." references."""
    content = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}_{m.group(1)}"', content)
    content = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}_{m.group(1)})', content)
    content = re.sub(r'xlink:href="#([^"]+)"', lambda m: f'xlink:href="#{prefix}_{m.group(1)}"', content)
    content = re.sub(r'(?<!xlink:)href="#([^"]+)"', lambda m: f'href="#{prefix}_{m.group(1)}"', content)
    return content


def main():
    with open(MAIN_SVG) as f:
        main = f.read()
    with open(SR_SVG) as f:
        sr_raw = f.read()
    with open(MUT_SVG) as f:
        mut_raw = f.read()
    with open(CONF_SVG) as f:
        conf_raw = f.read()

    sr_vw,   sr_vh,   sr_inner   = extract_inner(sr_raw)
    mut_vw,  mut_vh,  mut_inner  = extract_inner(mut_raw)
    conf_vw, conf_vh, conf_inner = extract_inner(conf_raw)

    sr_inner   = prefix_ids(sr_inner,   "sr")
    mut_inner  = prefix_ids(mut_inner,  "mut")
    conf_inner = prefix_ids(conf_inner, "conf")

    sr_tf   = fill_transform(sr_vw,   sr_vh,   SR_SLOT)
    mut_tf  = fill_transform(mut_vw,  mut_vh,  MUT_SLOT)
    conf_tf = fill_transform(conf_vw, conf_vh, CONF_SLOT)

    def kde_block(name, prefix, slot, tf, inner):
        clip_id = f"{prefix}_slot_clip"
        clip = (f'<clipPath id="{clip_id}">'
                f'<rect x="{slot["x"]}" y="{slot["y"]}" '
                f'width="{slot["w"]}" height="{slot["h"]}"/></clipPath>')
        return (f'<!-- kde_{name} inlined -->\n'
                f'<defs>{clip}</defs>\n'
                f'<g clip-path="url(#{clip_id})">'
                f'<g transform="{tf}">\n{inner}\n</g></g>')

    sr_block   = kde_block("singleresidue",   "sr",   SR_SLOT,   sr_tf,   sr_inner)
    mut_block  = kde_block("mutational",      "mut",  MUT_SLOT,  mut_tf,  mut_inner)
    conf_block = kde_block("configurational", "conf", CONF_SLOT, conf_tf, conf_inner)

    if 'xmlns:xlink' not in main:
        main = main.replace(
            '<svg xmlns="http://www.w3.org/2000/svg"',
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"',
            1
        )

    if MPLOT_STYLE not in main:
        main = main.replace(
            '<defs>',
            f'<defs>\n  <style type="text/css">{MPLOT_STYLE}</style>',
            1
        )

    main = re.sub(r'<image href="kde_singleresidue\.svg"[^/]*/>', sr_block, main)
    main = re.sub(r'<image href="kde_mutational\.svg"[^/]*/>', mut_block, main)
    main = re.sub(r'<image href="kde_configurational\.svg"[^/]*/>', conf_block, main)

    with open(OUT_SVG, 'w') as f:
        f.write(main)
    print(f"Written {OUT_SVG}")


if __name__ == "__main__":
    main()
