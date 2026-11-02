"""
Generate all paper figures for the PyTorch frustratometer re-implementation paper.

Figures produced
  fig1_schematic.png          — algorithmic workflow diagram
  fig2_scatter.png            — 1×3 hexbin scatter (all 3 modes)
  fig3_kdeplots.png           — 2×3 per-protein KDE (Pearson R + cosine/class-acc)
  fig4_speed.png              — 2-panel: absolute time + speedup vs protein length
  fig5_scaling.png            — log-log scaling with fitted power-law slope
  fig6_mutations.png          — mutation benchmark: scatter + KDE + (optional) per-sfam
  fig7_frustraevo.png         — FrustraEvo: IC agreement (panel A) + MSA timing (panel B)
  fig8_multichain.png         — 4Z66 nucleosome PyMOL image + per-mode scatter validation
  fig9_<family>.png           — per-protein representatives (bottom/median/top) for each protein family

Usage:
  python -m frustrai_fold.plots.paper_figures \\
      --results   baseline_comparison/results/run_info/all_results.csv \\
      --samples   baseline_comparison/results/run_info \\
      --mac       baseline_comparison/length_test/mac_timings.csv \\
      --mn5       baseline_comparison/length_test/mn5_timings.csv \\
      --mut-metrics baseline_comparison/mutants/plots/mutations_metrics.csv \\
      --mut-samples baseline_comparison/mutants \\
      --sfam-map  baseline_comparison/mutants/superfamily_map.csv \\
      --frustraevo-new-dir  baseline_comparison/FrustaEvo \\
      --frustraevo-orig-dir /path/to/seqdist3 \\
      --out       results/paper_figures
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns
from scipy.stats import linregress

from ..common import (
    PALETTE, MODE_THRESHOLDS, LO_SR, HI_SR, LO_PAIR, HI_PAIR,
    fi_class, compute_metrics, draw_kde, draw_confusion, draw_scatter_hexbin,
    draw_hexbin,
)
from ..mutations import (
    read_mutations_data, find_ref_mutations_dir, load_pred,
)
from .frustaevo_comparison import find_funfams, compare_funfam, collect_timings

MODES       = ["singleresidue", "mutational", "configurational"]
MODE_LABELS = {"singleresidue": "Single-residue",
               "mutational":    "Mutational",
               "configurational": "Configurational"}

# Domains dropped from every figure (FrustratometeR vs PyTorch disagreement with
# mean distance to the class boundary > 1 — see make_confusion.py --min-dist 1).
# --skip adds to this set.
EXCLUDE_DOMAINS = frozenset({"2vx2A01", "3bg3A02", "5facA02"})

TIER_DESCS = {"A": f"Bottom 5%", "B": "Median", "C": "Top 5%"}
_XLAB = "FrustratometeR"
_YLAB = "PyTorch Re-implementation"

sns.set_theme(style="whitegrid", font_scale=1.15)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _savefig(fig, path: Path, dpi: int = 600) -> None:
    """Save *fig* as PNG (600 dpi) and SVG."""
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def _drop_excluded(df: pd.DataFrame, exclude=EXCLUDE_DOMAINS) -> pd.DataFrame:
    """Drop rows for excluded domains from a per-protein/per-residue table.

    Uses the first of domain/stem/pdb columns, else the stem of 'structure'.
    """
    exclude = frozenset(exclude or ())
    if not exclude or df is None:
        return df
    for col in ("domain", "stem", "pdb"):
        if col in df.columns:
            return df[~df[col].astype(str).isin(exclude)]
    if "structure" in df.columns:
        dom = df["structure"].astype(str).map(lambda s: Path(s).name.split(".")[0])
        return df[~dom.isin(exclude)]
    return df


def _read_one_output_file(path_str: str):
    """Read one per-protein output file. Module-level so it is picklable.

    Returns (pred_list, true_list, gly_list, has_aa) or None on failure.
    """
    try:
        df = pd.read_csv(path_str, sep=" ", skipinitialspace=True, comment="#")
    except Exception:
        return None
    if "FrstIndex" not in df.columns or "TrueFrstIndex" not in df.columns:
        return None
    df["FrstIndex"]     = pd.to_numeric(df["FrstIndex"],     errors="coerce")
    df["TrueFrstIndex"] = pd.to_numeric(df["TrueFrstIndex"], errors="coerce")
    df = df.dropna(subset=["FrstIndex", "TrueFrstIndex"])
    # TrueFrstIndex==0 means FrustratometeR got NaN→0 via nan_to_num; drop those rows.
    # (PyTorch FrstIndex==0 is legitimate and kept.)
    df = df[df["TrueFrstIndex"] != 0.0]
    # Single-residue: "AA" column. Pair modes: "AA1"/"AA2" — GLY if either residue is GLY.
    if "AA" in df.columns:
        aa_upper = df["AA"].astype(str).str.upper()
        gly = aa_upper.isin(["GLY", "G"]).tolist()
        has_aa = True
    elif "AA1" in df.columns or "AA2" in df.columns:
        aa1 = df["AA1"].astype(str).str.upper() if "AA1" in df.columns else pd.Series(["X"] * len(df))
        aa2 = df["AA2"].astype(str).str.upper() if "AA2" in df.columns else pd.Series(["X"] * len(df))
        gly = (aa1.isin(["GLY", "G"]) | aa2.isin(["GLY", "G"])).tolist()
        has_aa = True
    else:
        gly = [False] * len(df)
        has_aa = False
    return df["FrstIndex"].tolist(), df["TrueFrstIndex"].tolist(), gly, has_aa


# FrustratometeR files read once (ref_dir joins) are kept here so the fr_rerun
# figure only has to read run 2 when --fr-run1 is the same tree as --ref-dir.
# Only key/FrstIndex/AA columns are kept to bound memory.
_FR_CACHE: dict = {}
_FR_CACHE_COLS = ("Res", "ChainRes", "Res1", "Res2", "ChainRes1", "ChainRes2",
                  "FrstIndex", "AA")


def _read_raw_output_file(path_str: str, cache: bool = False):
    """Read one output file without requiring TrueFrstIndex.

    Returns a DataFrame with FrstIndex + identifier columns, or None on failure
    (including a missing file — callers do not check existence first).
    FrstIndex==0 rows are kept (unlike _read_one_output_file which drops TrueFrstIndex==0).
    With cache=True the (slimmed) result is memoised in _FR_CACHE; treat it as read-only.
    """
    path_str = os.path.abspath(path_str)  # string-only normalisation, no stat
    if cache and path_str in _FR_CACHE:
        return _FR_CACHE[path_str]
    try:
        df = pd.read_csv(path_str, sep=" ", skipinitialspace=True, comment="#")
    except Exception:
        df = None
    if df is not None and "FrstIndex" in df.columns:
        df["FrstIndex"] = pd.to_numeric(df["FrstIndex"], errors="coerce")
        df = df.dropna(subset=["FrstIndex"])
        if cache:
            df = df[[c for c in df.columns if c in _FR_CACHE_COLS]]
    else:
        df = None
    if cache:
        _FR_CACHE[path_str] = df
    return df


def _iter_output_files(results_dir: Path, mode: str):
    """Yield paths to per-protein frustration output files for *mode*.

    Checks two layouts:
      • flat:  results_dir/{pdb}.frustratometer3.{mode}  (legacy)
      • done:  results_dir/{pdb}.done/FrustrationData/{pdb}.pdb_{mode}
    The done-subdir paths are yielded without checking they exist (that was
    one stat per protein on GPFS); readers return None for missing files.
    """
    suffixes = [f".frustratometer3.{mode}", f".pdb_{mode}"]
    # flat files directly in results_dir
    for suf in suffixes:
        for p in sorted(results_dir.glob(f"*{suf}")):
            if not p.name.endswith(f"_mutations{suf}"):
                yield p

    # done-subdir layout — no existence check
    for done_dir in sorted(results_dir.glob("*.done")):
        pdb_id = done_dir.stem
        yield done_dir / "FrustrationData" / f"{pdb_id}.pdb_{mode}"


def _load_output_files(samples_dir: Path, mode: str, exclude_domains: "frozenset[str] | None" = None):
    """Load pred/true/gly_mask directly from per-protein output files.

    Searches samples_dir itself first (new {pdb}.done/FrustrationData layout),
    then its parent (legacy flat layout).
    Files must have FrstIndex (pred) and TrueFrstIndex (true); AA is optional.
    Returns (pred, true, gly_mask) arrays, or (None, None, None) if nothing found.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    _excl = exclude_domains or frozenset()

    def _keep(p: Path) -> bool:
        return p.name.split(".")[0] not in _excl

    # Search samples_dir directly first (new output format), then parent (legacy)
    paths = [str(p) for p in _iter_output_files(samples_dir, mode) if _keep(p)]
    if not paths:
        paths = [str(p) for p in _iter_output_files(samples_dir.parent, mode) if _keep(p)]
    all_pred, all_true, all_gly = [], [], []
    has_aa = False
    n_files = 0

    if paths:
        n_workers = min(64, len(paths))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for result in pool.map(_read_one_output_file, paths):
                if result is None:
                    continue
                pred_l, true_l, gly_l, file_has_aa = result
                all_pred.extend(pred_l)
                all_true.extend(true_l)
                all_gly.extend(gly_l)
                has_aa = has_aa or file_has_aa
                n_files += 1

    print(f"  [output_files/{mode}] {n_files} files, {len(all_pred):,} residues, has_aa={has_aa}")
    if not all_pred:
        return None, None, None
    return (np.array(all_pred, dtype=float),
            np.array(all_true, dtype=float),
            np.array(all_gly, dtype=bool) if has_aa else None)


_PAIR_MODES = {"mutational", "configurational"}

# Join-key columns per mode (subset that must exist; ChainRes* are added when present)
_JOIN_KEYS = {
    "singleresidue":   ["Res"],
    "mutational":      ["Res1", "Res2"],
    "configurational": ["Res1", "Res2"],
}


def _build_file_index(results_dir: Path, mode: str) -> dict:
    """Return {stem: path} for all output files for *mode* found in *results_dir*."""
    idx = {}
    for p in _iter_output_files(results_dir, mode):
        # Stem is the part before the first dot
        stem = p.name.split(".")[0]
        idx[stem] = p
    return idx


def _load_output_files_with_ref(samples_dir: Path, ref_dir: Path, mode: str,
                                exclude_domains: "frozenset[str] | None" = None):
    """Load (pred, true, gly_mask) by joining our output files with a ref directory.

    pred = our FrstIndex, true = FrustratometeR FrstIndex from ref_dir.
    Residues are matched on (Res[, ChainRes]) for singleresidue or
    (Res1, Res2[, ChainRes1, ChainRes2]) for pair modes.
    Falls back to positional alignment when identifier columns are absent.
    """
    from concurrent.futures import ThreadPoolExecutor

    _excl = exclude_domains or frozenset()
    our_idx = _build_file_index(samples_dir, mode)
    ref_idx = _build_file_index(ref_dir,     mode)

    common = sorted((set(our_idx) & set(ref_idx)) - _excl)
    if not common:
        print(f"  [ref/{mode}] no matching stems between {samples_dir} and {ref_dir}")
        return None, None, None

    base_keys = _JOIN_KEYS.get(mode, ["Res"])

    def _join_pair(stem):
        our_df = _read_raw_output_file(str(our_idx[stem]))
        ref_df = _read_raw_output_file(str(ref_idx[stem]), cache=True)
        if our_df is None or ref_df is None or our_df.empty or ref_df.empty:
            return None

        # Drop FrustratometeR NaN→0 artefact rows from the ref side
        ref_df = ref_df[ref_df["FrstIndex"] != 0.0].copy()

        # Determine actual join keys (base + chain columns when present)
        chain_keys = [c for c in (["ChainRes"] if mode not in _PAIR_MODES
                                   else ["ChainRes1", "ChainRes2"])
                      if c in our_df.columns and c in ref_df.columns]
        keys = [k for k in base_keys if k in our_df.columns and k in ref_df.columns] + chain_keys

        if keys:
            for k in keys:
                our_df[k] = our_df[k].astype(str)
                ref_df[k] = ref_df[k].astype(str)
            merged = our_df[keys + ["FrstIndex"]].merge(
                ref_df[keys + ["FrstIndex"]].rename(columns={"FrstIndex": "FrstIndex_ref"}),
                on=keys, how="inner",
            )
            pred_l = merged["FrstIndex"].tolist()
            true_l = merged["FrstIndex_ref"].tolist()
            # GLY detection from our file (has AA columns)
            gly_l  = [False] * len(pred_l)
            has_aa = False
            if "AA" in our_df.columns:
                aa_col = our_df.set_index(keys)["AA"].reindex(
                    pd.MultiIndex.from_frame(merged[keys]) if len(keys) > 1
                    else merged[keys[0]]
                ).astype(str).str.upper()
                gly_l  = aa_col.isin(["GLY", "G"]).tolist()
                has_aa = True
            elif "AA1" in our_df.columns or "AA2" in our_df.columns:
                mk = merged[keys].copy()
                sub = our_df[keys + [c for c in ["AA1","AA2"] if c in our_df.columns]]
                m2  = mk.merge(sub, on=keys, how="left")
                aa1 = m2["AA1"].astype(str).str.upper() if "AA1" in m2.columns else pd.Series(["X"]*len(m2))
                aa2 = m2["AA2"].astype(str).str.upper() if "AA2" in m2.columns else pd.Series(["X"]*len(m2))
                gly_l  = (aa1.isin(["GLY","G"]) | aa2.isin(["GLY","G"])).tolist()
                has_aa = True
        else:
            # Positional fallback — truncate to shorter file
            n = min(len(our_df), len(ref_df))
            pred_l = our_df["FrstIndex"].iloc[:n].tolist()
            true_l = ref_df["FrstIndex"].iloc[:n].tolist()
            gly_l  = [False] * n
            has_aa = False

        return pred_l, true_l, gly_l, has_aa

    all_pred, all_true, all_gly = [], [], []
    has_aa = False
    n_files = 0

    n_workers = min(64, len(common))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_join_pair, common):
            if result is None:
                continue
            pred_l, true_l, gly_l, file_has_aa = result
            all_pred.extend(pred_l)
            all_true.extend(true_l)
            all_gly.extend(gly_l)
            has_aa = has_aa or file_has_aa
            n_files += 1

    print(f"  [ref/{mode}] {n_files}/{len(common)} stems matched, "
          f"{len(all_pred):,} residues, has_aa={has_aa}")
    if not all_pred:
        return None, None, None
    return (np.array(all_pred, dtype=float),
            np.array(all_true, dtype=float),
            np.array(all_gly, dtype=bool) if has_aa else None)


def _load_from_parquet(precomputed_dir: Path, mode: str,
                       exclude_domains: "frozenset[str] | None" = None):
    """Load (pred, true, gly_mask) from a pre-computed parquet file.

    Parquet must have columns: domain, pred, true, gly (written by
    paper_figures/precompute_cath_data.py).
    Returns (pred, true, gly_mask) arrays, or (None, None, None).
    """
    path = precomputed_dir / f"ptf_fr_{mode}.parquet"
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        return None, None, None
    except Exception as exc:
        print(f"  [parquet/{mode}] read failed: {exc}")
        return None, None, None
    if exclude_domains:
        df = df[~df["domain"].isin(exclude_domains)]
    pred = df["pred"].to_numpy(dtype=float)
    true = df["true"].to_numpy(dtype=float)
    gly  = df["gly"].to_numpy(dtype=bool) if "gly" in df.columns else None
    ok   = np.isfinite(pred) & np.isfinite(true)
    print(f"  [parquet/{mode}] {ok.sum():,} rows loaded from {path}")
    return pred[ok], true[ok], gly[ok] if gly is not None else None


_SAMPLES_CACHE: dict = {}


def _load_samples(samples_dir: Path, mode: str, ref_dir: Path | None = None,
                  precomputed_dir: Path | None = None,
                  exclude_domains: "frozenset[str] | None" = None):
    """Memoised wrapper — fig2 and fig2b load the same data; read it once.

    EXCLUDE_DOMAINS are always dropped, in addition to *exclude_domains*.
    """
    exclude_domains = EXCLUDE_DOMAINS | frozenset(exclude_domains or ())
    key = (str(samples_dir), mode, str(ref_dir), str(precomputed_dir),
           frozenset(exclude_domains or ()))
    if key not in _SAMPLES_CACHE:
        _SAMPLES_CACHE[key] = _load_samples_uncached(
            samples_dir, mode, ref_dir=ref_dir, precomputed_dir=precomputed_dir,
            exclude_domains=exclude_domains)
    return _SAMPLES_CACHE[key]


def _load_samples_uncached(samples_dir: Path, mode: str, ref_dir: Path | None = None,
                           precomputed_dir: Path | None = None,
                           exclude_domains: "frozenset[str] | None" = None):
    _excl = exclude_domains or frozenset()

    # Priority 0: pre-computed parquet (fastest — single file read)
    if precomputed_dir is not None:
        pred, true, gly_mask = _load_from_parquet(precomputed_dir, mode, _excl)
        if pred is not None:
            return pred, true, gly_mask
        print(f"  [samples/{mode}] parquet not found in {precomputed_dir}, falling back")

    # Priority 1: ref_dir supplied — join our output files with FrustratometeR ref files
    if ref_dir is not None:
        pred, true, gly_mask = _load_output_files_with_ref(samples_dir, ref_dir, mode,
                                                           exclude_domains=_excl)
        if pred is not None:
            return pred, true, gly_mask
        print(f"  [samples/{mode}] ref_dir join failed, falling back to TrueFrstIndex path")

    path = samples_dir / f"residue_samples_{mode}.csv"
    csv_pred, csv_true = None, None
    try:
        sdf = pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, IsADirectoryError):
        sdf = None
    if sdf is not None:
        sdf = sdf[sdf["pred"] != "pred"].copy()
        # Filter excluded domains if a domain/stem column exists
        for dom_col in ("domain", "stem", "pdb"):
            if dom_col in sdf.columns:
                sdf = sdf[~sdf[dom_col].isin(_excl)]
                break
        sdf["pred"] = pd.to_numeric(sdf["pred"], errors="coerce")
        sdf["true"] = pd.to_numeric(sdf["true"], errors="coerce")
        sdf = sdf.dropna(subset=["pred", "true"])
        csv_pred = sdf["pred"].to_numpy(dtype=float)
        csv_true = sdf["true"].to_numpy(dtype=float)

        if mode not in _PAIR_MODES:
            # Single-residue: one AA per row — "aa" column is sufficient.
            for col in ("aa", "AA", "resname", "ResName"):
                if col in sdf.columns:
                    aa_upper = sdf[col].astype(str).str.upper()
                    gly_mask = aa_upper.isin(["GLY", "G"]).to_numpy()
                    return csv_pred, csv_true, gly_mask

    # Pair modes (or no aa column): use output files which have AA1+AA2 per pair.
    print(f"  [samples/{mode}] loading output files for AA info…")
    pred, true, gly_mask = _load_output_files(samples_dir, mode, exclude_domains=_excl)
    if pred is not None:
        return pred, true, gly_mask

    # Output files also unavailable — return CSV data without GLY coloring
    return csv_pred, csv_true, None


def _norm_algo(name: str) -> str:
    u = name.upper()
    if "FRUSTRATOMETER" in u:
        return "FrustratometeR"
    if "PYTORCH" in u and "CPU" in u:
        return "PyTorch (CPU)"
    if "PYTORCH" in u:
        return "PyTorch (GPU)"
    return name


def _load_timings(mac_path, mn5_path):
    frames = []
    for path, source in [(mac_path, "Mac M2"), (mn5_path, "MN5 H100")]:
        if path and Path(path).exists():
            df = pd.read_csv(path)
            df["source"]    = source
            df["aa_length"] = pd.to_numeric(df["aa_length"], errors="coerce")
            df["time_s"]    = pd.to_numeric(df["time_s"],    errors="coerce")
            df["algo"]      = df["algorithm"].str.strip().apply(_norm_algo)
            frames.append(df.dropna(subset=["aa_length", "time_s"]))
    return pd.concat(frames, ignore_index=True).sort_values("aa_length") if frames else None


# ─── Figure 1: schematic ──────────────────────────────────────────────────────

def fig1_schematic(out_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    BOX_W, BOX_H = 1.8, 0.9
    BOX_Y = 2.8
    GAP   = 0.35
    COLS  = [0.3, 2.45, 4.6, 6.75, 9.0, 11.25]

    boxes = [
        ("PDB\nStructure",        COLS[0], "#BBDEFB"),
        ("Parse\nCoordinates",    COLS[1], "#C8E6C9"),
        ("Native\nContact Map",   COLS[2], "#C8E6C9"),
        ("Decoy\nSampling",       COLS[3], "#FFE0B2"),
        ("AWSEM\nEnergy",         COLS[4], "#FFE0B2"),
        ("Frustration\nIndex FI", COLS[5], "#F8BBD9"),
    ]

    def _box(ax, label, x, color):
        rect = mpatches.FancyBboxPatch(
            (x, BOX_Y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.08", facecolor=color,
            edgecolor="#555", linewidth=1.2, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x + BOX_W / 2, BOX_Y, label,
                ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

    for label, x, color in boxes:
        _box(ax, label, x, color)

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + BOX_W
        x1 = boxes[i + 1][1]
        ax.annotate("", xy=(x1, BOX_Y), xytext=(x0, BOX_Y),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5), zorder=5)

    # Output modes branching from FI box
    OUT_Y   = [3.95, 2.8, 1.65]
    OUT_COL = [PALETTE["singleresidue"], PALETTE["mutational"], PALETTE["configurational"]]
    OUT_LAB = ["Single-residue FI", "Mutational FI", "Configurational FI"]
    FI_X = COLS[5] + BOX_W / 2
    FI_Y = BOX_Y

    ax.annotate("", xy=(FI_X + 0.1, FI_Y), xytext=(FI_X - 0.1, FI_Y),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5), zorder=5)

    for oy, oc, ol in zip(OUT_Y, OUT_COL, OUT_LAB):
        ax.annotate("", xy=(12.5, oy), xytext=(FI_X + BOX_W / 2 + 0.05, FI_Y),
                    arrowprops=dict(arrowstyle="->", color=oc, lw=1.5,
                                    connectionstyle="arc3,rad=0.0"), zorder=5)
        rect = mpatches.FancyBboxPatch(
            (12.5, oy - 0.32), 0.45, 0.64,
            boxstyle="round,pad=0.05", facecolor=oc, alpha=0.25,
            edgecolor=oc, linewidth=1.2, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(12.52, oy, ol, ha="left", va="center",
                fontsize=8.5, color=oc, fontweight="bold", zorder=4)

    # Hardware annotation strip
    STRIP_H = 0.55
    STRIP_Y = 0.25
    hw_regions = [
        (COLS[0], COLS[1], "#E3F2FD", "Input"),
        (COLS[1], COLS[3] + BOX_W, "#E8F5E9", "CPU / GPU / MPS"),
        (COLS[3], COLS[5] + BOX_W, "#FFF3E0", "GPU-accelerated"),
    ]
    for x0, x1, col, lbl in hw_regions:
        w = x1 - x0
        rect = mpatches.FancyBboxPatch(
            (x0, STRIP_Y), w, STRIP_H,
            boxstyle="round,pad=0.05", facecolor=col,
            edgecolor="#aaa", linewidth=0.8, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x0 + w / 2, STRIP_Y + STRIP_H / 2, lbl,
                ha="center", va="center", fontsize=8, color="#444")

    ax.set_title("FrustrAI-Fold: PyTorch frustration calculation pipeline",
                 fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    out = out_dir / "fig1_schematic.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Figure 2: scatter ────────────────────────────────────────────────────────

def fig2_scatter(samples_dir: Path, out_dir: Path, results_csv: Path | None = None,
                 ref_dir: Path | None = None,
                 exclude_domains: "frozenset[str] | None" = None):
    from matplotlib.colors import LinearSegmentedColormap

    _excl = exclude_domains or frozenset()

    if results_csv is not None and Path(results_csv).exists():
        df_all = pd.read_csv(results_csv, low_memory=False)
        for c in ["pearson_r", "cosine_similarity"]:
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
        # Drop excluded domains from the per-protein metrics table
        df_all = _drop_excluded(df_all, EXCLUDE_DOMAINS | _excl)
    else:
        df_all = None

    has_cos = df_all is not None and "cosine_similarity" in df_all.columns

    # Per-mode colormaps: light-grey → PALETTE color (matches KDE fills in fig3)
    _cmaps = {m: LinearSegmentedColormap.from_list(f"_c_{m}", ["#d0d0d0", PALETTE[m]])
              for m in MODES}

    # Load all samples up-front; use fixed ±4 axis bounds
    all_samples = {m: _load_samples(samples_dir, m, ref_dir=ref_dir,
                                    exclude_domains=_excl) for m in MODES}
    shared_lim = (-4.0, 4.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5),
                             gridspec_kw={"wspace": 0.25})

    for col, mode in enumerate(MODES):
        color  = PALETTE[mode]

        ax = axes[col]
        pred, true, gly_mask = all_samples[mode]
        n_total = len(df_all[df_all["mode"] == mode]) if df_all is not None else 0
        n_res   = len(pred) if pred is not None else 0

        lo, hi = MODE_THRESHOLDS.get(mode, (LO_PAIR, HI_PAIR))

        # Per-protein average cosine similarity from results_csv (equal weight per domain).
        cos_line = None
        if has_cos:
            cos_vals = (df_all[df_all["mode"] == mode]["cosine_similarity"]
                        .dropna().to_numpy(dtype=float))
            if len(cos_vals) > 0:
                cos_line = f"Avg cos sim = {float(np.nanmean(cos_vals)):.4f}"

        ax.set_title(MODE_LABELS[mode], fontsize=11, fontweight="bold", pad=6)
        if pred is not None:
            draw_scatter_hexbin(ax, pred, true, title="",
                                cmap=_cmaps[mode], xlabel="", ylabel="",
                                line_color=color, extra_line=cos_line,
                                gly_mask=gly_mask, lo=lo, hi=hi)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel(_XLAB, fontsize=10)
        ax.set_ylabel(_YLAB if col == 0 else "", fontsize=10)
        if shared_lim is not None:
            ax.set_xlim(shared_lim)
            ax.set_ylim(shared_lim)

    # Per-mode domain counts (may differ when some modes have fewer validated domains)
    if df_all is not None:
        mode_n = [len(df_all[df_all["mode"] == m]) for m in MODES]
    else:
        mode_n = [sum(1 for _ in _iter_output_files(samples_dir, m)) for m in MODES]
    n_lo, n_hi = min(mode_n), max(mode_n)
    n_domains_str = f"{n_lo:,}–{n_hi:,}" if n_lo != n_hi else f"{n_lo:,}"
    n_res_total = sum(len(p) for p, *_ in all_samples.values() if p is not None) // len(MODES)
    fig.text(0.5, 1.02,
             "Numerical validation against FrustratometeR",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.01,
             f"CATH domains  ·  {n_res_total:,} residues  ·  {n_domains_str} proteins",
             ha="center", va="top", fontsize=10, color="#666")

    out = out_dir / "fig2_scatter.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Figure 2b: hexbin scatter ───────────────────────────────────────────────

def fig2_hexbin(samples_dir: Path, out_dir: Path, results_csv: Path | None = None,
                ref_dir: Path | None = None,
                exclude_domains: "frozenset[str] | None" = None):
    """Same layout as fig2_scatter but with hexbin density instead of scatter dots."""
    from matplotlib.colors import LinearSegmentedColormap

    _excl = exclude_domains or frozenset()

    if results_csv is not None and Path(results_csv).exists():
        df_all = pd.read_csv(results_csv, low_memory=False)
        for c in ["pearson_r", "cosine_similarity"]:
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
        df_all = _drop_excluded(df_all, EXCLUDE_DOMAINS | _excl)
    else:
        df_all = None

    has_cos = df_all is not None and "cosine_similarity" in df_all.columns

    _cmaps = {m: LinearSegmentedColormap.from_list(f"_c_{m}", ["#f0f0f0", PALETTE[m]])
              for m in MODES}

    all_samples = {m: _load_samples(samples_dir, m, ref_dir=ref_dir,
                                    exclude_domains=_excl) for m in MODES}
    shared_lim = (-4.0, 4.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             gridspec_kw={"wspace": 0.32})

    for col, mode in enumerate(MODES):
        color = PALETTE[mode]
        ax = axes[col]
        pred, true, gly_mask = all_samples[mode]

        lo, hi = MODE_THRESHOLDS.get(mode, (LO_PAIR, HI_PAIR))

        # Per-protein average cosine similarity from results_csv (equal weight per domain).
        cos_line = None
        if has_cos:
            cos_vals = (df_all[df_all["mode"] == mode]["cosine_similarity"]
                        .dropna().to_numpy(dtype=float))
            if len(cos_vals) > 0:
                cos_line = f"Avg cos sim = {float(np.nanmean(cos_vals)):.4f}"

        ax.set_title(MODE_LABELS[mode], fontsize=11, fontweight="bold", pad=6)
        if pred is not None:
            draw_hexbin(ax, pred, true, title="",
                        cmap=_cmaps[mode], xlabel="", ylabel="",
                        line_color=color, extra_line=cos_line,
                        gly_mask=gly_mask, lo=lo, hi=hi,
                        colorbar=(col == 2))
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel(_XLAB, fontsize=10)
        ax.set_ylabel(_YLAB if col == 0 else "", fontsize=10)
        ax.set_xlim(shared_lim)
        ax.set_ylim(shared_lim)

    # Per-mode domain counts (may differ when some modes have fewer validated domains)
    if df_all is not None:
        mode_n = [len(df_all[df_all["mode"] == m]) for m in MODES]
    else:
        mode_n = [sum(1 for _ in _iter_output_files(samples_dir, m)) for m in MODES]
    n_lo, n_hi = min(mode_n), max(mode_n)
    n_domains_str = f"{n_lo:,}–{n_hi:,}" if n_lo != n_hi else f"{n_lo:,}"
    n_res_total = sum(len(p) for p, *_ in all_samples.values() if p is not None) // len(MODES)
    fig.text(0.5, 1.02,
             "Numerical validation against FrustratometeR",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.01,
             f"CATH domains  ·  {n_res_total:,} residues  ·  {n_domains_str} proteins",
             ha="center", va="top", fontsize=10, color="#666")

    out = out_dir / "fig2b_hexbin.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Figure 3: per-protein KDE ────────────────────────────────────────────────

def _kde_ax(ax, vals, color, xlim=None, show_ylabel=True):
    """Draw a single publication-quality KDE panel."""
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    sns.kdeplot(v, ax=ax, color=color, fill=True, alpha=0.25, linewidth=2)
    med = float(np.nanmedian(v))
    ax.axvline(med, color="k", ls="--", lw=1.2)
    # Median annotation as an in-plot text box
    ax.text(0.04, 0.94, f"median = {med:.4f}", transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.85))
    if xlim is not None:
        ax.set_xlim(xlim)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=10)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=9)


def fig3_kdeplots(results_csv: Path, out_dir: Path,
                  samples_dir: Path | None = None):
    """3×3 figure: Pearson R KDE (row 0), cosine similarity KDE (row 1),
    confusion-matrix heatmap (row 2) — one column per frustration mode.
    Confusion matrices require samples_dir (residue_samples_*.csv files);
    falls back to a class-accuracy KDE when samples are unavailable.
    """
    df_all = _drop_excluded(pd.read_csv(results_csv, low_memory=False))
    for c in ["pearson_r", "class_accuracy", "cosine_similarity"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    has_cos     = "cosine_similarity" in df_all.columns
    has_samples = samples_dir is not None and Path(samples_dir).exists()

    # Consistent xlim per KDE row
    def _xlim(arr, pad=0.02):
        if len(arr) == 0:
            return (0, 1)
        lo = float(np.nanpercentile(arr, 0.5))
        hi = float(np.nanpercentile(arr, 99.9))
        span = hi - lo
        return (max(lo - span * pad, -0.02), min(hi + span * pad, 1.02))

    r_concat = np.concatenate([
        df_all[df_all["mode"] == m]["pearson_r"].dropna().to_numpy(dtype=float)
        for m in MODES
    ])
    r_xlim = _xlim(r_concat)

    cos_xlim = (0, 1)
    if has_cos:
        cos_concat = np.concatenate([
            df_all[df_all["mode"] == m]["cosine_similarity"].dropna().to_numpy(dtype=float)
            for m in MODES
        ])
        cos_xlim = _xlim(cos_concat)

    # Per-mode frustration thresholds for confusion matrix
    _thr = {
        "singleresidue":   (LO_SR,   HI_SR),
        "mutational":      (LO_PAIR, HI_PAIR),
        "configurational": (LO_PAIR, HI_PAIR),
    }

    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.3})
    fig.subplots_adjust(top=0.91)

    for col, mode in enumerate(MODES):
        df_mode = df_all[df_all["mode"] == mode]
        color   = PALETTE[mode]
        n_prot  = len(df_mode)
        show_y  = (col == 0)

        # Column header (top row only)
        axes[0, col].set_title(MODE_LABELS[mode], fontsize=13, fontweight="bold", pad=6)

        # ── Row 0: Pearson R KDE ──────────────────────────────────────────────
        ax = axes[0, col]
        r_vals = df_mode["pearson_r"].dropna().to_numpy(dtype=float)
        _kde_ax(ax, r_vals, color, xlim=r_xlim, show_ylabel=show_y)
        ax.set_xlabel("Pearson R", fontsize=10)

        # ── Row 1: Cosine similarity KDE ─────────────────────────────────────
        ax = axes[1, col]
        if has_cos and "cosine_similarity" in df_mode.columns:
            cos_vals = df_mode["cosine_similarity"].dropna().to_numpy(dtype=float)
            _kde_ax(ax, cos_vals, color, xlim=cos_xlim, show_ylabel=show_y)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            if show_y:
                ax.set_ylabel("Density", fontsize=10)
        ax.set_xlabel("Cosine similarity", fontsize=10)

        # ── Row 2: Confusion matrix (heatmap, colors match mode PALETTE) ────────
        ax = axes[2, col]
        lo, hi = _thr.get(mode, (LO_PAIR, HI_PAIR))
        from matplotlib.colors import LinearSegmentedColormap
        mode_cmap = LinearSegmentedColormap.from_list(
            f"conf_{mode}", ["#ffffff", color])
        if has_samples:
            pred, true, _ = _load_samples(Path(samples_dir), mode)
            if pred is not None:
                draw_confusion(ax, pred, true, lo=lo, hi=hi,
                               cmap=mode_cmap,
                               cbar=(col == 2),
                               xlabel="PyTorch Re-implementation",
                               ylabel="Original FrustratometeR" if show_y else "")
                if not show_y:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
        elif "class_accuracy" in df_mode.columns:
            acc_vals = df_mode["class_accuracy"].dropna().to_numpy(dtype=float)
            _kde_ax(ax, acc_vals, color, show_ylabel=show_y)
            ax.set_xlabel("Class accuracy", fontsize=10)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)

    # Left-margin row labels — anchored to actual axes positions after layout
    fig.canvas.draw()
    row_labels = ["Pearson R", "Cosine similarity",
                  "Class confusion" if has_samples else "Class accuracy"]
    x_anchor = axes[0, 0].get_position().x0 - 0.05
    for row, label in enumerate(row_labels):
        pos = axes[row, 0].get_position()
        yc  = pos.y0 + pos.height / 2
        fig.text(x_anchor, yc, label, va="center", ha="right",
                 rotation=90, fontsize=11, fontweight="bold")

    n_domains = len(df_all[df_all["mode"] == MODES[0]])
    fig.text(0.5, 0.975,
             "Per-protein agreement with FrustratometeR",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.963,
             f"CATH domains  ·  {n_domains:,} proteins",
             ha="center", va="top", fontsize=10, color="#666")

    out = out_dir / "fig3_kdeplots.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")
    if not has_cos:
        print("    note: cosine_similarity missing from all_results.csv — "
              "re-run the baseline to regenerate it with the current code")


# ─── Figure 4: speed comparison ───────────────────────────────────────────────

def fig4_speed(mac_path, mn5_path, out_dir: Path):
    df = _load_timings(mac_path, mn5_path)
    if df is None:
        print("  [skip] fig4: no timing data")
        return

    algo_order  = ["FrustratometeR", "PyTorch (CPU)", "PyTorch (GPU)"]
    algo_order  = [a for a in algo_order if a in df["algo"].unique()]
    palette     = dict(zip(algo_order, sns.color_palette("tab10", len(algo_order))))
    source_dash = {"MN5 H100": (1, 0), "Mac M2": (4, 2)}
    sources     = df["source"].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute time (log scale)
    ax = axes[0]
    sns.lineplot(data=df, x="aa_length", y="time_s",
                 hue="algo", hue_order=algo_order,
                 style="source", style_order=sources,
                 dashes={s: source_dash.get(s, (1, 0)) for s in sources},
                 markers=True, markersize=6, palette=palette, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Protein length (residues)")
    ax.set_ylabel("Wall time (s)  [log scale]")
    ax.set_title("Absolute compute time")
    ax.legend(title="Algorithm / hardware", fontsize=8, title_fontsize=8)

    # Right: speedup relative to FrustratometeR
    df_r = df[df["algo"] == "FrustratometeR"][["aa_length", "source", "time_s"]]\
             .rename(columns={"time_s": "ref_time"})
    df_s = df[df["algo"] != "FrustratometeR"].merge(
        df_r, on=["aa_length", "source"], how="inner")
    df_s["speedup"] = df_s["ref_time"] / df_s["time_s"]

    ax2 = axes[1]
    if not df_s.empty:
        speedup_algos = [a for a in algo_order if a != "FrustratometeR"]
        sns.lineplot(data=df_s, x="aa_length", y="speedup",
                     hue="algo", hue_order=speedup_algos,
                     style="source", style_order=sources,
                     dashes={s: source_dash.get(s, (1, 0)) for s in sources},
                     markers=True, markersize=6, palette=palette, ax=ax2)
        ax2.set_yscale("log")
        ax2.set_xlabel("Protein length (residues)")
        ax2.set_ylabel("Speedup vs FrustratometeR  [log scale]")
        ax2.set_title("Speedup relative to FrustratometeR")
        ax2.axhline(1, color="k", lw=0.8, ls="--", alpha=0.5)
        ax2.legend(title="Algorithm / hardware", fontsize=8, title_fontsize=8)
    else:
        ax2.text(0.5, 0.5, "need FrustratometeR rows in both CSVs",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle("Computation time: FrustratometeR vs PyTorch Re-implementation",
                 fontsize=13, fontweight="bold", y=1.01)
    sns.despine()
    fig.tight_layout()
    out = out_dir / "fig4_speed.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Figure 5: log-log scaling ────────────────────────────────────────────────

def fig5_scaling(mac_path, mn5_path, out_dir: Path):
    df = _load_timings(mac_path, mn5_path)
    if df is None:
        print("  [skip] fig5: no timing data")
        return

    algo_order = ["FrustratometeR", "PyTorch (CPU)", "PyTorch (GPU)"]
    algo_order = [a for a in algo_order if a in df["algo"].unique()]
    palette    = dict(zip(algo_order, sns.color_palette("tab10", len(algo_order))))
    source_dash= {"MN5 H100": (1, 0), "Mac M2": (4, 2)}
    sources    = df["source"].unique().tolist()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.lineplot(data=df, x="aa_length", y="time_s",
                 hue="algo", hue_order=algo_order,
                 style="source", style_order=sources,
                 #dashes={s: source_dash.get(s, (1, 0)) for s in sources},
                 markers=True, markersize=7, palette=palette, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Annotate power-law slope for each algo × source
    for algo in algo_order:
        for src in sources:
            sub = df[(df["algo"] == algo) & (df["source"] == src)].sort_values("aa_length")
            if len(sub) < 3:
                continue
            logx = np.log10(sub["aa_length"].to_numpy(dtype=float))
            logy = np.log10(sub["time_s"].to_numpy(dtype=float))
            ok   = np.isfinite(logx) & np.isfinite(logy)
            if ok.sum() < 3:
                continue
            slope, intercept, *_ = linregress(logx[ok], logy[ok])
            # Place label near the last data point
            x_last = sub["aa_length"].iloc[-1]
            y_last = sub["time_s"].iloc[-1]
            ax.text(x_last * 1.05, y_last,
                    f"α={slope:.2f}", fontsize=7,
                    color=palette.get(algo, "k"), va="center")

    ax.set_xlabel("Protein length (residues)  [log scale]")
    ax.set_ylabel("Wall time (s)  [log scale]")
    ax.set_title("Scaling behaviour (log–log)\nslope α gives T ~ Lᵅ")
    ax.legend(title="Algorithm / hardware", fontsize=8, title_fontsize=8,
              bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine()
    fig.tight_layout()
    out = out_dir / "fig5_scaling.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Figure 6: mutational scanning ───────────────────────────────────────────

def _estimate_orig_mut_time(mut_dir: Path) -> float | None:
    """Estimate FrustratometeR wall time from mtime spread of MutationsData files."""
    files = sorted(mut_dir.glob("singleresidue_Res*_threading_*.txt"))
    if len(files) < 2:
        return None
    mtimes = [f.stat().st_mtime for f in files]
    elapsed = max(mtimes) - min(mtimes)
    return elapsed if elapsed > 0 else None


def _aa_length_from_pred(pred_df: pd.DataFrame) -> int:
    """Count unique residue positions in a singleresidue_mutations pred file."""
    return int(pred_df["Res"].nunique()) if "Res" in pred_df.columns else 0


def _interp_reimpl_time(lengths: np.ndarray, timing_df: pd.DataFrame) -> np.ndarray:
    """Interpolate PyTorch re-implementation time at each aa_length from timing data."""
    df = timing_df.copy()
    df["algo"] = df["algorithm"].str.strip().apply(_norm_algo)
    pytorch = df[df["algo"].isin(["PyTorch (GPU)", "PyTorch (CPU)"])]
    if pytorch.empty:
        return np.full(len(lengths), np.nan)
    ref = (pytorch.groupby("aa_length")["time_s"]
           .median().reset_index().sort_values("aa_length"))
    return np.interp(lengths,
                     ref["aa_length"].to_numpy(dtype=float),
                     ref["time_s"].to_numpy(dtype=float))


def _compute_mut_metrics(pred_dir: Path, ref_dir: Path):
    """Scan pred_dir for *.singleresidue_mutations, merge with ref MutationsData,
    return (df_metrics, all_pred, all_true, gly_mask).
    df_metrics includes aa_length and orig_time_s (estimated from MutationsData mtimes).
    gly_mask is a bool array True where the native AA is GLY; None if AA_native unavailable."""
    # Collect prediction files from both old and new path layouts (deduplicated).
    # Old: pred_dir/*.frustratometer3.singleresidue_mutations
    # New: pred_dir/*.done/FrustrationData/*.pdb_singleresidue_mutations
    seen: set[Path] = set()
    pred_files: list[tuple[Path, str]] = []  # (path, stem)
    for pf in sorted(pred_dir.glob("*.frustratometer3.singleresidue_mutations")):
        if pf not in seen:
            seen.add(pf)
            pred_files.append((pf, pf.name.split(".frustratometer3")[0]))
    for pf in sorted(pred_dir.glob("*.done/FrustrationData/*.pdb_singleresidue_mutations")):
        if pf not in seen:
            seen.add(pf)
            pred_files.append((pf, pf.stem))  # e.g. "6bmeA00.pdb_singleresidue_mutations" → stem "6bmeA00"

    if not pred_files:
        print(f"  [fig6] no singleresidue_mutations files found in {pred_dir}")
        return None, None, None, None

    per_protein, all_pred, all_true, all_gly = [], [], [], []
    has_gly_info = False
    for pf, stem in pred_files:
        pred_df = load_pred(pf)
        if pred_df is None:
            continue
        mut_dir = find_ref_mutations_dir(stem, ref_dir)
        if mut_dir is None:
            continue

        orig_time = _estimate_orig_mut_time(mut_dir)

        ref_df = read_mutations_data(mut_dir)
        if ref_df is None or ref_df.empty:
            continue
        ref_df = ref_df.rename(columns={"AA": "AA_mutant",
                                        "FrstIndex": "FrstIndex_ref"})
        ref_df["Res"]           = pd.to_numeric(ref_df["Res"],           errors="coerce")
        ref_df["FrstIndex_ref"] = pd.to_numeric(ref_df["FrstIndex_ref"], errors="coerce")
        merged = pred_df.merge(
            ref_df[["Res", "ChainRes", "AA_mutant", "FrstIndex_ref"]],
            on=["Res", "ChainRes", "AA_mutant"], how="inner",
        )

        if merged.empty:
            continue
        p = merged["FrstIndex"].to_numpy(dtype=float)
        t = merged["FrstIndex_ref"].to_numpy(dtype=float)
        m = compute_metrics(p, t)
        per_protein.append({
            "stem":        stem,
            "aa_length":   _aa_length_from_pred(pred_df),
            "orig_time_s": orig_time,
            **m,
        })
        all_pred.extend(p.tolist())
        all_true.extend(t.tolist())
        if "AA_native" in merged.columns:
            all_gly.extend(merged["AA_native"].astype(str).str.upper().isin(["GLY", "G"]).tolist())
            has_gly_info = True
        else:
            all_gly.extend([False] * len(p))

    if not per_protein:
        return None, None, None, None
    gly_mask = np.asarray(all_gly, dtype=bool) if has_gly_info else None
    return (pd.DataFrame(per_protein),
            np.asarray(all_pred, dtype=float),
            np.asarray(all_true, dtype=float),
            gly_mask)


def fig6_mutations(mut_metrics_csv: Path, mut_samples_dir: Path | None,
                   mut_ref_dir: Path | None, sfam_map_csv: Path | None,
                   out_dir: Path,
                   mac_path: str | None = None, mn5_path: str | None = None):
    from matplotlib.colors import LinearSegmentedColormap

    # ── Resolve metrics CSV or compute on the fly ─────────────────────────────
    pred_all, true_all, gly_all = None, None, None
    df_m = None

    if mut_metrics_csv and Path(mut_metrics_csv).is_file():
        df_m = pd.read_csv(mut_metrics_csv)
    elif mut_samples_dir:
        ref_dir = Path(mut_ref_dir) if mut_ref_dir else Path(mut_samples_dir)
        print(f"  fig6: no metrics CSV — computing from raw files in {mut_samples_dir}")
        df_m, pred_all, true_all, gly_all = _compute_mut_metrics(
            Path(mut_samples_dir), ref_dir)
        if df_m is not None:
            csv_out = out_dir / "mutations_metrics.csv"
            df_m.to_csv(csv_out, index=False)
            print(f"  fig6: wrote computed metrics to {csv_out}")

    if df_m is None or df_m.empty:
        print("  [skip] fig6: no mutations data found")
        return

    for c in ["pearson_r", "cosine_similarity", "class_accuracy"]:
        if c in df_m.columns:
            df_m[c] = pd.to_numeric(df_m[c], errors="coerce")

    # Interpolate reimpl times from length_test timing CSV
    if "aa_length" in df_m.columns:
        timing_frames = [pd.read_csv(p) for p in [mn5_path]
                         if p and Path(p).is_file()]
        if timing_frames:
            timing_df = pd.concat(timing_frames, ignore_index=True)
            timing_df["aa_length"] = pd.to_numeric(timing_df["aa_length"], errors="coerce")
            timing_df["time_s"]    = pd.to_numeric(timing_df["time_s"],    errors="coerce")
            timing_df = timing_df.dropna(subset=["aa_length", "time_s"])
            lengths = df_m["aa_length"].to_numpy(dtype=float)
            df_m["reimpl_time_s"] = _interp_reimpl_time(lengths, timing_df)

    color     = PALETTE["mutations"]
    cmap_mut  = LinearSegmentedColormap.from_list("_c_mut",  ["#f0f0f0", color])
    conf_cmap = LinearSegmentedColormap.from_list("_cm_mut", ["#ffffff", color])
    has_cos   = "cosine_similarity" in df_m.columns

    # If pred/true weren't built above, try loading from a pooled samples file
    if pred_all is None and mut_samples_dir:
        p = Path(mut_samples_dir)
        candidates = [
            p / "residue_samples_singleresidue_mutations.csv",
            p / "residue_samples_mutations.csv",
        ] + sorted(p.glob("residue_samples_*mutations*.csv"))
        for f in candidates:
            if f.exists():
                sdf = pd.read_csv(f, low_memory=False)
                sdf = sdf[sdf["pred"] != "pred"].copy()
                sdf["pred"] = pd.to_numeric(sdf["pred"], errors="coerce")
                sdf["true"] = pd.to_numeric(sdf["true"], errors="coerce")
                sdf = sdf.dropna(subset=["pred", "true"])
                pred_all = sdf["pred"].to_numpy(dtype=float)
                true_all = sdf["true"].to_numpy(dtype=float)
                for col in ("AA_native", "aa_native", "AA", "aa"):
                    if col in sdf.columns:
                        gly_all = sdf[col].astype(str).str.upper().isin(["GLY", "G"]).to_numpy()
                        break
                break

    # ── Single-row layout: scatter | Pearson R | cosine | confusion ─────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5),
                             gridspec_kw={"wspace": 0.3})

    n_prot = len(df_m)
    n_res  = len(pred_all) if pred_all is not None else 0

    # ── Scatter ───────────────────────────────────────────────────────────────
    ax_scatter = axes[0]
    if pred_all is not None:
        cos_line = None
        if has_cos:
            cos_mean = float(df_m["cosine_similarity"].dropna().mean())
            if np.isfinite(cos_mean):
                cos_line = f"Avg cos sim = {cos_mean:.4f}"
        draw_scatter_hexbin(ax_scatter, pred_all, true_all, title="",
                            cmap=cmap_mut, xlabel=_XLAB, ylabel=_YLAB,
                            line_color=color, extra_line=cos_line,
                            gly_mask=gly_all, lo=LO_SR, hi=HI_SR)
        combined = np.concatenate([pred_all, true_all])
        combined = combined[np.isfinite(combined)]
        lo  = float(np.nanpercentile(combined, 0.5))
        hi  = float(np.nanpercentile(combined, 99.5))
        pad = (hi - lo) * 0.03
        ax_scatter.set_xlim(lo - pad, hi + pad)
        ax_scatter.set_ylim(lo - pad, hi + pad)
    else:
        ax_scatter.text(0.5, 0.5, "no residue samples\n(pass --mut-samples)",
                        ha="center", va="center", transform=ax_scatter.transAxes)
    ax_scatter.set_xlabel(_XLAB, fontsize=10)
    ax_scatter.set_ylabel(_YLAB, fontsize=10)

    # ── Pearson R KDE ─────────────────────────────────────────────────────────
    r_vals = df_m["pearson_r"].dropna().to_numpy(dtype=float)
    _kde_ax(axes[1], r_vals, color, show_ylabel=True)
    axes[1].set_xlabel("Pearson R", fontsize=10)

    # ── Cosine similarity KDE ─────────────────────────────────────────────────
    if has_cos:
        cos_vals = df_m["cosine_similarity"].dropna().to_numpy(dtype=float)
        _kde_ax(axes[2], cos_vals, color, show_ylabel=False)
    else:
        axes[2].text(0.5, 0.5, "no data", ha="center", va="center",
                     transform=axes[2].transAxes)
    axes[2].set_xlabel("Cosine similarity", fontsize=10)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if pred_all is not None:
        draw_confusion(axes[3], pred_all, true_all,
                       lo=LO_SR, hi=HI_SR,
                       cmap=conf_cmap, cbar=True,
                       xlabel="PyTorch Re-implementation",
                       ylabel="Original FrustratometeR")
    else:
        axes[3].text(0.5, 0.5, "no data", ha="center", va="center",
                     transform=axes[3].transAxes)

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.5, 1.02,
             "Mutational Scanning Benchmark",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.01,
             f"CATH domains  ·  {n_res:,} residues  ·  {n_prot:,} proteins",
             ha="center", va="top", fontsize=10, color="#666")

    out = out_dir / "fig6_mutations.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")

    fig6b_mut_timing(df_m, out_dir)


# ─── Figure 6b: mutational scanning timing ────────────────────────────────────

def fig6b_mut_timing(df_m: pd.DataFrame, out_dir: Path):
    """Separate timing figure for lab meeting.

    Original: wall time estimated from MutationsData file mtime spread.
    Re-impl:  lower bound interpolated from MN5 length benchmark (basic frustration,
              excludes mutation overhead — actual time will be slightly higher).
    """
    color = PALETTE["mutations"]

    has_timing = ("orig_time_s"   in df_m.columns and df_m["orig_time_s"].notna().any() and
                  "reimpl_time_s" in df_m.columns and df_m["reimpl_time_s"].notna().any())
    if not has_timing:
        print("  [skip] fig6b: no timing data in metrics dataframe")
        return

    tdf = df_m[["aa_length", "orig_time_s", "reimpl_time_s"]].dropna()
    tdf = tdf.sort_values("aa_length")
    if tdf.empty:
        print("  [skip] fig6b: timing columns all NaN")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.35})

    # ── Panel A: wall time vs protein length ──────────────────────────────────
    ax = axes[0]
    ax.scatter(tdf["aa_length"], tdf["orig_time_s"],
               color="#888", s=25, alpha=0.6, zorder=3,
               label="FrustratometeR\n(est. from file mtimes)")
    ax.scatter(tdf["aa_length"], tdf["reimpl_time_s"],
               color=color, s=25, alpha=0.8, zorder=3,
               label="PyTorch Re-impl.\n(lower bound — basic frustration,\nexcl. mutation overhead)")

    # Fit lines
    for vals, c, ls in [(tdf["orig_time_s"], "#888", "--"),
                         (tdf["reimpl_time_s"], color, "-")]:
        ok = np.isfinite(vals) & np.isfinite(tdf["aa_length"])
        if ok.sum() > 1:
            from scipy.stats import linregress
            slope, intercept, *_ = linregress(
                np.log(tdf["aa_length"][ok]), np.log(vals[ok]))
            xs = np.linspace(tdf["aa_length"].min(), tdf["aa_length"].max(), 200)
            ax.plot(xs, np.exp(intercept) * xs ** slope, color=c, ls=ls, lw=1.5, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Protein length (residues)", fontsize=11)
    ax.set_ylabel("Wall time (s)  [log scale]", fontsize=11)
    ax.set_title("Mutational scanning wall time vs length", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)

    # ── Panel B: speedup (orig / reimpl) ─────────────────────────────────────
    ax = axes[1]
    speedup = tdf["orig_time_s"] / tdf["reimpl_time_s"]
    ok = np.isfinite(speedup) & (speedup > 0)
    ax.scatter(tdf["aa_length"][ok], speedup[ok],
               color=color, s=25, alpha=0.8, zorder=3)
    ax.axhline(1, color="k", lw=0.8, ls="--", alpha=0.5)
    med_speedup = float(np.nanmedian(speedup[ok]))
    ax.axhline(med_speedup, color=color, lw=1.2, ls=":", alpha=0.8,
               label=f"median speedup = {med_speedup:.0f}×")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Protein length (residues)", fontsize=11)
    ax.set_ylabel("Speedup (FrustratometeR / Re-impl.)  [log]", fontsize=11)
    ax.set_title("Estimated speedup", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    fig.text(0.5, 1.02, "Mutational Scanning: Speed Comparison",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.01,
             "Re-impl. timing is a lower bound (basic frustration benchmark, MN5 H100)",
             ha="center", va="top", fontsize=9, color="#888", style="italic")

    out = out_dir / "fig6b_mut_timing.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Figure 7: FrustraEvo — IC agreement + timing ────────────────────────────

def fig7_frustraevo(new_dir: Path | None, orig_dir: Path | None, out_dir: Path):
    """
    Panel A: per-position IC hexbin scatter (re-implementation vs FrustratometeR).
    Panel B: per-funfam timing scatter (new vs orig, log-log).
    Panel C: wall time vs MSA length (log y-scale).

    All data collected by integrating frustaevo_comparison logic directly.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if new_dir is None or not new_dir.exists():
        print("  [skip] fig7: --frustraevo-new-dir not provided or not found")
        return
    if orig_dir is None or not orig_dir.exists():
        print("  [skip] fig7: --frustraevo-orig-dir not provided or not found")
        return

    # ── Collect IC and timing data ────────────────────────────────────────────
    funfams = list(find_funfams(new_dir))
    print(f"  fig7: found {len(funfams)} completed funfams")

    all_summary, all_scatter = [], []
    for funfam, new_out in funfams:
        orig_out = orig_dir / f"FrustraEvo_{funfam}" / "OutPutFiles"
        if not orig_out.is_dir():
            continue
        s, sc = compare_funfam(funfam, new_out, orig_out)
        all_summary.extend(s)
        all_scatter.extend(sc)

    if not all_summary:
        print("  [skip] fig7: no matching funfam data found")
        return

    summary_df = pd.DataFrame(all_summary)
    csv_out = out_dir / "frustaevo_ic_comparison.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"  fig7: wrote {len(summary_df)} rows → {csv_out}")

    scatter_df = pd.DataFrame(all_scatter)
    size_map   = (summary_df[summary_df["metric"] == "IC_SingleRes"]
                  .set_index("funfam")["n_positions"].to_dict())
    timing_df  = collect_timings(funfams, new_dir, orig_dir, size_map)

    n_funfams  = len(funfams)
    n_pos_total = len(scatter_df)

    color_reimpl = PALETTE["singleresidue"]
    color_orig   = "#888888"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             gridspec_kw={"wspace": 0.3})

    # ── Panel A: IC hexbin scatter ────────────────────────────────────────────
    ax = axes[0]
    ic_sub = (scatter_df[scatter_df["metric"] == "IC_SingleRes"]
              if not scatter_df.empty else pd.DataFrame())
    if not ic_sub.empty:
        orig_ic = ic_sub["orig_val"].to_numpy(dtype=float)
        new_ic  = ic_sub["new_val"].to_numpy(dtype=float)
        cmap_ic = LinearSegmentedColormap.from_list("_ic", ["#f0f0f0", color_reimpl])
        # avg cosine across funfams
        cos_vals = (summary_df[summary_df["metric"] == "IC_SingleRes"]["cosine_sim"]
                    .dropna().to_numpy(dtype=float))
        cos_line = (f"Avg cos sim = {float(np.nanmean(cos_vals)):.4f}"
                    if len(cos_vals) else None)
        draw_scatter_hexbin(ax, pred=new_ic, true=orig_ic, title="",
                            cmap=cmap_ic,
                            xlabel="Original FrustratometeR ICTot",
                            ylabel="Re-implementation ICTot",
                            line_color=color_reimpl,
                            extra_line=cos_line)
    else:
        ax.text(0.5, 0.5, "No IC data found", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Per-position IC agreement\n(IC_SingleRes ICTot)",
                 fontsize=11, fontweight="bold")

    # ── Panel B: per-funfam timing (log-log scatter) ──────────────────────────
    ax = axes[1]
    if not timing_df.empty:
        both = timing_df.dropna(subset=["new_time", "orig_time"])
        if not both.empty:
            n_pos_arr = both["n_positions"].fillna(0).to_numpy(dtype=float)
            sc = ax.scatter(both["orig_time"], both["new_time"],
                            c=n_pos_arr, cmap="viridis",
                            vmin=n_pos_arr.min(), vmax=n_pos_arr.max(),
                            s=45, alpha=0.75, zorder=3)
            tmin = min(both["orig_time"].min(), both["new_time"].min())
            tmax = max(both["orig_time"].max(), both["new_time"].max())
            ax.plot([tmin, tmax], [tmin, tmax], "k--", lw=0.8, alpha=0.5)
            speedup = float((both["orig_time"] / both["new_time"]).median())
            ax.text(0.04, 0.94, f"median speedup = {speedup:.1f}×",
                    transform=ax.transAxes, fontsize=9, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#ccc", alpha=0.85))
            plt.colorbar(sc, ax=ax, label="MSA length (positions)", pad=0.02)
        else:
            ax.text(0.5, 0.5, "No paired timing data", ha="center", va="center",
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Original FrustratometeR time (s)", fontsize=10)
    ax.set_ylabel("Re-implementation time (s)", fontsize=10)
    ax.set_title("Timing per funfam", fontsize=11, fontweight="bold")

    # ── Panel C: time vs MSA length ───────────────────────────────────────────
    ax = axes[2]
    if not timing_df.empty:
        has_size = timing_df.dropna(subset=["n_positions"])
        sns.scatterplot(data=has_size.dropna(subset=["new_time"]),
                        x="n_positions", y="new_time",
                        color=color_reimpl, label="Re-implementation",
                        alpha=0.7, s=35, ax=ax)
        sns.scatterplot(data=has_size.dropna(subset=["orig_time"]),
                        x="n_positions", y="orig_time",
                        color=color_orig, label="FrustratometeR",
                        alpha=0.7, s=35, marker="^", ax=ax)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_yscale("log")
    ax.set_xlabel("MSA length (positions)", fontsize=10)
    ax.set_ylabel("Time (s)  [log scale]", fontsize=10)
    ax.set_title("Scaling: time vs MSA length", fontsize=11, fontweight="bold")

    # ── Two-line suptitle ─────────────────────────────────────────────────────
    fig.text(0.5, 1.02,
             "FrustraEvo: IC agreement and timing comparison",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.01,
             f"{n_funfams} funfams  ·  {n_pos_total:,} IC positions",
             ha="center", va="top", fontsize=10, color="#666")

    out = out_dir / "fig7_frustraevo.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── Figure 8: multichain validation ─────────────────────────────────────────

def _read_pred_file(path: Path, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Read FrstIndex (pred) and TrueFrstIndex (true) from a .frustratometer3.{mode} file."""
    try:
        df = pd.read_csv(path, sep=" ", comment="#")
    except Exception as e:
        print(f"  [fig8] could not read {path.name}: {e}")
        return None
    if "FrstIndex" not in df.columns or "TrueFrstIndex" not in df.columns:
        print(f"  [fig8] {path.name} missing FrstIndex/TrueFrstIndex columns")
        return None
    pred = pd.to_numeric(df["FrstIndex"],     errors="coerce").to_numpy(dtype=float)
    true = pd.to_numeric(df["TrueFrstIndex"], errors="coerce").to_numpy(dtype=float)
    ok   = np.isfinite(pred) & np.isfinite(true)
    return pred[ok], true[ok]


def _try_render_pymol(multichain_dir: Path) -> Path | None:
    """Run generate_pymol_figure.py via the current Python interpreter (uses pymol2 headless API)."""
    import subprocess, sys
    script  = multichain_dir / "generate_pymol_figure.py"
    out_png = multichain_dir / "4z66_pymol.png"
    if out_png.exists():
        print(f"  fig8: using existing PyMOL image {out_png}")
        return out_png
    if not script.exists():
        print("  fig8: generate_pymol_figure.py not found — skipping structure image")
        return None
    print("  fig8: rendering PyMOL figure (headless) …")
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            check=True, timeout=300,
            capture_output=True, text=True,
        )
        if result.stdout:
            print(f"  fig8: {result.stdout.strip()}")
        return out_png if out_png.exists() else None
    except subprocess.CalledProcessError as e:
        print(f"  fig8: pymol render failed:\n{e.stderr[-500:] if e.stderr else ''}")
        return None
    except Exception as e:
        print(f"  fig8: pymol render error: {e}")
        return None


def fig8_multichain(multichain_dir: Path, out_dir: Path):
    """
    Left panel: PyMOL structure image (4Z66 nucleosome, chains coloured by histone type).
    Right 3 panels: hexbin scatter (like fig2) reading FrstIndex/TrueFrstIndex directly
    from the .frustratometer3.{mode} pred files (correctly paired per-row).
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec

    # ── Find pred files ───────────────────────────────────────────────────────
    stems = set()
    for f in multichain_dir.glob("*.frustratometer3.singleresidue"):
        stem = f.name.split(".frustratometer3")[0]
        if all((multichain_dir / f"{stem}.frustratometer3.{m}").is_file()
               for m in MODES):
            stems.add(stem)

    if not stems:
        print(f"  [skip] fig8: no complete .frustratometer3.{{mode}} set in {multichain_dir}")
        return

    stem = sorted(stems)[0]
    print(f"  fig8: using {stem}")

    # ── Load scatter data ─────────────────────────────────────────────────────
    _cmaps = {m: LinearSegmentedColormap.from_list(f"_c_{m}", ["#d0d0d0", PALETTE[m]])
              for m in MODES}

    all_samples = {}
    for mode in MODES:
        path = multichain_dir / f"{stem}.frustratometer3.{mode}"
        res  = _read_pred_file(path, mode)
        if res is not None:
            all_samples[mode] = res

    if not all_samples:
        print("  [skip] fig8: no valid data loaded")
        return

    combined = np.concatenate([p for p, _ in all_samples.values()] +
                               [t for _, t in all_samples.values()])
    combined = combined[np.isfinite(combined)]
    lo  = float(np.nanpercentile(combined, 0.5))
    hi  = float(np.nanpercentile(combined, 99.5))
    pad = (hi - lo) * 0.03
    shared_lim = (lo - pad, hi + pad)

    # ── Render PyMOL image (optional) ────────────────────────────────────────
    pymol_png = _try_render_pymol(multichain_dir)
    has_struct = pymol_png is not None and pymol_png.exists()

    # ── Layout ───────────────────────────────────────────────────────────────
    if has_struct:
        fig = plt.figure(figsize=(18, 5))
        gs  = gridspec.GridSpec(1, 4, figure=fig,
                                width_ratios=[1.5, 1, 1, 1],
                                wspace=0.3)
        ax_struct = fig.add_subplot(gs[0, 0])
        scatter_axes = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
    else:
        fig, scatter_axes = plt.subplots(1, 3, figsize=(13, 5),
                                         gridspec_kw={"wspace": 0.25})

    # ── Structure panel ───────────────────────────────────────────────────────
    if has_struct:
        img = plt.imread(str(pymol_png))
        ax_struct.imshow(img)
        ax_struct.axis("off")

        def _lighten(hex_color, factor=0.40):
            import matplotlib.colors as mc
            r, g, b = mc.to_rgb(hex_color)
            return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)

        legend_items = [
            ("H3.2 — chain A", PALETTE["singleresidue"]),
            ("H3.2 — chain E", _lighten(PALETTE["singleresidue"])),
            ("H4 — chain B",   PALETTE["mutational"]),
            ("H4 — chain F",   _lighten(PALETTE["mutational"])),
            ("H2A — chain C",  PALETTE["configurational"]),
            ("H2A — chain G",  _lighten(PALETTE["configurational"])),
            ("H2B — chain D",  PALETTE["mutations"]),
            ("H2B — chain H",  _lighten(PALETTE["mutations"])),
            ("DNA (I, J)",     "#AAAAAA"),
        ]
        handles = [plt.Line2D([0], [0], color=c, lw=4, label=lbl)
                   for lbl, c in legend_items]
        ax_struct.legend(handles=handles, loc="upper center",
                         bbox_to_anchor=(0.5, -0.06),
                         ncol=2, fontsize=7.5, frameon=False)
        ax_struct.set_title("4Z66 nucleosome\n(8-chain histone octamer + 2 DNA strands)",
                            fontsize=10, fontweight="bold", pad=4)

    # ── Scatter panels ────────────────────────────────────────────────────────
    for col, mode in enumerate(MODES):
        ax    = scatter_axes[col]
        color = PALETTE[mode]

        if mode not in all_samples:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(MODE_LABELS[mode], fontsize=11, fontweight="bold", pad=6)
            continue

        pred, true = all_samples[mode]
        m = compute_metrics(pred, true)
        cos_line = (f"Avg cos sim = {m['cosine_similarity']:.4f}"
                    if np.isfinite(m["cosine_similarity"]) else None)

        ax.set_title(MODE_LABELS[mode], fontsize=11, fontweight="bold", pad=6)
        draw_scatter_hexbin(ax, pred=pred, true=true, title="",
                            cmap=_cmaps[mode], xlabel="", ylabel="",
                            line_color=color, extra_line=cos_line)
        ax.set_xlabel(_XLAB, fontsize=10)
        ax.set_ylabel(_YLAB if col == 0 else "", fontsize=10)
        ax.set_xlim(shared_lim)
        ax.set_ylim(shared_lim)

    # ── Titles ────────────────────────────────────────────────────────────────
    n_res   = len(all_samples.get("singleresidue", ([], []))[0])
    n_pairs = len(all_samples.get("mutational",    ([], []))[0])

    fig.text(0.5, 1.02,
             "Multichain validation: 4Z66 nucleosome",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.01,
             f"8 histone chains + 2 DNA strands  ·  {n_res:,} residues  ·  {n_pairs:,} contact pairs",
             ha="center", va="top", fontsize=10, color="#666")

    out = out_dir / "fig8_multichain.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")



def fig_density_slide(out_dir: Path) -> None:
    """Slide-style figure summarising the residue density (rho) calculation."""
    fig = plt.figure(figsize=(16, 9), facecolor="white")

    # ── title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.95, "Residue Density  ρ  —  Local Burial Indicator",
             ha="center", va="top", fontsize=22, fontweight="bold", color="#1a1a2e")

    # ── left panel: smooth well plot ──────────────────────────────────────────
    ax_well = fig.add_axes([0.05, 0.18, 0.38, 0.68])

    d = np.linspace(0, 12, 500)
    eta, rmin, rmax = 5.0, 4.5, 6.5
    v_theta = 0.25 * (1 + np.tanh(eta * (d - rmin))) * (1 + np.tanh(eta * (rmax - d)))

    ax_well.plot(d, v_theta, color="#2196F3", lw=3, zorder=3)
    ax_well.fill_between(d, v_theta, alpha=0.15, color="#2196F3")

    # annotate well edges
    for xv, lbl in [(rmin, f"$r_{{min}}$ = {rmin} Å"), (rmax, f"$r_{{max}}$ = {rmax} Å")]:
        ax_well.axvline(xv, color="#888", lw=1.2, linestyle="--", zorder=1)
        ax_well.text(xv, 1.05, lbl, ha="center", va="bottom", fontsize=11, color="#555")

    # annotate peak
    d_peak = (rmin + rmax) / 2
    ax_well.annotate("peak ≈ 1.0", xy=(d_peak, 1.0), xytext=(d_peak + 1.5, 0.88),
                     arrowprops=dict(arrowstyle="->", color="#444"),
                     fontsize=11, color="#444")

    # broken hard-gate region shading
    d_gate = np.linspace(rmin - 0.8, rmin, 100)
    v_gate = 0.25 * (1 + np.tanh(eta * (d_gate - rmin))) * (1 + np.tanh(eta * (rmax - d_gate)))
    ax_well.fill_between(d_gate, v_gate, alpha=0.35, color="#E53935",
                         label="Hard-gate bug: zeroed here\n(d < $r_{min}$ but $v_θ$ > 0)")
    ax_well.legend(loc="upper right", fontsize=10, framealpha=0.9)

    ax_well.set_xlabel("Cβ – Cβ distance  $d_{ij}$  (Å)", fontsize=13)
    ax_well.set_ylabel("$v_θ(d)$", fontsize=13)
    ax_well.set_title("Smooth tanh contact well", fontsize=13, fontweight="bold")
    ax_well.set_xlim(0, 12)
    ax_well.set_ylim(-0.05, 1.25)
    ax_well.spines["top"].set_visible(False)
    ax_well.spines["right"].set_visible(False)

    # ── right panel: formulas + bullets ───────────────────────────────────────
    ax_txt = fig.add_axes([0.50, 0.10, 0.47, 0.80])
    ax_txt.axis("off")

    def _box(y, title, body, title_color="#1565C0"):
        ax_txt.text(0.0, y, title, transform=ax_txt.transAxes,
                    fontsize=13, fontweight="bold", color=title_color, va="top")
        ax_txt.text(0.0, y - 0.06, body, transform=ax_txt.transAxes,
                    fontsize=12, va="top", color="#222",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#BDBDBD", lw=0.8))

    _box(0.98, "Per-residue density  (local burial)",
         r"$\rho_i = \sum_{j,\,|i-j|>2}\ v_\theta(d_{ij})$"
         "\n\n"
         r"$v_\theta(d) = \frac{1}{4}"
         r"(1+\tanh[\eta(d-r_{min})])"
         r"(1+\tanh[\eta(r_{max}-d)])$"
         "\n\n"
         r"$\eta=5.0,\quad r_{min}=4.5\,\AA,\quad r_{max}=6.5\,\AA$")

    _box(0.55, "Downstream use: σ_water(i, j)",
         r"$\sigma_w = \frac{1}{4}"
         r"(1-\tanh[\eta_\sigma(\rho_i-\rho_0)])"
         r"(1-\tanh[\eta_\sigma(\rho_j-\rho_0)])$"
         "\n\nDetermines whether a contact is water-mediated or direct.",
         title_color="#6a1b9a")

    _box(0.28, "Two bugs in the initial PyTorch port  →  fixed",
         "[1] Wrong Cb coords -- IPA-rotated virtual Cb instead of real PDB Cb\n"
         "    Fix: use raw coordinate tensor (X_raw) for distance computation\n\n"
         "[2] Hard in-well gate  (d >= r_min) & (d <= r_max)  zeroed valid pairs\n"
         "    Fix: remove gate -- let tanh decay handle boundaries smoothly",
         title_color="#B71C1C")

    # ── footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.03,
             "Correct implementation: all-vs-all PyTorch · real Cβ/Cα coords · "
             "no hard gate · |Δres| > 2 exclusion · 9.5 Å neighbor cutoff (≈ 0 beyond)",
             ha="center", fontsize=10, color="#555", style="italic")

    out_dir.mkdir(parents=True, exist_ok=True)
    _savefig(fig, out_dir / "density_slide.png")
    plt.close(fig)
    print(f"  wrote {out_dir / 'density_slide.png'}")


def fig9_per_protein(
    per_protein_img_dir: Path | None,
    results_csv: Path | None,
    out_dir: Path,
    families: list[str] | None = None,
    mode: str = "singleresidue",
    bottom_pct: float = 5.0,
    top_pct: float = 5.0,
):
    """Per-protein comparison: bottom / median / top percentile by Pearson R.

    Loads pre-generated *_per_protein.png images from per_protein_img_dir/<family>/
    and assembles a 3-row figure with circled tier labels (A/B/C).
    Representatives are selected from results_csv (all_results.csv) by Pearson R.
    """
    import matplotlib.image as mpimg

    if families is None:
        families = ["beta_lactamase", "globin", "top7"]

    if per_protein_img_dir is None or not per_protein_img_dir.exists():
        print("  fig9: per_protein_img_dir not found — skip")
        return
    if results_csv is None or not results_csv.exists():
        print("  fig9: results_csv not found — skip")
        return

    df = _drop_excluded(pd.read_csv(results_csv))
    df_mode = df[df["mode"] == mode].copy()

    def _extract_domain(s: str) -> str:
        return Path(s).stem.split(".")[0]

    df_mode = df_mode.copy()
    df_mode["domain"] = df_mode["structure"].apply(_extract_domain)
    df_mode = df_mode.dropna(subset=["pearson_r"])

    tier_defs = [
        ("A", bottom_pct / 100.0,        TIER_DESCS["A"]),
        ("B", 0.50,                       TIER_DESCS["B"]),
        ("C", 1.0 - top_pct / 100.0,     TIER_DESCS["C"]),
    ]

    for family in families:
        fam_img_dir = per_protein_img_dir / family
        if not fam_img_dir.exists():
            print(f"  fig9: {family} image dir not found — skip")
            continue

        avail = {p.stem.replace("_per_protein", "")
                 for p in fam_img_dir.glob("*_per_protein.png")}
        if not avail:
            print(f"  fig9: no per_protein images in {fam_img_dir} — skip")
            continue

        fam_df = df_mode[df_mode["domain"].isin(avail)].copy()
        if fam_df.empty:
            print(f"  fig9: no metric data for {family} domains — skip")
            continue
        fam_df = fam_df.sort_values("pearson_r").reset_index(drop=True)

        reps = []
        seen_domains: set[str] = set()
        for tier, q, tier_desc in tier_defs:
            target = fam_df["pearson_r"].quantile(q)
            cand   = fam_df[~fam_df["domain"].isin(seen_domains)]
            if cand.empty:
                cand = fam_df
            idx    = (cand["pearson_r"] - target).abs().idxmin()
            domain = cand.loc[idx, "domain"]
            r_val  = float(cand.loc[idx, "pearson_r"])
            reps.append(dict(tier=tier, tier_desc=tier_desc,
                             domain=domain, pearson_r=r_val))
            seen_domains.add(domain)
            print(f"  fig9  {family:20s}  tier {tier} ({tier_desc:10s}): "
                  f"{domain}  R={r_val:.4f}")

        n_rows = len(reps)
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5.5 * n_rows), squeeze=False)

        for row_idx, rep in enumerate(reps):
            ax  = axes[row_idx, 0]
            png = fam_img_dir / f"{rep['domain']}_per_protein.png"
            if png.exists():
                img = mpimg.imread(str(png))
                ax.imshow(img)
            ax.axis("off")

            # circled tier label (A / B / C)
            ax.annotate(
                rep["tier"],
                xy=(0.005, 0.5), xycoords="axes fraction",
                fontsize=15, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=2),
            )
            ax.annotate(
                f"{rep['tier_desc']}  (R = {rep['pearson_r']:.4f})",
                xy=(0.04, 0.97), xycoords="axes fraction",
                fontsize=10, ha="left", va="top", style="italic", color="#444",
            )

        fig.suptitle(
            f"{family.replace('_', ' ').title()}  —  "
            "Original FrustratometeR vs PyTorch Re-implementation\n"
            f"Representatives by Pearson R per protein  ({mode})",
            fontsize=12, fontweight="bold", y=1.005,
        )
        plt.tight_layout(rect=[0, 0, 1, 1])
        out = out_dir / f"fig9_{family}.png"
        _savefig(fig, out)
        plt.close(fig)
        print(f"  wrote {out}")


# ─── Figure FR-rerun: FrustratometeR run-vs-run reproducibility ───────────────

def _read_fr_fi(path: Path, mode: str, cache: bool = False) -> pd.DataFrame | None:
    """Read a FrustratometeR FrustrationData file, return DataFrame with key+FrstIndex.

    Returns None if the file is missing/unreadable. With cache=True, files already
    read as the ref side of fig1/fig2 (--ref-dir) are reused instead of re-read.
    """
    return _read_raw_output_file(str(path), cache=cache)


def _merge_key(mode: str, df1: pd.DataFrame, df2: pd.DataFrame) -> list[str]:
    """Return the join key columns present in both dataframes."""
    if mode == "singleresidue":
        base = ["Res"]
        extended = ["Res", "ChainRes"]
    else:
        base = ["Res1", "Res2"]
        extended = ["Res1", "Res2", "ChainRes1", "ChainRes2"]
    if all(k in df1.columns and k in df2.columns for k in extended):
        return extended
    return base


def _load_fr_one_protein(args):
    """Read and merge one protein's FI for a given mode.

    Args is a tuple (f1, f2, mode) — must be a module-level function so it is
    picklable by multiprocessing.  Returns (mode, fi1, fi2, gly_mask) or None.
    gly_mask is a bool array (True where native AA is GLY); None for pair modes.
    """
    f1, f2, mode = args
    # Run 1 is normally the --ref-dir tree: reuse what fig1/fig2 already read.
    # Only run 2 is actually read from disk here.
    df1 = _read_fr_fi(Path(f1), mode, cache=True)
    if df1 is None:
        return None
    df2 = _read_fr_fi(Path(f2), mode)
    if df2 is None:
        return None
    key = _merge_key(mode, df1, df2)
    if not all(k in df1.columns and k in df2.columns for k in key):
        return None
    # Include AA column from df1 when available (singleresidue has "AA")
    extra = [c for c in ("AA",) if c in df1.columns]
    merged = (df1[key + ["FrstIndex"] + extra].rename(columns={"FrstIndex": "fi1"})
              .merge(df2[key + ["FrstIndex"]].rename(columns={"FrstIndex": "fi2"}),
                     on=key, how="inner"))
    fi1 = merged["fi1"].to_numpy(dtype=float)
    fi2 = merged["fi2"].to_numpy(dtype=float)
    ok  = np.isfinite(fi1) & np.isfinite(fi2)
    gly_mask = None
    if "AA" in merged.columns:
        aa_upper = merged["AA"].astype(str).str.upper()
        gly_mask = aa_upper.isin(["GLY", "G"]).to_numpy()[ok]
    return mode, fi1[ok], fi2[ok], gly_mask


def _load_all_fr_pairs(run1_dir: Path, run2_dir: Path, workers: int | None = None):
    """Load FrstIndex pairs for all modes and proteins in a thread pool.

    Returns dict {mode: (fi_run1, fi_run2, n_proteins)}.
    Uses threads (not processes) — GPFS reads are I/O-bound so threads beat
    processes by avoiding spawn overhead and pickle serialisation of results.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Build task list across all modes (glob only; no per-file existence checks —
    # missing files just make _load_fr_one_protein return None)
    done_dirs = sorted(run1_dir.glob("*.done"))
    print(f"  found {len(done_dirs)} *.done dirs in run1_dir", flush=True)

    tasks = []
    for done1 in done_dirs:
        pdb_id = done1.stem
        if pdb_id in EXCLUDE_DOMAINS:
            continue
        for mode in MODES:
            f1 = done1 / "FrustrationData" / f"{pdb_id}.pdb_{mode}"
            f2 = run2_dir / f"{pdb_id}.done" / "FrustrationData" / f"{pdb_id}.pdb_{mode}"
            tasks.append((str(f1), str(f2), mode))

    n_workers = max(1, min(workers or 64, len(done_dirs)))

    print(f"  {len(tasks)} tasks across {n_workers} threads", flush=True)
    buckets: dict[str, tuple[list, list, list, int]] = {m: ([], [], [], 0) for m in MODES}
    if not tasks:
        return {m: (np.empty(0), np.empty(0), 0, None) for m in MODES}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_load_fr_one_protein, tasks):
            if result is None:
                continue
            mode, fi1, fi2, gly = result
            a1, a2, ag, n = buckets[mode]
            a1.append(fi1)
            a2.append(fi2)
            if gly is not None:
                ag.append(gly)
            buckets[mode] = (a1, a2, ag, n + 1)

    out = {}
    for m, (a1, a2, ag, n) in buckets.items():
        gly_mask = np.concatenate(ag) if ag else None
        out[m] = (np.concatenate(a1) if a1 else np.empty(0),
                  np.concatenate(a2) if a2 else np.empty(0),
                  n, gly_mask)
    return out


def fig_fr_rerun(run1_dir: Path, run2_dir: Path, out_dir: Path,
                 run1_label: str = "Run 1", run2_label: str = "Run 2"):
    """1×3 hexbin scatter comparing two FrustratometeR output directories.

    For each mode, reads {pdb_id}.done/FrustrationData/{pdb_id}.pdb_{mode} from
    both dirs, merges on residue (or pair) keys, and plots FI run1 vs FI run2.
    Points should lie on the diagonal if the reruns are identical.
    """
    from matplotlib.colors import LinearSegmentedColormap

    _cmaps = {m: LinearSegmentedColormap.from_list(f"_c_{m}", ["#d0d0d0", PALETTE[m]])
              for m in MODES}

    print("  loading FrustratometeR pairs (all modes, multiprocessing)…")
    mode_data = _load_all_fr_pairs(run1_dir, run2_dir)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), gridspec_kw={"wspace": 0.28},
                             layout="constrained")

    total_prot = 0
    for col, mode in enumerate(MODES):
        ax    = axes[col]
        color = PALETTE[mode]

        fi1, fi2, n_prot, gly_mask = mode_data[mode]
        total_prot = max(total_prot, n_prot)
        lo, hi = MODE_THRESHOLDS.get(mode, (LO_PAIR, HI_PAIR))

        if len(fi1) == 0:
            ax.text(0.5, 0.5, "no matched data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        else:
            m = compute_metrics(fi2, fi1)
            cos_line = (f"cos sim = {m['cosine_similarity']:.6f}"
                        if np.isfinite(m["cosine_similarity"]) else None)
            draw_scatter_hexbin(ax, pred=fi2, true=fi1, title="",
                                cmap=_cmaps[mode], xlabel="", ylabel="",
                                line_color=color, extra_line=cos_line,
                                gly_mask=gly_mask, lo=lo, hi=hi)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

        ax.set_title(MODE_LABELS[mode], fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel(run1_label, fontsize=10)
        ax.set_ylabel(run2_label if col == 0 else "", fontsize=10)

    fig.text(0.5, 1.02,
             f"FrustratometeR reproducibility: {run1_label} vs {run2_label}",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.text(0.5, 1.005,
             f"{total_prot} proteins",
             ha="center", va="top", fontsize=10, color="#666")

    out = out_dir / "fig_fr_rerun.png"
    _savefig(fig, out)
    plt.close(fig)
    print(f"  wrote {out}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=None,
                   help="frustration output directory containing {pdb}.done/ subdirs "
                        "(new format). Sets --samples to this dir and makes --results optional.")
    p.add_argument("--results",
                   default=None,
                   help="all_results.csv from frustratometer3 test run (optional with --output-dir)")
    p.add_argument("--samples",
                   default="baseline_comparison/frustratometer3/results/run_info",
                   help="directory with residue_samples_*.csv (legacy; prefer --output-dir)")
    p.add_argument("--mac",
                   default="baseline_comparison/length_test/mac_timings.csv")
    p.add_argument("--mn5",
                   default="baseline_comparison/length_test/mn5_timings.csv")
    p.add_argument("--mut-metrics",
                   default="baseline_comparison/mutants/mutations_metrics.csv",
                   help="precomputed mutations_metrics.csv (computed on the fly if absent)")
    p.add_argument("--mut-samples",
                   default="baseline_comparison/mutants",
                   help="directory with *.frustratometer3.singleresidue_mutations files")
    p.add_argument("--mut-ref-dir",
                   default="../dompdb_v4.4_mutants",
                   help="root dir with <stem>.done/MutationsData/ reference trees")
    p.add_argument("--sfam-map",
                   default=None,
                   help="CSV with columns stem,superfamily (optional)")
    p.add_argument("--multichain",
                   default="baseline_comparison/multichain",
                   help="directory containing .frustratometer3.{mode} files for multichain validation")
    p.add_argument("--frustraevo-new-dir",
                   default=None,
                   help="Dir with per-funfam subdirs from the re-implementation FrustraEvo run")
    p.add_argument("--frustraevo-orig-dir",
                   default=None,
                   help="Dir containing original FrustraEvo_* runs")
    p.add_argument("--out",
                   default="results/paper_figures")
    p.add_argument("--per-protein-imgs",
                   default="baseline_comparison/frustratometer3/plots/per_protein",
                   help="directory with per-family per-protein comparison images")
    p.add_argument("--per-protein-families", nargs="+",
                   default=None,
                   help="families to include in fig9 (default: beta_lactamase globin top7)")
    p.add_argument("--figs",  nargs="+",
                   default=["1", "2", "2b", "3", "4", "5", "6", "7", "8", "9"],
                   help="which figures to generate (default: all). "
                        "Use '2b' for hexbin scatter, 'fr_rerun' for reproducibility.")
    p.add_argument("--fr-run1",
                   default=None,
                   help="Root dir of the first  FrustratometeR run (contains *.done/ subdirs)")
    p.add_argument("--fr-run2",
                   default=None,
                   help="Root dir of the second FrustratometeR run (rerun to compare)")
    p.add_argument("--fr-run1-label", default="Run 1",
                   help="Legend label for the first  FrustratometeR run  (default: 'Run 1')")
    p.add_argument("--fr-run2-label", default="Rerun",
                   help="Legend label for the second FrustratometeR run (default: 'Rerun')")
    p.add_argument("--ref-dir",
                   default=None,
                   help="Root dir of FrustratometeR reference outputs (*.done/FrustrationData/ "
                        "layout). When supplied, fig2/fig2b join our FrstIndex against the ref "
                        "FrstIndex instead of requiring TrueFrstIndex in our output files. "
                        "Useful when --compare_to_original was not used during the run.")
    p.add_argument("--skip", nargs="+", default=[], metavar="DOMAIN",
                   help="Domain IDs to exclude from all figures (e.g. --skip 2vx2A01 5zwnx00).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = set(args.figs)

    # --output-dir overrides --samples and makes --results optional
    samples_dir    = Path(args.output_dir) if args.output_dir else Path(args.samples)
    results_csv    = Path(args.results) if args.results else None
    ref_dir        = Path(args.ref_dir)  if args.ref_dir  else None
    exclude_domains = frozenset(args.skip) if args.skip else None

    print("Generating paper figures →", out_dir)

    if "1" in figs:
        print("\n── Figure 1: schematic ──")
        fig1_schematic(out_dir)

    if "2" in figs:
        print("\n── Figure 2: scatter ──")
        fig2_scatter(samples_dir, out_dir, results_csv=results_csv, ref_dir=ref_dir,
                     exclude_domains=exclude_domains)

    if "2b" in figs:
        print("\n── Figure 2b: hexbin ──")
        fig2_hexbin(samples_dir, out_dir, results_csv=results_csv, ref_dir=ref_dir,
                    exclude_domains=exclude_domains)

    if "3" in figs:
        print("\n── Figure 3: per-protein KDEs ──")
        fig3_kdeplots(results_csv, out_dir,
                      samples_dir=samples_dir)

    if "4" in figs:
        print("\n── Figure 4: speed comparison ──")
        fig4_speed(args.mac, args.mn5, out_dir)

    if "5" in figs:
        print("\n── Figure 5: log-log scaling ──")
        fig5_scaling(args.mac, args.mn5, out_dir)

    if "6" in figs:
        print("\n── Figure 6: mutational scanning ──")
        fig6_mutations(
            Path(args.mut_metrics) if args.mut_metrics else None,
            Path(args.mut_samples) if args.mut_samples else None,
            Path(args.mut_ref_dir) if args.mut_ref_dir else None,
            Path(args.sfam_map)    if args.sfam_map    else None,
            out_dir,
            mac_path=args.mac,
            mn5_path=args.mn5,
        )

    if "8" in figs:
        print("\n── Figure 8: multichain ──")
        fig8_multichain(
            Path(args.multichain) if args.multichain else Path("baseline_comparison/multichain"),
            out_dir,
        )

    if "7" in figs:
        print("\n── Figure 7: FrustraEvo ──")
        fig7_frustraevo(
            Path(args.frustraevo_new_dir) if args.frustraevo_new_dir else None,
            Path(args.frustraevo_orig_dir) if args.frustraevo_orig_dir else None,
            out_dir,
        )

    if "9" in figs:
        print("\n── Figure 9: per-protein representatives ──")
        fig9_per_protein(
            Path(args.per_protein_imgs) if args.per_protein_imgs else None,
            results_csv,
            out_dir,
            families=args.per_protein_families,
        )

    if "density" in figs:
        print("\n── Density slide ──")
        fig_density_slide(out_dir)

    if "fr_rerun" in figs:
        print("\n── FrustratometeR reproducibility (fr_rerun) ──")
        if args.fr_run1 and args.fr_run2:
            fig_fr_rerun(
                Path(args.fr_run1), Path(args.fr_run2), out_dir,
                run1_label=args.fr_run1_label,
                run2_label=args.fr_run2_label,
            )
        else:
            print("  [skip] fr_rerun: --fr-run1 and --fr-run2 are required")

    print(f"\nDone. All figures written to {out_dir}/")


if __name__ == "__main__":
    main()
