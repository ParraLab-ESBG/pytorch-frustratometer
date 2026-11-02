"""
Investigate configurational off-diagonal outliers: residue pairs where
PyTorch FrstIndex deviates most from R.

Produces:
  offdiagonal_scatter_{pid}.png  — scatter coloured by seq-sep and Cα distance
  offdiagonal_pairs_{pid}.csv    — ranked off-diagonal pair table

Usage:
    python -m frustrai_fold.plots.offdiagonal_analysis \\
        --dompdb  ../dompdb_v4.4 \\
        --pid     2hytA00 \\
        --out_dir baseline_comparison/frustratometer3/offdiagonal \\
        --decoys  1000

    # Debug intermediates for a specific pair:
    python -m frustrai_fold.plots.offdiagonal_analysis \\
        --dompdb  ../dompdb_v4.4 \\
        --pid     2hytA00 \\
        --debug_pair 12,45
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

RC = {
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
}

# ── PDB helpers ───────────────────────────────────────────────────────────────

def parse_ca_positions(pdb_path: Path) -> dict[int, np.ndarray]:
    """Return {resnum: [x,y,z]} for Cα atoms (first model, first chain)."""
    ca = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line[:4] not in ("ATOM", "HETA"):
                continue
            if line[12:16].strip() != "CA":
                continue
            resnum = int(line[22:26])
            if resnum not in ca:
                ca[resnum] = np.array([float(line[30:38]),
                                        float(line[38:46]),
                                        float(line[46:54])])
    return ca


def ca_distance(ca: dict[int, np.ndarray], r1: int, r2: int) -> float:
    if r1 in ca and r2 in ca:
        return float(np.linalg.norm(ca[r1] - ca[r2]))
    return np.nan


# ── R file loader ─────────────────────────────────────────────────────────────

def load_r_configurational(dompdb_dir: Path, pid: str) -> pd.DataFrame | None:
    path = dompdb_dir / f"{pid}.done" / "FrustrationData" / f"{pid}.pdb_configurational"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    for col in ("Res1", "Res2", "NativeEnergy", "DecoyEnergy", "SDEnergy", "FrstIndex"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Res1", "Res2", "FrstIndex"])
    return df[df["FrstIndex"] != 0.0].reset_index(drop=True)


# ── PyTorch model runner ──────────────────────────────────────────────────────

def run_pytorch(pdb_path: Path, n_decoys: int, use_c_rand: bool,
                capture_intermediates: bool = False):
    """Run configurational model; return (pairs_df, intermediates_dict)."""
    import torch
    from pytorch_frustratometer.coefficients import Coefficients
    from pytorch_frustratometer.data.dataset import (
        FrustratometerDataset, collate_frustratometer,
    )
    from pytorch_frustratometer.algorithm.configurational import ConfigurationalFrustratometer
    from torch.utils.data import DataLoader

    coeff = Coefficients.from_dict({
        "mode": "configurational",
        "tert_frust_ndecoys": n_decoys,
        "min_seq_distance": 3,
    })
    dataset = FrustratometerDataset(
        [str(pdb_path)], coeff, compare_to_original=True,
        dtype=torch.float32,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=collate_frustratometer)
    model = ConfigurationalFrustratometer(coeff, dtype=torch.float32)
    model.use_c_rand = use_c_rand

    rows, intermediates = [], {}

    for batch in loader:
        with torch.no_grad():
            result = model(batch)

        fi         = result["configurational"][0]   # [E]
        mask       = batch["configurational_frust_mask"][0].bool()
        edge_index = batch["edge_index"][0]          # [2, E]
        ri         = batch.get("residue_index", [None])[0]

        if capture_intermediates:
            intermediates["native_energy"]  = result.get("_configurational_native_energy", [None])[0]  # [E]
            intermediates["decoy_stats"]    = result.get("_configurational_decoy_stats",   [None])[0]  # [2]
            intermediates["edge_index"]     = edge_index
            intermediates["residue_index"]  = ri
            intermediates["fi"]             = fi
            intermediates["mask"]           = mask
            # R reference fields loaded by dataset
            intermediates["r_native"]  = batch.get("configurational_native_energy",  [None])[0]
            intermediates["r_decoy"]   = batch.get("configurational_decoy_energy",   [None])[0]
            intermediates["r_sd"]      = batch.get("configurational_sd_energy",      [None])[0]
            intermediates["r_fi"]      = batch.get("configurational_frust_true",     [None])[0]

        for e in range(fi.shape[0]):
            if not mask[e]:
                continue
            src, dst = edge_index[0, e].item(), edge_index[1, e].item()
            rows.append({
                "Res1":      ri[src].item() if ri is not None else src,
                "Res2":      ri[dst].item() if ri is not None else dst,
                "FrstIndex": fi[e].item(),
                "edge_idx":  e,
            })

    return pd.DataFrame(rows), intermediates


# ── debug dump for a single pair ──────────────────────────────────────────────

def debug_pair(dompdb_dir: Path, pid: str, res1: int, res2: int,
               n_decoys: int, use_c_rand: bool):
    import torch
    pdb = dompdb_dir / f"{pid}.done" / "FrustrationData" / f"{pid}.pdb"
    r_df = load_r_configurational(dompdb_dir, pid)
    pt_df, inter = run_pytorch(pdb, n_decoys, use_c_rand, capture_intermediates=True)

    print(f"\n{'='*65}")
    print(f"DEBUG PAIR  {pid}  Res1={res1}  Res2={res2}  "
          f"{'c_rand' if use_c_rand else 'torch_rand'}  D={n_decoys}")
    print(f"{'='*65}")

    # ── R reference ───────────────────────────────────────────────────────────
    r_row = r_df[(r_df["Res1"] == res1) & (r_df["Res2"] == res2)]
    if r_row.empty:
        r_row = r_df[(r_df["Res1"] == res2) & (r_df["Res2"] == res1)]
    if r_row.empty:
        print("  WARNING: pair not found in R reference file")
        r_nat = r_dec = r_sd = r_fi = float("nan")
    else:
        r_row = r_row.iloc[0]
        r_nat = r_row.get("NativeEnergy", float("nan"))
        r_dec = r_row.get("DecoyEnergy",  float("nan"))
        r_sd  = r_row.get("SDEnergy",     float("nan"))
        r_fi  = r_row.get("FrstIndex",    float("nan"))

    # ── PyTorch ───────────────────────────────────────────────────────────────
    pt_row = pt_df[(pt_df["Res1"] == res1) & (pt_df["Res2"] == res2)]
    if pt_row.empty:
        pt_row = pt_df[(pt_df["Res1"] == res2) & (pt_df["Res2"] == res1)]
    if pt_row.empty:
        print("  WARNING: pair not in PyTorch output")
        return

    edge_e = int(pt_row.iloc[0]["edge_idx"])
    pt_fi  = pt_row.iloc[0]["FrstIndex"]

    nat_t  = inter["native_energy"]
    stats  = inter["decoy_stats"]
    pt_nat = nat_t[edge_e].item()   if nat_t  is not None else float("nan")
    pt_dec = stats[0].item()        if stats   is not None else float("nan")
    pt_sd  = stats[1].item()        if stats   is not None else float("nan")

    # Dataset-loaded R fields (may be None if dataset didn't load them)
    ri_nat = inter["r_native"]
    ri_dec = inter["r_decoy"]
    ri_sd  = inter["r_sd"]
    ri_fi  = inter["r_fi"]

    ri = inter["residue_index"]
    src_idx, dst_idx = inter["edge_index"][0, edge_e].item(), inter["edge_index"][1, edge_e].item()
    pt_res1 = ri[src_idx].item() if ri is not None else src_idx
    pt_res2 = ri[dst_idx].item() if ri is not None else dst_idx

    print(f"\n  {'Field':<22} {'R reference':>18}  {'PyTorch':>18}  {'Δ':>12}")
    print(f"  {'-'*22}  {'-'*18}  {'-'*18}  {'-'*12}")

    def row(name, rv, pv):
        d = pv - rv if (np.isfinite(rv) and np.isfinite(pv)) else float("nan")
        print(f"  {name:<22}  {rv:>18.6f}  {pv:>18.6f}  {d:>+12.6f}")

    row("NativeEnergy",     r_nat,  pt_nat)
    row("DecoyEnergy (mean)", r_dec, pt_dec)
    row("SDEnergy (std)",   r_sd,   pt_sd)
    row("FrstIndex",        r_fi,   pt_fi)

    print(f"\n  Computed FI (PyTorch):  ({pt_nat:.6f} - {pt_dec:.6f}) / {pt_sd:.6f}"
          f"  =  {(pt_nat - pt_dec) / pt_sd if pt_sd else float('nan'):.6f}")

    # Dataset-loaded R fields (alignment check)
    if ri_nat is not None and ri_fi is not None:
        # find the same edge in the dataset-provided tensors
        edge_mask = inter["mask"]
        print(f"\n  Dataset R fields @ edge {edge_e}:")
        print(f"    NativeEnergy  = {ri_nat[edge_e].item():.6f}" if ri_nat is not None else "    NativeEnergy  = N/A")
        print(f"    DecoyEnergy   = {ri_dec[edge_e].item():.6f}" if ri_dec is not None else "    DecoyEnergy   = N/A")
        print(f"    SDEnergy      = {ri_sd[edge_e].item():.6f}"  if ri_sd  is not None else "    SDEnergy      = N/A")
        print(f"    FrstIndex     = {ri_fi[edge_e].item():.6f}"  if ri_fi  is not None else "    FrstIndex     = N/A")

    print(f"\n  Diagnosis:")
    if abs(r_nat - pt_nat) > 0.01:
        print(f"    !! NativeEnergy differs by {r_nat - pt_nat:+.4f} → energy formula issue")
    elif abs(r_dec - pt_dec) > 0.05:
        print(f"    !! DecoyEnergy (background mean) differs by {r_dec - pt_dec:+.4f} → decoy sampling / RNG issue")
    elif abs(r_sd - pt_sd) > 0.05:
        print(f"    !! SDEnergy differs by {r_sd - pt_sd:+.4f} → decoy sampling / RNG issue")
    else:
        print(f"    ✓ All intermediates agree — FI difference is numerical noise")


# ── scatter figure ────────────────────────────────────────────────────────────

def make_scatter(merged: pd.DataFrame, ca: dict, pid: str, out_dir: Path):
    """2-panel scatter: left coloured by seq-sep, right by Cα distance."""
    if merged.empty:
        print(f"  No merged pairs to plot for {pid}", flush=True)
        return

    merged = merged.copy()
    merged["seq_sep"]  = (merged["Res2"] - merged["Res1"]).abs()
    merged["ca_dist"]  = merged.apply(
        lambda r: ca_distance(ca, int(r["Res1"]), int(r["Res2"])), axis=1)
    merged["abs_err"]  = (merged["FrstIndex"] - merged["TrueFrstIndex"]).abs()

    lim = (-5, 5)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

        for ax, col, label, cmap in [
            (axes[0], "seq_sep",  "Sequence separation |Res2−Res1|", "plasma"),
            (axes[1], "ca_dist", "Cα distance (Å)",                  "viridis"),
        ]:
            valid = merged[col].notna()
            sc = ax.scatter(
                merged.loc[valid, "TrueFrstIndex"],
                merged.loc[valid, "FrstIndex"],
                c=merged.loc[valid, col],
                cmap=cmap, s=6, alpha=0.6, linewidths=0,
                vmin=merged.loc[valid, col].quantile(0.02),
                vmax=merged.loc[valid, col].quantile(0.98),
            )
            ax.plot(lim, lim, "k--", lw=0.8, zorder=0)
            plt.colorbar(sc, ax=ax, shrink=0.8, label=label)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("R FrstIndex"); ax.set_ylabel("PyTorch FrstIndex")
            ax.set_aspect("equal")
            r   = np.corrcoef(merged["FrstIndex"], merged["TrueFrstIndex"])[0, 1]
            mae = merged["abs_err"].mean()
            ax.text(0.03, 0.97, f"r={r:.3f}  MAE={mae:.3f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

        fig.suptitle(f"Configurational FrstIndex — {pid}", fontsize=11)
        out = out_dir / f"offdiagonal_scatter_{pid}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        out_pdf = out.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {out}", flush=True)


# ── off-diagonal pair table ───────────────────────────────────────────────────

def offdiagonal_table(merged: pd.DataFrame, ca: dict, pid: str,
                       out_dir: Path, threshold: float = 0.5, top: int = 20):
    if merged.empty:
        return
    merged = merged.copy()
    merged["seq_sep"]  = (merged["Res2"] - merged["Res1"]).abs()
    merged["ca_dist"]  = merged.apply(
        lambda r: ca_distance(ca, int(r["Res1"]), int(r["Res2"])), axis=1)
    merged["error"]    = merged["FrstIndex"] - merged["TrueFrstIndex"]
    merged["abs_err"]  = merged["error"].abs()
    merged = merged.sort_values("abs_err", ascending=False)

    out_csv = out_dir / f"offdiagonal_pairs_{pid}.csv"
    merged.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}  ({len(merged)} pairs)", flush=True)

    outliers = merged[merged["abs_err"] >= threshold].head(top)
    if outliers.empty:
        print(f"  No pairs with abs_err >= {threshold}", flush=True)
        return

    print(f"\n  Top off-diagonal pairs (abs_err >= {threshold}):", flush=True)
    cols = ["Res1", "Res2", "TrueFrstIndex", "FrstIndex", "abs_err",
            "seq_sep", "ca_dist"]
    cols = [c for c in cols if c in outliers.columns]
    print(outliers[cols].head(top).to_string(index=False, float_format="{:.3f}".format), flush=True)

    # Summary: are high-error pairs long-range?
    print(f"\n  Correlation of abs_err vs seq_sep:  "
          f"r={merged[['abs_err','seq_sep']].corr().iloc[0,1]:.3f}", flush=True)
    if merged["ca_dist"].notna().any():
        print(f"  Correlation of abs_err vs ca_dist:  "
              f"r={merged[['abs_err','ca_dist']].dropna().corr().iloc[0,1]:.3f}", flush=True)

    # Breakdown by sequence-separation bin
    bins = [0, 5, 12, 24, 50, 9999]
    labels = ["2-4", "5-11", "12-23", "24-49", "50+"]
    merged["sep_bin"] = pd.cut(merged["seq_sep"], bins=bins, labels=labels)
    g = merged.groupby("sep_bin", observed=True)["abs_err"].agg(["mean", "median", "count"])
    print(f"\n  Mean abs_err by sequence-separation bin:\n{g.to_string()}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb",      required=True, type=Path)
    ap.add_argument("--pid",         required=True)
    ap.add_argument("--out_dir",     default="offdiagonal", type=Path)
    ap.add_argument("--decoys",      default=1000, type=int)
    ap.add_argument("--use_c_rand",  action="store_true")
    ap.add_argument("--threshold",   default=0.5, type=float,
                    help="abs_err threshold for off-diagonal pair table")
    ap.add_argument("--top",         default=30, type=int)
    ap.add_argument("--debug_pair",  default=None,
                    help="Res1,Res2  — dump intermediates for this pair")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pid = args.pid

    # ── debug mode ────────────────────────────────────────────────────────────
    if args.debug_pair:
        r1, r2 = map(int, args.debug_pair.split(","))
        debug_pair(args.dompdb, pid, r1, r2, args.decoys, args.use_c_rand)
        return

    pdb = args.dompdb / f"{pid}.done" / "FrustrationData" / f"{pid}.pdb"
    if not pdb.exists():
        print(f"ERROR: PDB not found: {pdb}"); sys.exit(1)

    print(f"Loading R reference for {pid}...", flush=True)
    r_df = load_r_configurational(args.dompdb, pid)
    if r_df is None or r_df.empty:
        print("ERROR: R file not found or empty"); sys.exit(1)
    print(f"  {len(r_df)} pairs", flush=True)

    print(f"Running PyTorch ({'c_rand' if args.use_c_rand else 'torch_rand'}, D={args.decoys})...", flush=True)
    pt_df, _ = run_pytorch(pdb, args.decoys, args.use_c_rand)
    print(f"  {len(pt_df)} pairs", flush=True)

    merged = pt_df.merge(r_df[["Res1", "Res2", "FrstIndex"]].rename(
                             columns={"FrstIndex": "TrueFrstIndex"}),
                          on=["Res1", "Res2"], how="inner")
    print(f"  {len(merged)} matched pairs", flush=True)

    ca = parse_ca_positions(pdb)

    make_scatter(merged, ca, pid, args.out_dir)
    offdiagonal_table(merged, ca, pid, args.out_dir,
                       threshold=args.threshold, top=args.top)


if __name__ == "__main__":
    main()
