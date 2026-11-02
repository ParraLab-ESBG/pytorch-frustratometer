"""
Map per-residue |ΔFI| (PyTorch vs R) onto 3D structure:
  • Writes a PDB with |ΔFI| in the B-factor column
  • Writes a PyMOL .pml script to colour by B-factor (blue=low, red=high)
  • Prints a residue-level summary: where do outliers cluster?

For configurational/mutational modes, |ΔFI| per residue = max over all
pairs involving that residue.  For single-residue it is direct.

Usage:
    python -m frustrai_fold.plots.structural_outlier_map \\
        --dompdb  ../dompdb_v4.4 \\
        --pid     2hytA00 \\
        --mode    configurational \\
        --out_dir baseline_comparison/frustratometer3/structures \\
        --decoys  1000

    # Multiple PIDs:
    python -m frustrai_fold.plots.structural_outlier_map \\
        --dompdb  ../dompdb_v4.4 \\
        --pid     2hytA00 1abc 2xyz \\
        --mode    configurational \\
        --out_dir ...
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── R loaders ─────────────────────────────────────────────────────────────────

def _load_r(dompdb_dir: Path, pid: str, mode: str) -> pd.DataFrame | None:
    suffix = {"configurational": "pdb_configurational",
               "mutational":      "pdb_mutational",
               "singleresidue":   "pdb_singleresidue"}[mode]
    path = dompdb_dir / f"{pid}.done" / "FrustrationData" / f"{pid}.{suffix}"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    if mode == "singleresidue":
        for c in ("Res", "FrstIndex"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Res", "FrstIndex"])
        return df[df["FrstIndex"] != 0.0]
    else:
        for c in ("Res1", "Res2", "FrstIndex"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Res1", "Res2", "FrstIndex"])
        return df[df["FrstIndex"] != 0.0]


# ── PyTorch runner ────────────────────────────────────────────────────────────

def run_pytorch(pdb_path: Path, mode: str, n_decoys: int,
                use_c_rand: bool) -> pd.DataFrame:
    import torch
    from pytorch_frustratometer.coefficients import Coefficients
    from pytorch_frustratometer.data.dataset import (
        FrustratometerDataset, collate_frustratometer,
    )
    from torch.utils.data import DataLoader

    coeff = Coefficients.from_dict({
        "mode": mode, "tert_frust_ndecoys": n_decoys, "min_seq_distance": 3,
    })
    dataset = FrustratometerDataset(
        [str(pdb_path)], coeff, compare_to_original=True, dtype=torch.float32,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0,
                        collate_fn=collate_frustratometer)

    if mode == "configurational":
        from pytorch_frustratometer.algorithm.configurational import ConfigurationalFrustratometer
        model = ConfigurationalFrustratometer(coeff, dtype=torch.float32)
        model.use_c_rand = use_c_rand
    else:
        from pytorch_frustratometer.frustratometer import Frustratometer3
        model = Frustratometer3(coeff, dtype=torch.float32)
        model.awsem.use_c_rand = use_c_rand

    rows = []
    for batch in loader:
        with torch.no_grad():
            result = model(batch)
        ri = batch.get("residue_index", [None])[0]

        if mode == "singleresidue":
            fi   = result["singleresidue"][0]
            mask = batch["single_frust_mask"][0].bool()
            for l in range(fi.shape[0]):
                if mask[l]:
                    rows.append({"Res": ri[l].item() if ri is not None else l,
                                  "FrstIndex": fi[l].item()})
        else:
            key      = mode
            mask_key = f"{key}_frust_mask"
            fi        = result[key][0]
            mask      = batch[mask_key][0].bool()
            ei        = batch["edge_index"][0]
            for e in range(fi.shape[0]):
                if mask[e]:
                    src, dst = ei[0, e].item(), ei[1, e].item()
                    rows.append({"Res1": ri[src].item() if ri is not None else src,
                                  "Res2": ri[dst].item() if ri is not None else dst,
                                  "FrstIndex": fi[e].item()})
    return pd.DataFrame(rows)


# ── per-residue |ΔFI| ─────────────────────────────────────────────────────────

def per_residue_dfi(pt_df: pd.DataFrame, r_df: pd.DataFrame,
                    mode: str) -> pd.Series:
    """Return Series indexed by residue number, value = max |ΔFI| over pairs."""
    if mode == "singleresidue":
        merged = pt_df.merge(r_df[["Res", "FrstIndex"]].rename(
                                 columns={"FrstIndex": "TrueFrstIndex"}),
                              on="Res", how="inner")
        merged["abs_err"] = (merged["FrstIndex"] - merged["TrueFrstIndex"]).abs()
        return merged.groupby("Res")["abs_err"].max()
    else:
        merged = pt_df.merge(r_df[["Res1", "Res2", "FrstIndex"]].rename(
                                 columns={"FrstIndex": "TrueFrstIndex"}),
                              on=["Res1", "Res2"], how="inner")
        merged["abs_err"] = (merged["FrstIndex"] - merged["TrueFrstIndex"]).abs()
        # take max |ΔFI| over all pairs touching each residue
        s1 = merged.groupby("Res1")["abs_err"].max()
        s2 = merged.groupby("Res2")["abs_err"].max()
        return pd.concat([s1, s2]).groupby(level=0).max()


# ── PDB B-factor writer ───────────────────────────────────────────────────────

def write_bfactor_pdb(pdb_in: Path, pdb_out: Path,
                       dfi: pd.Series, missing_val: float = 0.0):
    """Write a copy of pdb_in with B-factor replaced by dfi[resnum]."""
    max_dfi = dfi.max() if len(dfi) > 0 else 1.0
    lines_out = []
    with open(pdb_in) as fh:
        for line in fh:
            if line[:4] in ("ATOM", "HETA"):
                resnum = int(line[22:26])
                val = dfi.get(resnum, missing_val)
                # scale to 0-99.99 for visualisation clarity
                scaled = min(val / max(max_dfi, 1e-6) * 99.99, 99.99)
                line = line[:60] + f"{scaled:6.2f}" + line[66:]
            lines_out.append(line)
    with open(pdb_out, "w") as fh:
        fh.writelines(lines_out)


# ── PyMOL script writer ───────────────────────────────────────────────────────

def write_pymol_script(pml_out: Path, pdb_out: Path,
                        pid: str, mode: str, max_dfi: float):
    """Write a .pml that loads the B-factor PDB and colours by |ΔFI|."""
    pdb_rel = pdb_out.name
    with open(pml_out, "w") as fh:
        fh.write(f"""\
# PyMOL script — {pid} {mode} |ΔFI| map
# B-factor is scaled:  bfactor=99.99 → max|ΔFI|={max_dfi:.3f}

load {pdb_rel}, {pid}

hide everything
show cartoon, {pid}
set cartoon_transparency, 0.1

# Colour: blue=low error, white=intermediate, red=high error
spectrum b, blue_white_red, {pid}, minimum=0, maximum=99.99

# Show side-chains of high-error residues (b > 66)
select outliers, {pid} and b > 66
show sticks, outliers
label outliers and name CA, "%s%s" % (resi, resn)

set label_size, 14
set label_color, black
zoom {pid}
""")


# ── residue-level summary ─────────────────────────────────────────────────────

def residue_summary(dfi: pd.Series, pdb_path: Path, top: int = 20):
    """Print where high-error residues fall relative to structure context."""
    if dfi.empty:
        return

    # Try to get secondary structure context from PDB HELIX/SHEET records
    helix, sheet = set(), set()
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith("HELIX"):
                    for r in range(int(line[21:25]), int(line[33:37]) + 1):
                        helix.add(r)
                elif line.startswith("SHEET"):
                    for r in range(int(line[22:26]), int(line[33:37]) + 1):
                        sheet.add(r)
    except Exception:
        pass

    # Termini: first/last 10 residues
    all_res = sorted(dfi.index)
    term = set(all_res[:10]) | set(all_res[-10:])

    def sse(r):
        if r in helix:  return "helix"
        if r in sheet:  return "strand"
        if r in term:   return "terminus"
        return "loop/other"

    df = dfi.reset_index()
    df.columns = ["Res", "max_dfi"]
    df["SSE"] = df["Res"].apply(sse)
    df = df.sort_values("max_dfi", ascending=False)

    print(f"\n  Top {min(top, len(df))} residues by max |ΔFI|:", flush=True)
    print(df.head(top).to_string(index=False, float_format="{:.4f}".format), flush=True)

    # Breakdown by SSE
    g = df.groupby("SSE")["max_dfi"].agg(["count", "mean", "max"])
    g.columns = ["n_residues", "mean_dfi", "max_dfi"]
    print(f"\n  |ΔFI| breakdown by structural context:\n{g.to_string()}", flush=True)

    # Correlation with residue number (termini / chain position effect)
    r_val = np.corrcoef(df["Res"], df["max_dfi"])[0, 1]
    print(f"\n  Corr(residue number, max |ΔFI|): {r_val:.3f}"
          f"  (nonzero → error tracks chain position)", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb",     required=True, type=Path)
    ap.add_argument("--pid",        required=True, nargs="+")
    ap.add_argument("--mode",       default="configurational",
                    choices=["configurational", "mutational", "singleresidue"])
    ap.add_argument("--out_dir",    default="structural_maps", type=Path)
    ap.add_argument("--decoys",     default=1000, type=int)
    ap.add_argument("--use_c_rand", action="store_true")
    ap.add_argument("--top",        default=20, type=int)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng_label = "c_rand" if args.use_c_rand else "torch_rand"

    for pid in args.pid:
        print(f"\n{'='*60}\n{pid}  mode={args.mode}  rng={rng_label}", flush=True)
        pdb = args.dompdb / f"{pid}.done" / "FrustrationData" / f"{pid}.pdb"
        if not pdb.exists():
            print(f"  PDB not found: {pdb}"); continue

        r_df = _load_r(args.dompdb, pid, args.mode)
        if r_df is None or r_df.empty:
            print(f"  R reference not found"); continue
        print(f"  R pairs/residues: {len(r_df)}", flush=True)

        print(f"  Running PyTorch...", flush=True)
        pt_df = run_pytorch(pdb, args.mode, args.decoys, args.use_c_rand)
        print(f"  PyTorch pairs/residues: {len(pt_df)}", flush=True)

        dfi = per_residue_dfi(pt_df, r_df, args.mode)
        print(f"  Residues with |ΔFI| data: {len(dfi)}", flush=True)
        print(f"  Global  mean={dfi.mean():.4f}  "
              f"median={dfi.median():.4f}  max={dfi.max():.4f}", flush=True)

        # Write B-factor PDB
        pdb_out = args.out_dir / f"{pid}.{args.mode}.dfi.pdb"
        write_bfactor_pdb(pdb, pdb_out, dfi)
        print(f"  Wrote B-factor PDB → {pdb_out}", flush=True)

        # Write PyMOL script
        pml_out = args.out_dir / f"{pid}.{args.mode}.dfi.pml"
        write_pymol_script(pml_out, pdb_out, pid, args.mode, float(dfi.max()))
        print(f"  Wrote PyMOL script → {pml_out}", flush=True)

        residue_summary(dfi, pdb, top=args.top)


if __name__ == "__main__":
    main()
