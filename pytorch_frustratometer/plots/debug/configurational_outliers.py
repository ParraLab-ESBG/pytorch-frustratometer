"""
Extract configurational frustration outliers: pairs where PyTorch re-implementation
deviates most from the original R Frustratometer3 output.

Reads all *.frustratometer3.configurational files, computes |FrstIndex - TrueFrstIndex|,
joins the original R density values from the corresponding
  <dompdb_dir>/<protein_id>.done/FrustrationData/<protein_id>.pdb_configurational
files, and writes the top outliers to a CSV.

Columns in output:
  protein_id, Res1, Res2, ChainRes1, ChainRes2, AA1, AA2, Welltype
  FrstIndex       — PyTorch re-implementation
  TrueFrstIndex   — original R output  (same as R_FrstIndex, sanity check)
  error           — FrstIndex - TrueFrstIndex
  abs_error
  FrstState, TrueFrstState
  DensityRes1/2   — PyTorch re-implementation density
  R_DensityRes1/2 — original R density (from dompdb_dir)
  NativeEnergy

Usage:
    python -m frustrai_fold.plots.configurational_outliers \
        --results_dir /path/to/baseline_comparison/frustratometer3/results \
        --dompdb_dir  /path/to/dompdb_v4.4 \
        --out         configurational_outliers.csv \
        --top         1000
        # or --threshold 0.5  (alternative to --top)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ── loader ────────────────────────────────────────────────────────────────────

def load_one(path: Path, dompdb_dir: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#")
    except Exception as e:
        print(f"ERROR reading {path.name}: {e}", flush=True)
        return None

    # Configurational files omit DecoyEnergy/SDEnergy → 14 data cols mapped to
    # a 16-col header, so cols 9-13 are shifted left by 2:
    #   header col  9 "DecoyEnergy" → contains Welltype
    #   header col 10 "SDEnergy"    → contains PyTorch FrstIndex
    #   header col 11 "Welltype"    → contains PyTorch FrstState
    #   header col 12 "FrstIndex"   → contains R TrueFrstIndex
    #   header col 13 "FrstState"   → contains R TrueFrstState
    #   header cols 14-15            → all-NaN (no data)
    if "TrueFrstIndex" in df.columns and df["TrueFrstIndex"].isna().all():
        df = df.drop(columns=["TrueFrstIndex", "TrueFrstState"], errors="ignore")
        df = df.rename(columns={
            "DecoyEnergy": "Welltype",
            "SDEnergy":    "FrstIndex",
            "Welltype":    "FrstState",
            "FrstIndex":   "TrueFrstIndex",
            "FrstState":   "TrueFrstState",
        })

    needed = {"Res1", "Res2", "FrstIndex", "TrueFrstIndex", "DensityRes1", "DensityRes2"}
    if not needed.issubset(df.columns):
        print(f"MISSING COLS {path.name}: {needed - set(df.columns)}", flush=True)
        return None
    for col in ("Res1", "Res2", "FrstIndex", "TrueFrstIndex", "DensityRes1", "DensityRes2"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["FrstIndex", "TrueFrstIndex"])
    if len(df) == 0:
        return None
    df["protein_id"] = path.name.split(".")[0]

    protein_id = df["protein_id"].iloc[0]
    r_path = dompdb_dir / f"{protein_id}.done" / "FrustrationData" / f"{protein_id}.pdb_configurational"
    if r_path.exists():
        try:
            r_df = pd.read_csv(r_path, sep=r"\s+", comment="#")
            for col in ("Res1", "Res2", "DensityRes1", "DensityRes2"):
                r_df[col] = pd.to_numeric(r_df[col], errors="coerce")
            r_df = (r_df[["Res1", "Res2", "DensityRes1", "DensityRes2"]]
                    .dropna()
                    .rename(columns={"DensityRes1": "R_DensityRes1",
                                     "DensityRes2": "R_DensityRes2"}))
            df = df.merge(r_df, on=["Res1", "Res2"], how="left")
        except Exception as e:
            print(f"ERROR reading R file for {protein_id}: {e}", flush=True)
            df["R_DensityRes1"] = float("nan")
            df["R_DensityRes2"] = float("nan")
    else:
        df["R_DensityRes1"] = float("nan")
        df["R_DensityRes2"] = float("nan")
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=Path,
                        help="Directory with *.frustratometer3.configurational files")
    parser.add_argument("--dompdb_dir", required=True, type=Path,
                        help="dompdb_v4.4 root — used to find R *.pdb_configurational files")
    parser.add_argument("--out", default=None, type=Path,
                        help="Output CSV (default: <results_dir>/../configurational_outliers.csv)")
    parser.add_argument("--top", default=None, type=int,
                        help="Keep top N pairs by absolute error")
    parser.add_argument("--threshold", default=None, type=float,
                        help="Keep pairs where |error| >= threshold (ignored when --top is set)")
    parser.add_argument("--n_jobs", default=-1, type=int,
                        help="Parallel workers for joblib (default: -1 = all CPUs)")
    args = parser.parse_args()

    results_dir = args.results_dir
    dompdb_dir  = args.dompdb_dir
    out_path    = args.out or results_dir.parent / "configurational_outliers.csv"

    files = sorted(results_dir.glob("*.frustratometer3.configurational"))
    print(f"Found {len(files)} configurational files", flush=True)

    results = Parallel(n_jobs=args.n_jobs, prefer="threads", verbose=1)(
        delayed(load_one)(f, dompdb_dir) for f in files
    )

    chunks = [r for r in results if r is not None]
    print(f"Loaded {len(chunks)}/{len(files)} files successfully", flush=True)
    if not chunks:
        print("No data loaded — check --results_dir path")
        return

    r_missing = sum(1 for r in chunks if r["R_DensityRes1"].isna().all())
    if r_missing:
        print(f"Warning: {r_missing} proteins had no R configurational file in {dompdb_dir}",
              flush=True)

    print("Concatenating...", flush=True)
    all_df = pd.concat(chunks, ignore_index=True)
    all_df["error"]     = all_df["FrstIndex"] - all_df["TrueFrstIndex"]
    all_df["abs_error"] = all_df["error"].abs()

    p95 = np.percentile(all_df["abs_error"].dropna(), 95)
    print(f"Total pairs: {len(all_df):,}  |  R missing: {r_missing}", flush=True)
    print(f"Error stats:  mean={all_df['abs_error'].mean():.4f}  "
          f"median={all_df['abs_error'].median():.4f}  "
          f"p95={p95:.4f}  max={all_df['abs_error'].max():.4f}", flush=True)

    # Filter
    if args.top is not None:
        outliers = all_df.nlargest(args.top, "abs_error")
        print(f"Keeping top {args.top} pairs by abs_error", flush=True)
    elif args.threshold is not None:
        outliers = all_df[all_df["abs_error"] >= args.threshold]
        print(f"Keeping {len(outliers):,} pairs with abs_error >= {args.threshold}", flush=True)
    else:
        sigma3 = all_df["abs_error"].mean() + 3 * all_df["abs_error"].std()
        outliers = all_df[all_df["abs_error"] >= sigma3]
        print(f"Keeping {len(outliers):,} pairs >= 3σ  (threshold={sigma3:.4f})", flush=True)

    # Remove TrueFrstIndex == 0.0 artifacts (C++ printf NaN → "0.000" on old glibc)
    n_artifacts = (outliers["TrueFrstIndex"] == 0.0).sum()
    outliers = outliers[outliers["TrueFrstIndex"] != 0.0].reset_index(drop=True)
    print(f"Removed {n_artifacts:,} TrueFrstIndex=0 artifacts  →  {len(outliers):,} real outliers remain",
          flush=True)

    keep_cols = [
        "protein_id", "Res1", "Res2", "ChainRes1", "ChainRes2",
        "AA1", "AA2", "Welltype",
        "FrstIndex", "TrueFrstIndex", "error", "abs_error",
        "FrstState", "TrueFrstState",
        "DensityRes1", "DensityRes2",
        "R_DensityRes1", "R_DensityRes2",
        "NativeEnergy",
    ]
    keep_cols = [c for c in keep_cols if c in outliers.columns]
    outliers = outliers[keep_cols].sort_values("abs_error", ascending=False).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    outliers.to_csv(out_path, index=False)
    print(f"\nWrote {len(outliers):,} outlier pairs → {out_path}", flush=True)
    print(outliers.head(20).to_string())


if __name__ == "__main__":
    main()
