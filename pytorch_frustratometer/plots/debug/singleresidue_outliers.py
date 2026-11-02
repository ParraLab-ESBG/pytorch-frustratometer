"""
Extract single-residue frustration outliers: residues where PyTorch re-implementation
deviates most from the original R Frustratometer3 output.

Usage:
    python -m frustrai_fold.plots.singleresidue_outliers \
        --results_dir /path/to/baseline_comparison/frustratometer3/results \
        --dompdb_dir  /path/to/dompdb_v4.4 \
        --out         singleresidue_outliers.csv \
        --threshold   1.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def load_one(path: Path, dompdb_dir: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#")
    except Exception as e:
        print(f"ERROR reading {path.name}: {e}", flush=True)
        return None

    needed = {"Res", "FrstIndex", "TrueFrstIndex"}
    if not needed.issubset(df.columns):
        print(f"MISSING COLS {path.name}: {needed - set(df.columns)}", flush=True)
        return None
    for col in ("Res", "FrstIndex", "TrueFrstIndex"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["FrstIndex", "TrueFrstIndex"])
    if len(df) == 0:
        return None
    df["protein_id"] = path.name.split(".")[0]

    protein_id = df["protein_id"].iloc[0]
    r_path = dompdb_dir / f"{protein_id}.done" / "FrustrationData" / f"{protein_id}.pdb_singleresidue"
    if r_path.exists():
        try:
            r_df = pd.read_csv(r_path, sep=r"\s+", comment="#")
            for col in ("Res", "DensityRes"):
                r_df[col] = pd.to_numeric(r_df[col], errors="coerce")
            r_df = (r_df[["Res", "DensityRes"]]
                    .dropna()
                    .rename(columns={"DensityRes": "R_DensityRes"}))
            df = df.merge(r_df, on="Res", how="left")
        except Exception as e:
            print(f"ERROR reading R file for {protein_id}: {e}", flush=True)
            df["R_DensityRes"] = float("nan")
    else:
        df["R_DensityRes"] = float("nan")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=Path)
    parser.add_argument("--dompdb_dir",  required=True, type=Path)
    parser.add_argument("--out",         default=None, type=Path)
    parser.add_argument("--top",         default=None, type=int)
    parser.add_argument("--threshold",   default=None, type=float)
    parser.add_argument("--n_jobs",      default=-1, type=int)
    args = parser.parse_args()

    results_dir = args.results_dir
    dompdb_dir  = args.dompdb_dir
    out_path    = args.out or results_dir.parent / "singleresidue_outliers.csv"

    files = sorted(results_dir.glob("*.frustratometer3.singleresidue"))
    print(f"Found {len(files)} singleresidue files", flush=True)

    results = Parallel(n_jobs=args.n_jobs, prefer="threads", verbose=1)(
        delayed(load_one)(f, dompdb_dir) for f in files
    )
    chunks = [r for r in results if r is not None]
    print(f"Loaded {len(chunks)}/{len(files)} files successfully", flush=True)
    if not chunks:
        print("No data loaded — check --results_dir path")
        return

    all_df = pd.concat(chunks, ignore_index=True)
    all_df["error"]     = all_df["FrstIndex"] - all_df["TrueFrstIndex"]
    all_df["abs_error"] = all_df["error"].abs()

    p95 = np.percentile(all_df["abs_error"].dropna(), 95)
    r_missing = sum(1 for r in chunks if r["R_DensityRes"].isna().all())
    print(f"Total residues: {len(all_df):,}  |  R missing: {r_missing}", flush=True)
    print(f"Error stats:  mean={all_df['abs_error'].mean():.4f}  "
          f"median={all_df['abs_error'].median():.4f}  "
          f"p95={p95:.4f}  max={all_df['abs_error'].max():.4f}", flush=True)

    if args.top is not None:
        outliers = all_df.nlargest(args.top, "abs_error")
        print(f"Keeping top {args.top} residues by abs_error", flush=True)
    elif args.threshold is not None:
        outliers = all_df[all_df["abs_error"] >= args.threshold]
        print(f"Keeping {len(outliers):,} residues with abs_error >= {args.threshold}", flush=True)
    else:
        sigma3 = all_df["abs_error"].mean() + 3 * all_df["abs_error"].std()
        outliers = all_df[all_df["abs_error"] >= sigma3]
        print(f"Keeping {len(outliers):,} residues >= 3σ  (threshold={sigma3:.4f})", flush=True)

    n_artifacts = (outliers["TrueFrstIndex"] == 0.0).sum()
    outliers = outliers[outliers["TrueFrstIndex"] != 0.0].reset_index(drop=True)
    print(f"Removed {n_artifacts:,} TrueFrstIndex=0 artifacts  →  {len(outliers):,} remain",
          flush=True)

    keep_cols = [
        "protein_id", "Res", "ChainRes", "AA",
        "FrstIndex", "TrueFrstIndex", "error", "abs_error",
        "FrstState", "TrueFrstState",
        "DensityRes", "R_DensityRes",
        "NativeEnergy",
    ]
    keep_cols = [c for c in keep_cols if c in outliers.columns]
    outliers = outliers[keep_cols].sort_values("abs_error", ascending=False).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    outliers.to_csv(out_path, index=False)
    print(f"\nWrote {len(outliers):,} outlier residues → {out_path}", flush=True)
    print(outliers.head(20).to_string())


if __name__ == "__main__":
    main()
