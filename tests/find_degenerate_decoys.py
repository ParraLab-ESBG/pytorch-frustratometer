"""
Identify positions where either our model or FrustrateR had a degenerate
decoy distribution (std ≈ 0 → FI = NaN, silently converted to 0).

Two sources of false zeros:
- Ours (pre-fix): SDEnergy < threshold — our decoy std was 0 so
  (mean - native) / std was NaN → nan_to_num → 0.
- FrustrateR: TrueFrstIndex == 0.0 exactly — FrustrateR's R code has no
  std=0 guard, so it produces NaN; parse_singleresidue_frustration applies
  nan_to_num(…, nan=0.0) → 0. Also catches literal NaN rows.

Usage:
    # Scan a single file
    python find_degenerate_decoys.py path/to/protein_singleresidue

    # Scan a whole results directory
    python find_degenerate_decoys.py /path/to/frust_verify_results/

    # Change the SDEnergy threshold
    python find_degenerate_decoys.py /path/to/results/ --threshold 1e-6

    # Save output to CSV
    python find_degenerate_decoys.py /path/to/results/ --out degenerate.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SD_THRESHOLD_DEFAULT = 1e-8


def find_degenerate_positions(frust_file, sd_threshold=SD_THRESHOLD_DEFAULT):
    """
    Returns a DataFrame of degenerate rows from a single *_singleresidue
    comparison file (must have SDEnergy and TrueFrstIndex columns).

    Columns added:
        our_degenerate  — SDEnergy < threshold (our model had std=0)
        ref_degenerate  — TrueFrstIndex is 0.0 or NaN (FrustrateR had NaN→0)
        both_degenerate — both flags set
    """
    df = pd.read_csv(frust_file, sep=" ", comment="#")

    if "SDEnergy" not in df.columns or "TrueFrstIndex" not in df.columns:
        raise ValueError(
            f"{frust_file}: needs SDEnergy and TrueFrstIndex columns "
            "(must be a --compare_to_original output file)"
        )

    our_degen = df["SDEnergy"].abs() < sd_threshold
    ref_degen = (df["TrueFrstIndex"] == 0.0) | df["TrueFrstIndex"].isna()

    keep_cols = [c for c in ["Res", "ChainRes", "AA", "NativeEnergy",
                              "SDEnergy", "FrstIndex", "TrueFrstIndex"] if c in df.columns]
    out = df[keep_cols].copy()
    out["our_degenerate"]  = our_degen
    out["ref_degenerate"]  = ref_degen
    out["both_degenerate"] = our_degen & ref_degen

    return out[our_degen | ref_degen].reset_index(drop=True)


def scan_directory(results_dir, pattern="*_singleresidue", sd_threshold=SD_THRESHOLD_DEFAULT):
    """
    Scan all comparison files matching `pattern` under `results_dir`.
    Returns a DataFrame with an added `protein` column.
    """
    results_dir = Path(results_dir)
    frames = []
    skipped = []
    for fpath in sorted(results_dir.rglob(pattern)):
        try:
            sub = find_degenerate_positions(fpath, sd_threshold=sd_threshold)
            if len(sub):
                sub.insert(0, "protein", fpath.stem.replace("_singleresidue", ""))
                frames.append(sub)
        except (ValueError, KeyError) as e:
            skipped.append((fpath.name, str(e)))

    if skipped:
        print(f"Skipped {len(skipped)} files (not comparison format):")
        for name, err in skipped[:5]:
            print(f"  {name}: {err}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df):
    if df.empty:
        print("No degenerate positions found.")
        return
    n_our  = df["our_degenerate"].sum()
    n_ref  = df["ref_degenerate"].sum()
    n_both = df["both_degenerate"].sum()
    n_prot = df["protein"].nunique() if "protein" in df.columns else 1
    print(f"{'Proteins affected:':35s} {n_prot}")
    print(f"{'Our model degenerate (SDEnergy≈0):':35s} {n_our}")
    print(f"{'FrustrateR degenerate (TrueFI=0/NaN):':35s} {n_ref}")
    print(f"{'Both degenerate:':35s} {n_both}")
    print(f"{'Total flagged rows:':35s} {len(df)}")
    print()
    if "protein" in df.columns:
        worst = (df.groupby("protein")[["our_degenerate", "ref_degenerate"]]
                   .sum().sum(axis=1).sort_values(ascending=False).head(10))
        print("Top proteins by flagged positions:")
        for prot, count in worst.items():
            print(f"  {prot}: {int(count)}")
    else:
        print(df.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Single *_singleresidue file or a results directory")
    parser.add_argument("--threshold", type=float, default=SD_THRESHOLD_DEFAULT,
                        help="SDEnergy cutoff for our model's degenerate std (default: 1e-8)")
    parser.add_argument("--pattern", default="*_singleresidue",
                        help="Glob pattern when scanning a directory (default: *_singleresidue)")
    parser.add_argument("--out", default=None,
                        help="Optional CSV path to save the flagged rows")
    parser.add_argument("--both_only", action="store_true",
                        help="Only show rows where BOTH systems were degenerate")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file():
        df = find_degenerate_positions(path, sd_threshold=args.threshold)
        if "protein" not in df.columns:
            df.insert(0, "protein", path.stem.replace("_singleresidue", ""))
    elif path.is_dir():
        df = scan_directory(path, pattern=args.pattern, sd_threshold=args.threshold)
    else:
        raise FileNotFoundError(args.path)

    if args.both_only and not df.empty:
        df = df[df["both_degenerate"]]

    summarize(df)

    if args.out and not df.empty:
        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
