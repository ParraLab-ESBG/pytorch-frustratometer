"""
bulk_fi_comparison.py

Compute per-protein MAE between PyTorch output and FrustratometeR reference
across all proteins in a directory.  Produces summary statistics and optional
scatter/histogram plots.

The script expects the directory layout:
    <pytorch_dir>/<pdb_id>.done/FrustrationData/<pdb_id>.pdb_configurational
    <ref_dir>/<pdb_id>.done/FrustrationData/<pdb_id>.pdb_configurational

If <ref_dir> is omitted, <pytorch_dir> files are assumed to contain a
TrueFrstIndex column (written when compare_to_original was used at run time).

Usage:
    # Two separate output dirs:
    python tests/bulk_fi_comparison.py \
        --pytorch-dir /path/to/pytorch/output \
        --ref-dir     /path/to/frustratometer_R/output \
        --mode        configurational \
        --workers     32 \
        --out         bulk_fi_stats.csv \
        [--plot       bulk_fi.png]

    # Single dir with TrueFrstIndex column already in files:
    python tests/bulk_fi_comparison.py \
        --pytorch-dir /path/to/pytorch/output \
        --mode        configurational \
        --workers     32
"""
from __future__ import annotations
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

MODE_SUFFIX = {
    "singleresidue":  "singleresidue",
    "mutational":     "mutational",
    "configurational": "configurational",
}


import io as _io

_WANT_COLS = frozenset({
    "Res", "ChainRes",                           # singleresidue
    "Res1", "Res2", "ChainRes1", "ChainRes2",    # mutational / configurational
    "FrstIndex", "TrueFrstIndex", "Welltype",
    "NativeEnergy", "SDEnergy", "DensityRes1", "DensityRes2",
})
_FLOAT_COLS = frozenset({"FrstIndex", "TrueFrstIndex", "NativeEnergy", "SDEnergy", "DensityRes1", "DensityRes2"})


def _read_file(path: Path) -> pd.DataFrame | None:
    try:
        # Single read syscall — critical on GPFS where open() latency is high.
        raw = path.read_bytes()
        lines = raw.split(b"\n")
        data = [l for l in lines if l and not l.startswith(b"#")]
        if len(data) < 2:
            return None
        header = data[0].decode().split()
        usecols = [c for c in header if c in _WANT_COLS]
        if "FrstIndex" not in usecols:
            return None
        df = pd.read_csv(
            _io.BytesIO(b"\n".join(data)),
            sep=" ", skipinitialspace=True,
            usecols=usecols,
            dtype={c: "float32" for c in usecols if c in _FLOAT_COLS},
            engine="c",
        )
        return df
    except Exception:
        return None


def _is_singleresidue(df: pd.DataFrame) -> bool:
    return "Res" in df.columns and "Res1" not in df.columns


def _normalize_chains(df: pd.DataFrame) -> pd.DataFrame:
    """Remap chain columns to canonical A, B, C… so our output and
    FrustratometeR output (which may use original chain IDs like '3') match."""
    df = df.copy()
    if _is_singleresidue(df):
        if "ChainRes" not in df.columns:
            return df
        chains = sorted(df["ChainRes"].astype(str).unique())
        chain_map = {c: chr(ord("A") + i) for i, c in enumerate(chains)}
        df["ChainRes"] = df["ChainRes"].astype(str).map(chain_map)
    else:
        if "ChainRes1" not in df.columns:
            return df
        all_chains = sorted(
            set(df["ChainRes1"].astype(str).unique()) |
            set(df["ChainRes2"].astype(str).unique())
        )
        chain_map = {c: chr(ord("A") + i) for i, c in enumerate(all_chains)}
        df["ChainRes1"] = df["ChainRes1"].astype(str).map(chain_map)
        df["ChainRes2"] = df["ChainRes2"].astype(str).map(chain_map)
    return df


def _make_key(df: pd.DataFrame) -> pd.Series:
    if _is_singleresidue(df):
        r = df["Res"].astype(str)
        c = df["ChainRes"].astype(str) if "ChainRes" in df.columns else pd.Series(["A"] * len(df))
        return pd.Series(list(zip(r, c)))
    r1 = df["Res1"].astype(str)
    r2 = df["Res2"].astype(str)
    c1 = df["ChainRes1"].astype(str)
    c2 = df["ChainRes2"].astype(str)
    k     = pd.Series(list(zip(r1, r2, c1, c2)))
    k_rev = pd.Series(list(zip(r2, r1, c2, c1)))
    return k.where(r1 <= r2, k_rev)


def compare_one(pytorch_file: Path, ref_file: Path | None) -> dict | None:
    """Read both files then compare. Parallelism comes from the outer pool."""
    pt  = _read_file(pytorch_file)
    ref = _read_file(ref_file) if ref_file is not None else None
    return _compare_dfs(pt, ref, pytorch_file)


_FI_LO, _FI_HI = -1.0, 0.78  # configurational class boundaries


def _fi_class(arr):
    cls = np.ones(len(arr), dtype=np.int8)
    cls[arr <  _FI_LO] = 0
    cls[arr >= _FI_HI] = 2
    return cls


_EXTRA_COLS = ("NativeEnergy", "DensityRes1", "DensityRes2")


def _compare_dfs(pt: pd.DataFrame, ref: pd.DataFrame | None,
                 pytorch_file: Path) -> dict | None:
    """Compare already-loaded DataFrames. Pure CPU — no I/O."""
    if pt is None or "FrstIndex" not in pt.columns:
        return None

    n_pt  = len(pt)  if pt  is not None else 0
    n_ref = len(ref) if ref is not None else 0

    if ref is None:
        if "TrueFrstIndex" not in pt.columns:
            return None
        keep = ["FrstIndex", "TrueFrstIndex", "Welltype"] + [c for c in _EXTRA_COLS if c in pt.columns]
        merged = pt[keep].dropna(subset=["FrstIndex", "TrueFrstIndex"])
        merged = merged.rename(columns={"FrstIndex": "fi_ours", "TrueFrstIndex": "fi_ref"})
        # When ref is embedded in pt, density/energy come from a single source
        for col in _EXTRA_COLS:
            if col in merged.columns:
                merged[f"{col}_ours"] = merged[col]
                merged[f"{col}_ref"]  = merged[col]
                merged.drop(columns=[col], inplace=True)
        n_ref = len(merged)
    else:
        if "FrstIndex" not in ref.columns:
            return None
        pt  = _normalize_chains(pt)
        ref = _normalize_chains(ref)
        pt["_key"]  = _make_key(pt)
        ref["_key"] = _make_key(ref)

        ref_extra = [c for c in _EXTRA_COLS if c in ref.columns]
        pt_extra  = [c for c in _EXTRA_COLS if c in pt.columns]

        ref_wt = ["Welltype"] if "Welltype" in ref.columns else []
        pt_wt  = ["Welltype"] if "Welltype" in pt.columns  else []
        ref_rename = {"FrstIndex": "fi_ref"}
        pt_rename  = {"FrstIndex": "fi_ours"}
        if ref_wt:
            ref_rename["Welltype"] = "wt_ref"
        if pt_wt:
            pt_rename["Welltype"] = "wt_ours"
        for col in ref_extra:
            ref_rename[col] = f"{col}_ref"
        for col in pt_extra:
            pt_rename[col] = f"{col}_ours"

        merged = pd.merge(
            ref[["_key", "FrstIndex"] + ref_wt + ref_extra].rename(columns=ref_rename),
            pt[["_key",  "FrstIndex"] + pt_wt  + pt_extra ].rename(columns=pt_rename),
            on="_key", how="inner",
        )
        if len(merged) == 0:
            return None
        if "wt_ref" in merged.columns:
            merged["Welltype"] = merged["wt_ref"]

    fi_ref  = merged["fi_ref"].values.astype(float)
    fi_ours = merged["fi_ours"].values.astype(float)
    ok = np.isfinite(fi_ref) & np.isfinite(fi_ours)
    if ok.sum() < 2:
        return None

    fi_ref, fi_ours = fi_ref[ok], fi_ours[ok]
    wt   = merged["Welltype"].values[ok] if "Welltype" in merged.columns else None
    diff = fi_ours - fi_ref

    # Pearson R is invariant to linear transforms. Since structures are
    # identical (same E_native), FI_ours = (σ_ref/σ_ours)·FI_ref + const,
    # so R must equal 1.0 for any correctly matched protein.
    # R < 1 signals a pair-matching bug, not a physics difference.
    if fi_ref.std() < 1e-12 or fi_ours.std() < 1e-12:
        r = float("nan")  # degenerate protein: all contacts share same FI
    else:
        r = float(np.corrcoef(fi_ref, fi_ours)[0, 1])

    # Fraction of pairs that matched between our output and FR output.
    # Low frac_matched explains low R (misaligned pair sets).
    frac_matched = float(ok.sum()) / max(n_pt, n_ref, 1)

    # Classification accuracy: class = {0=HF, 1=neutral, 2=MF}
    class_acc = float((_fi_class(fi_ours) == _fi_class(fi_ref)).mean())

    def _col(name):
        """Extract an extra column (ref or ours) filtered by ok, or nan array."""
        col = f"{name}_ref" if f"{name}_ref" in merged.columns else None
        arr_ref = merged[col].values.astype(float)[ok] if col else np.full(ok.sum(), np.nan)
        col = f"{name}_ours" if f"{name}_ours" in merged.columns else None
        arr_ours = merged[col].values.astype(float)[ok] if col else np.full(ok.sum(), np.nan)
        return arr_ref, arr_ours

    native_e_ref,  native_e_ours  = _col("NativeEnergy")
    density1_ref,  density1_ours  = _col("DensityRes1")
    density2_ref,  density2_ours  = _col("DensityRes2")

    stats = {
        "pdb_id":        pytorch_file.stem,
        "n_pairs":       int(ok.sum()),
        "n_pt":          n_pt,
        "n_ref":         n_ref,
        "frac_matched":  frac_matched,
        "mae":           float(np.abs(diff).mean()),
        "rmse":          float(np.sqrt((diff ** 2).mean())),
        "bias":          float(diff.mean()),
        "bias_abs":      float(np.abs(diff.mean())),
        "pearson_r":     r,
        "class_acc":     class_acc,
        "fi_ref":        fi_ref,
        "fi_ours":       fi_ours,
        "native_energy_ref":  native_e_ref,
        "native_energy_ours": native_e_ours,
        "density1_ref":       density1_ref,
        "density1_ours":      density1_ours,
        "density2_ref":       density2_ref,
        "density2_ours":      density2_ours,
    }

    if wt is not None:
        for wt_name in ("short", "long", "water-mediated"):
            mask = wt == wt_name
            if mask.sum() > 0:
                key = wt_name.replace("-", "_")
                stats[f"mae_{key}"] = float(np.abs(diff[mask]).mean())
                stats[f"n_{key}"]   = int(mask.sum())

    # Per-protein FI² mean — needed for the correct bias prediction formula.
    # Var(FI_ours - FI_ref) = (E[FI²] + 2) / n_decoys  (from δμ and δσ noise).
    stats["fi_sq_mean"] = float((fi_ref ** 2).mean())

    # Count unique residues from pair columns (for mutational bias formula).
    # Contacts share residues (~6 contacts/residue on average), so the effective
    # independent n for the per-protein mean bias is n_residues, not n_contacts.
    if "Res1" in pt.columns and "Res2" in pt.columns:
        c1 = pt["ChainRes1"] if "ChainRes1" in pt.columns else pd.Series(["A"] * len(pt))
        c2 = pt["ChainRes2"] if "ChainRes2" in pt.columns else pd.Series(["A"] * len(pt))
        unique_res = set(zip(pt["Res1"].astype(str), c1.astype(str))) | \
                     set(zip(pt["Res2"].astype(str), c2.astype(str)))
        stats["n_residues"] = len(unique_res)

    # SDEnergy from ref gives σ_decoy; store for diagnostics.
    src = ref if ref is not None else pt
    if src is not None and "SDEnergy" in src.columns:
        vals = src["SDEnergy"].dropna()
        if len(vals) > 0:
            stats["sd_e_mean"] = float(vals.mean())

    return stats


def investigate_protein(
    pdb_id: str,
    pytorch_dir: Path,
    ref_dir: Path,
    mode: str = "configurational",
    top_n: int = 20,
):
    """Deep-dive into one protein: break NativeEnergy discrepancy by (AA1, AA2, Welltype)."""
    suffix = MODE_SUFFIX[mode]
    pt_file  = pytorch_dir / f"{pdb_id}.done" / "FrustrationData" / f"{pdb_id}.pdb_{suffix}"
    ref_file = ref_dir     / f"{pdb_id}.done" / "FrustrationData" / f"{pdb_id}.pdb_{suffix}"

    for p, label in [(pt_file, "pytorch"), (ref_file, "ref")]:
        if not p.exists():
            print(f"  MISSING {label}: {p}")
            return

    # Read with all columns (not just _WANT_COLS) so we get AA1, AA2
    def _read_full(path):
        raw = path.read_bytes()
        lines = [l for l in raw.split(b"\n") if l and not l.startswith(b"#")]
        return pd.read_csv(_io.BytesIO(b"\n".join(lines)), sep=r"\s+")

    pt  = _read_full(pt_file)
    ref = _read_full(ref_file)

    # Normalise chain IDs before merging
    pt  = _normalize_chains(pt)
    ref = _normalize_chains(ref)
    pt["_key"]  = _make_key(pt)
    ref["_key"] = _make_key(ref)

    # Keep all columns we need from each side
    _want = ["_key", "FrstIndex", "NativeEnergy", "Welltype", "AA1", "AA2",
             "DensityRes1", "DensityRes2", "DecoyEnergy", "SDEnergy"]
    ref_cols = [c for c in _want if c in ref.columns]
    pt_cols  = [c for c in _want if c in pt.columns]

    _ref_rename = {"FrstIndex": "fi_ref",  "NativeEnergy": "ne_ref"}
    _pt_rename  = {"FrstIndex": "fi_ours", "NativeEnergy": "ne_ours"}
    for col in ("DensityRes1", "DensityRes2", "DecoyEnergy", "SDEnergy"):
        _ref_rename[col] = f"{col}_ref"
        _pt_rename[col]  = f"{col}_ours"

    merged = pd.merge(
        ref[ref_cols].rename(columns=_ref_rename),
        pt[pt_cols  ].rename(columns=_pt_rename),
        on="_key", how="inner", suffixes=("_ref", "_ours"),
    )

    if len(merged) == 0:
        print(f"  No matching pairs for {pdb_id} after normalization.")
        return

    # Resolve AA1/AA2: prefer ref columns, fall back to ours
    for aa in ("AA1", "AA2"):
        ref_col, our_col = f"{aa}_ref", f"{aa}_ours"
        if ref_col in merged.columns:
            merged[aa] = merged[ref_col]
        elif our_col in merged.columns:
            merged[aa] = merged[our_col]

    wt_col = "Welltype_ref" if "Welltype_ref" in merged.columns else "Welltype"
    if wt_col in merged.columns:
        merged["Welltype"] = merged[wt_col]

    merged["ne_diff"]  = merged["ne_ours"] - merged["ne_ref"]
    merged["fi_diff"]  = merged["fi_ours"] - merged["fi_ref"]
    merged["aa_pair"]  = merged["AA1"].astype(str) + "-" + merged["AA2"].astype(str)

    print(f"\n{'='*60}")
    print(f"Investigation: {pdb_id}  ({len(merged)} matched pairs)")
    print(f"{'='*60}")

    # Overall NativeEnergy
    print(f"\n── NativeEnergy ───────────────────────────────────────────")
    print(f"  ref  mean={merged['ne_ref'].mean():.4f}  std={merged['ne_ref'].std():.4f}")
    print(f"  ours mean={merged['ne_ours'].mean():.4f}  std={merged['ne_ours'].std():.4f}")
    print(f"  diff mean={merged['ne_diff'].mean():.4f}  std={merged['ne_diff'].std():.4f}")
    ne_r = float(np.corrcoef(merged["ne_ref"], merged["ne_ours"])[0, 1])
    print(f"  Pearson R(ne_ref, ne_ours)  = {ne_r:.6f}")
    fi_r = float(np.corrcoef(merged["fi_ref"], merged["fi_ours"])[0, 1])
    print(f"  Pearson R(fi_ref, fi_ours)  = {fi_r:.6f}")

    # By welltype
    if "Welltype" in merged.columns:
        print(f"\n── NativeEnergy diff by welltype ──────────────────────────")
        for wt in merged["Welltype"].unique():
            sub = merged[merged["Welltype"] == wt]
            r = float(np.corrcoef(sub["ne_ref"], sub["ne_ours"])[0, 1]) if len(sub) > 1 else float("nan")
            print(f"  {wt:20s}  n={len(sub):5d}  "
                  f"ne_diff mean={sub['ne_diff'].mean():+.4f}  std={sub['ne_diff'].std():.4f}  "
                  f"R={r:.4f}")

    # By AA pair — which pairs have the largest mean |ne_diff|
    print(f"\n── NativeEnergy diff by AA pair (top {top_n} worst) ────────")
    grp = (merged.groupby("aa_pair")["ne_diff"]
           .agg(n="count", mean="mean", std="std", abs_mean=lambda x: x.abs().mean())
           .reset_index()
           .sort_values("abs_mean", ascending=False))
    print(grp.head(top_n).to_string(index=False))

    # Are all pairs for a given AA type equally shifted? (points to gamma table bug)
    print(f"\n── AA types present in mismatched pairs (ne_diff ≠ 0) ─────")
    bad = merged[merged["ne_diff"].abs() > 1e-4]
    print(f"  {len(bad):,} pairs with |ne_diff| > 1e-4  "
          f"({100*len(bad)/len(merged):.1f}%)")
    for aa in ("AA1", "AA2"):
        if aa in bad.columns:
            vc = bad[aa].value_counts().head(10)
            print(f"  {aa} distribution in bad pairs: {dict(vc)}")

    # Check if ne_diff is constant per AA pair (gamma lookup bug)
    print(f"\n── Is ne_diff constant per AA pair? (gamma mismatch signature) ─")
    for wt in sorted(merged["Welltype"].unique() if "Welltype" in merged.columns else []):
        sub = merged[merged["Welltype"] == wt]
        per_pair_std = sub.groupby("aa_pair")["ne_diff"].std().dropna()
        near_zero = (per_pair_std < 1e-4).sum()
        print(f"  {wt}: {near_zero}/{len(per_pair_std)} AA pairs have std(ne_diff) < 1e-4  "
              f"→ {'likely gamma table mismatch' if near_zero > len(per_pair_std)*0.8 else 'not a simple gamma bug'}")

    # Density comparison — burial mismatch is the next candidate
    dens_cols = [c for c in merged.columns if c.startswith("Density")]
    if dens_cols:
        print(f"\n── Density (burial) comparison ─────────────────────────────")
        for side in ("Res1", "Res2"):
            rc = f"DensityRes{side[-1]}_ref"
            oc = f"DensityRes{side[-1]}_ours"
            if rc not in merged.columns or oc not in merged.columns:
                continue
            d_ref  = merged[rc].values.astype(float)
            d_ours = merged[oc].values.astype(float)
            ok_d   = np.isfinite(d_ref) & np.isfinite(d_ours)
            diff_d = (d_ours - d_ref)[ok_d]
            dr = float(np.corrcoef(d_ref[ok_d], d_ours[ok_d])[0, 1]) if ok_d.sum() > 1 else float("nan")
            print(f"  Density{side}: R={dr:.4f}  diff mean={diff_d.mean():+.4f}  "
                  f"std={diff_d.std():.4f}  max|diff|={np.abs(diff_d).max():.4f}  "
                  f"disagree>{0.01:.0%}: {(np.abs(diff_d)>0.01).mean()*100:.1f}%")
        # Does density diff predict ne_diff?
        d1r = f"DensityRes1_ref";  d1o = f"DensityRes1_ours"
        d2r = f"DensityRes2_ref";  d2o = f"DensityRes2_ours"
        if all(c in merged.columns for c in (d1r, d1o, d2r, d2o)):
            dd = ((merged[d1o] - merged[d1r]) + (merged[d2o] - merged[d2r])).values.astype(float)
            ne = merged["ne_diff"].values.astype(float)
            ok_d2 = np.isfinite(dd) & np.isfinite(ne)
            if ok_d2.sum() > 2:
                rd = float(np.corrcoef(dd[ok_d2], ne[ok_d2])[0, 1])
                print(f"\n  R(Δdensity_sum, Δne) = {rd:.4f}  "
                      f"{'← density mismatch drives ne_diff' if abs(rd) > 0.6 else '← density NOT the main cause'}")
        # Per-residue density — group by Res1 to get unique residue densities
        if "DensityRes1_ref" in merged.columns and "DensityRes1_ours" in merged.columns:
            # Each residue appears in multiple contacts; take first occurrence
            per_res = (merged.groupby("_key")
                       .agg(d1_ref=("DensityRes1_ref", "first"),
                            d1_ours=("DensityRes1_ours", "first"))
                       .reset_index())
            d1r = per_res["d1_ref"].values.astype(float)
            d1o = per_res["d1_ours"].values.astype(float)
            ok  = np.isfinite(d1r) & np.isfinite(d1o)
            print(f"\n  Per-residue DensityRes1 distribution (unique residues ≈ {ok.sum()}):")
            print(f"    ref  mean={d1r[ok].mean():.3f}  median={np.median(d1r[ok]):.3f}  "
                  f"p25={np.percentile(d1r[ok], 25):.2f}  p75={np.percentile(d1r[ok], 75):.2f}")
            print(f"    ours mean={d1o[ok].mean():.3f}  median={np.median(d1o[ok]):.3f}  "
                  f"p25={np.percentile(d1o[ok], 25):.2f}  p75={np.percentile(d1o[ok], 75):.2f}")
            n_zero_ref  = (d1r[ok] < 0.01).sum()
            n_zero_ours = (d1o[ok] < 0.01).sum()
            print(f"    ref  n_zero(ρ<0.01): {n_zero_ref}  ours n_zero(ρ<0.01): {n_zero_ours}")

    # Compare which residues appear in ref vs ours (any that ref has but ours doesn't, or vice versa)
    print(f"\n── Residue coverage comparison ─────────────────────────────")
    def _res_set(df_raw, tag):
        """Unique (res, chain) pairs present in the raw file (before merge)."""
        r1 = set(zip(df_raw["Res1"].astype(str), df_raw["ChainRes1"].astype(str)))
        r2 = set(zip(df_raw["Res2"].astype(str), df_raw["ChainRes2"].astype(str)))
        all_res = r1 | r2
        return all_res

    ref_res  = _res_set(ref, "ref")
    pt_res   = _res_set(pt,  "ours")
    only_ref  = ref_res  - pt_res
    only_ours = pt_res   - ref_res
    print(f"  Unique residues in ref output:  {len(ref_res)}")
    print(f"  Unique residues in ours output: {len(pt_res)}")
    print(f"  Only in ref (missing from ours): {len(only_ref)}")
    print(f"  Only in ours (missing from ref): {len(only_ours)}")
    if only_ref:
        sample = sorted(only_ref)[:10]
        print(f"  Sample residues only in ref: {sample}")
    else:
        print(f"\n  (DensityRes1/2 not in ref file)")

    # DecoyEnergy / SDEnergy comparison
    for field in ("DecoyEnergy", "SDEnergy"):
        rc, oc = f"{field}_ref", f"{field}_ours"
        if rc in merged.columns and oc in merged.columns:
            d_ref  = merged[rc].values.astype(float)
            d_ours = merged[oc].values.astype(float)
            ok_f   = np.isfinite(d_ref) & np.isfinite(d_ours)
            diff_f = (d_ours - d_ref)[ok_f]
            dr = float(np.corrcoef(d_ref[ok_f], d_ours[ok_f])[0, 1]) if ok_f.sum() > 1 else float("nan")
            n_const_ref  = (merged[rc].nunique() == 1)
            n_const_ours = (merged[oc].nunique() == 1)
            print(f"\n── {field} ──────────────────────────────────────────────────")
            print(f"  ref  mean={np.nanmean(d_ref):.4f}  const={'YES' if n_const_ref else 'no'}")
            print(f"  ours mean={np.nanmean(d_ours):.4f}  const={'YES' if n_const_ours else 'no'}")
            print(f"  R={dr:.4f}  diff mean={diff_f.mean():+.4f}  std={diff_f.std():.4f}")


def find_pairs(pytorch_dir: Path, ref_dir: Path | None, mode: str) -> list[tuple]:
    suffix = MODE_SUFFIX[mode]
    pairs = []
    for done_dir in sorted(pytorch_dir.glob("*.done")):
        pdb_id = done_dir.stem
        pt_file = done_dir / "FrustrationData" / f"{pdb_id}.pdb_{suffix}"
        if not pt_file.exists():
            continue
        if ref_dir is not None:
            ref_file = ref_dir / f"{pdb_id}.done" / "FrustrationData" / f"{pdb_id}.pdb_{suffix}"
            if not ref_file.exists():
                continue
        else:
            ref_file = None
        pairs.append((pt_file, ref_file))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorch-dir", default=None)
    ap.add_argument("--ref-dir",     default=None)
    ap.add_argument("--mode",        default="configurational",
                    choices=list(MODE_SUFFIX.keys()))
    ap.add_argument("--workers",     type=int, default=32)
    ap.add_argument("--out",         default=None, help="CSV to save per-protein stats")
    ap.add_argument("--plot",        default=None, help="PNG path for summary plots")
    ap.add_argument("--n-decoys",    type=int, default=4000,
                    help="n_decoys used when computing the PyTorch FI values "
                         "(needed for the theoretical bias prediction)")
    ap.add_argument("--max-proteins",type=int, default=None,
                    help="Limit number of proteins (for quick checks)")
    ap.add_argument("--precomputed-stats", default=None, metavar="CSV",
                    help="bulk_stats.csv written by --save_bulk_stats during the main run. "
                         "When provided, skip re-reading all frustration files (instant).")
    ap.add_argument("--investigate", default=None, metavar="PDB_ID",
                    help="Deep-dive into one protein: NativeEnergy diff by AA pair / welltype.")
    args = ap.parse_args()

    if args.investigate:
        if not args.pytorch_dir or not args.ref_dir:
            sys.exit("--investigate requires both --pytorch-dir and --ref-dir")
        investigate_protein(
            args.investigate,
            Path(args.pytorch_dir),
            Path(args.ref_dir),
            mode=args.mode,
        )
        return

    if args.precomputed_stats:
        df = pd.read_csv(args.precomputed_stats)
        if "mode" in df.columns:
            df = df[df["mode"] == args.mode].copy()
        # Normalise column names to match _compare_dfs output
        df = df.rename(columns={"structure": "pdb_id", "class_accuracy": "class_acc"})
        df["pdb_id"] = df["pdb_id"].apply(lambda x: Path(x).stem)
        if "bias" in df.columns:
            df["bias_abs"] = df["bias"].abs()
        if args.max_proteins:
            df = df.head(args.max_proteins)
        print(f"Loaded {len(df):,} precomputed records from {args.precomputed_stats}")
    else:
        if not args.pytorch_dir:
            sys.exit("--pytorch-dir is required unless --precomputed-stats is given")
        pytorch_dir = Path(args.pytorch_dir)
        ref_dir     = Path(args.ref_dir) if args.ref_dir else None

        print(f"Scanning {pytorch_dir} for {args.mode} output ...")
        pairs = find_pairs(pytorch_dir, ref_dir, args.mode)
        if args.max_proteins:
            pairs = pairs[:args.max_proteins]
        print(f"  Found {len(pairs):,} protein pairs to compare")

        if not pairs:
            sys.exit("No pairs found — check --pytorch-dir / --ref-dir paths")

        records = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(compare_one, pt, ref): (pt, ref)
                    for pt, ref in pairs}
            with tqdm(total=len(pairs), unit="protein") as bar:
                for fut in as_completed(futs):
                    result = fut.result()
                    if result is not None:
                        records.append(result)
                    bar.update(1)

        print(f"  {len(pairs):,} proteins processed, {len(records):,} valid", flush=True)

        if not records:
            sys.exit("No valid comparisons produced.")

        df = pd.DataFrame(records)
    n_prot  = len(df)
    n_pairs = int(df["n_pairs"].sum())

    # ── aggregate all individual pair errors for a global R² ─────────────────
    # We stored per-protein (bias, rmse, n_pairs); reconstruct the pooled stats
    # from the per-protein moments (exact for bias; approximate for R²).
    # For a cleaner global R², collect all FI values via a second pass if needed.
    # Here we report per-protein moments which are well-defined.

    bias_mean = float(df["bias"].mean())
    bias_std  = float(df["bias"].std())

    r_vals    = df["pearson_r"].dropna().values
    r_qs      = np.percentile(r_vals, [5, 25, 50, 75, 95])

    acc_mean  = float(df["class_acc"].mean()) if "class_acc" in df else float("nan")

    # Theoretical σ_bias from decoy sampling noise:
    #   singleresidue / mutational: each residue/pair draws its OWN independent decoys
    #     (distance fixed at native geometry; only AA types randomized per pair)
    #     → per-protein mean bias averages n independent values → divide by sqrt(n)
    #     σ_bias = sqrt((E[FI²]+2) / (n_decoys × n_per_protein))
    #   configurational: ONE shared global (AA, distance) pool per protein
    #     → all contacts share the same μ_decoy / σ_decoy noise; averaging over
    #        contacts does NOT reduce it → no division by sqrt(n_contacts)
    #     σ_bias = sqrt((E[FI²]+2) / n_decoys)
    n_decoys = args.n_decoys
    if "fi_sq_mean" in df.columns:
        if args.mode == "configurational":
            df["bias_pred"] = np.sqrt((df["fi_sq_mean"] + 2.0) / n_decoys)
            bias_pred_mean  = float(df["bias_pred"].mean())
            bias_pred_str   = (f"{bias_pred_mean:.4f}  "
                               f"(= mean sqrt((E[FI²]+2)/n_decoys), n_decoys={n_decoys}; "
                               f"shared pool → no /√n_contacts)")
        elif args.mode == "mutational" and "n_residues" in df.columns:
            # Mutational decoys are per-contact but contacts share residues (~6/residue),
            # so effective independent n ≈ n_residues, not n_contacts.
            df["bias_pred"] = np.sqrt((df["fi_sq_mean"] + 2.0) / n_decoys) / np.sqrt(df["n_residues"].clip(lower=1))
            bias_pred_mean  = float(df["bias_pred"].mean())
            bias_pred_str   = (f"{bias_pred_mean:.4f}  "
                               f"(= mean sqrt((E[FI²]+2)/(n_decoys·n_residues)), n_decoys={n_decoys}; "
                               f"contacts correlated via shared residues)")
        elif "n_pairs" in df.columns:
            df["bias_pred"] = np.sqrt((df["fi_sq_mean"] + 2.0) / n_decoys) / np.sqrt(df["n_pairs"].clip(lower=1))
            bias_pred_mean  = float(df["bias_pred"].mean())
            n_label = "n_residues" if args.mode == "singleresidue" else "n_contacts"
            bias_pred_str   = (f"{bias_pred_mean:.4f}  "
                               f"(= mean sqrt((E[FI²]+2)/(n_decoys·{n_label})), n_decoys={n_decoys})")
        else:
            df["bias_pred"] = np.sqrt((df["fi_sq_mean"] + 2.0) / n_decoys)
            bias_pred_mean  = float(df["bias_pred"].mean())
            bias_pred_str   = (f"{bias_pred_mean:.4f}  "
                               f"(= mean sqrt((E[FI²]+2)/n_decoys), n_decoys={n_decoys})")
    else:
        bias_pred_str = f"{np.sqrt(2.0/n_decoys):.4f}  (= sqrt(2/n_decoys), rough)"

    print(f"\n{'='*60}")
    print(f"Bulk comparison: {n_prot:,} proteins, {n_pairs:,} pairs, mode={args.mode}")
    print(f"{'='*60}")

    print(f"\n── Agreement ──────────────────────────────────────────────")
    print(f"  Per-protein MAE    mean={df['mae'].mean():.4f}  "
          f"median={df['mae'].median():.4f}  "
          f"p95={df['mae'].quantile(0.95):.4f}  max={df['mae'].max():.4f}")
    print(f"  Per-protein RMSE   mean={df['rmse'].mean():.4f}  "
          f"median={df['rmse'].median():.4f}")
    print(f"  Classification acc mean={acc_mean:.4f}  "
          f"(fraction of pairs with correct HF/neutral/MF label)")

    print(f"\n── Pearson R (within-protein FI ordering) ─────────────────")
    print(f"  p5={r_qs[0]:.4f}  p25={r_qs[1]:.4f}  "
          f"median={r_qs[2]:.4f}  p75={r_qs[3]:.4f}  p95={r_qs[4]:.4f}")
    print(f"  mean={r_vals.mean():.4f}  min={r_vals.min():.4f}")
    print(f"  Since structures are identical, E_native is identical per pair.")
    print(f"  FI_ours = (σ_ref/σ_ours)·FI_ref + const  →  R must equal 1.0.")
    print(f"  Any protein with R < 1 has a pair-matching bug (wrong edge set).")
    n_r_lt1 = int((r_vals < 0.999).sum())
    if "frac_matched" in df.columns:
        low_r = df[df["pearson_r"].lt(0.999)]
        fm_low = low_r["frac_matched"].mean() if len(low_r) else float("nan")
        print(f"  Proteins with R < 0.999: {n_r_lt1:,}  "
              f"(mean frac_matched={fm_low:.3f} vs overall {df['frac_matched'].mean():.3f})")

    print(f"\n── Bias = per-protein constant offset ─────────────────────")
    print(f"  mean bias        = {bias_mean:.5f}  (≈ 0 → no systematic over/under-estimation)")
    print(f"  std  bias        = {bias_std:.4f}  (spread of per-protein offsets)")
    print(f"  |bias| ≈ MAE     = {df['bias_abs'].mean():.4f}  → error is almost entirely "
          f"a per-protein constant")
    print(f"  predicted σ_bias = {bias_pred_str}")
    print(f"  ratio obs/pred   = "
          f"{bias_std / (df['bias_pred'].mean() if 'bias_pred' in df.columns else 1/np.sqrt(n_decoys)):.2f}"
          f"  (≈ 1.0 → decoy sampling noise fully explains per-protein bias spread)")

    for wt in ("short", "long", "water_mediated"):
        col = f"mae_{wt}"
        if col in df.columns:
            print(f"  MAE {wt:20s}: mean={df[col].mean():.4f}  "
                  f"median={df[col].median():.4f}")

    print(f"\n── Outliers ────────────────────────────────────────────────")
    cols_show = ["pdb_id", "n_pairs", "n_pt", "n_ref", "frac_matched",
                 "mae", "bias", "pearson_r", "class_acc"]
    cols_show = [c for c in cols_show if c in df.columns]
    print(f"Top 10 worst by MAE:")
    print(df.nlargest(10, "mae")[cols_show].to_string(index=False))
    print(f"\nBottom 10 by Pearson R  (R<1 → pair-matching issue, not physics):")
    print(df.nsmallest(10, "pearson_r")[cols_show].to_string(index=False))
    if "frac_matched" in df.columns:
        low_match = df[df["frac_matched"] < 0.9]
        print(f"\n{len(low_match):,} proteins with frac_matched < 0.90  "
              f"(likely different edge sets — need investigation)")

    # Proteins where R < 0.999 AND every pair matched (n_pairs == n_pt == n_ref)
    # are the only truly worrying cases: full pair overlap yet FI values don't
    # line up → genuine E_native difference, not a missing-pair artefact.
    # Proteins with frac_matched < 1.0 can show R < 0.999 simply because
    # the few unmatched pairs tend to be extreme-FI outliers.
    if "frac_matched" in df.columns and "n_pt" in df.columns and "n_ref" in df.columns:
        full_match = (df["n_pairs"] == df["n_pt"]) & (df["n_pairs"] == df["n_ref"])
        genuine = df[full_match & df["pearson_r"].lt(0.999)]
        partial_r = df[~full_match & df["pearson_r"].lt(0.999)]
        if len(partial_r):
            print(f"\n  ({len(partial_r):,} more proteins have R<0.999 but frac_matched<1.0 "
                  f"— explained by non-random missing pairs, not energy bugs)")
        if len(genuine):
            print(f"\n── Genuine energy mismatches (R<0.999 AND all pairs matched) ──")
            print(f"  {len(genuine):,} proteins — real E_native difference despite full pair set")
            print(genuine[cols_show].sort_values("pearson_r").to_string(index=False))
        else:
            print(f"\n  No proteins with R<0.999 AND full pair match — algorithm is correct.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nPer-protein stats → {args.out}")

    if args.plot:
        _make_plots(df, args.plot, args.mode, n_decoys=n_decoys)


def _make_plots(df: pd.DataFrame, out_path: str, mode: str, n_decoys: int = 4000):
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"PyTorch vs FrustratometeR — {mode} frustration index\n"
        f"{len(df):,} proteins · {df['n_pairs'].sum():,} contact pairs · "
        f"n_decoys={n_decoys}",
        fontsize=11,
    )

    # ── 1. Pearson R histogram ────────────────────────────────────────────────
    ax = axes[0, 0]
    r_vals = df["pearson_r"].dropna().values
    ax.hist(r_vals, bins=60, color="#9B59B6", edgecolor="none", alpha=0.8)
    for pct, ls, label in [(50, "--", "median"), (5, ":", "p5")]:
        v = np.percentile(r_vals, pct)
        ax.axvline(v, lw=1.5, ls=ls, color="darkviolet",
                   label=f"{label}={v:.4f}")
    ax.set_xlabel("Per-protein Pearson R")
    ax.set_ylabel("# proteins")
    ax.set_title("Within-protein FI ordering\n"
                 "R=1.0 expected (same structures) — R<1 → pair-matching bug")
    ax.legend(fontsize=8)
    ax.set_xlim(left=max(r_vals.min() - 0.05, -1.05))

    # ── 2. Bias histogram + theoretical Gaussian overlay ─────────────────────
    ax = axes[0, 1]
    bias = df["bias"].values
    ax.hist(bias, bins=60, color="#E74C3C", edgecolor="none", alpha=0.75,
            density=True, label="observed")
    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.axvline(bias.mean(), color="darkred", lw=1.5,
               label=f"mean={bias.mean():.4f}")
    # Theoretical Gaussian from sampling noise
    if "bias_pred" in df.columns:
        sigma_pred = float(df["bias_pred"].mean())
    else:
        sigma_pred = 1.0 / np.sqrt(n_decoys)
    x_g = np.linspace(bias.min(), bias.max(), 200)
    ax.plot(x_g, norm.pdf(x_g, 0, sigma_pred), "k-", lw=1.5,
            label=f"N(0, σ={sigma_pred:.3f})\ntheory: SDEnergy/√{n_decoys}")
    ax.set_xlabel("Per-protein bias  (ours − ref, FI units)")
    ax.set_ylabel("density")
    ax.set_title("Bias = per-protein constant offset\n"
                 "(mean ≈ 0 → no systematic error)")
    ax.legend(fontsize=7.5)

    # ── 3. |bias| vs predicted bias (from sampling theory) ───────────────────
    ax = axes[0, 2]
    if "bias_pred" in df.columns:
        ax.scatter(df["bias_pred"], df["bias_abs"], s=2, alpha=0.2, color="#E67E22")
        lim = max(df["bias_pred"].max(), df["bias_abs"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y = x")
        r_bp = float(np.corrcoef(df["bias_pred"], df["bias_abs"])[0, 1])
        ax.set_xlabel("Predicted |bias|  =  SDEnergy / √n_decoys")
        ax.set_ylabel("Observed |bias|  (FI units)")
        ax.set_title(f"Sampling-noise theory vs observation\nR={r_bp:.3f}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "SDEnergy column not available\n(need FrustratometeR ref files)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Predicted vs observed |bias|")

    # ── 4. Classification accuracy histogram ──────────────────────────────────
    ax = axes[1, 0]
    if "class_acc" in df.columns:
        acc = df["class_acc"].dropna().values
        ax.hist(acc, bins=40, color="#27AE60", edgecolor="none", alpha=0.8)
        ax.axvline(np.median(acc), color="darkgreen", lw=1.5, ls="--",
                   label=f"median={np.median(acc):.3f}")
        ax.axvline(acc.mean(), color="darkgreen", lw=1.5, ls="-",
                   label=f"mean={acc.mean():.3f}")
        ax.set_xlabel("Per-protein class accuracy  (HF / neutral / MF)")
        ax.set_ylabel("# proteins")
        ax.set_title("Frustration class agreement\n"
                     "(robust to constant per-protein FI offset)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.02)
    else:
        ax.set_visible(False)

    # ── 5. MAE vs n_pairs (log-log, expect slope ≈ 0) ────────────────────────
    # A per-protein CONSTANT offset gives slope = 0, not −0.5.
    # Slope ≈ 0 is the expected signature of decoy-sampling noise.
    ax = axes[1, 1]
    ax.scatter(df["n_pairs"], df["mae"], s=2, alpha=0.25, color="#2ECC71")
    log_n = np.log(df["n_pairs"].values)
    log_m = np.log(df["mae"].values)
    slope, intercept = np.polyfit(log_n, log_m, 1)
    x_fit = np.logspace(np.log10(df["n_pairs"].min()), np.log10(df["n_pairs"].max()), 100)
    ax.plot(x_fit, np.exp(intercept) * x_fit ** slope, "r-", lw=1.5,
            label=f"fit slope={slope:.2f}\n(expect ≈0 for constant offset;\n−0.5 for per-pair noise)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n_pairs")
    ax.set_ylabel("MAE (FI units)")
    ax.set_title("MAE vs protein size\n"
                 "slope≈0 expected: offset = global (μ,σ), not per-pair noise")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2)

    # ── 6. MAE by well type ───────────────────────────────────────────────────
    ax = axes[1, 2]
    wt_cols = {"short": "mae_short", "long": "mae_long",
               "water-\nmediated": "mae_water_mediated"}
    data_box   = [df[c].dropna().values for c in wt_cols.values() if c in df.columns]
    labels_box = [k for k, c in wt_cols.items() if c in df.columns]
    if data_box:
        bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True,
                        medianprops={"color": "black", "lw": 2},
                        flierprops={"marker": ".", "ms": 2, "alpha": 0.3})
        for patch, col in zip(bp["boxes"], ["#3498DB", "#E74C3C", "#2ECC71"]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
    ax.set_ylabel("MAE (FI units)")
    ax.set_title("MAE by contact type")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")


if __name__ == "__main__":
    main()
