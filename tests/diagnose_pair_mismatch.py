"""
diagnose_pair_mismatch.py

For proteins with low frac_matched, identify WHICH residues cause the pair
set difference between our PyTorch output and the FrustratometeR reference.

For each protein it reports:
  - Pairs in PT only (we have, FR doesn't): likely residues we include that FR skips
  - Pairs in REF only (FR has, we don't): residues FR includes that we skip
  - If --pdb-dir is given: shows the residue names at those positions in the PDB

Usage:
    python tests/diagnose_pair_mismatch.py \
        --stats       bulk_stats.csv \
        --pytorch-dir /path/to/pytorch/output \
        --ref-dir     /path/to/frustratometer_R/output \
        --mode        configurational \
        --pdb-dir     /path/to/preprocessed/pdbs \   # optional
        --frac-thresh 0.90 \                          # proteins below this
        --n-worst     20                              # how many to show
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MODE_SUFFIX = {
    "singleresidue":   "singleresidue",
    "mutational":      "mutational",
    "configurational": "configurational",
}


def _make_key(df: pd.DataFrame) -> pd.Series:
    r1 = df["Res1"].astype(str)
    r2 = df["Res2"].astype(str)
    c1 = df["ChainRes1"].astype(str) if "ChainRes1" in df.columns else pd.Series(["A"] * len(df))
    c2 = df["ChainRes2"].astype(str) if "ChainRes2" in df.columns else pd.Series(["A"] * len(df))
    k     = list(zip(r1, r2, c1, c2))
    k_rev = list(zip(r2, r1, c2, c1))
    return pd.Series([a if a[0] <= a[1] else b for a, b in zip(k, k_rev)])


def read_frustration_file(path: Path) -> pd.DataFrame | None:
    try:
        raw = path.read_bytes()
        lines = [l for l in raw.split(b"\n") if l and not l.startswith(b"#")]
        if len(lines) < 2:
            return None
        return pd.read_csv(
            __import__("io").BytesIO(b"\n".join(lines)),
            sep=" ", skipinitialspace=True,
        )
    except Exception:
        return None


def load_pdb_resnames(pdb_path: Path) -> dict[tuple[int, str], str]:
    """Return {(resnum, chain): resname} for all ATOM records."""
    mapping: dict[tuple[int, str], str] = {}
    try:
        for line in pdb_path.read_text().splitlines():
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            record = line[:6].strip()
            resname = line[17:20].strip()
            chain   = line[21:22].strip() or "A"
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            mapping[(resnum, chain)] = f"{resname}({'ATOM' if record == 'ATOM' else 'HETATM'})"
    except Exception:
        pass
    return mapping


def diagnose(pdb_id: str, pt_file: Path, ref_file: Path,
             pdb_path: Path | None = None) -> None:
    pt  = read_frustration_file(pt_file)
    ref = read_frustration_file(ref_file)
    if pt is None or ref is None:
        print(f"  [SKIP] could not read files for {pdb_id}")
        return

    pt["_key"]  = _make_key(pt)
    ref["_key"] = _make_key(ref)

    pt_keys  = set(pt["_key"])
    ref_keys = set(ref["_key"])

    pt_only  = pt_keys  - ref_keys   # we have, FR doesn't
    ref_only = ref_keys - pt_keys    # FR has, we don't

    n_pt  = len(pt_keys)
    n_ref = len(ref_keys)
    n_matched = len(pt_keys & ref_keys)
    frac = n_matched / max(n_pt, n_ref)

    print(f"\n{'─'*60}")
    print(f"  {pdb_id}  n_pt={n_pt}  n_ref={n_ref}  matched={n_matched}  frac={frac:.3f}")
    print(f"  PT-only pairs: {len(pt_only)}   REF-only pairs: {len(ref_only)}")

    # Tally which residue numbers appear in the exclusive sets
    def residue_counts(keys):
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for k in keys:
            r1, r2, c1, c2 = k
            counts[(r1, c1)] += 1
            counts[(r2, c2)] += 1
        return sorted(counts.items(), key=lambda x: -x[1])

    pdb_map = load_pdb_resnames(pdb_path) if pdb_path else {}

    if pt_only:
        print(f"\n  Top residues appearing in PT-only pairs (we have, FR doesn't):")
        for (res, ch), cnt in residue_counts(pt_only)[:12]:
            label = pdb_map.get((int(res), ch), "?") if pdb_map else ""
            print(f"    res={res} chain={ch}  count={cnt:3d}  {label}")

    if ref_only:
        print(f"\n  Top residues appearing in REF-only pairs (FR has, we don't):")
        for (res, ch), cnt in residue_counts(ref_only)[:12]:
            label = pdb_map.get((int(res), ch), "?") if pdb_map else ""
            print(f"    res={res} chain={ch}  count={cnt:3d}  {label}")

    # Summarise which non-standard residue types are involved
    if pdb_map and pt_only:
        resnames_pt_only: dict[str, int] = defaultdict(int)
        for (res, ch), cnt in residue_counts(pt_only):
            rn = pdb_map.get((int(res) if res.isdigit() else -1, ch), "?")
            resnames_pt_only[rn] += cnt
        print(f"\n  Non-standard types in PT-only residues:")
        for rn, cnt in sorted(resnames_pt_only.items(), key=lambda x: -x[1]):
            if not rn.startswith(("ALA", "ARG", "ASN", "ASP", "CYS", "GLN",
                                   "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
                                   "MET", "PHE", "PRO", "SER", "THR", "TRP",
                                   "TYR", "VAL", "?")):
                print(f"    {rn}: {cnt} appearances")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats",        required=True,
                    help="bulk_stats.csv from bulk_fi_comparison.py")
    ap.add_argument("--pytorch-dir",  required=True)
    ap.add_argument("--ref-dir",      required=True)
    ap.add_argument("--mode",         default="configurational",
                    choices=list(MODE_SUFFIX.keys()))
    ap.add_argument("--pdb-dir",      default=None,
                    help="Dir with preprocessed PDB files (to identify residue types)")
    ap.add_argument("--frac-thresh",  type=float, default=0.90)
    ap.add_argument("--n-worst",      type=int, default=43,
                    help="Show this many worst proteins")
    ap.add_argument("--sort-by",      default="frac_matched",
                    choices=["frac_matched", "n_pairs", "mae"])
    args = ap.parse_args()

    df = pd.read_csv(args.stats)
    bad = df[df["frac_matched"] < args.frac_thresh].copy()
    if args.sort_by == "frac_matched":
        bad = bad.sort_values("frac_matched")
    elif args.sort_by == "n_pairs":
        bad = bad.sort_values("n_pairs", ascending=False)
    else:
        bad = bad.sort_values("mae", ascending=False)
    bad = bad.head(args.n_worst)

    pytorch_dir = Path(args.pytorch_dir)
    ref_dir     = Path(args.ref_dir)
    pdb_dir     = Path(args.pdb_dir) if args.pdb_dir else None
    suffix      = MODE_SUFFIX[args.mode]

    print(f"Diagnosing {len(bad)} proteins with frac_matched < {args.frac_thresh}")
    print(f"(sorted by {args.sort_by})")

    overall_pt_only_types:  dict[str, int] = defaultdict(int)
    overall_ref_only_types: dict[str, int] = defaultdict(int)

    for _, row in bad.iterrows():
        pdb_id  = row["pdb_id"]
        pt_file = pytorch_dir / f"{pdb_id}.done" / "FrustrationData" / f"{pdb_id}.pdb_{suffix}"
        ref_file = ref_dir / f"{pdb_id}.done" / "FrustrationData" / f"{pdb_id}.pdb_{suffix}"

        # Find PDB file
        pdb_path = None
        if pdb_dir:
            for candidate in [
                pdb_dir / f"{pdb_id}.done" / "FrustrationData" / f"{pdb_id}.pdb",
                pdb_dir / f"{pdb_id}.done" / f"{pdb_id}.pdb",
                pdb_dir / f"{pdb_id}.pdb",
            ]:
                if candidate.is_file():
                    pdb_path = candidate
                    break

        if not pt_file.exists() or not ref_file.exists():
            print(f"\n  [SKIP] {pdb_id} — output files not found")
            continue

        diagnose(pdb_id, pt_file, ref_file, pdb_path)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
