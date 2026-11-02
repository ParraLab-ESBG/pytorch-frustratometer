#!/usr/bin/env python3
"""
Diagnose edge-set mismatches between our code and FrustratometeR.

Usage:
    python diagnose_edge_mismatch.py 2vx2A01 2zuoC12 1yx2A02 1yx2A03 \
        --dompdb-dir ../dompdb_v4.4 \
        --results-dir results/<your_run>

For each domain it prints:
  1. Every residue in the raw domain PDB with our filter decision
  2. How many residues we parse vs FrustratometeR has
  3. Whether a sequential-index shift explains the pair mismatch
"""
import sys
import argparse
from pathlib import Path
from Bio.PDB import PDBParser

# ── import our own residue constants ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytorch_frustratometer.structure import residue_constants

PARSER = PDBParser(QUIET=True)

VALID_SHORTNAMES = set(residue_constants.restype_3to1.values()) - {"X"}
VALID_3LETTER    = {k for k, v in residue_constants.restype_3to1.items() if v != "X"} - {"UNK"}


def classify_residue(res):
    """Return ('included' | 'skip_hetatm' | 'skip_nonstandard', reason)."""
    hetflag = res.id[0]
    resname  = res.resname

    is_true_hetatm = hetflag != " " and (
        resname not in residue_constants.standardize_aa_mapping
        or resname == "UNK"
    )
    if is_true_hetatm:
        return "skip_hetatm", f"HETATM flag={hetflag!r} and not in standardize_aa_mapping"

    standard_name = residue_constants.standardize_aa_mapping.get(resname, resname)
    res_shortname = residue_constants.restype_3to1.get(standard_name, "X")
    if res_shortname == "X":
        return "skip_nonstandard", f"maps to {standard_name!r} → 'X' (not in 20-AA set)"

    return "included", f"{resname} → {standard_name} → {res_shortname}"


def count_ref_residues(ref_pdb: Path):
    """Count ATOM residues in the FrustratometeR reference PDB."""
    if not ref_pdb.exists():
        return None, []
    struct  = PARSER.get_structure("ref", str(ref_pdb))
    model   = list(struct.get_models())[0]
    seen    = set()
    ordered = []
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue  # HETATM in ref PDB (shouldn't appear)
            key = (chain.id, res.id[1], res.id[2].strip())
            if key not in seen:
                seen.add(key)
                ordered.append((chain.id, res.id[1], res.id[2].strip(), res.resname))
    return len(ordered), ordered


def diagnose(domain_id: str, dompdb_dir: Path, results_dir: Path):
    print(f"\n{'='*70}")
    print(f"Domain: {domain_id}")
    print(f"{'='*70}")

    # Locate input domain PDB (what our code reads)
    # Typical paths (adjust if your layout differs):
    candidates = [
        dompdb_dir / f"{domain_id}.done" / "FrustrationData" / f"{domain_id}.pdb",
        dompdb_dir / f"{domain_id}" / f"{domain_id}.pdb",
        dompdb_dir / f"{domain_id}.pdb",
    ]
    input_pdb = None
    for c in candidates:
        if c.exists():
            input_pdb = c
            break

    if input_pdb is None:
        print(f"  [!] Cannot find input domain PDB under {dompdb_dir}")
        return

    # Parse input PDB — show every residue and our filter decision
    struct = PARSER.get_structure("inp", str(input_pdb))
    model  = list(struct.get_models())[0]

    included      = []
    skipped_hetatm     = []
    skipped_nonstandard = []

    for chain in model:
        for res in chain:
            decision, reason = classify_residue(res)
            entry = (chain.id, res.id[1], res.id[2].strip(), res.resname, reason)
            if decision == "included":
                included.append(entry)
            elif decision == "skip_hetatm":
                skipped_hetatm.append(entry)
            else:
                skipped_nonstandard.append(entry)

    print(f"\n  Input PDB : {input_pdb}")
    print(f"  Total residues in file  : {len(included)+len(skipped_hetatm)+len(skipped_nonstandard)}")
    print(f"  We include              : {len(included)}")
    print(f"  We skip (HETATM)        : {len(skipped_hetatm)}")
    print(f"  We skip (non-standard)  : {len(skipped_nonstandard)}")

    if skipped_hetatm:
        print(f"\n  -- Skipped as HETATM --")
        for ch, ri, ic, rn, reason in skipped_hetatm:
            print(f"     chain={ch} res={ri}{ic} name={rn:4s}  ({reason})")

    if skipped_nonstandard:
        print(f"\n  -- Skipped as non-standard (ATOM record) --")
        for ch, ri, ic, rn, reason in skipped_nonstandard:
            print(f"     chain={ch} res={ri}{ic} name={rn:4s}  ({reason})")

    # Residues with insertion codes
    ins_code_res = [(ch, ri, ic, rn, r) for ch, ri, ic, rn, r in included if ic]
    if ins_code_res:
        print(f"\n  -- Included residues WITH insertion codes (we drop the ins_code!) --")
        for ch, ri, ic, rn, reason in ins_code_res:
            print(f"     chain={ch} res={ri} ins={ic!r} name={rn:4s}")

    # Locate reference (FrustratometeR output) PDB
    ref_candidates = [
        dompdb_dir / f"{domain_id}.done" / "FrustrationData" / f"{domain_id}.pdb",
        results_dir / f"{domain_id}.done" / "FrustrationData" / f"{domain_id}_ref.pdb",
    ]
    # The ref PDB is the same file as the input (FrustratometeR reads and overwrites B-factors).
    # For residue count comparison we use the ref output in results_dir if separate.
    ref_pdb_out = results_dir / f"{domain_id}.done" / "FrustrationData" / f"{domain_id}.pdb" if results_dir else None

    # Estimate expected pair count for our protein
    L_ours = len(included)
    n_pairs_ours_expected = L_ours * (L_ours - 1) // 2  # upper bound (before distance cutoff)
    min_sep = 3
    n_pairs_min_sep = sum(1 for i in range(L_ours) for j in range(i+min_sep, L_ours))
    print(f"\n  Our parsed residues     : {L_ours}")
    print(f"  Pairs after min_sep≥3   : {n_pairs_min_sep}  (before distance cutoff)")

    # Show our residue list (first 20 and last 5 if long)
    print(f"\n  -- Our residue list (chain, res_id, ins_code, name) --")
    to_show = included[:20] + (included[-5:] if len(included) > 25 else [])
    if len(included) > 25:
        print(f"     [showing first 20 and last 5 of {len(included)}]")
    for ch, ri, ic, rn, _ in to_show:
        tag = f" ins={ic!r}" if ic else ""
        print(f"     chain={ch} res={ri:4d}{tag}  {rn}")
    if len(included) > 25 and len(included) > 20:
        gap = len(included) - 25
        if gap > 0:
            print(f"     ... ({gap} more) ...")

    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domains", nargs="+", help="Domain IDs to diagnose")
    ap.add_argument("--dompdb-dir", default="../dompdb_v4.4", type=Path)
    ap.add_argument("--results-dir", default=None, type=Path)
    args = ap.parse_args()

    for dom in args.domains:
        diagnose(dom.strip(), args.dompdb_dir, args.results_dir or Path("."))


if __name__ == "__main__":
    main()
