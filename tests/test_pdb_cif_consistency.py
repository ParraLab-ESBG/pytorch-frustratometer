"""
Test that PDB and CIF inputs produce identical frustration results for the same structure.

201_1A0F_A is used because its CIF has a genuine label_seq_id gap (chain A: jump of 2,
i.e. one missing residue). This triggered a PDB vs CIF mismatch (issue 3) before the fix:
the CIF path passed raw label_seq_id (with gaps) as residue_index, causing
_insert_gap_residues to inflate L relative to the PDB path (which used sequential 1..n).
The fix makes both paths use sequential 1..n per-chain, so they produce identical results.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PDB_FILE = REPO / "tests" / "length_test" / "201_1A0F_A.pdb"
CIF_FILE = REPO / "tests" / "length_test" / "201_1A0F_A.cif"

from pytorch_frustratometer.structure.load_structure import load_structure
from pytorch_frustratometer.coefficients import Coefficients


def _load(path):
    cfg = Coefficients()
    return load_structure(path, cfg=cfg, rotate=False, backbone_method="ideal")


def test_pdb_cif_same_length():
    """Both formats should parse the same number of residues."""
    pdb = _load(PDB_FILE)
    cif = _load(CIF_FILE)
    L_pdb = int(pdb[3].sum())  # seq_mask
    L_cif = int(cif[3].sum())
    assert L_pdb == L_cif, f"Length mismatch: PDB={L_pdb}, CIF={L_cif}"


def test_pdb_cif_same_contact_count():
    """Both formats should produce the same number of contacts."""
    pdb = _load(PDB_FILE)
    cif = _load(CIF_FILE)
    E_pdb = pdb[5].shape[1]  # edge_index [2, E]
    E_cif = cif[5].shape[1]
    assert E_pdb == E_cif, f"Contact count mismatch: PDB={E_pdb}, CIF={E_cif}"


def test_pdb_cif_same_distances():
    """Distance matrices should be identical (same coordinates, same contacts)."""
    pdb = _load(PDB_FILE)
    cif = _load(CIF_FILE)
    # distance tensor index 4
    assert pdb[4].shape == cif[4].shape, (
        f"Distance shape mismatch: PDB={pdb[4].shape}, CIF={cif[4].shape}"
    )
    assert torch.allclose(pdb[4], cif[4], atol=1e-4), "Distance matrices differ"


if __name__ == "__main__":
    print("Running PDB/CIF consistency checks...")
    for name, fn in [
        ("same_length",        test_pdb_cif_same_length),
        ("same_contact_count", test_pdb_cif_same_contact_count),
        ("same_distances",     test_pdb_cif_same_distances),
    ]:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            print(f"  ERROR {name}: {e}")
