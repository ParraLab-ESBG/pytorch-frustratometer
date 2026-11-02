"""Verify warnings for residues excluded because backbone repair is incomplete."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from pytorch_frustratometer.structure import parse, protein, residue_constants


def _protein_with_gap(missing: tuple[str, ...], unknown: bool) -> protein.Protein:
    """Create an originally incomplete residue after a gap, with author numbering."""
    mask = np.ones((3, residue_constants.atom_type_num), dtype=np.float32)
    for name in missing:
        mask[1, residue_constants.atom_order[name]] = 0
    positions = np.full((3, residue_constants.atom_type_num, 3), 2.0)
    positions[mask == 0] = 0
    return protein.Protein(
        atom_positions=positions, atom_mask=mask,
        aatype=np.array([0, 20 if unknown else 0, 0]),
        residue_index=np.array([1, 3, 4]), chain_index=np.zeros(3, dtype=int),
        b_factors=np.zeros_like(mask), hetatm=np.zeros(3, dtype=int),
        chain_id_map=["B"], auth_seq_id=np.array([10, 265, 266]),
        ins_code=np.array(["", "A", ""]),
    )


@pytest.mark.parametrize("missing,repaired,unknown,reason", [
    (("N", "CA"), False, False, "incomplete"),
    (("N",), False, False, "incomplete"),
    (("CA",), False, False, "incomplete"),
    (("N", "CA", "C"), False, False, "incomplete"),
    (("N", "CA"), True, False, None),
    (("N", "CA"), False, True, None),
    (("C",), False, False, "forced"),
    (("C",), True, False, "forced"),
    ((), True, False, None),
])
def test_reconstruction_exclusion_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    missing: tuple[str, ...], repaired: bool, unknown: bool, reason: str | None,
) -> None:
    """Report failures once, keep original identities, and preserve returned data."""
    original = _protein_with_gap(missing, unknown)
    source = tmp_path / "input.pdb"
    source.write_text("", encoding="utf-8")
    reconstructed: dict[str, torch.Tensor] = {}

    def reconstruct(positions, mask, seq_mask, **kwargs):
        """Simulate repair while leaving the inserted gap excluded."""
        output_mask = seq_mask.clone()
        output_mask[2] = repaired
        reconstructed["positions"] = positions.clone()
        reconstructed["mask"] = output_mask.clone()
        return positions, output_mask

    def identity(positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Avoid unrelated ideal atom placement in this logging test."""
        return positions

    monkeypatch.setattr(parse, "reconstruct_missing_backbone", reconstruct)
    monkeypatch.setattr(parse, "place_missing_cb", identity)
    monkeypatch.setattr(parse, "place_missing_o", identity)
    with caplog.at_level(logging.WARNING, logger=parse.__name__):
        result = parse.get_parsed_inputs(
            source, parser=lambda *args: original, backbone_method="auto",
            save_raw_coordinates=True,
        )
    records = [record for record in caplog.records if record.name == parse.__name__]
    assert len(records) == int(reason is not None)
    if reason is not None:
        message = records[0].getMessage()
        assert str(source) in message
        assert "chain B, internal residue_index=3, author residue=265A" in message
        assert f"original backbone missing {', '.join(missing)}" in message
        assert "original present atoms=[" in message
        if reason == "incomplete":
            assert "method=auto; backbone reconstruction incomplete; excluded from calculations" in message
        else:
            assert "C coordinates zeroed" in message
            assert "backbone reconstruction incomplete" not in message
    expected_mask = reconstructed["mask"].clone()
    expected_mask[1] = False  # Inserted gap is always excluded.
    if unknown or reason == "forced":
        expected_mask[2] = False
    assert torch.equal(result[3], expected_mask)
    assert torch.equal(result[1], reconstructed["positions"])


def test_auto_c_only_residue_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Exercise real ideal reconstruction for the C-only backbone seen at LEU A265."""
    original = _protein_with_gap(("N", "CA"), False)
    source = tmp_path / "input.pdb"
    source.write_text("", encoding="utf-8")

    def identity(positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Leave raw reconstruction coordinates unchanged during atom placement."""
        return positions

    monkeypatch.setattr(parse, "place_missing_cb", identity)
    monkeypatch.setattr(parse, "place_missing_o", identity)
    with caplog.at_level(logging.WARNING, logger=parse.__name__):
        result = parse.get_parsed_inputs(
            source, parser=lambda *args: original, backbone_method="auto",
            save_raw_coordinates=True,
        )
    records = [record for record in caplog.records if record.name == parse.__name__]
    assert len(records) == 1
    assert "author residue=265A" in records[0].getMessage()
    assert "backbone reconstruction incomplete" in records[0].getMessage()
    assert not result[3][2]
