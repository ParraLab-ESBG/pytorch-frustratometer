"""Check exclusion diagnostics without requiring MODELLER or real structures."""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pytorch_frustratometer.structure import parse, protein, residue_constants


def _original_protein(missing: tuple[str, ...], unknown: bool = False) -> protein.Protein:
    """Build a structure with a sequence gap before the partially missing residue."""
    mask = np.ones((3, len(residue_constants.atom_types)), dtype=np.float32)
    for name in missing:
        mask[1, residue_constants.atom_types.index(name)] = 0
    return protein.Protein(
        atom_positions=np.full((3, mask.shape[1], 3), 2.0),
        aatype=np.array([0, 20 if unknown else 0, 0]),
        atom_mask=mask,
        residue_index=np.array([1, 3, 4]),
        chain_index=np.zeros(3, dtype=int),
        b_factors=np.zeros_like(mask),
        hetatm=np.zeros(3, dtype=int),
        chain_id_map=["B"],
        auth_seq_id=np.array([10, 42, 43]),
        ins_code=np.array(["", "A", ""]),
    )


def _stub_atom_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep atom placement independent of the exclusion behavior under test."""
    def identity(positions: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return coordinates unchanged."""
        return positions

    monkeypatch.setattr(parse, "place_missing_cb", identity)
    monkeypatch.setattr(parse, "place_missing_o", identity)


@pytest.mark.parametrize("missing,unknown,excluded", [
    (("C",), False, True),
    (("N", "C"), False, True),
    (("CA", "C"), False, True),
    (("N",), False, False),
    (("CA",), False, False),
    ((), False, False),
    (("C",), True, False),
])
def test_normal_exclusion_logging(tmp_path, monkeypatch, caplog, missing, unknown, excluded):
    """Log only targeted residues and retain original labels after gap insertion."""
    original = _original_protein(missing, unknown)
    source = tmp_path / "input.pdb"
    source.write_text("", encoding="utf-8")
    _stub_atom_placement(monkeypatch)

    def reconstruct(positions, mask, seq_mask, **kwargs):
        """Simulate successful reconstruction of every missing backbone."""
        return torch.full_like(positions, 2.0), torch.ones_like(seq_mask)

    monkeypatch.setattr(parse, "reconstruct_missing_backbone", reconstruct)
    with caplog.at_level(logging.WARNING, logger=parse.__name__):
        result = parse.get_parsed_inputs(
            source, parser=lambda *args: original, save_raw_coordinates=True,
        )
    records = [record for record in caplog.records if record.name == parse.__name__]
    assert len(records) == int(excluded)
    assert result[3].tolist() == [True, False, not (excluded or unknown), True]
    if excluded:
        message = records[0].getMessage()
        assert str(source) in message
        assert "chain B, internal residue_index=3, author residue=42A" in message
        assert f"original backbone missing {', '.join(missing)}" in message
        assert "original present atoms=[" in message
        assert "path=normal; C coordinates zeroed" in message
        assert torch.equal(result[1][2, 2], torch.zeros(3, dtype=result[1].dtype))


def test_modeller_exclusion_logging(tmp_path, monkeypatch, caplog):
    """Match original residue keys after MODELLER changes array positions."""
    original = _original_protein(("C",))
    order = np.array([2, 0, 1])
    completed = replace(
        original, aatype=original.aatype[order],
        residue_index=original.residue_index[order],
        atom_mask=np.ones_like(original.atom_mask),
        auth_seq_id=original.auth_seq_id[order], ins_code=original.ins_code[order],
    )
    source = tmp_path / "input.pdb"
    source.write_text("", encoding="utf-8")
    _stub_atom_placement(monkeypatch)
    positions = torch.from_numpy(completed.atom_positions.copy())
    monkeypatch.setattr(parse, "_reconstruct_with_modeller", lambda *args, **kwargs: (positions, completed))
    with caplog.at_level(logging.WARNING, logger=parse.__name__):
        result = parse.get_parsed_inputs(
            source, parser=lambda *args: original, backbone_method="modeller",
            save_raw_coordinates=True,
        )
    records = [record for record in caplog.records if record.name == parse.__name__]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "author residue=42A" in message
    assert "original backbone missing C" in message
    assert "original present atoms=[N, CA, CB, O," in message
    assert "path=modeller; excluded after reconstruction" in message
    assert result[3].tolist() == [True, True, False]
    assert torch.equal(result[1], positions)


def test_description_without_author_metadata():
    """Use explicit internal identifiers when optional author metadata is absent."""
    original = replace(_original_protein(("C",)), chain_id_map=None, auth_seq_id=None, ins_code=None)
    message = parse._describe_partial_backbone(original, 1, Path("input.pdb"))
    assert "chain_index 0, internal residue_index=3" in message
    assert "author residue=" not in message
