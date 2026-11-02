"""Regression checks for explicit reconstruction dispatch without silent fallback."""

import builtins
from pathlib import Path

import numpy as np
import pytest
import torch

from pytorch_frustratometer.structure import parse, place_ideal_atoms, protein


@pytest.fixture
def original_protein() -> protein.Protein:
    """Provide nondegenerate coordinates with one residue missing N and CA."""
    positions = np.random.default_rng(0).normal(size=(3, 37, 3)) + 10
    mask = np.ones((3, 37), dtype=np.float32)
    mask[1, :2] = 0
    positions[1, :2] = 0
    return protein.Protein(
        atom_positions=positions, atom_mask=mask, aatype=np.zeros(3, dtype=int),
        residue_index=np.array([1, 2, 3]), chain_index=np.zeros(3, dtype=int),
        b_factors=np.zeros_like(mask), hetatm=np.zeros(3, dtype=int),
        chain_id_map=["A"], auth_seq_id=np.array([264, 265, 266]),
    )


@pytest.fixture
def no_modeller(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate missing MODELLER regardless of the machine's installed packages."""
    original_import = builtins.__import__

    def import_without_modeller(name, *args, **kwargs):
        """Reject MODELLER imports while preserving all other imports."""
        if name == "modeller" or name.startswith("modeller."):
            raise ModuleNotFoundError("simulated unavailable MODELLER")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_modeller)
    monkeypatch.delenv("MODELLER_DIR", raising=False)


def _forbid_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a MODELLER failure never attempts either geometry implementation."""
    def unexpected_geometry(*args, **kwargs):
        """Fail if a reconstruction fallback is called."""
        pytest.fail("Geometry reconstruction must not run for explicit MODELLER")

    monkeypatch.setattr(place_ideal_atoms, "_vectorized_ideal_geometry", unexpected_geometry)
    monkeypatch.setattr(place_ideal_atoms, "_place_fourth_atom_frame2seq", unexpected_geometry)
    monkeypatch.setattr(place_ideal_atoms, "_place_C_from_N_and_Nnext", unexpected_geometry)


def test_missing_modeller_raises_at_import_boundary(no_modeller) -> None:
    """The helper raises an actionable error retaining the underlying import cause."""
    with pytest.raises(ImportError, match="explicitly use backbone_method='ideal'") as error:
        place_ideal_atoms._reconstruct_with_modeller("unused.pdb")
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize("entrypoint", ["parser", "reconstruction"])
def test_missing_modeller_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, original_protein: protein.Protein,
    no_modeller, entrypoint: str,
) -> None:
    """Both public reconstruction paths propagate missing-MODELLER errors."""
    source = tmp_path / "input.pdb"
    source.write_text("", encoding="utf-8")
    _forbid_geometry(monkeypatch)
    with pytest.raises(ImportError, match="MODELLER is not installed or not found"):
        if entrypoint == "parser":
            parse.get_parsed_inputs(source, parser=lambda *args: original_protein,
                                    backbone_method="modeller")
        else:
            place_ideal_atoms.reconstruct_missing_backbone(
                torch.from_numpy(original_protein.atom_positions),
                torch.from_numpy(original_protein.atom_mask[:, :3]).bool(),
                torch.tensor([True, False, True]), pdb_file=str(source),
                backbone_method="modeller",
            )


def test_auto_and_ideal_are_identical(original_protein: protein.Protein) -> None:
    """Dispatch aliases yield identical coordinates and residue validity masks."""
    positions = torch.from_numpy(original_protein.atom_positions)
    mask = torch.from_numpy(original_protein.atom_mask[:, :3]).bool()
    seq_mask = mask.all(dim=-1)
    auto = place_ideal_atoms.reconstruct_missing_backbone(
        positions, mask, seq_mask.clone(), backbone_method="auto",
    )
    ideal = place_ideal_atoms.reconstruct_missing_backbone(
        positions, mask, seq_mask.clone(), backbone_method="ideal",
    )
    assert torch.equal(auto[0], ideal[0])
    assert torch.equal(auto[1], ideal[1])
    assert not auto[1][1]  # C-only residue is not repaired by vectorized geometry.


@pytest.mark.parametrize("method,pdb_file", [("sequential", "input.pdb"), (None, None), ("modeller", None)])
def test_invalid_dispatch_fails(
    monkeypatch: pytest.MonkeyPatch, method: str | None, pdb_file: str | None,
) -> None:
    """Unsupported methods and missing MODELLER input cannot reach sequential code."""
    _forbid_geometry(monkeypatch)
    with pytest.raises(ValueError):
        place_ideal_atoms.reconstruct_missing_backbone(
            torch.zeros(1, 37, 3), torch.zeros(1, 3, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool), pdb_file=pdb_file, backbone_method=method,
        )


def test_successful_modeller_returns_completed_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful MODELLER uses its completed coordinates without geometry fallback."""
    completed = torch.ones(2, 37, 3)
    monkeypatch.setattr(place_ideal_atoms, "_reconstruct_with_modeller",
                        lambda *args, **kwargs: (completed, None))
    _forbid_geometry(monkeypatch)
    positions, seq_mask = place_ideal_atoms.reconstruct_missing_backbone(
        torch.zeros(2, 37, 3), torch.zeros(2, 3, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool), pdb_file="input.pdb", backbone_method="modeller",
    )
    assert positions is completed
    assert seq_mask.tolist() == [True, True]


def test_trusted_input_bypasses_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, original_protein: protein.Protein,
) -> None:
    """The parser accepts the dataset's no-reconstruction mode via trust_all_atoms."""
    def unexpected_reconstruction(*args, **kwargs):
        """Fail if trusted input attempts backbone reconstruction."""
        pytest.fail("Trusted input must not invoke reconstruction")

    source = tmp_path / "input.pdb"
    source.write_text("", encoding="utf-8")
    monkeypatch.setattr(parse, "reconstruct_missing_backbone", unexpected_reconstruction)
    result = parse.get_parsed_inputs(
        source, parser=lambda *args: original_protein, trust_all_atoms=True,
        backbone_method=None, save_raw_coordinates=True,
    )
    assert torch.equal(result[1], torch.from_numpy(original_protein.atom_positions))
    assert result[3].tolist() == [True, True, True]
