"""Contract: residue density and frustration reproduce reference Frustratometer outputs.

The references in tests/data/frustratometeR are outputs of the AWSEM Frustratometer:

- the Frustratometer web server, for single-chain 1r69 and 1l63, copied from the Python Frustratometer tests;
- frustratometeR, the Frustratometer R package, for 1r69 split into two chains whose ends are in contact.

Its SeqDist option (3 or 12) controls both the contact network filter (|i-j| >= SeqDist) and the burial
density — frustratometeR applies the same value to both. PTF decouples them: min_seq_distance=3 for contacts,
burial_min_sep for burial density. The burial_min_sep values in CONFIGS are the PTF settings that reproduce the
burial densities frustratometeR computes with the corresponding SeqDist (see tests/data/frustratometeR/README.md).

Tolerances follow the Python Frustratometer's tests: density 1e-3 and native energies 1.5e-3 (the file
precision), frustration indices 0.3 (the references sample decoys, and so does configurational frustration).

tests/data/lammps holds a direct AWSEM LAMMPS run of 6u5e with 10^6 decoys per contact, close to the limit that exact
mutational statistics compute. Its tolerances are the measured agreement with margin.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner, Result

from pytorch_frustratometer import Coefficients, Frustratometer3, FrustratometerDataset, collate_frustratometer
from pytorch_frustratometer.cli import cli
from pytorch_frustratometer.structure.geometry import get_CB_coords
from pytorch_frustratometer.structure.protein import PDB_CHAIN_IDS

DATA = Path(__file__).parent / "data"
REFERENCES = DATA / "frustratometeR"
LAMMPS_MUTATIONAL = (DATA / "lammps"
                     / "6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat")

DENSITY_ATOL = 1e-3
ENERGY_ATOL = 1.5e-3
FRUSTRATION_ATOL = 0.3

# Exact statistics agree with LAMMPS within 0.0046 (frustration index), 0.020 (decoy mean) and 0.014 (decoy std).
LAMMPS_FRUSTRATION_ATOL = 0.01
LAMMPS_DECOY_MEAN_ATOL = 0.05
LAMMPS_DECOY_STD_ATOL = 0.03
COORDINATE_ATOL = 1e-3

MODES = ("singleresidue", "mutational", "configurational")

# Reference directory -> (structure, electrostatics K, burial_min_sep that reproduces its density).
CONFIGS = {
    "1r69_kelec_4.15_seqsep_3": ("1r69", 4.15, 1),
    "1l63_kelec_0_seqsep_3": ("1l63", 0.0, 1),
    "1l63_kelec_0_seqsep_12": ("1l63", 0.0, 12),
    "1r69_split_kelec_0_seqsep_3": ("1r69_split", 0.0, 1),
    "1r69_split_kelec_0_seqsep_12": ("1r69_split", 0.0, 12),
}

SPLIT_CONFIGS = ["1r69_split_kelec_0_seqsep_3", "1r69_split_kelec_0_seqsep_12"]
CHAIN_BREAK = {("A", 30), ("B", 31)}  # the residues on either side of the 1r69_split break
CHAIN_BREAK_BACKGROUND = pytest.mark.xfail(
    strict=True,
    reason="the mutational background counts a residue that is consecutive across a chain break twice",
)


@dataclass
class Run:
    residues: dict[tuple[str, int], int]            # (chain, residue number) -> position
    contacts: dict[tuple[str, int, str, int], int]  # (chain 1, residue 1, chain 2, residue 2) -> contact
    density: np.ndarray                             # [R]
    coordinates: np.ndarray                         # [R, 3] CB (CA for glycine)
    native_energy: dict[str, np.ndarray]            # mode -> [R] or [E]
    decoy_stats: dict[str, np.ndarray]              # mode -> decoy mean and std, [R, 2], [E, 2] or [2]
    frustration: dict[str, np.ndarray]              # mode -> [R] or [E]


def compute(structure: str, k_elec: float = 0.0, **settings) -> Run:
    """Run the CLI's model (Frustratometer3) in float64 on CPU with seed 0."""
    electrostatics = dict(use_electrostatics=k_elec > 0)
    if k_elec > 0:
        electrostatics.update(k_PlusPlus=k_elec, k_MinusMinus=k_elec, k_PlusMinus=k_elec)
    coefficients = Coefficients(**{"mode": "all", **electrostatics, **settings})
    model = Frustratometer3(coefficients, dtype=torch.float64)
    dataset = FrustratometerDataset([str(DATA / f"{structure}.pdb")], coefficients, dtype=torch.float64)
    batch = collate_frustratometer([dataset[0]])
    torch.manual_seed(0)
    with torch.no_grad():
        result = model(batch)

    chain_map = batch.get("chain_id_map")
    chain_map = chain_map[0] if chain_map is not None else None
    chains = [chain_map[c] if chain_map else PDB_CHAIN_IDS[c] for c in batch["chain_index"][0].tolist()]
    numbers = batch["residue_index"][0].tolist()
    residues = {(chain, number): i for i, (chain, number) in enumerate(zip(chains, numbers))}
    contacts = {(chains[a], numbers[a], chains[b], numbers[b]): e
                for e, (a, b) in enumerate(batch["edge_index"][0].T.tolist()) if a >= 0}
    modes = [m for m in MODES if m in result]
    coordinates = get_CB_coords(batch["atom_positions"][:, :, :5], batch["aa_type"])[0]
    return Run(residues=residues, contacts=contacts, density=batch["residue_density"][0].numpy(),
               coordinates=coordinates.numpy(),
               native_energy={m: result[f"_{m}_native_energy"][0].numpy() for m in modes},
               decoy_stats={m: result[f"_{m}_decoy_stats"][0].numpy() for m in modes},
               frustration={m: result[m][0].numpy() for m in modes})


def density(structure: str, **settings) -> Run:
    """A cheap run for density-only checks: single-residue mode, 10 decoys, electrostatics off."""
    return compute(structure, **{"mode": "singleresidue", "tert_frust_ndecoys": 10, **settings})


@functools.cache
def matched(config: str) -> Run:
    """All three modes for a reference configuration, with the separation and electrostatics it was made with."""
    structure, k_elec, burial_min_sep = CONFIGS[config]
    return compute(structure, k_elec, burial_min_sep=burial_min_sep)


def reference(config: str, mode: str) -> pd.DataFrame:
    """A reference output file. Leading '#' is dropped from the header (the web server writes #Res, the R
    package Res)."""
    (path,) = (REFERENCES / config / "FrustrationData").glob(f"*.pdb_{mode}")
    return pd.read_csv(path, sep=r"\s+").rename(columns=lambda name: name.lstrip("#"))


def residue_rows(run: Run, ref: pd.DataFrame) -> list[int]:
    return [run.residues[(chain, number)] for chain, number in zip(ref["ChainRes"], ref["Res"])]


def contact_keys(ref: pd.DataFrame) -> list[tuple[str, int, str, int]]:
    return list(zip(ref["ChainRes1"], ref["Res1"], ref["ChainRes2"], ref["Res2"]))


def rows(run: Run, config: str, mode: str) -> tuple[pd.DataFrame, list[int]]:
    """The reference file of a mode and the matching positions in our residue or contact arrays."""
    ref = reference(config, mode)
    if mode == "singleresidue":
        return ref, residue_rows(run, ref)
    return ref, [run.contacts[key] for key in contact_keys(ref)]


def at_chain_break(config: str, mode: str, ref: pd.DataFrame) -> np.ndarray:
    """Mutational contacts of the split structure that touch the chain break, where the mutational
    background is known to be wrong (see test_mutational_native_energy_at_chain_break)."""
    if mode != "mutational" or config not in SPLIT_CONFIGS:
        return np.zeros(len(ref), dtype=bool)
    return np.array([key[:2] in CHAIN_BREAK or key[2:] in CHAIN_BREAK for key in contact_keys(ref)])


def max_density_error(run: Run, config: str) -> float:
    ref = reference(config, "singleresidue")
    return float(np.abs(run.density[residue_rows(run, ref)] - ref["DensityRes"].to_numpy()).max())


def run_cli(tmp_path: Path, structure: str, *options: str) -> Result:
    args = ["run", "-s", str(DATA / f"{structure}.pdb"), "-m", "singleresidue", "-o", str(tmp_path),
            "--decoys", "10", "--accelerator", "cpu", "--num-workers", "0", "--no-pdb-write", *options]
    return CliRunner().invoke(cli, args)


def assert_written_density(tmp_path: Path, result: Result, config: str) -> None:
    assert result.exit_code == 0, result.output
    (path,) = tmp_path.glob("**/*.pdb_singleresidue")
    written = pd.read_csv(path, sep=r"\s+").rename(columns=lambda name: name.lstrip("#"))
    ours = dict(zip(zip(written["ChainRes"], written["Res"]), written["DensityRes"]))
    ref = reference(config, "singleresidue")
    np.testing.assert_allclose([ours[key] for key in zip(ref["ChainRes"], ref["Res"])],
                               ref["DensityRes"].to_numpy(), atol=DENSITY_ATOL)


@pytest.mark.parametrize("config", ["1r69_kelec_4.15_seqsep_3", "1l63_kelec_0_seqsep_3",
                                    "1r69_split_kelec_0_seqsep_3"])
def test_default_density_matches_seqdist3(config: str) -> None:
    """Feature: with default settings, the residue density is the SeqDist=3 density (same-chain pairs with
    |i-j| >= 2, pairs in different chains at any separation)."""
    structure, k_elec, _ = CONFIGS[config]
    run = density(structure, use_electrostatics=k_elec > 0)
    assert max_density_error(run, config) <= DENSITY_ATOL


@pytest.mark.parametrize("config", ["1l63_kelec_0_seqsep_12", "1r69_split_kelec_0_seqsep_12"])
def test_burial_min_sep_12_matches_seqdist12(config: str) -> None:
    """Feature: burial_min_sep=12 reproduces the SeqDist=12 density."""
    assert max_density_error(matched(config), config) <= DENSITY_ATOL


def test_separation_is_inclusive() -> None:
    """Feature: burial_min_sep is the smallest same-chain separation that counts (|i-j| > burial_min_sep)."""
    run = density("1l63", burial_min_sep=1)
    assert max_density_error(run, "1l63_kelec_0_seqsep_3") <= DENSITY_ATOL


def test_fixtures_distinguish_neighboring_separations() -> None:
    """Feature: the 1l63 references fail for a separation one off in either direction, so an off-by-one
    error cannot pass the density tests."""
    assert max_density_error(density("1l63", burial_min_sep=2), "1l63_kelec_0_seqsep_3") > 0.5
    for burial_min_sep in (11, 13):
        assert max_density_error(density("1l63", burial_min_sep=burial_min_sep), "1l63_kelec_0_seqsep_12") > 0.5


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("mode", ["mutational", "configurational"])
def test_pair_outputs_carry_the_same_densities(config: str, mode: str) -> None:
    """Feature: the contacts and the densities of both residues of every contact equal the reference pair file."""
    run = matched(config)
    ref = reference(config, mode)
    keys = contact_keys(ref)
    assert set(keys) == set(run.contacts)
    first = [run.residues[(chain, number)] for chain, number, _, _ in keys]
    second = [run.residues[(chain, number)] for _, _, chain, number in keys]
    np.testing.assert_allclose(run.density[first], ref["DensityRes1"].to_numpy(), atol=DENSITY_ATOL)
    np.testing.assert_allclose(run.density[second], ref["DensityRes2"].to_numpy(), atol=DENSITY_ATOL)


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("mode", MODES)
def test_native_energies_match_references(config: str, mode: str) -> None:
    """Feature: the native energy of every residue or contact equals the reference, in every mode."""
    run = matched(config)
    ref, positions = rows(run, config, mode)
    compared = ~at_chain_break(config, mode, ref)
    np.testing.assert_allclose(run.native_energy[mode][positions][compared],
                               ref["NativeEnergy"].to_numpy()[compared], atol=ENERGY_ATOL)


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("mode", MODES)
def test_frustration_indices_match_references(config: str, mode: str) -> None:
    """Feature: the frustration index of every residue or contact reproduces the reference, in every mode."""
    run = matched(config)
    ref, positions = rows(run, config, mode)
    compared = ~at_chain_break(config, mode, ref)
    np.testing.assert_allclose(run.frustration[mode][positions][compared],
                               ref["FrstIndex"].to_numpy()[compared], atol=FRUSTRATION_ATOL)


@CHAIN_BREAK_BACKGROUND
@pytest.mark.parametrize("config", SPLIT_CONFIGS)
def test_mutational_native_energy_at_chain_break(config: str) -> None:
    """Feature: the mutational native energy of a contact touching a chain break equals the reference."""
    run = matched(config)
    ref, positions = rows(run, config, "mutational")
    touching = at_chain_break(config, "mutational", ref)
    assert touching.any()
    np.testing.assert_allclose(run.native_energy["mutational"][positions][touching],
                               ref["NativeEnergy"].to_numpy()[touching], atol=ENERGY_ATOL)


def test_exact_mutational_matches_lammps_million_decoys() -> None:
    """Feature: exact mutational frustration reproduces a direct AWSEM LAMMPS run with 10^6 decoys per contact
    (6u5e chain A, electrostatics off): its contacts, densities, native energies, decoy statistics and frustration
    indices. The reference numbers residues by their 1-based position in the chain."""
    run = compute("6u5e", mode="mutational")
    ref = pd.read_csv(LAMMPS_MUTATIONAL, sep=r"\s+")
    first, second = ref["i"].to_numpy() - 1, ref["j"].to_numpy() - 1
    np.testing.assert_allclose(run.coordinates[first], ref[["xi", "yi", "zi"]].to_numpy(), atol=COORDINATE_ATOL)
    np.testing.assert_allclose(run.coordinates[second], ref[["xj", "yj", "zj"]].to_numpy(), atol=COORDINATE_ATOL)
    by_position = sorted(run.residues, key=run.residues.get)
    keys = [(*by_position[i], *by_position[j]) for i, j in zip(first, second)]
    assert set(keys) <= set(run.contacts)
    contacts = [run.contacts[key] for key in keys]
    np.testing.assert_allclose(run.density[first], ref["rho_i"].to_numpy(), atol=DENSITY_ATOL)
    np.testing.assert_allclose(run.density[second], ref["rho_j"].to_numpy(), atol=DENSITY_ATOL)
    np.testing.assert_allclose(run.native_energy["mutational"][contacts], ref["native_energy"].to_numpy(),
                               atol=ENERGY_ATOL)
    mean, std = run.decoy_stats["mutational"][contacts].T
    np.testing.assert_allclose(mean, ref["<decoy_energies>"].to_numpy(), atol=LAMMPS_DECOY_MEAN_ATOL)
    np.testing.assert_allclose(std, ref["std(decoy_energies)"].to_numpy(), atol=LAMMPS_DECOY_STD_ATOL)
    np.testing.assert_allclose(run.frustration["mutational"][contacts], ref["f_ij"].to_numpy(),
                               atol=LAMMPS_FRUSTRATION_ATOL)


def test_residues_in_different_chains_count_at_any_separation() -> None:
    """Feature: a pair of residues in different chains counts toward the density at any sequence separation."""
    split = density("1r69_split", burial_min_sep=12)
    unsplit = density("1r69", burial_min_sep=12)
    assert ("A", 30, "B", 31) in split.contacts
    assert split.density[split.residues[("A", 30)]] - unsplit.density[unsplit.residues[("A", 30)]] > 0.5
    assert max_density_error(split, "1r69_split_kelec_0_seqsep_12") <= DENSITY_ATOL


def test_cutoff_shorter_than_density_reach_is_rejected() -> None:
    """Feature: a contact cutoff that would drop residue pairs from the density is an error."""
    with pytest.raises(ValueError, match=r"7\.5"):
        Coefficients(tert_frust_cutoff=6.0)
    Coefficients(tert_frust_cutoff=7.5)


def test_cli_rejects_cutoff_shorter_than_density_reach(tmp_path: Path) -> None:
    """Feature: the CLI reports a cutoff too short for the density as a usage error."""
    result = run_cli(tmp_path, "1r69", "--cutoff", "6")
    assert result.exit_code != 0
    assert "7.5" in result.output


def test_longer_cutoff_does_not_change_density() -> None:
    """Feature: the residue density does not depend on the contact cutoff beyond the density's reach."""
    at_9_5 = density("1l63", tert_frust_cutoff=9.5)
    at_12 = density("1l63", tert_frust_cutoff=12.0)
    np.testing.assert_allclose(at_12.density, at_9_5.density, rtol=0, atol=1e-9)


def test_cli_burial_min_sep_reaches_written_density(tmp_path: Path) -> None:
    """Feature: --burial-min-sep sets the density written to the output files."""
    result = run_cli(tmp_path, "1l63", "--burial-min-sep", "12", "--electrostatics", "false")
    assert_written_density(tmp_path, result, "1l63_kelec_0_seqsep_12")


def test_cli_default_writes_web_server_density(tmp_path: Path) -> None:
    """Feature: the CLI's default density is the web server's SeqDist=3 density."""
    result = run_cli(tmp_path, "1l63", "--electrostatics", "false")
    assert_written_density(tmp_path, result, "1l63_kelec_0_seqsep_3")


def test_min_seq_sep_does_not_change_density(tmp_path: Path) -> None:
    """Feature: --min-seq-sep does not change the residue density."""
    result = run_cli(tmp_path, "1l63", "--min-seq-sep", "5", "--electrostatics", "false")
    assert_written_density(tmp_path, result, "1l63_kelec_0_seqsep_3")


def test_burial_min_sep_survives_yaml_and_set(tmp_path: Path) -> None:
    """Feature: burial_min_sep can be set from a YAML file and with --set on the command line."""
    path = tmp_path / "coefficients.yaml"
    path.write_text("burial_min_sep: 12\n")
    assert Coefficients.from_yaml(path).burial_min_sep == 12
    output = tmp_path / "out"
    result = run_cli(output, "1l63", "--set", "burial_min_sep=12", "--electrostatics", "false")
    assert_written_density(output, result, "1l63_kelec_0_seqsep_12")
