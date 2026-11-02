"""Contract: decoys are drawn only from the valid standard residues of each protein.

A valid standard residue is inside seq_mask and has one of the 20 standard amino acids (aa_type < 20).
Gaps, padding, excluded residues and unknown residues (aa_type 20) are not part of the model. No sampler
draws them, for their amino acid, their density or their distances, with torch random numbers or with the
C random stream (use_c_rand). LAMMPS draws its decoys the same way, from the residues of its model.
"""
from __future__ import annotations

import ctypes

import pytest
import torch

from pytorch_frustratometer import Coefficients
from pytorch_frustratometer.algorithm import AWSEM
from pytorch_frustratometer.algorithm.configurational import sample_configurational_decoys
from pytorch_frustratometer.algorithm.mutational import sample_mutational_decoys
from pytorch_frustratometer.algorithm.singleresidue import sample_decoy_aa_from_sequence

# Amino-acid indices, in the package's one-letter alphabetical order.
ALA, ASP, GLU, LYS, ARG, TRP = 0, 2, 3, 8, 14, 18
UNKNOWN = 20

CUTOFF = 9.5
N_EDGES = 3


def geometry(length):
    """Distinct densities per residue and distinct distances per residue pair, all within the cutoff.

    The values are exact in float32, so they compare equal whatever precision a sampler keeps.
    """
    position = torch.arange(length, dtype=torch.float64)
    i, j = position[:, None], position[None, :]
    distance = 3.0 + 0.5 * torch.minimum(i, j) + 0.0625 * torch.maximum(i, j)
    return distance.unsqueeze(0), (1.0 + position).unsqueeze(0)


def single_residue_aa(aa_type, seq_mask, n_decoys, use_c_rand=False, distribution="sequence"):
    return sample_decoy_aa_from_sequence(aa_type, n_decoys, seq_mask, use_c_rand=use_c_rand,
                                         decoy_distribution=distribution)


def mutational_aa(aa_type, seq_mask, n_decoys, use_c_rand=False, distribution="sequence"):
    return sample_mutational_decoys(aa_type, seq_mask, N_EDGES, n_decoys, use_c_rand=use_c_rand,
                                    decoy_distribution=distribution)


def configurational(aa_type, seq_mask, n_decoys, use_c_rand=False, distribution="sequence"):
    distance, density = geometry(aa_type.shape[1])
    return sample_configurational_decoys(distance, density, aa_type, seq_mask, n_decoys, cutoff=CUTOFF,
                                         use_c_rand=use_c_rand, decoy_distribution=distribution)


def glycine_corrected_configurational(aa_type, seq_mask, n_decoys, use_c_rand=False):
    assert not use_c_rand, "the glycine-corrected sampler has no C random stream"
    distance, density = geometry(aa_type.shape[1])
    batch = {"aa_type": aa_type, "seq_mask": seq_mask, "distance": distance, "residue_density": density}
    model = AWSEM(Coefficients(mode="configurational"), dtype=torch.float64)
    valid = seq_mask[0] & (aa_type[0] < 20)
    glycine_pair_distances = [torch.empty(0, dtype=torch.float64)]  # no residue is glycine
    other_pair_distances = [distance[0][valid[:, None] & valid[None, :]]]
    return model._sample_configurational_decoys_gly_corrected(batch, n_decoys, CUTOFF, glycine_pair_distances,
                                                               other_pair_distances)


def configurational_aa(*args, **kwargs):
    return configurational(*args, **kwargs)[2]


def glycine_corrected_configurational_aa(*args, **kwargs):
    return glycine_corrected_configurational(*args, **kwargs)[2]


@pytest.mark.parametrize("draw, use_c_rand", [
    pytest.param(single_residue_aa, False, id="single-residue"),
    pytest.param(single_residue_aa, True, id="single-residue-c-rand"),
    pytest.param(mutational_aa, False, id="mutational"),
    pytest.param(mutational_aa, True, id="mutational-c-rand"),
    pytest.param(configurational_aa, False, id="configurational"),
    pytest.param(configurational_aa, True, id="configurational-c-rand"),
    pytest.param(glycine_corrected_configurational_aa, False, id="glycine-corrected-configurational"),
])
def test_decoys_are_drawn_only_from_valid_standard_residues(draw, use_c_rand):
    """Feature: every sampler draws decoy amino acids from the valid standard residues only.

    The valid standard residues are all lysine. An unknown residue inside seq_mask and a tryptophan outside
    it are not part of the model, so every decoy is lysine.
    """
    torch.manual_seed(0)
    aa_type = torch.tensor([[TRP, UNKNOWN, LYS, LYS, LYS, LYS]])
    seq_mask = torch.tensor([[False, True, True, True, True, True]])

    decoys = draw(aa_type, seq_mask, 2000, use_c_rand=use_c_rand)

    assert decoys.numel() > 0
    assert torch.all(decoys == LYS)


@pytest.mark.parametrize("use_c_rand", [False, True], ids=["torch", "c-rand"])
def test_configurational_decoy_geometry_comes_from_valid_standard_residues(use_c_rand):
    """Feature: configurational decoys take their densities and distances from valid standard residues.

    Every residue has its own density and every pair its own distance, so each decoy value identifies
    where it was drawn. Residue 1 is unknown and residue 0 is outside seq_mask.
    """
    torch.manual_seed(0)
    aa_type = torch.tensor([[LYS, UNKNOWN, ASP, GLU, LYS, ARG]])
    seq_mask = torch.tensor([[False, True, True, True, True, True]])
    valid = torch.tensor([2, 3, 4, 5])
    distance, density = geometry(aa_type.shape[1])
    off_diagonal = valid[:, None] != valid[None, :]

    decoy_distance, decoy_density, _ = configurational(aa_type, seq_mask, 2000, use_c_rand=use_c_rand)

    assert torch.all(torch.isin(decoy_density, density[0, valid]))
    assert torch.all(torch.isin(decoy_distance, distance[0][valid][:, valid][off_diagonal]))


@pytest.mark.parametrize("draw", [
    pytest.param(single_residue_aa, id="single-residue"),
    pytest.param(mutational_aa, id="mutational"),
    pytest.param(configurational_aa, id="configurational"),
])
def test_pseudocount_weights_count_only_valid_standard_residues(draw):
    """Feature: the pseudocount distribution counts only the valid standard residues.

    With four lysines and four unknown residues, the alanine weight is (0 + 1) / (4 + 20) = 1/24.
    """
    torch.manual_seed(0)
    aa_type = torch.tensor([[LYS] * 4 + [UNKNOWN] * 4])
    seq_mask = torch.ones_like(aa_type, dtype=torch.bool)

    decoys = draw(aa_type, seq_mask, 20000, distribution="pseudocount")

    assert (decoys == ALA).double().mean().item() == pytest.approx(1 / 24, abs=0.01)


def test_c_rand_mutational_decoys_follow_the_c_random_stream():
    """Feature: with use_c_rand, mutational decoys reproduce the C++ random stream over the valid residues.

    LAMMPS draws each decoy residue as rand() % n over the n residues of its model, after srand(1): contact
    by contact, decoy by decoy, first endpoint first. Residues outside seq_mask are not in that model.
    """
    aa_type = torch.tensor([[TRP, ASP, GLU, TRP, LYS, ARG]])
    seq_mask = torch.tensor([[False, True, True, False, True, True]])
    valid = [ASP, GLU, LYS, ARG]
    n_decoys = 50

    decoys = sample_mutational_decoys(aa_type, seq_mask, N_EDGES, n_decoys, use_c_rand=True)

    libc = ctypes.CDLL("libc.so.6")
    libc.srand(1)
    expected = torch.tensor([[[valid[libc.rand() % len(valid)] for _endpoint in range(2)]
                              for _decoy in range(n_decoys)]
                             for _edge in range(N_EDGES)])
    assert torch.equal(decoys[:, 0].permute(1, 2, 0), expected)
