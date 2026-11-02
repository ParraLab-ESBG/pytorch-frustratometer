# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Full PyTorch implementation of single-residue, mutational, and configurational frustration modes
- GPU-accelerated batch processing via `FrustratometerDataset` and `DataLoader`
- Multi-chain PDB support with chain-aware residue lookup
- AWSEM potential with burial and water-mediated terms
- `Coefficients` dataclass for unified hyperparameter management
- CLI (`pytorch_frustratometer`) with YAML config and per-field overrides
- PyMOL `.pml` script generation for visualising frustration on 3D structure
- Plotting subpackage: frustration distributions, outlier maps, FrustraPy/FrustraEvo comparisons
- Subpackage layout: `algorithm/`, `data/`, `features/`, `structure/`, `utils/`, `plots/`
- Optional `[plots]` extra for matplotlib/seaborn/umap dependencies
- Parity tests against Frustratometer web-server and frustratometeR outputs: residue density, native energies
  and frustration indices in every mode, for single-chain structures and a chain break in contact
- Exact (decoy-free) mutational frustration: the decoy mean and standard deviation of each contact are the weighted
  moments over all 20 x 20 amino-acid pairs, as single-residue frustration does over the 20 amino acids
- Plain functions `decoy_aa_frequencies`, `decoy_aa_weights` and `decoy_statistics`
  (`pytorch_frustratometer.algorithm`)

### Changed
- The residue density counts same-chain pairs with `|i-j| > burial_min_sep`.
  The default `burial_min_sep` is `1`, equivalent to the old `>= 2` convention with default `2`
- `--min-seq-sep` no longer changes the residue density
- A `tert_frust_cutoff` (`--cutoff`) shorter than the residue density's reach, `max(well_rmax[0], burial_rmax) +
  5 / eta` (7.5 A by default), raises an error instead of silently dropping residue pairs from the density
- Mutational frustration is exact, as single-residue frustration. Results carry no sampling noise, and the decoy
  standard deviation is the population value (divides by N), as in LAMMPS. `--decoys` sets the decoy count of
  configurational frustration and of `--use-c-rand` runs only
- `--use-c-rand` mutational statistics use the population standard deviation, as in LAMMPS
- MJ mutational frustration honors `--decoy-distribution` and is exact
- Mutational decoy energies and their background water are evaluated in chunks of `DECOY_CHUNK_ELEMENTS` contacts x decoys, which bounds the memory of large structures with the same results

### Fixed
- The residue density matches the AWSEM LAMMPS implementation and the Frustratometer web server
- Decoys are drawn only from valid standard residues, as in LAMMPS. An unknown residue (`aa_type` 20) inside `seq_mask` was drawn as alanine, and with `--use-c-rand` mutational decoys were drawn by tensor position, so a residue excluded from the calculation shifted the draws
- `Frustratometer("singleresidue" | "mutational" | "configurational")` returns `Frustratometer3`, the model the CLI uses, instead of raising a method-resolution-order `TypeError`
- `SingleResidueFrustratometer`, `MutationalFrustratometer` and `ConfigurationalFrustratometer` can be built,
  and so their module CLIs run: their bases repeated a mode mixin that `AWSEM` already holds (`TypeError`).
  Their densities now count different-chain pairs as `Frustratometer3` does, and the single-residue module
  CLI no longer stops with a `NameError` after writing its mutation table
