# pytorch-frustratometer

A PyTorch implementation of the [FrustratometeR](https://github.com/proteinphysiologylab/frustratometeR) for GPU-accelerated protein frustration analysis.

The Frustratometer quantifies local energetic frustration in protein structures by measuring how optimal each residue or contact is relative to the sequence and structural context. This package re-implements the algorithm in PyTorch, enabling batch processing and GPU acceleration across four frustration modes: single-residue, mutational, configurational, and smooth.

![Frustration modes overview](docs/fig1_latest_7oct2026.png)

## Installation

```bash
pip install pytorch-frustratometer
```

Plotting tools (matplotlib, seaborn, joblib, biotite) are included in the standard install.

### From source with uv (recommended)

[uv](https://docs.astral.sh/uv/) will install PyTorch with CUDA 12.8 support and resolve `torch-cluster` automatically:

```bash
git clone https://github.com/ParraLab-ESBG/pytorch-frustratometer.git
cd pytorch-frustratometer
uv sync
```

> **CPU-only / different CUDA version**: remove or edit the `[tool.uv.sources]` and `[[tool.uv.index]]` blocks in `pyproject.toml` and install PyTorch manually following [pytorch.org/get-started](https://pytorch.org/get-started/locally/) before running `uv sync`.

Activate the environment and run:

```bash
source .venv/bin/activate
pytorch_frustratometer --help
```

Or run without activating:

```bash
uv run pytorch_frustratometer --structure my_protein.pdb
```

### From source with pip

Activate your Python 3.12 environment, then clone the repository:

```bash
git clone https://github.com/ParraLab-ESBG/pytorch-frustratometer.git
cd pytorch-frustratometer
```

For Linux with an NVIDIA GPU and a driver supporting CUDA 12.8:

```bash
python -m pip install -r requirements-cu128.txt
```

For CPU-only installation on Linux or Windows:

```bash
python -m pip install -r requirements-cpu.txt
```

Run these commands from the repository root. Both profiles install the project
in editable mode with matching PyTorch and prebuilt `torch-cluster` wheels.

---

## Command-line usage

```bash
pytorch_frustratometer --structure my_protein.pdb
```

This runs all three standard modes (singleresidue, mutational, configurational) by default.
To request specific modes, pass `--mode` one or more times:

```bash
pytorch_frustratometer -s my_protein.pdb -m configurational -o ./results/
pytorch_frustratometer -s my_protein.pdb -m singleresidue -m smooth -o ./results/
```

Multiple structures or a directory:

```bash
pytorch_frustratometer -s /path/to/pdbs/ -m mutational -m singleresidue -o ./results/
```

### Options

#### Input / output

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--structure` | `-s` | *(required)* | PDB/mmCIF file(s), directory, or text file listing one path per line |
| `--output-dir` | `-o` | `frustratometer_YYYYMMDD_HHMMSS_<pid>/` in cwd | Directory for output files |
| `--mode` | `-m` | `all` | `singleresidue`, `mutational`, `configurational`, `smooth`, or `all`; repeatable; `all` = s+m+c |
| `--chain` | | all chains | Chain(s) to extract, e.g. `--chain A`; repeat to align with multiple `--structure` values |
| `--force` | | | Recompute even if a `<stem>.done/` output directory already exists |
| `--verbose` | `-v` | | Print each output file as it is written |

#### Frustration parameters

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--decoys` / `--ndecoys` | `-n` | `100000` | Number of sampled decoys for configurational frustration and `--use-c-rand` runs. Single-residue and mutational frustration are exact. See [Exact single-residue and mutational frustration](#exact-single-residue-and-mutational-frustration) |
| `--cutoff` | | `9.5` | Contact distance cutoff in Å |
| `--min-seq-sep` | | `3` | Minimum sequence separation \|i−j\| >= value for a contact to be considered |
| `--burial-min-sep` | | `1` | Minimum sequence separation: a residue pair counts toward burial density when \|i-j\| > value |
| `--decoy-distribution` | | `sequence` | AA decoy sampling strategy — see [Decoy distributions](#decoy-distributions) |
| `--electrostatics` | | `true` | `true`/`false` to enable/disable; a single number sets all K values; `K_PP,K_MM,K_PM` sets them individually |

#### Output files

| Option | Default | Description |
|--------|---------|-------------|
| `--DMFS` / `--deep-mutational-frustration-scanning` | | Write full per-residue / per-contact mutation energy tables (20 AA × L or 20×20 × E). `--save-mutations` is a backwards-compatible alias. |
| `--write-pymol` | | Write a PyMOL `.pml` colouring script alongside each output file |
| `--write-mvsj` | | Write a [MolViewSpec](https://molviewspec.org) `.mvsj` file for each output |
| `--no-pdb-write` | | Skip writing the per-protein PDB with frustration index in B-factors. Also suppresses the repair log and residue mapping files (see [Output structure](#output-structure)) |
| `--shard-output` | | Write results as shards to this directory instead of per-protein `.done/` directories — see [Large-scale dataset processing](#large-scale-dataset-processing-shards) |
| `--shard-format` | `parquet` | Shard format: `parquet`, `hdf5`, or `pt.zst` |
| `--shard-rank` | `0` | Worker rank in a multi-process shard run |
| `--shard-world-size` | `1` | Total number of parallel shard workers |
| `--shard-size` | `200` | Proteins per shard for normal-length (≤ 500 residues) proteins |
| `--shard-size-large` | `16` | Proteins per shard for large (> 500 residues) proteins |
| `--shard-save-structure` | | Save backbone atom positions (N, CA, C, CB, O — columns `x_N`/`y_N`/`z_N` … `x_O`/`y_O`/`z_O`) in a `structure` table (parquet/hdf5) or `backbone_positions` tensor (pt.zst) — lets FrustrAI-Fold reload structures without the original PDB files |
| `--shard-save-sidechain-centroid` | | When `--shard-save-structure` is set, also save the sidechain centroid (average of non-backbone heavy atoms) as `x_SC`/`y_SC`/`z_SC` columns or a `sidechain_centroid` tensor |

#### Visualization

| Option | Default | Description |
|--------|---------|-------------|
| `--draw-figure` | | Generate frustration plots and NxN contact maps per protein |
| `--image-format` | `png` | Figure format(s); repeat for multiple: `--image-format png --image-format svg` |
| `--render-structure` | | Add a PyMOL 3-D structure panel to figures (requires `pymol` in PATH) |

#### Computation

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--dtype` | | `float32` | Floating-point precision (`float32` or `float64`) |
| `--batch-size` | `-b` | `1` | Structures per GPU batch — increase for throughput on large datasets |
| `--num-workers` | | `4` | DataLoader worker processes for parallel structure loading |
| `--devices` | | `auto` | GPUs to use: `auto`, an integer count, or comma-separated IDs (e.g. `0,1`) |
| `--accelerator` | | `auto` | PyTorch Lightning accelerator: `auto`, `gpu`, `cpu`, `tpu` |
| `--strategy` | | `auto` | PyTorch Lightning strategy: `auto`, `ddp`, `ddp_spawn`, `dp` |

#### Configuration

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | | YAML file with `Coefficients` field overrides (applied before `--set`) |
| `--set` | | | Override any `Coefficients` field inline: `--set k_water=0.5 --set well_rmax="[6.5,9.5]"` |

### Exact single-residue and mutational frustration

Single-residue and mutational frustration use no random decoys. Single-residue frustration
evaluates all 20 amino acids at each position, and mutational frustration all 20 x 20 amino-acid
pairs at each contact. The decoy mean and standard deviation are the weighted moments over the
decoy distribution: the limit of infinitely many sampled decoys.

Configurational frustration samples `--decoys` decoys (100 000 by default), one decoy set per
protein. `--use-c-rand` samples `--decoys` decoys in every mode, to reproduce the C random stream
of the AWSEM LAMMPS code.

### Decoy distributions

The `--decoy-distribution` option controls how amino acid decoys are drawn when computing
the frustration index.  All modes (singleresidue, mutational, configurational) respect it.
For single-residue and mutational frustration it sets the weight of each decoy amino acid.

| Value | Description |
|-------|-------------|
| `sequence` *(default)* | Sample uniformly from the protein's own sequence composition. AA types absent from the sequence are never drawn as decoys. |
| `pseudocount` | Add 1 virtual count per AA type before normalising. Every one of the 20 standard AAs can appear as a decoy — useful when a protein has unusual or very short sequences. |
| `sequence_exclude_native` | Like `sequence`, but the native AA at each position is excluded (decoy ≠ native). |
| `pseudocount_exclude_native` | Like `pseudocount`, but the native AA at each position is excluded. |

```bash
# Pseudocount — all 20 AAs can appear even if absent from the sequence
pytorch_frustratometer -s my_protein.pdb --decoy-distribution pseudocount

# Exclude the native AA from the decoy distribution
pytorch_frustratometer -s my_protein.pdb --decoy-distribution sequence_exclude_native

# Via Python API
from pytorch_frustratometer import Frustratometer, Coefficients

coeff  = Coefficients.from_dict({"tert_frust_decoy_distribution": "pseudocount"})
model  = Frustratometer("all", coefficients=coeff)
result = model.from_pdb("my_protein.pdb")
```

### Deep Mutational Frustration Scanning (DMFS)

The increased computational efficiency of PyTorch Frustratometer enables ultra-fast Deep Mutational
Frustration Scanning (DMFS), named in analogy to Deep Mutational Scanning (DMS) experiments, in
which changes in local frustration are calculated for every position and all alternative amino acid
identities for a given structure.

DMFS writes the full exhaustive mutation energy tables — every amino acid substitution at every
position or contact — alongside the standard frustration output:

```bash
pytorch_frustratometer -s my_protein.pdb --DMFS
```

Three tables are written per protein (one per mode that was run):

| File | Contents |
|------|----------|
| `<stem>.pdb_singleresidue_mutations` | 20 AA × L rows — single-residue substitution energies |
| `<stem>.pdb_mutational_mutations` | 20×20 AA × E rows — pairwise contact substitution energies |
| `<stem>.pdb_configurational_mutations` | configurational substitution energies |

When writing shards, DMFS tables are stored in the `singleresidue_mutations` / `mutational_mutations`
mode tables rather than separate files.

```bash
# DMFS on a whole directory, sharded
pytorch_frustratometer -s /path/to/pdbs/ --DMFS --shard-output ./shards/

# Via Python API
from pytorch_frustratometer import Frustratometer, Coefficients

coeff = Coefficients.from_dict({"run_dmfs": True})
model = Frustratometer("all", coefficients=coeff)
result = model.from_pdb("my_protein.pdb")
```

### Output structure

Each structure produces a `<stem>.done/` directory:

```
<output_dir>/
└── <stem>.done/
    ├── FrustrationData/
    │   ├── <stem>.pdb                         # structure with frustration index in B-factors
    │   │                                      #   (skipped with --no-pdb-write)
    │   ├── <stem>.pdb.repair.log              # human-readable log of every structural modification
    │   │                                      #   made during PDB parsing (gaps inserted, backbone
    │   │                                      #   atoms reconstructed, CB/O placed, residues excluded)
    │   │                                      #   (skipped with --no-pdb-write)
    │   ├── <stem>.pdb.repair_map.tsv          # TSV mapping new_idx ↔ orig_idx for every residue,
    │   │                                      #   with chain, auth_seq, and status columns
    │   │                                      #   (skipped with --no-pdb-write)
    │   ├── <stem>.pdb_singleresidue            # per-residue frustration TSV
    │   ├── <stem>.pdb_mutational               # per-contact mutational frustration TSV
    │   ├── <stem>.pdb_configurational          # per-contact configurational frustration TSV
    │   ├── <stem>.pdb_smooth                   # per-contact smooth frustration TSV
    │   ├── <stem>.pdb_mutational_5adens        # residue density file (mutational)
    │   ├── <stem>.pdb_configurational_5adens   # residue density file (configurational)
    │   ├── <stem>.pdb_singleresidue_mutations  # 20-AA × L mutation energy table (--DMFS)
    │   ├── <stem>.pdb_mutational_mutations     # 20×20 × E pair mutation table (--DMFS)
    │   └── <stem>.pdb_configurational_mutations  # configurational mutation table (--DMFS)
    └── Visualizations/                        # created by --draw-figure, --write-pymol, or --write-mvsj
        ├── <stem>_singleresidue.<fmt>          # per-residue frustration line plot (--draw-figure)
        ├── <stem>_mutational.<fmt>             # NxN contact map (--draw-figure)
        ├── <stem>_configurational.<fmt>        # NxN contact map (--draw-figure)
        ├── <stem>.pdb_singleresidue.pml        # PyMOL colouring script (--write-pymol)
        ├── <stem>.pdb_mutational.pml
        ├── <stem>.pdb_configurational.pml
        ├── <stem>.pdb_singleresidue.mvsj       # MolViewSpec scene (--write-mvsj)
        ├── <stem>.pdb_mutational.mvsj
        └── <stem>.pdb_configurational.mvsj
```

Only the files for the requested modes are written. The `_5adens` density files are always produced alongside `_mutational` and `_configurational` outputs.

#### Repair log and residue mapping

When PDB output is enabled (i.e. `--no-pdb-write` is **not** set), two additional files are written alongside the output PDB recording every structural modification made during parsing:

**`<stem>.pdb.repair.log`** — one entry per modification, e.g.:
```
GAP_INSERTED: chain_index=0 residue_index=45 (auth 47)
BACKBONE_RECONSTRUCTED: chain_index=0 residue_index=46 (auth 48) missing=N method=ideal
EXCLUDED_PARTIAL_BB: chain_index=0 residue_index=51 (auth 53) (C missing in original PDB; C zeroed, residue excluded)
CB_PLACED: chain_index=0 residue_index=12 (auth 14)
O_PLACED: chain_index=0 residue_index=12 (auth 14)
```

Entry types:

| Entry | Meaning |
|-------|---------|
| `GAP_INSERTED` | Entire residue absent from PDB (jump in residue numbering); placeholder added |
| `BACKBONE_RECONSTRUCTED` | N/CA/C atom(s) were missing and rebuilt from ideal geometry; `missing=` lists which |
| `EXCLUDED_PARTIAL_BB` | Residue had N or CA but no C — excluded from calculations, C coordinate zeroed |
| `CB_PLACED` | β-carbon placed using ideal geometry |
| `O_PLACED` | Backbone carbonyl O placed using ideal geometry |
| `MODELLER_ADDED` / `MODELLER_REMOVED` | Residues added or dropped by MODELLER reconstruction |
| `EXCLUDED_C_ZERO` | On a re-run with a completed PDB, residue skipped because C is at origin (marked missing in the prior run) |

**`<stem>.pdb.repair_map.tsv`** — one row per residue in the output array, with columns:

| Column | Meaning |
|--------|---------|
| `new_idx` | 0-based position in the repaired output array |
| `orig_idx` | 0-based position in the original PDB array; **blank** for inserted gap slots |
| `chain` | Chain letter |
| `res_idx` | Sequential residue index |
| `auth_seq` | Original PDB residue number |
| `ins_code` | Insertion code (blank if none) |
| `status` | `original` / `gap_inserted` / `excluded` |

Use this table to map frustration scores back to original PDB residue numbers, or to identify which output positions correspond to synthesised gap residues.

#### Figures

When `--draw-figure` is set, one figure per mode is written into `Visualizations/`:

- **Singleresidue**: per-residue frustration index line plot
- **Mutational / configurational**: NxN contact map with per-residue mean on top

Adding `--render-structure` embeds a PyMOL cartoon coloured by frustration class alongside the plot (requires `pymol` in PATH):

```bash
uv sync --extra pymol
# or
pip install "pytorch-frustratometer[pymol]"
```

Multiple formats can be requested together:

```bash
pytorch_frustratometer -s proteins/ -m all --draw-figure --image-format png --image-format svg -o results/
```

Full option reference:

```bash
pytorch_frustratometer --help
```

---

## Large-scale dataset processing (shards)

For datasets with thousands to millions of structures, writing one `.done/` directory per protein is slow and creates too many small files. The `--shard-output` flag switches to a shard-based output: frustration results are batched and written as compact files to a single directory, which is far more efficient on parallel filesystems (Lustre, GPFS) and downstream ML pipelines.

```bash
# Run on a directory of PDBs, writing parquet shards
pytorch_frustratometer \
    --structure /path/to/pdbs/ \
    --mode singleresidue --mode mutational \
    --shard-output /scratch/shards/ \
    --shard-format parquet \
    --batch-size 8
```

### Shard formats

| Format | Extension | When to use |
|--------|-----------|-------------|
| `parquet` *(default)* | `.parquet` | Best for downstream analysis (pandas, Polars, DuckDB, PyArrow). zstd-compressed, column-oriented. |
| `hdf5` | `.h5` | HDF5 via pandas/PyTables. Requires `pip install tables`. Read with `pd.read_hdf(path, key=mode)`. |
| `pt.zst` | `.pt.zst` | Legacy batched-tensor format used by FrustrAI-Fold. Stores raw PyTorch tensors compressed with zstd. |

### Output layout

Shards are split by mode and by a size bucket (proteins ≤ 500 residues go to `normal`; larger proteins go to `large`). A `metadata.json` summary is written on completion.

```
/scratch/shards/
├── shard_singleresidue_normal_0000.parquet   # proteins 1–200
├── shard_singleresidue_normal_0001.parquet   # proteins 201–400
├── shard_singleresidue_large_0000.parquet    # proteins > 500 residues
├── shard_mutational_normal_0000.parquet
├── shard_mutational_large_0000.parquet
├── shard_structure_normal_0000.parquet       # backbone coords (--shard-save-structure)
└── metadata.json
```

With multiple workers the rank is embedded in the filename:

```
shard_singleresidue_normal_rank0_0000.parquet
shard_singleresidue_normal_rank1_0000.parquet
```

### Schema

**`singleresidue`** — one row per residue:

| Column | Type | Description |
|--------|------|-------------|
| `entry` | str | Protein identifier (input filename stem) |
| `Res` | int32 | Residue index (0-based) |
| `ChainRes` | str | Chain + residue label |
| `AA` | int8 | Amino acid index (0–19) |
| `DensityRes` | float32 | Local residue density |
| `NativeEnergy` | float32 | Native sequence energy |
| `DecoyEnergy` | float32 | Mean decoy energy |
| `SDEnergy` | float32 | Standard deviation of decoy energies |
| `FrstIndex` | float32 | Frustration index |
| `FrstState` | int8 | Classification: −1 highly, 0 neutral, 1 minimally frustrated |

**`mutational` and `configurational`** — one row per contact pair:

| Column | Type | Description |
|--------|------|-------------|
| `entry` | str | Protein identifier |
| `Res1`, `Res2` | int32 | Residue indices of the contact pair |
| `ChainRes1`, `ChainRes2` | str | Chain + residue labels |
| `AA1`, `AA2` | int8 | Amino acid indices |
| `DensityRes1`, `DensityRes2` | float32 | Local densities |
| `NativeEnergy` | float32 | Native contact energy |
| `DecoyEnergy` | float32 | Mean decoy contact energy |
| `SDEnergy` | float32 | Std dev of decoy contact energies |
| `Welltype` | int8 | Well type index |
| `FrstIndex` | float32 | Frustration index |
| `FrstState` | int8 | Classification: −1 highly, 0 neutral, 1 minimally frustrated |

**`singleresidue_mutations`** *(requires `--DMFS`)* — one row per residue × mutant AA (L × 20 rows per protein):

| Column | Type | Description |
|--------|------|-------------|
| `entry` | str | Protein identifier |
| `Res` | int32 | Residue index |
| `ChainRes` | str | Chain + residue label |
| `AA_native` | int8 | Native amino acid index |
| `AA_mutant` | int8 | Mutant amino acid index (0–19) |
| `NativeEnergy` | float32 | Native energy at this position |
| `MutantEnergy` | float32 | Energy of the mutant |
| `ScanFrstIndex` | float32 | Per-mutant frustration index |
| `ScanFrstState` | int8 | Classification |

### Reading shards

```python
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

shard_dir = Path("/scratch/shards")

# Read all singleresidue shards into one DataFrame
sr = pd.concat(
    [pd.read_parquet(p) for p in sorted(shard_dir.glob("shard_singleresidue_normal_*.parquet"))],
    ignore_index=True,
)
# FrstState: -1 = highly frustrated, 0 = neutral, 1 = minimally frustrated
highly = sr[sr["FrstState"] == -1]
```

### Parallel runs (SLURM / multi-node)

Use `--shard-rank` and `--shard-world-size` to split a large dataset across multiple jobs. Each worker processes a disjoint slice of the input and writes its own shard stream. Rank is embedded in filenames so outputs from all workers coexist in the same directory and can be loaded together.

```bash
# Submit one job per rank
for RANK in 0 1 2 3; do
  sbatch --wrap "pytorch_frustratometer \
      --structure /path/to/pdbs/ \
      --shard-output /scratch/shards/ \
      --shard-rank $RANK --shard-world-size 4"
done
```

---

## Python API

### Single structure

`Frustratometer(mode, ...)` is a factory that returns the right model for the chosen mode.
Use `from_pdb` to run on a single file:

```python
from pytorch_frustratometer import Frustratometer, Coefficients

# Default: singleresidue + mutational + configurational
model  = Frustratometer("all")
result = model.from_pdb("my_protein.pdb")

fi_sr   = result["singleresidue"]    # [B, L] per-residue frustration index
fi_mut  = result["mutational"]       # [B, E] per-contact frustration index
fi_conf = result["configurational"]  # [B, E] per-contact frustration index
```

### Smooth frustration (fast pairwise, no decoys)

```python
from pytorch_frustratometer import Frustratometer

model  = Frustratometer("smooth")
result = model.from_pdb("my_protein.pdb")

fi_smooth = result["smooth"]   # [B, E] -(Fi + Fj) per contact edge
```

### Batch processing with DataLoader

```python
from pytorch_frustratometer import (
    Frustratometer, Coefficients,
    FrustratometerDataset, collate_frustratometer,
)
from torch.utils.data import DataLoader

coeff   = Coefficients.from_dict({"mode": "configurational", "tert_frust_ndecoys": 1000})
dataset = FrustratometerDataset(["protein1.pdb", "protein2.pdb"], coeff)
loader  = DataLoader(dataset, batch_size=4, collate_fn=collate_frustratometer)
model   = Frustratometer("configurational", coefficients=coeff)

for batch in loader:
    result = model(batch)
    print(result["configurational"])   # [B, E] frustration index per contact edge
```

`FrustratometerDataset` also accepts a directory of PDB/mmCIF files or a plain-text file
listing one path per line — the same inputs accepted by the CLI `--structure` option.

---

## Frustration modes

| Mode | Output | Description |
|------|--------|-------------|
| `singleresidue` | Per-residue index | How optimal is this residue in its local context? |
| `mutational` | Per-contact index | Averaged over all amino acid substitutions at each contact |
| `configurational` | Per-contact index | Averaged over decoy backbone conformations |
| `smooth` | Per-contact index | F̃ᵢⱼ = −(Fᵢ + Fⱼ) derived from the single-residue index |

Each mode classifies contacts/residues as **highly frustrated**, **neutral**, or **minimally frustrated** using the standard Frustratometer thresholds.

The `smooth` mode ([Guan et al. 2024](https://doi.org/10.1073/pnas.2410662121)) is a fast, decoy-free pairwise index derived directly from single-residue scores. It runs singleresidue once and requires no additional sampling, making it much cheaper than mutational or configurational. It is not included in `all` and must be requested explicitly.

---

## Package layout

```
pytorch_frustratometer/
├── algorithm/          # Core models: Frustratometer3, ConfigurationalFrustratometer, etc.
├── data/               # FrustratometerDataset, IO parsers, AWSEM gamma coefficients (.dat)
├── features/           # Feature computation: distances, neighbor graphs, geometry
├── structure/          # PDB parsing, residue constants, ideal atom placement
├── utils/              # Loss functions, frustration weights, shared utilities
├── plots/              # Visualization: frustration maps, outlier analysis, comparisons
└── coefficients.py     # Coefficients dataclass (hyperparameters for all modes)
```

---

## Citation

If you use this package, please cite the original Frustratometer:

> Ferreiro DU, Hegler JA, Komives EA, Wolynes PG. "Localizing frustration in native proteins and protein assemblies." *PNAS* (2007). https://doi.org/10.1073/pnas.0709915104

> Parra RG et al. "Protein Frustratometer 2: a tool to localize energetic frustration in protein molecules, now with electrostatics." *Nucleic Acids Research* (2016). https://doi.org/10.1093/nar/gkw304

If you use the `smooth` mode, please also cite:

> Guan X et al. "Predicting protein conformational motions using energetic frustration analysis and AlphaFold2." *PNAS* (2024). https://doi.org/10.1073/pnas.2410662121

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see [LICENSE](LICENSE). You are free to share and adapt this work, provided appropriate credit is given.
