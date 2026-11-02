# AWSEM LAMMPS reference outputs

Reference outputs for `tests/test_frustratometer_parity.py`, from direct runs of AWSEM in LAMMPS.

| File | Structure | Mode | Decoys per contact | Electrostatics |
|---|---|---|---|---|
| `6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat` | `6u5e.pdb`, chain A | mutational | 10^6 | off |

## Source

Copied without changes from the Python Frustratometer repository (github.com/cabb99/Frustratometer),
`tests/data/`: the `.dat` file from commit 87db488 and `6u5e.pdb` (here `tests/data/6u5e.pdb`) from commit
5f29641. The Python Frustratometer's tests compare their mutational frustration with the same file.

## Columns

One row per contact:

| Column | Content |
|---|---|
| `i`, `j` | 1-based positions of the two residues in the chain |
| `i_chain`, `j_chain` | chain numbers |
| `xi yi zi`, `xj yj zj` | CB coordinates (CA for glycine), in A |
| `r_ij` | distance between them, in A |
| `rho_i`, `rho_j` | residue densities |
| `a_i`, `a_j` | one-letter amino acids |
| `native_energy` | native energy of the contact |
| `<decoy_energies>`, `std(decoy_energies)` | mean and standard deviation of the decoy energies |
| `f_ij` | frustration index |
