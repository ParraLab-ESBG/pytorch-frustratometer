# Frustratometer reference outputs

Reference outputs for `tests/test_frustratometer_parity.py`, from the AWSEM Frustratometer. Its `SeqDist`
option (3 or 12) changes only which residue pairs count toward the residue density:
- `SeqDist=3`: same-chain pairs with `|i-j| >= 2`;
- `SeqDist=12`: same-chain pairs with `|i-j| >= 13`.

Pairs of residues in different chains count at any separation. The contacts are the same at both values:
`|i-j| >= 2` within a chain, closer than 9.5 A.

| Directory | Source | Structure | Electrostatics K | `SeqDist` | `burial_min_sep` |
|---|---|---|---|---|---|
| `1r69_kelec_4.15_seqsep_3` | web server | `1r69.pdb` | 4.15 | 3 | 2 |
| `1l63_kelec_0_seqsep_3` | web server | `1l63.pdb` | 0 (off) | 3 | 2 |
| `1l63_kelec_0_seqsep_12` | web server | `1l63.pdb` | 0 (off) | 12 | 13 |
| `1r69_split_kelec_0_seqsep_3` | R package, `SeqDist=3` binary of commit 11edd32 | `1r69_split.pdb` | off | 3 | 2 |
| `1r69_split_kelec_0_seqsep_12` | R package | `1r69_split.pdb` | off | 12 | 13 |

## Sources

- **Web server.** Outputs of the Frustratometer web server, copied without changes (directory and file
  names included) from the Python Frustratometer repository, `tests/data/` at commit 5f29641
  (github.com/cabb99/Frustratometer). The file names are server job IDs. The headers start with `#Res`.
- **R package.** frustratometeR 0.1.0 (github.com/proteinphysiologylab/frustratometeR, commit fe2d6aa), run
  with R 4.3.3. The headers start with `Res`. Each mode and `SeqDist` was a separate call:

  ```r
  frustratometeR::calculate_frustration(PdbFile = "1r69_split.pdb", Mode = <"singleresidue"|"mutational"|"configurational">,
                                        SeqDist = <3|12>, Electrostatics_K = NULL,
                                        Graphics = FALSE, Visualization = FALSE, ResultsDir = <one directory per call>)
  ```

- **The `SeqDist=3` split references** used the package's earlier `SeqDist=3` LAMMPS binary,
  `inst/Scripts/lmp_serial_3` at commit 11edd32 (2020-09-30).
  - The LAMMPS input template was the package's own, except for `atom_style peptide`, the atom style that
    binary reads. The template used it until commit c3df550.
  - That binary, run the same way on `1r69.pdb`, reproduces the web server's `1r69_kelec_0_seqsep_3` files
    exactly, frustration indices included.

## Structures

- `1r69.pdb` and `1l63.pdb` are copied from the same Python Frustratometer commit.
- `1r69_split.pdb` is `1r69.pdb` with residues 31-63 relabeled as chain B, and a TER record
  (`TER 228 SER A 30`) between the chains. Residue numbers, coordinates and every other record are
  unchanged. The two sides of the break, A30 and B31, are in contact. This script makes it:

  ```python
  out, last_chain_a_atom = [], None
  for line in open("1r69.pdb"):
      is_residue_record = line.startswith(("ATOM  ", "HETATM", "TER   "))
      residue = int(line[22:26]) if is_residue_record else None
      if is_residue_record and line[21] == "A" and 31 <= residue <= 63:
          if last_chain_a_atom is not None:
              serial = int(last_chain_a_atom[6:11]) + 1
              out.append(f"TER   {serial:5d}      {last_chain_a_atom[17:20]} A{last_chain_a_atom[22:26]}".ljust(80) + "\n")
              last_chain_a_atom = None
          line = line[:21] + "B" + line[22:]
      elif line.startswith("ATOM  ") and line[21] == "A" and residue < 31:
          last_chain_a_atom = line
      out.append(line)
  open("1r69_split.pdb", "w").writelines(out)
  ```
