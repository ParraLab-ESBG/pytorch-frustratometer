# code credit: https://github.com/google-deepmind/alphafold

"""Protein data type."""
import dataclasses
import io
from typing import Any, Mapping, Optional
import numpy as np

from . import residue_constants


FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
  """Protein structure representation."""

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: np.ndarray  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray  # [num_res]

  # 0-indexed number corresponding to the chain in the protein that this residue
  # belongs to.
  chain_index: np.ndarray  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray  # [num_res, num_atom_type]

  hetatm: np.ndarray # [num_res]

  # Original chain letter for each unique chain_index integer, sorted alphabetically.
  # chain_id_map[i] == the PDB chain letter corresponding to chain_index value i.
  chain_id_map: np.ndarray = None  # [num_unique_chains], dtype=str

  # Author (PDB/auth) sequence numbers — the residue numbers printed in the PDB file.
  # residue_index holds sequential per-chain 1-based indices (label_seq_id style) to
  # prevent large auth_seq_id gaps from inflating L; auth_seq_id carries the original
  # numbers for output labelling so results match FrustratometeR's column headers.
  auth_seq_id: np.ndarray = None  # [num_res], int

  # Insertion codes (e.g. '' for normal residues, 'A' for 40A).
  ins_code: np.ndarray = None    # [num_res], str

  def __post_init__(self):
    pass  # unlimited chains supported; PDB_MAX_CHAINS limit removed


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None, fn: Optional[str] = None) -> Protein:
  try:
    from Bio.PDB import PDBParser
  except ImportError:
    raise ImportError(
        "biopython is required for from_pdb_string(). "
        "Use from_pdb_biotite() instead, or install biopython."
    )
  pdb_fh = io.StringIO(pdb_str)
  parser = PDBParser(QUIET=True)
  try:
    structure = parser.get_structure('none', pdb_fh)
  except Exception as e:
    assert 0, (str(e), fn)
  return from_biopython(structure, chain_id=chain_id, fn=fn)

def from_cif_string(pdb_str: str, chain_id: Optional[str] = None, fn: Optional[str] = None) -> Protein:
  try:
    from Bio.PDB.MMCIFParser import MMCIFParser
  except ImportError:
    raise ImportError(
        "biopython is required for from_cif_string(). "
        "Use from_cif_biotite() instead, or install biopython."
    )
  pdb_fh = io.StringIO(pdb_str)
  parser = MMCIFParser(QUIET=True)
  structure = parser.get_structure('none', pdb_fh)
  return from_biopython(structure, chain_id=chain_id, fn=fn)

def _from_biotite_array(arr, fn=None, chain_id: Optional[str] = None, label_to_auth=None) -> Protein:
  """Convert a biotite AtomArray (model already selected) to Protein — fully vectorized."""
  import biotite.structure as struc

  if chain_id is not None:
    arr = arr[np.isin(arr.chain_id, chain_id.split(','))]

  # MSE → MET; clear hetero flag so the HETATM filter below keeps them
  mse_mask = arr.res_name == "MSE"
  arr.res_name[mse_mask] = "MET"
  arr.atom_name[mse_mask & (arr.atom_name == "SE")] = "SD"
  arr.hetero[mse_mask] = False

  # CATH domain chain from filename
  fn_chain = None
  if fn is not None:
    from pathlib import Path as _Path
    stem = _Path(str(fn)).stem
    if len(stem) >= 6 and stem[:4].isalnum() and stem[4].isalpha() and stem[5:7].isdigit():
      fn_chain = stem[4].upper()

  # Per-residue metadata at residue start indices
  res_starts  = struc.get_residue_starts(arr)
  n_res       = len(res_starts)
  n_atoms     = len(arr)

  res_names_r = arr.res_name[res_starts]
  res_ids_r   = arr.res_id[res_starts]
  chain_ids_r = arr.chain_id[res_starts]
  hetero_r    = arr.hetero[res_starts]
  ins_r       = getattr(arr, 'ins_code', np.full(n_atoms, ''))[res_starts]

  chain_labels_r = np.where(np.char.strip(chain_ids_r) != '',
                             np.char.strip(chain_ids_r),
                             fn_chain or 'A')

  # Residue-level filters
  std_names  = np.array([residue_constants.standardize_aa_mapping.get(n, n) for n in res_names_r])
  one_letter = np.array([residue_constants.restype_3to1.get(n, 'X') for n in std_names])
  is_hetatm_bad = hetero_r & np.array(
      [n not in residue_constants.standardize_aa_mapping or n == "UNK" for n in res_names_r])
  keep_res = ~is_hetatm_bad & (one_letter != 'X')

  # De-duplicate: keep first occurrence of each (chain, res_id, ins_code)
  keys = np.stack([chain_labels_r, res_ids_r.astype(str), ins_r], axis=1)
  _, first_occ = np.unique(keys, axis=0, return_index=True)
  is_first = np.zeros(n_res, dtype=bool)
  is_first[first_occ] = True
  keep_res &= is_first

  # Atom → residue index
  atom_res_idx = np.searchsorted(res_starts, np.arange(n_atoms), side='right') - 1

  # Atom name → 37-slot index (-1 = not in our set)
  atom_slots = np.array([residue_constants.atom_order.get(name, -1) for name in arr.atom_name])

  keep_atom   = keep_res[atom_res_idx] & (atom_slots >= 0)
  res_remap   = np.cumsum(keep_res) - 1
  new_res_idx = res_remap[atom_res_idx[keep_atom]]
  new_slots   = atom_slots[keep_atom]
  n_kept      = int(keep_res.sum())

  # Scatter into [n_kept, 37, *] arrays
  pos_out  = np.zeros((n_kept, residue_constants.atom_type_num, 3))
  mask_out = np.zeros((n_kept, residue_constants.atom_type_num))
  bfac_out = np.zeros((n_kept, residue_constants.atom_type_num))

  pos_out[new_res_idx, new_slots]  = arr.coord[keep_atom]
  bfac_out[new_res_idx, new_slots] = getattr(arr, 'b_factor', np.zeros(n_atoms))[keep_atom]
  is_unk = res_names_r[atom_res_idx[keep_atom]] == "UNK"
  mask_out[new_res_idx, new_slots] = np.where(is_unk, 0.0, 1.0)

  kr          = keep_res
  aatype_out  = np.array([residue_constants.AA_TO_ID[c] for c in one_letter[kr]])
  chain_str   = chain_labels_r[kr]
  unique_chains = np.unique(chain_str)
  chain_id_mapping = {c: i for i, c in enumerate(unique_chains)}
  chain_index = np.array([chain_id_mapping[c] for c in chain_str])
  # Remap label chain IDs to auth chain IDs for output (if mapping provided)
  if label_to_auth is not None:
    unique_chains = np.array([label_to_auth.get(c, c) for c in unique_chains])

  assert n_kept > 1, f"No standard residues parsed from {fn}"

  raw_res_ids = res_ids_r[kr]   # label_seq_id (CIF) or auth_seq_id (PDB)

  # Always use sequential 1..n per-chain so PDB and CIF produce identical
  # residue_index values for the same set of present residues.  Using raw
  # label_seq_id on the CIF path caused _insert_gap_residues to fire (inflating
  # L relative to PDB) for structures with genuinely missing residues.
  auth_seq_id_out   = None if label_to_auth is not None else raw_res_ids
  seq_residue_index = np.zeros(n_kept, dtype=int)
  for ci_val in np.unique(chain_index):
      m = chain_index == ci_val
      seq_residue_index[m] = np.arange(1, int(m.sum()) + 1)

  return Protein(
      atom_positions=pos_out,
      atom_mask=mask_out,
      aatype=aatype_out,
      residue_index=seq_residue_index,
      chain_index=chain_index,
      b_factors=bfac_out,
      hetatm=np.zeros(n_kept, dtype=int),
      chain_id_map=unique_chains,
      auth_seq_id=auth_seq_id_out,
      ins_code=np.array([c.strip() for c in ins_r[kr].astype(str)], dtype='U4'),
  )


def from_pdb_biotite(fn, chain_id: Optional[str] = None, altloc: str = 'first') -> Protein:
  """Parse PDB with biotite."""
  import biotite.structure.io.pdb as pdb_io
  pdb_file = pdb_io.PDBFile.read(str(fn))
  arr = pdb_io.get_structure(pdb_file, model=1,
                              extra_fields=['b_factor', 'occupancy'],
                              altloc=altloc)
  return _from_biotite_array(arr, fn=fn, chain_id=chain_id)


def from_cif_biotite(fn, chain_id: Optional[str] = None, altloc: str = 'first') -> Protein:
  """Parse mmCIF with biotite."""
  import biotite.structure.io.pdbx as pdbx_io
  try:
    pdbx_file = pdbx_io.CIFFile.read(str(fn))
  except AttributeError:
    pdbx_file = pdbx_io.PDBxFile.read(str(fn))
  # Use label fields (sequential label_seq_id) to avoid OOM from auth_seq_id gaps
  arr = pdbx_io.get_structure(pdbx_file, model=1,
                               extra_fields=['b_factor', 'occupancy'],
                               altloc=altloc,
                               use_author_fields=False)
  # Build label_asym_id → auth_asym_id mapping so chain_id_map uses original chain letters
  label_to_auth = None
  try:
    block = next(iter(pdbx_file.values()))
    atom_site = block["atom_site"]
    label_ids = atom_site["label_asym_id"].as_array(str)
    auth_ids  = atom_site["auth_asym_id"].as_array(str)
    label_to_auth = {}
    for l, a in zip(label_ids, auth_ids):
      if l not in label_to_auth:
        label_to_auth[l] = a
  except Exception:
    pass
  # Issue 5: chain_id is an auth chain ID; translate to label_asym_id for filtering
  # since the array was read with use_author_fields=False.
  label_chain_id = chain_id
  if chain_id is not None and label_to_auth is not None:
    auth_to_label = {v: k for k, v in label_to_auth.items()}
    parts = []
    for c in chain_id.split(','):
      lbl = auth_to_label.get(c)
      if lbl is None:
        raise ValueError(
            f"Chain '{c}' not found in {fn}. "
            f"Available auth chains: {sorted(set(label_to_auth.values()))}"
        )
      parts.append(lbl)
    label_chain_id = ','.join(parts)
  return _from_biotite_array(arr, fn=fn, chain_id=label_chain_id, label_to_auth=label_to_auth)

def from_biopython(structure, chain_id: Optional[str] = None, fn: Optional[str] = None) -> Protein:
  models = list(structure.get_models())
  model = models[0]
  atom_positions = []
  aatype = []
  atom_mask = []
  residue_index = []
  chain_ids = []
  b_factors = []
  hetatm = []

  # If the filename follows CATH domain naming (e.g. 1abcA00.pdb), extract the
  # chain letter from position 4 of the stem so blank-chain PDBs get labelled
  # with the correct chain rather than defaulting to 'A'.
  fn_chain = None
  if fn is not None:
    from pathlib import Path as _Path
    stem = _Path(fn).stem
    if len(stem) >= 6 and stem[:4].isalnum() and stem[4].isalpha() and stem[5:7].isdigit():
      fn_chain = stem[4].upper()
  chain_id = chain_id.split(',')
  seen_positions: set = set()
  for chain in model:
    raw_chain = chain.id.strip()
    # Blank chain ID: prefer the chain encoded in the filename, fall back to 'A'.
    chain_label = raw_chain or fn_chain or 'A'
    if chain_id is not None and chain_label not in chain_id:
      continue
    for res in chain:
      # Only include residues that map to one of the 20 standard amino acids, matching
      # FrustratometeR's ATOM.only + 20-AA filter behavior. Two cases to exclude:
      #   1. HETATM ligands/cofactors absent from standardize_aa_mapping (e.g. LLP) —
      #      they shift sequential indices and corrupt the min_sep check.
      #   2. ATOM-record non-standard residues (e.g. selenocysteine SEC, PYL) that are
      #      not in the 20-AA set — FrustratometeR excludes these from its domain PDB.
      is_true_hetatm = res.id[0] != " " and (
          res.resname not in residue_constants.standardize_aa_mapping
          or res.resname == "UNK"
      )
      if is_true_hetatm:
        continue

      standard_name = residue_constants.standardize_aa_mapping.get(res.resname, res.resname)
      res_shortname = residue_constants.restype_3to1.get(standard_name, 'X')
      if res_shortname == 'X':
        continue  # ATOM non-standard residue (SEC, PYL, etc.) not in 20-AA set


      restype_idx = residue_constants.AA_TO_ID.get(res_shortname)

      pos = np.zeros((residue_constants.atom_type_num, 3))
      mask = np.zeros((residue_constants.atom_type_num,))
      res_b_factors = np.zeros((residue_constants.atom_type_num,))
      # DisorderedResidue (all atoms have alt-loc conformers A+B): iterating the
      # DisorderedResidue yields child Residue objects, not atoms. Unwrap to get
      # atoms from the selected (default 'A') conformer.
      atom_source = res.disordered_get() if res.is_disordered() == 2 else res
      for atom in atom_source:
        if res.resname == "MSE" and atom.name == "SE":
          # Convert MSE to MET by replacing Se with S.
          atom.name = "SD"
        
        if atom.name not in residue_constants.atom_types:
          continue
        pos[residue_constants.atom_order[atom.name]] = atom.coord
        mask[residue_constants.atom_order[atom.name]] = 1. if res.resname!="UNK" else 0.
        res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor

      # if np.sum(mask) < 0.5:
      #   continue
      
      pos_key = (chain_label, res.id[1], res.id[2])
      if pos_key in seen_positions:
        continue
      seen_positions.add(pos_key)

      aatype.append(restype_idx)
      atom_positions.append(pos)
      atom_mask.append(mask)
      residue_index.append(res.id[1])
      chain_ids.append(chain_label)
      b_factors.append(res_b_factors)
      hetatm.append(0)  # all parsed residues are in standardize_aa_mapping → not HETATM

  unique_chain_ids = np.unique(chain_ids)
  chain_id_mapping = {cid: i for i, cid in enumerate(unique_chain_ids)}
  chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

  auth_residue_index = np.array(residue_index)   # original auth_seq_id for output

  # Sequential per-chain 1-based index (label_seq_id style) for residue_index so
  # that large auth_seq_id gaps (renumbered chains, disordered regions) don't inflate L.
  seq_residue_index = np.zeros(len(chain_ids), dtype=int)
  for ci_val in np.unique(chain_index):
      m = chain_index == ci_val
      seq_residue_index[m] = np.arange(1, int(m.sum()) + 1)

  assert len(atom_positions) > 1

  return Protein(
      atom_positions=np.array(atom_positions),
      atom_mask=np.array(atom_mask),
      aatype=np.array(aatype),
      residue_index=seq_residue_index,
      chain_index=chain_index,
      b_factors=np.array(b_factors),
      hetatm=np.array(hetatm),
      chain_id_map=unique_chain_ids,
      auth_seq_id=auth_residue_index)


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')
