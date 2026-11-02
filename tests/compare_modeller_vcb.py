"""
compare_modeller_vcb.py — Compare MODELLER's CB placement to our compute_ideal_vcb

For each mutant PDB saved by FrustratometeR (after complete_pdb), reads MODELLER's
CB at the mutated position and compares it to our ideal vCB computed from the native
backbone.  Also reports rho at the mutated position using both CB sources.

Key question: at native GLY positions, does MODELLER's CB match our vCB?

Usage:
    python tests/compare_modeller_vcb.py \\
        --native   path/to/8dc1A01.pdb \\
        --mut-dir  path/to/8dc1A01_A.done \\
        --positions 175 207        # 1-indexed; default = GLY positions only

    # Compare all available positions:
    python tests/compare_modeller_vcb.py \\
        --native  path/to/8dc1A01.pdb \\
        --mut-dir path/to/8dc1A01_A.done \\
        --all-positions
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_cb_ca_from_pdb(pdb_path, resnum):
    """Return (CB_xyz or None, CA_xyz or None, resname or None) for the given residue number."""
    cb = ca = resname = None
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            rnum = int(line[22:26].strip())
            if rnum != resnum:
                continue
            if resname is None:
                resname = line[17:20].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if atom == "CA":
                ca = np.array([x, y, z])
            elif atom == "CB":
                cb = np.array([x, y, z])
    return cb, ca, resname


def read_native_resname(pdb_path, resnum):
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and int(line[22:26].strip()) == resnum:
                return line[17:20].strip()
    return None


def scan_mut_dir(mut_dir):
    """Return list of (resnum, mutant_aa1, pdb_path) for every mutant PDB found.

    Searches two locations within each per-mutation *_<chain>.done/ directory:
      1. directly inside the done dir  (*_?.done/*.pdb)
      2. inside its FrustrationData/   (*_?.done/FrustrationData/*.pdb)

    FrustratometeR's calculate_frustration moves all *.pdb to FrustrationData/
    before returning, so older runs store the complete_pdb-processed structure
    there rather than at the top level of the done dir.

    Files without a trailing single-char chain suffix (e.g. 8dc1A01_A_175_ALA.pdb)
    are pre-complete_pdb saves and are excluded by the regex.
    """
    pattern = re.compile(r"_(\d+)_([A-Z]+)_[A-Za-z0-9]\.pdb$")
    seen = {}  # (resnum, aa1) -> path; prefer top-level over FrustrationData
    for glob_pat in ["*_?.done/*.pdb", "*_?.done/FrustrationData/*.pdb"]:
        for path in sorted(Path(mut_dir).glob(glob_pat)):
            m = pattern.search(path.name)
            if not m:
                continue
            resnum = int(m.group(1))
            aa1 = THREE_TO_ONE.get(m.group(2))
            if aa1 and (resnum, aa1) not in seen:
                seen[(resnum, aa1)] = str(path)
    return [(r, aa, p) for (r, aa), p in sorted(seen.items())]


def load_native(native_pdb):
    """Load native PDB; return (batch, vcb_all, mpnn_vcb_all, native_centroid, cfg, params, model, native_energy)."""
    from pytorch_frustratometer import FrustratometerDataset, collate_frustratometer, Frustratometer
    from pytorch_frustratometer.coefficients import Coefficients
    from pytorch_frustratometer.structure.geometry import compute_ideal_vcb, compute_mpnn_vcb, get_CB_coords

    cfg = Coefficients()
    cfg.save_mutations = True
    ds  = FrustratometerDataset([native_pdb], cfg)
    batch = collate_frustratometer([ds[0]])

    vcb_all      = compute_ideal_vcb(batch)                # [1, L, 3]
    mpnn_vcb_all = compute_mpnn_vcb(batch)                 # [1, L, 3]

    atom_pos = batch["atom_positions"]  # [1, L, 6, 3]
    aatype   = batch["aa_type"]         # [1, L]
    native_centroid = get_CB_coords(atom_pos[:, :, :5], aatype).float()  # [1, L, 3]

    eta    = float(cfg.eta)
    rmin   = float(cfg.well_rmin[0])
    rmax   = float(cfg.well_rmax[0])
    minsep = int(cfg.min_seq_distance)
    cutoff = float(cfg.tert_frust_cutoff)
    params = dict(eta=eta, rmin=rmin, rmax=rmax, minsep=minsep, cutoff=cutoff)

    model = Frustratometer("singleresidue", coefficients=cfg)
    if "residue_density" not in batch:
        with torch.no_grad():
            model.compute_densities(batch)
    native_energy = model.compute_singleresidue_native_ixn(batch, batch["aa_type"])

    return batch, vcb_all, mpnn_vcb_all, native_centroid, cfg, params, model, native_energy


def compute_rho_full(cb_xyz, idx, batch, native_centroid, params):
    """Return (rho_scalar, v_theta [L], active_mask [L], dist [L])."""
    ri_all   = batch["residue_index"][0]
    seq_mask = batch["seq_mask"][0].bool()
    hetatm   = batch["hetatm"][0].bool()
    edge_mask = seq_mask & ~hetatm

    query = torch.tensor(cb_xyz, dtype=torch.float32).unsqueeze(0)  # [1, 3]
    nat_j = native_centroid[0]                                       # [L, 3]

    dist    = torch.cdist(query, nat_j.float())[0]   # [L]
    t_min   = torch.tanh(params["eta"] * (dist - params["rmin"]))
    t_max   = torch.tanh(params["eta"] * (params["rmax"] - dist))
    v_theta = 0.25 * (1 + t_min) * (1 + t_max)

    ri_q   = ri_all[idx]
    sep    = (ri_q - ri_all).abs()
    active = edge_mask & (sep > params["minsep"]) & (dist <= params["cutoff"])

    rho = (v_theta * active.float()).sum().item()
    return rho, v_theta, active, dist


def compute_rho(cb_xyz, idx, batch, native_centroid, params):
    return compute_rho_full(cb_xyz, idx, batch, native_centroid, params)[0]




def compute_fi_with_cb(mod_cb_xyz, idx, batch, model, native_energy):
    """
    Estimate FI at `idx` using mod_cb_xyz as the CB for water-distance calculation.

    Replaces the vCB at position idx with mod_cb_xyz, recomputes
    _compute_gly_water_correction, recomputes stats, returns FI for each AA.
    Returns dict {aa1: fi_value} or None on error.
    """
    try:
        import torch
        from pytorch_frustratometer.structure.geometry import compute_ideal_vcb
    except ImportError as e:
        print(f"    (cannot compute FI: {e})")
        return None

    from pytorch_frustratometer.structure.residue_constants import ID_TO_AA

    vcb_all  = compute_ideal_vcb(batch, dtype=torch.float32)   # [1, L, 3]
    vcb_mod  = vcb_all.clone()
    vcb_mod[0, idx] = torch.tensor(mod_cb_xyz, dtype=torch.float32)

    try:
        rho_vcb, rho_ca, vcb_hybrid = model._compute_gly_alt_densities(batch)
        vcb_hybrid_mod = vcb_hybrid.clone()
        vcb_hybrid_mod[0, idx] = torch.tensor(mod_cb_xyz, dtype=torch.float32)

        stats_mod, E_mod, _ = model.compute_singleresidue_stats_fast(
            batch, native_energy,
            rho_vcb=rho_vcb, rho_ca=rho_ca,
            vcb=vcb_hybrid_mod, gly_water_corr=True,
        )
        d_mean = float(stats_mod[0, idx, 0])
        d_std  = float(stats_mod[0, idx, 1])
        result = {}
        for x in range(20):
            e_x = float(E_mod[x, 0, idx])
            fi  = (d_mean - e_x) / max(d_std, 1e-8)
            result[ID_TO_AA[x]] = fi
        return result
    except Exception as e:
        print(f"    (FI computation failed: {e})")
        return None


def load_mut_ref(mut_ref_dir):
    """Load FrustratometeR FI reference from a MutationsData/ directory."""
    if mut_ref_dir is None or not os.path.isdir(mut_ref_dir):
        return {}
    try:
        from pytorch_frustratometer.data.io import read_mutations_data
        df = read_mutations_data(mut_ref_dir)
        result = {}
        for _, row in df.iterrows():
            result[(int(row["Res"]), str(row["AA"]))] = float(row["FrstIndex"])
        return result
    except Exception as e:
        print(f"  (could not load mut-ref: {e})")
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native",        default=None,
                    help="Native PDB file (default: FrustrationData/<base>.pdb inside --mut-dir)")
    ap.add_argument("--mut-dir",      required=True, help="Directory with MODELLER mutant PDBs")
    ap.add_argument("--positions",    type=int, nargs="*", default=None,
                    help="1-indexed residue numbers to compare (default: GLY positions only)")
    ap.add_argument("--all-positions", action="store_true",
                    help="Compare all positions with available mutant PDBs")
    ap.add_argument("--mut-ref",      default=None,
                    help="MutationsData/ directory with FrustratometeR FI reference values")
    ap.add_argument("--show-neighbors", action="store_true",
                    help="Print per-neighbor contact differences between vCB sources")
    ap.add_argument("--fi",           action="store_true",
                    help="Compute FI estimate using MODELLER CB (slow — one model pass per position)")
    args = ap.parse_args()

    if not os.path.isdir(args.mut_dir):
        sys.exit(f"Mutant directory not found: {args.mut_dir}")

    # Auto-detect native from FrustrationData if not supplied.
    if args.native is None:
        frusta_dir = Path(args.mut_dir) / "FrustrationData"
        candidates = sorted(frusta_dir.glob("*.pdb")) if frusta_dir.is_dir() else []
        candidates = [p for p in candidates if not re.search(r"_\d+_[A-Z]+", p.name)]
        if not candidates:
            sys.exit("Could not auto-detect native PDB in FrustrationData/. Use --native.")
        args.native = str(candidates[0])
        print(f"Auto-detected native PDB: {args.native}")

    if not os.path.isfile(args.native):
        sys.exit(f"Native PDB not found: {args.native}")

    native_gly = set()
    native_res = {}
    with open(args.native) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                rnum = int(line[22:26].strip())
                rname = line[17:20].strip()
                native_res[rnum] = rname
                if rname == "GLY":
                    native_gly.add(rnum)

    mutants = scan_mut_dir(args.mut_dir)
    if not mutants:
        sys.exit("No mutant PDB files found.")

    avail_pos = {r for r, _, _ in mutants}

    if args.positions:
        show_pos = set(args.positions)
    elif args.all_positions:
        show_pos = avail_pos
    else:
        show_pos = native_gly

    target_pos = sorted(show_pos & avail_pos)
    if not target_pos:
        print("No mutant PDBs found for the requested positions.")
        print(f"Available: {sorted(avail_pos)}")
        print(f"GLY positions: {sorted(native_gly)}")
        return

    ref_fi = load_mut_ref(args.mut_ref)

    print(f"Native PDB : {args.native}")
    print(f"GLY positions (1-indexed): {sorted(native_gly)}")
    print(f"Positions with mutant PDBs: {sorted(avail_pos)}")
    print(f"Positions to compare: {target_pos}")
    print("\nLoading native structure …")

    batch, vcb_all, mpnn_vcb_all, native_centroid, cfg, params, model, native_energy = \
        load_native(args.native)
    ri_all  = batch["residue_index"][0]
    aa_seq  = batch.get("sequence", ["?"] * ri_all.shape[0])

    print()
    for resnum in target_pos:
        native_name = native_res.get(resnum, "?")
        is_gly = native_name == "GLY"

        matches = (ri_all == resnum).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            print(f"Res {resnum}: not found in batch, skipping.")
            continue
        idx = matches[0].item()

        ca_xyz       = batch["atom_positions"][0, idx, 1].numpy()
        vcb_xyz      = vcb_all[0, idx].numpy()
        mpnn_vcb_xyz = mpnn_vcb_all[0, idx].numpy()

        rho_nat_batch = batch["residue_density"][0, idx].item() \
            if "residue_density" in batch else float("nan")
        rho_vcb_ours = compute_rho(vcb_xyz,      idx, batch, native_centroid, params)
        rho_mpnn     = compute_rho(mpnn_vcb_xyz, idx, batch, native_centroid, params)

        print(f"=== Res {resnum:3d}  native={native_name}{' (GLY)' if is_gly else ''} ===")
        print(f"  CA xyz            : {ca_xyz}")
        print(f"  our vCB (CHARMM)  : {vcb_xyz}  |CA→vCB| = {np.linalg.norm(vcb_xyz - ca_xyz):.4f} Å")
        print(f"  ProteinMPNN vCB   : {mpnn_vcb_xyz}  |CA→vCB| = {np.linalg.norm(mpnn_vcb_xyz - ca_xyz):.4f} Å")
        print(f"  |CHARMM - MPNN|   : {np.linalg.norm(vcb_xyz - mpnn_vcb_xyz):.4f} Å")
        print(f"  rho_native (batch): {rho_nat_batch:.4f}")
        print(f"  rho (CHARMM vCB)  : {rho_vcb_ours:.4f}")
        print(f"  rho (MPNN vCB)    : {rho_mpnn:.4f}")

        pos_mutants = sorted((aa, p) for r, aa, p in mutants if r == resnum and aa != "G")
        if not pos_mutants:
            print("  No non-GLY mutant PDBs at this position.\n")
            continue

        # Collect MODELLER CBs (most are the same ideal geometry; use first non-None)
        mod_cb_consensus = None
        mod_cbs = {}
        for aa, pdb_path in pos_mutants:
            cb, _, _ = read_cb_ca_from_pdb(pdb_path, resnum)
            if cb is not None:
                mod_cbs[aa] = cb
                if mod_cb_consensus is None:
                    mod_cb_consensus = cb

        if mod_cb_consensus is not None:
            rho_mod_consensus = compute_rho(mod_cb_consensus, idx, batch, native_centroid, params)
            d_ca_mod = np.linalg.norm(mod_cb_consensus - ca_xyz)
            d_vcb_mod = np.linalg.norm(vcb_xyz - mod_cb_consensus)
            print(f"\n  MODELLER CB (from first mutant):")
            print(f"    xyz             : {mod_cb_consensus}  |CA→CB| = {d_ca_mod:.4f} Å")
            print(f"    |CHARMM - MOD|  : {d_vcb_mod:.4f} Å")
            print(f"    |MPNN - MOD|    : {np.linalg.norm(mpnn_vcb_xyz - mod_cb_consensus):.4f} Å")
            print(f"    rho (MOD CB)    : {rho_mod_consensus:.4f}  (vs CHARMM: {rho_mod_consensus-rho_vcb_ours:+.4f}, vs MPNN: {rho_mod_consensus-rho_mpnn:+.4f})")

            if args.show_neighbors:
                print(f"\n  Contact differences (CHARMM vCB vs MODELLER CB):")
                ri_all_np = ri_all.numpy()
                for cb_a, cb_b, la, lb in [
                    (vcb_xyz, mod_cb_consensus, "CHARMM", "MODELLER"),
                    (mpnn_vcb_xyz, mod_cb_consensus, "MPNN", "MODELLER"),
                ]:
                    _, theta_a, active_a, dist_a = compute_rho_full(cb_a, idx, batch, native_centroid, params)
                    _, theta_b, active_b, dist_b = compute_rho_full(cb_b, idx, batch, native_centroid, params)
                    only_a = active_a & ~active_b
                    only_b = active_b & ~active_a
                    print(f"  [{la} vs {lb}]")
                    if not (only_a.any() or only_b.any()):
                        print(f"    (identical contact set)")
                    for mask, dist_x, theta_x, tag in [
                        (only_a, dist_a, theta_a, f"only {la}"),
                        (only_b, dist_b, theta_b, f"only {lb}"),
                    ]:
                        for j in mask.nonzero(as_tuple=True)[0].tolist():
                            rj = ri_all_np[j]
                            print(f"    {tag:>14}  res={rj:4d}  dist={dist_x[j]:.3f}Å  θ={theta_x[j]:.4f}")

        # Per-AA table
        fi_hdr = ""
        if ref_fi:
            fi_hdr = f"  {'FI_FR':>8}  {'FI_CHARMM':>10}  {'err_C':>7}"
            if args.fi and mod_cb_consensus is not None:
                fi_hdr += f"  {'FI_MOD':>8}  {'err_M':>7}"
        print(f"\n  {'AA':>4}  {'|CA→MOD|':>9}  {'|vCB-MOD|':>10}  {'rho(MOD)':>9}  {'rho_diff':>9}{fi_hdr}")

        fi_mod_by_aa = None
        if args.fi and mod_cb_consensus is not None:
            print("  (computing FI with MODELLER CB …)")
            fi_mod_by_aa = compute_fi_with_cb(
                mod_cb_consensus, idx, batch, model, native_energy
            )

        for aa, pdb_path in pos_mutants:
            if aa not in mod_cbs:
                print(f"  {aa:>4}  (no CB found in {Path(pdb_path).name})")
                continue
            mod_cb   = mod_cbs[aa]
            d_ca     = np.linalg.norm(mod_cb - ca_xyz)
            d_vcb    = np.linalg.norm(vcb_xyz - mod_cb)
            rho_mod  = compute_rho(mod_cb, idx, batch, native_centroid, params)
            row = f"  {aa:>4}  {d_ca:9.4f}  {d_vcb:10.4f}  {rho_mod:9.4f}  {rho_mod-rho_vcb_ours:+9.4f}"
            if ref_fi:
                fi_fr = ref_fi.get((resnum, aa), float("nan"))
                fi_ours = float("nan")  # placeholder — ours not recomputed here
                err_c = float("nan")
                row += f"  {fi_fr:>8.3f}  {'':>10}  {'':>7}"
                if fi_mod_by_aa is not None:
                    fi_m  = fi_mod_by_aa.get(aa, float("nan"))
                    err_m = abs(fi_m - fi_fr) if not (np.isnan(fi_m) or np.isnan(fi_fr)) else float("nan")
                    row += f"  {fi_m:>8.3f}  {err_m:>7.3f}"
            print(row)
        print()


if __name__ == "__main__":
    main()
