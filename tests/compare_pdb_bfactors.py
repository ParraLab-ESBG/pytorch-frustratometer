"""
compare_pdb_bfactors.py

Compare B-factors (frustration index) and atom coordinates between the PyTorch
frustratometer PDB output and the original FrustratometeR PDB output, and
diagnose potential causes of density calculation differences.

Sections:
  1. B-factor comparison      — frustration index per residue (CA atom)
  2. Coordinate superimposition — Kabsch alignment on common CA atoms
  3. Atom position comparison — CA, CB (non-GLY), GLY CA separately
  4. Density diagnostics      — HETATM/MSE, alt-locs, occupancy,
                                missing CB, residue set differences,
                                borderline density pairs (4.5–6.5 Å shell)
  5. Frustration pair-set     — reads <stem>.pdb_<mode> txt files, compares
                                pair sets (frac_matched diagnosis), reports
                                missing pairs with HETATM/distance info, and
                                compares NativeEnergy for matched pairs

Usage:
    # Single protein:
    python tests/compare_pdb_bfactors.py \
        results/frustration_data_investigate_77/4l1gA00.done/FrustrationData/

    python tests/compare_pdb_bfactors.py --ours a.pdb --ref b.pdb

    # All 77 outliers, output to file (with frustration pair diagnosis):
    python tests/compare_pdb_bfactors.py \
        --domains outliers_domains.txt \
        --results-dir results/frustration_data_investigate_77 \
        --ref-base ../dompdb_v4.4 \
        --frust-mode configurational \
        > compare_77.txt 2>&1
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

RMIN, RMAX = 4.5, 6.5   # AWSEM burial density well
MIN_SEP     = 3          # sequence separation exclusion

STANDARD_AA = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
    "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
    "THR","TRP","TYR","VAL",
    # common synonyms accepted by our parser
    "HIE","HID","HIP","CYX","MSE","SEP","TPO","PTR",
}


# ---------------------------------------------------------------------------
# PDB parsing — stores record_type alongside each atom
# ---------------------------------------------------------------------------

def parse_pdb(path: Path) -> dict[tuple, dict]:
    """
    Returns {(chain, resnum, icode, atom_name): record}
    where record = {xyz, bfac, resname, record_type, altloc, occupancy}.
    """
    records = {}
    with open(path) as fh:
        for line in fh:
            rec_type = line[:6].rstrip()
            if rec_type not in ("ATOM", "HETATM"):
                continue
            atom_name = line[12:16].strip()
            altloc    = line[16].strip()
            resname   = line[17:20].strip()
            chain     = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            try:
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            try:
                occupancy = float(line[54:60])
            except ValueError:
                occupancy = 1.0
            try:
                bfac = float(line[60:66])
            except ValueError:
                bfac = float("nan")
            key = (chain, resnum, icode, atom_name)
            # keep first alt-loc seen (usually 'A' or blank)
            if key not in records:
                records[key] = {
                    "xyz": xyz, "bfac": bfac, "resname": resname,
                    "record_type": rec_type, "altloc": altloc,
                    "occupancy": occupancy,
                }
    return records


def residue_atoms(records: dict, chain, resnum, icode) -> dict[str, dict]:
    prefix = (chain, resnum, icode)
    return {k[3]: v for k, v in records.items() if k[:3] == prefix}


def all_residues(records: dict) -> list[tuple]:
    return sorted({k[:3] for k in records})


def density_coord(records: dict, key: tuple):
    """CA for GLY, CB for all others.  Returns xyz or None."""
    ch, rn, ic = key
    atoms = residue_atoms(records, ch, rn, ic)
    if not atoms:
        return None
    resname = next(iter(atoms.values()))["resname"]
    atom = "CA" if resname == "GLY" else "CB"
    if atom in atoms:
        return atoms[atom]["xyz"]
    if "CA" in atoms:   # fallback
        return atoms["CA"]["xyz"]
    return None


# ---------------------------------------------------------------------------
# Kabsch superimposition
# ---------------------------------------------------------------------------

def kabsch(P, Q):
    p_c, q_c = P.mean(0), Q.mean(0)
    H = (P - p_c).T @ (Q - q_c)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T
    return R, q_c - p_c @ R.T


def xform(xyz, R, t):
    return xyz @ R.T + t


# ---------------------------------------------------------------------------
# Section 1 — B-factor comparison
# ---------------------------------------------------------------------------

def section_bfactors(ours_rec, ref_rec):
    print("\n══ 1. B-FACTOR COMPARISON (frustration index, CA atom) ══════════════")

    def ca_bfac(rec):
        return {k[:3]: v["bfac"] for k, v in rec.items() if k[3] == "CA"}

    ob, rb = ca_bfac(ours_rec), ca_bfac(ref_rec)
    common = sorted(set(ob) & set(rb))
    only_o = sorted(set(ob) - set(rb))
    only_r = sorted(set(rb) - set(ob))

    print(f"  common={len(common)}  only-ours={len(only_o)}  only-ref={len(only_r)}")
    if only_o: print(f"    Only in ours: {only_o[:8]}{'...' if len(only_o)>8 else ''}")
    if only_r: print(f"    Only in ref:  {only_r[:8]}{'...' if len(only_r)>8 else ''}")
    if not common:
        print("  ERROR: no common CA residues"); return

    o = np.array([ob[k] for k in common])
    r = np.array([rb[k] for k in common])
    d = o - r
    if len(common) > 1 and o.std() > 0 and r.std() > 0:
        corr = np.corrcoef(o, r)[0, 1]
    else:
        corr = float("nan")
    print(f"  n={len(common)}  MAE={np.mean(np.abs(d)):.4f}  RMSE={np.sqrt(np.mean(d**2)):.4f}"
          f"  R={corr:.6f}  mean_Δ={d.mean():+.4f}")
    order = np.argsort(np.abs(d))[::-1]
    print(f"\n  {'chain':>5} {'res':>5} {'ic':>2}   {'ours':>8} {'ref':>8} {'Δ':>8}")
    for i in order[:10]:
        ch, rn, ic = common[i]
        print(f"  {ch:>5} {rn:>5} {ic or ' ':>2}   {o[i]:>8.3f} {r[i]:>8.3f} {d[i]:>+8.3f}")


# ---------------------------------------------------------------------------
# Section 2 — Superimposition
# ---------------------------------------------------------------------------

def section_superimpose(ours_rec, ref_rec):
    print("\n══ 2. COORDINATE SUPERIMPOSITION (Kabsch on common CA) ══════════════")
    ours_ca = {k[:3] for k in ours_rec if k[3] == "CA"}
    ref_ca  = {k[:3] for k in ref_rec  if k[3] == "CA"}
    common  = sorted(ours_ca & ref_ca)
    if len(common) < 3:
        print("  Not enough common CA atoms"); return None, None

    P = np.array([ours_rec[(ch,rn,ic,"CA")]["xyz"] for ch,rn,ic in common])
    Q = np.array([ref_rec[ (ch,rn,ic,"CA")]["xyz"] for ch,rn,ic in common])
    pre  = np.sqrt(np.mean(np.sum((P-Q)**2, axis=1)))
    R, t = kabsch(P, Q)
    d    = np.sqrt(np.sum((xform(P,R,t)-Q)**2, axis=1))
    post = np.sqrt(np.mean(d**2))

    print(f"  Common CA: {len(common)}   pre-RMSD={pre:.4f} Å   post-RMSD={post:.4f} Å")
    if post > 0.01:
        print(f"  WARNING: structures differ after alignment (RMSD={post:.4f} Å)")
        order = np.argsort(d)[::-1]
        print(f"  {'chain':>5} {'res':>5} {'ic':>2}   {'dist_Å':>8}")
        for i in order[:10]:
            ch, rn, ic = common[i]
            print(f"  {ch:>5} {rn:>5} {ic or ' ':>2}   {d[i]:>8.4f}")
    else:
        print("  Structures are essentially identical after alignment")
    return R, t


# ---------------------------------------------------------------------------
# Section 3 — Atom position comparison
# ---------------------------------------------------------------------------

def section_atom_positions(ours_rec, ref_rec, R, t):
    print("\n══ 3. ATOM POSITION COMPARISON ══════════════════════════════════════")

    def adist(key, atom):
        ch, rn, ic = key
        k1, k2 = (ch,rn,ic,atom), (ch,rn,ic,atom)
        if k1 not in ours_rec or k2 not in ref_rec: return None
        o = ours_rec[k1]["xyz"]
        r = ref_rec[k2]["xyz"]
        if R is not None: o = xform(o[None], R, t)[0]
        return float(np.linalg.norm(o - r))

    common = sorted(set(all_residues(ours_rec)) & set(all_residues(ref_rec)))
    gly, non_gly = [], []
    for key in common:
        ch, rn, ic = key
        atoms = residue_atoms(ours_rec, ch, rn, ic)
        (gly if (atoms and next(iter(atoms.values()))["resname"]=="GLY") else non_gly).append(key)

    print(f"  GLY={len(gly)}  non-GLY={len(non_gly)}")

    # CA (all)
    ca_d = [d for k in common if (d:=adist(k,"CA")) is not None]
    if ca_d:
        a = np.array(ca_d)
        print(f"\n  CA (all):  mean={a.mean():.4f}  max={a.max():.4f}  n>0.01Å={np.sum(a>0.01)}")

    # GLY CA
    if gly:
        gd = [d for k in gly if (d:=adist(k,"CA")) is not None]
        cb_o = sum("CB" in residue_atoms(ours_rec,*k) for k in gly)
        cb_r = sum("CB" in residue_atoms(ref_rec, *k) for k in gly)
        print(f"\n  GLY CA:    ", end="")
        if gd:
            a = np.array(gd)
            print(f"mean={a.mean():.4f}  max={a.max():.4f}  n>0.01Å={np.sum(a>0.01)}")
            big = [(k,d) for k,d in zip(gly,[adist(k,'CA') for k in gly])
                   if d is not None and d>0.01]
            for k,d in sorted(big, key=lambda x:-x[1]):
                print(f"    chain={k[0]} res={k[1]}{k[2] or ''}  CA_dist={d:.4f} Å")
        else:
            print("n/a")
        print(f"  GLY CB present — ours: {cb_o}/{len(gly)}   ref: {cb_r}/{len(gly)}")

    # non-GLY CB
    if non_gly:
        cbd = [(k,d) for k in non_gly if (d:=adist(k,"CB")) is not None]
        miss_o = sum("CB" not in residue_atoms(ours_rec,*k) for k in non_gly)
        miss_r = sum("CB" not in residue_atoms(ref_rec, *k) for k in non_gly)
        print(f"\n  non-GLY CB (n={len(cbd)}):", end=" ")
        if cbd:
            a = np.array([d for _,d in cbd])
            print(f"mean={a.mean():.4f}  max={a.max():.4f}  n>0.01Å={np.sum(a>0.01)}")
            if miss_o or miss_r:
                print(f"    Missing CB — ours: {miss_o}   ref: {miss_r}")
            order = np.argsort(a)[::-1]
            if a.max() > 0.01:
                print(f"    Top CB outliers: {'chain':>5} {'res':>5} {'ic':>2} {'resname':>7} {'dist':>8}")
                for i in order[:10]:
                    k, d = cbd[i]
                    rn_str = next(iter(residue_atoms(ours_rec,*k).values()))["resname"]
                    print(f"      {k[0]:>5} {k[1]:>5} {k[2] or ' ':>2} {rn_str:>7} {d:>8.4f}")
        else:
            print("no common CB")


# ---------------------------------------------------------------------------
# Section 4 — Density diagnostics
# ---------------------------------------------------------------------------

def section_density_diagnostics(ours_rec, ref_rec, R, t):
    print("\n══ 4. DENSITY DIAGNOSTICS ════════════════════════════════════════════")

    # ── 4a. HETATM / non-standard residues ──────────────────────────────────
    print("\n  ── 4a. HETATM / non-standard residues ──")
    for label, rec in [("ours", ours_rec), ("ref", ref_rec)]:
        hetatm = {k[:3]: v["resname"] for k, v in rec.items()
                  if v["record_type"] == "HETATM"}
        non_std = {k[:3]: v["resname"] for k, v in rec.items()
                   if v["resname"] not in STANDARD_AA}
        uniq_hetatm = {rn: keys for rn, keys in
                       {v: [k for k in hetatm if hetatm[k]==v] for v in set(hetatm.values())}.items()}
        print(f"  {label}: HETATM residues={len(set(hetatm))}  non-std residues={len(set(non_std))}")
        if hetatm:
            summary = {}
            for k, rn in hetatm.items():
                summary.setdefault(rn, []).append(k)
            for rn, keys in sorted(summary.items()):
                print(f"    {rn}: {len(keys)} residue(s) — e.g. {keys[0]}")

    # ── 4b. Alt-loc atoms ───────────────────────────────────────────────────
    print("\n  ── 4b. Alt-loc atoms ──")
    for label, rec in [("ours", ours_rec), ("ref", ref_rec)]:
        alt = [(k, v["altloc"]) for k, v in rec.items() if v["altloc"]]
        print(f"  {label}: {len(alt)} atoms with alt-loc"
              + (f" — e.g. {alt[:3]}" if alt else ""))

    # ── 4c. Low-occupancy atoms ─────────────────────────────────────────────
    print("\n  ── 4c. Low-occupancy atoms (< 1.0) ──")
    for label, rec in [("ours", ours_rec), ("ref", rec_ref := ref_rec)]:
        low = [(k, v["occupancy"]) for k, v in rec.items() if v["occupancy"] < 0.99]
        print(f"  {label}: {len(low)} atoms with occupancy < 1.0"
              + (f" — e.g. {low[:3]}" if low else ""))

    # ── 4d. Residues included in density (CA for GLY, CB for others) ────────
    print("\n  ── 4d. Residues counted in density computation ──")
    ours_density_res, ref_density_res = set(), set()
    for label, rec, out in [("ours", ours_rec, ours_density_res),
                             ("ref",  ref_rec,  ref_density_res)]:
        for key in all_residues(rec):
            ch, rn, ic = key
            atoms = residue_atoms(rec, ch, rn, ic)
            if not atoms: continue
            resname = next(iter(atoms.values()))["resname"]
            # our code includes everything in standardize_aa_mapping
            # ref uses ATOM.only → excludes pure HETATM residues like MSE
            has_density_atom = ("CA" in atoms) if resname == "GLY" else ("CB" in atoms or "CA" in atoms)
            if has_density_atom:
                out.add(key)
        print(f"  {label}: {len(out)} residues with a density atom")

    only_ours_d = sorted(ours_density_res - ref_density_res)
    only_ref_d  = sorted(ref_density_res  - ours_density_res)
    print(f"  Extra in ours (not ref): {len(only_ours_d)}")
    if only_ours_d:
        for key in only_ours_d[:10]:
            atoms = residue_atoms(ours_rec, *key)
            rn_str = next(iter(atoms.values()))["resname"] if atoms else "?"
            rec_type = next(iter(atoms.values()))["record_type"] if atoms else "?"
            print(f"    {key}  resname={rn_str}  record={rec_type}")
    print(f"  Extra in ref  (not ours): {len(only_ref_d)}")
    if only_ref_d:
        for key in only_ref_d[:10]:
            atoms = residue_atoms(ref_rec, *key)
            rn_str = next(iter(atoms.values()))["resname"] if atoms else "?"
            print(f"    {key}  resname={rn_str}")

    # ── 4e. Borderline density pairs (one side 4.5–6.5 Å, other side not) ──
    print("\n  ── 4e. Borderline density pairs (near 4.5 or 6.5 Å boundary) ──")
    print("    (pairs where ours and ref disagree on inclusion in density shell)")

    # Build sequential-position ordered list for seq-sep filter
    ours_keys_ord = list(all_residues(ours_rec))
    ref_keys_ord  = list(all_residues(ref_rec))
    ours_pos = {k: i for i, k in enumerate(ours_keys_ord)}
    ref_pos  = {k: i for i, k in enumerate(ref_keys_ord)}

    # Collect density coords
    ours_dc = {k: c for k in ours_keys_ord if (c:=density_coord(ours_rec, k)) is not None}
    ref_dc  = {k: c for k in ref_keys_ord  if (c:=density_coord(ref_rec,  k)) is not None}

    common_dc = sorted(set(ours_dc) & set(ref_dc))

    # Apply transform to ours coordinates
    def ours_xyz(k):
        c = ours_dc[k]
        if R is not None:
            return xform(c[None], R, t)[0]
        return c

    borderline = []   # (ki, kj, d_ours, d_ref, in_ours, in_ref)
    for a, ki in enumerate(common_dc):
        pi_o = ours_pos.get(ki, a)
        pi_r = ref_pos.get(ki, a)
        xi_o = ours_xyz(ki)
        xi_r = ref_dc[ki]
        for b, kj in enumerate(common_dc):
            if b <= a: continue
            pj_o = ours_pos.get(kj, b)
            pj_r = ref_pos.get(kj, b)
            if abs(pi_o - pj_o) <= MIN_SEP and abs(pi_r - pj_r) <= MIN_SEP:
                continue
            xj_o = ours_xyz(kj)
            xj_r = ref_dc[kj]
            d_o = float(np.linalg.norm(xi_o - xj_o))
            d_r = float(np.linalg.norm(xi_r - xj_r))
            in_o = RMIN <= d_o <= RMAX
            in_r = RMIN <= d_r <= RMAX
            if in_o != in_r:
                borderline.append((ki, kj, d_o, d_r, in_o, in_r))

    print(f"    Pairs where ours and ref disagree: {len(borderline)}")
    if borderline:
        # Sort by how close to boundary the disagreeing one is
        def boundary_margin(entry):
            _, _, d_o, d_r, in_o, in_r = entry
            d = d_o if not in_o else d_r
            return min(abs(d - RMIN), abs(d - RMAX))

        borderline.sort(key=boundary_margin)
        print(f"    {'chain_i':>7} {'res_i':>5} {'chain_j':>7} {'res_j':>5}"
              f"   {'d_ours':>7} {'d_ref':>7} {'in_ours':>7} {'in_ref':>7}")
        for ki, kj, d_o, d_r, in_o, in_r in borderline[:20]:
            print(f"    {ki[0]:>7} {ki[1]:>5}  {kj[0]:>7} {kj[1]:>5}"
                  f"   {d_o:>7.3f} {d_r:>7.3f} {'Y' if in_o else 'N':>7} {'Y' if in_r else 'N':>7}")

    # ── 4f. Per-residue density count comparison ────────────────────────────
    print("\n  ── 4f. Per-residue density count comparison ──")
    print("    (number of neighbors in [4.5, 6.5] Å shell, excl. min_sep=3)")

    def compute_density(dc, pos_map):
        keys = list(dc)
        rho = {}
        for a, ki in enumerate(keys):
            pi = pos_map.get(ki, a)
            xi = dc[ki]
            count = 0.0
            for b, kj in enumerate(keys):
                if a == b: continue
                pj = pos_map.get(kj, b)
                if abs(pi - pj) <= MIN_SEP: continue
                d = float(np.linalg.norm(xi - dc[kj]))
                if RMIN <= d <= RMAX:
                    count += 0.25 * (1 + np.tanh(5*(d-RMIN))) * (1 + np.tanh(5*(RMAX-d)))
            rho[ki] = count
        return rho

    rho_ours = compute_density({k: ours_xyz(k) for k in common_dc}, ours_pos)
    rho_ref  = compute_density(ref_dc,                               ref_pos)

    diff_rho = {k: rho_ours[k] - rho_ref[k] for k in common_dc}
    diffs = np.array(list(diff_rho.values()))
    if len(diffs) == 0:
        print("    n=0  (no common density residues)")
        return
    print(f"    n={len(diffs)}  mean_Δρ={diffs.mean():+.4f}  "
          f"max|Δρ|={np.max(np.abs(diffs)):.4f}  "
          f"n|Δρ|>0.1: {np.sum(np.abs(diffs)>0.1)}")

    order = np.argsort(np.abs(diffs))[::-1]
    keys_list = list(diff_rho)
    print(f"\n    Top density outliers:")
    print(f"    {'chain':>5} {'res':>5} {'resname':>7}  {'ρ_ours':>8} {'ρ_ref':>8} {'Δρ':>8}")
    for i in order[:15]:
        k = keys_list[i]
        ch, rn, ic = k
        atoms = residue_atoms(ours_rec, ch, rn, ic) or residue_atoms(ref_rec, ch, rn, ic)
        rn_str = next(iter(atoms.values()))["resname"] if atoms else "?"
        print(f"    {ch:>5} {rn:>5} {rn_str:>7}  "
              f"{rho_ours[k]:>8.3f} {rho_ref[k]:>8.3f} {diff_rho[k]:>+8.3f}")


# ---------------------------------------------------------------------------
# Section 5 — Frustration pair-set comparison
# ---------------------------------------------------------------------------

def _read_frust_txt(path: Path):
    """
    Parse a frustration txt file (pdb_configurational / pdb_mutational /
    pdb_singleresidue).  Returns a DataFrame with at least Res1, Res2,
    ChainRes1, ChainRes2, FrstIndex, NativeEnergy columns (whatever is present).
    Returns None if file missing or unparseable.
    """
    if not path.exists():
        return None
    try:
        lines = path.read_text().splitlines()
        data  = [l for l in lines if l.strip() and not l.startswith("#")]
        if len(data) < 2:
            return None
        import io, pandas as pd
        df = pd.read_csv(io.StringIO("\n".join(data)), sep=r"\s+", engine="python")
        return df
    except Exception as e:
        print(f"    WARNING: could not parse {path}: {e}")
        return None


def _pair_key(r1, r2, c1, c2):
    """Canonical (sorted) pair key so (i,j) == (j,i)."""
    if (str(c1), int(r1)) <= (str(c2), int(r2)):
        return (int(r1), int(r2), str(c1), str(c2))
    return (int(r2), int(r1), str(c2), str(c1))


def _normalize_chain_map(df):
    """Remap chain IDs to A,B,C,… order so ours and ref match."""
    for col in ("ChainRes1", "ChainRes2"):
        if col not in df.columns:
            return df
    all_chains = sorted(
        set(df["ChainRes1"].astype(str)) | set(df["ChainRes2"].astype(str))
    )
    cmap = {c: chr(ord("A") + i) for i, c in enumerate(all_chains)}
    df = df.copy()
    df["ChainRes1"] = df["ChainRes1"].astype(str).map(cmap)
    df["ChainRes2"] = df["ChainRes2"].astype(str).map(cmap)
    return df


def section_frust_pairs(ours_frust: Path, ref_frust: Path, ours_rec: dict,
                        ref_rec: dict, mode: str):
    print(f"\n══ 5. FRUSTRATION PAIR-SET COMPARISON ({mode}) ══════════════════════")

    df_o = _read_frust_txt(ours_frust)
    df_r = _read_frust_txt(ref_frust)

    if df_o is None:
        print(f"  SKIP: ours file not found: {ours_frust}"); return
    if df_r is None:
        print(f"  SKIP: ref file not found: {ref_frust}"); return

    is_pair = "Res2" in df_o.columns  # False for singleresidue

    if not is_pair:
        # ── Singleresidue: compare per-residue ──
        df_o = _normalize_chain_map(df_o)
        df_r = _normalize_chain_map(df_r)
        key_o = set(zip(df_o["Res1"].astype(int), df_o["ChainRes1"].astype(str)))
        key_r = set(zip(df_r["Res1"].astype(int), df_r["ChainRes1"].astype(str)))
        only_o = sorted(key_o - key_r)
        only_r = sorted(key_r - key_o)
        print(f"  Residues — ours={len(key_o)}  ref={len(key_r)}  "
              f"only-ours={len(only_o)}  only-ref={len(only_r)}")
        if only_r:
            print(f"  In ref but not ours (first 10): {only_r[:10]}")
        if only_o:
            print(f"  In ours but not ref (first 10): {only_o[:10]}")
        return

    # ── Pair mode ────────────────────────────────────────────────────────────
    df_o = _normalize_chain_map(df_o)
    df_r = _normalize_chain_map(df_r)

    keys_o = {_pair_key(r["Res1"], r["Res2"], r["ChainRes1"], r["ChainRes2"])
              for _, r in df_o.iterrows()}
    keys_r = {_pair_key(r["Res1"], r["Res2"], r["ChainRes1"], r["ChainRes2"])
              for _, r in df_r.iterrows()}

    only_r = sorted(keys_r - keys_o)   # in ref but not ours
    only_o = sorted(keys_o - keys_r)   # in ours but not ref
    common  = keys_o & keys_r
    frac    = len(common) / len(keys_r) if keys_r else float("nan")

    print(f"  Pairs — ours={len(keys_o)}  ref={len(keys_r)}  "
          f"common={len(common)}  frac_matched={frac:.4f}")
    print(f"  Only in ref (missing from ours): {len(only_r)}")
    print(f"  Only in ours (extra vs ref):     {len(only_o)}")

    # ── Annotate missing pairs with residue info ─────────────────────────────
    def residue_info(resnum, chain, records):
        """Return (resname, is_hetatm, density_coord_xyz)."""
        # Find residue by chain+resnum (ignore icode for txt-file matching)
        matches = [(k, v) for k, v in records.items()
                   if k[0] == chain and k[1] == resnum and k[3] == "CA"]
        if not matches:
            return "?", "?", None
        k, v = matches[0]
        resname   = v["resname"]
        is_hetatm = v["record_type"] == "HETATM"
        # density atom: CA for GLY, CB otherwise
        atom = "CA" if resname == "GLY" else "CB"
        dk = (k[0], k[1], k[2], atom)
        xyz = records[dk]["xyz"] if dk in records else v["xyz"]
        return resname, is_hetatm, xyz

    def pair_distance(r1, r2, c1, c2, records):
        _, _, xyz1 = residue_info(r1, c1, records)
        _, _, xyz2 = residue_info(r2, c2, records)
        if xyz1 is None or xyz2 is None:
            return float("nan")
        return float(np.linalg.norm(xyz1 - xyz2))

    if only_r:
        print(f"\n  ── Pairs in ref but NOT in ours (first 20) ──")
        print(f"  {'res1':>5} {'ch1':>3} {'res2':>5} {'ch2':>3}"
              f"  {'rn1':>6} {'H1':>2}  {'rn2':>6} {'H2':>2}  {'d_ref_Å':>8}")
        for r1, r2, c1, c2 in only_r[:20]:
            rn1, h1, _ = residue_info(r1, c1, ref_rec)
            rn2, h2, _ = residue_info(r2, c2, ref_rec)
            d = pair_distance(r1, r2, c1, c2, ref_rec)
            print(f"  {r1:>5} {c1:>3} {r2:>5} {c2:>3}"
                  f"  {rn1:>6} {'H' if h1 else '.':>2}  {rn2:>6} {'H' if h2 else '.':>2}  {d:>8.3f}")

        # Summarise: how many missing pairs touch a HETATM residue?
        hetatm_missing = sum(
            1 for r1, r2, c1, c2 in only_r
            if residue_info(r1, c1, ref_rec)[1] is True
            or residue_info(r2, c2, ref_rec)[1] is True
        )
        print(f"\n  Pairs in ref-only touching a HETATM residue: "
              f"{hetatm_missing}/{len(only_r)}")

    if only_o:
        print(f"\n  ── Pairs in ours but NOT in ref (first 10) ──")
        print(f"  {'res1':>5} {'ch1':>3} {'res2':>5} {'ch2':>3}"
              f"  {'rn1':>6} {'H1':>2}  {'rn2':>6} {'H2':>2}  {'d_ours_Å':>9}")
        for r1, r2, c1, c2 in only_o[:10]:
            rn1, h1, _ = residue_info(r1, c1, ours_rec)
            rn2, h2, _ = residue_info(r2, c2, ours_rec)
            d = pair_distance(r1, r2, c1, c2, ours_rec)
            print(f"  {r1:>5} {c1:>3} {r2:>5} {c2:>3}"
                  f"  {rn1:>6} {'H' if h1 else '.':>2}  {rn2:>6} {'H' if h2 else '.':>2}  {d:>9.3f}")

    # ── NativeEnergy comparison for matched pairs ─────────────────────────────
    if "NativeEnergy" in df_o.columns and "NativeEnergy" in df_r.columns:
        print(f"\n  ── NativeEnergy comparison (matched pairs, n={len(common)}) ──")

        # Build key→NativeEnergy maps
        def key_map(df):
            m = {}
            for _, row in df.iterrows():
                k = _pair_key(row["Res1"], row["Res2"], row["ChainRes1"], row["ChainRes2"])
                m[k] = float(row["NativeEnergy"])
            return m

        ne_o = key_map(df_o)
        ne_r = key_map(df_r)
        common_sorted = sorted(common)
        diff_ne = np.array([ne_o[k] - ne_r[k] for k in common_sorted
                            if k in ne_o and k in ne_r])

        if len(diff_ne):
            r_ne = np.corrcoef([ne_o[k] for k in common_sorted if k in ne_o and k in ne_r],
                               [ne_r[k] for k in common_sorted if k in ne_o and k in ne_r])[0, 1]
            print(f"  R(NativeEnergy)={r_ne:.6f}  "
                  f"MAE={np.mean(np.abs(diff_ne)):.5f}  "
                  f"max|Δ|={np.max(np.abs(diff_ne)):.5f}  "
                  f"mean_Δ={diff_ne.mean():+.5f}")

            if np.max(np.abs(diff_ne)) > 1e-4:
                order = np.argsort(np.abs(diff_ne))[::-1]
                print(f"  Top 10 NativeEnergy outlier pairs:")
                print(f"  {'res1':>5} {'ch1':>3} {'res2':>5} {'ch2':>3}"
                      f"  {'NE_ours':>10} {'NE_ref':>10} {'Δ':>10}")
                for idx in order[:10]:
                    k = common_sorted[idx]
                    r1, r2, c1, c2 = k
                    print(f"  {r1:>5} {c1:>3} {r2:>5} {c2:>3}"
                          f"  {ne_o[k]:>10.5f} {ne_r[k]:>10.5f} {diff_ne[idx]:>+10.5f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def find_pair(directory: Path, ref_base: Path | None = None):
    """
    Locate (ours, ref) PDB files.

    ours: <directory>/<stem>.pdb  (non-corrected file in FrustrationData dir)
    ref:  if ref_base given  → <ref_base>/<stem>.done/FrustrationData/<stem>.pdb
          otherwise fallback → <directory>/<stem>.frustratometer_corrected.pdb
    """
    # Ours: any .pdb in directory that isn't frustratometer_corrected
    pdbs = list(directory.glob("*.pdb"))
    ours_pdbs = [p for p in pdbs if "frustratometer_corrected" not in p.name]
    if not ours_pdbs:
        raise FileNotFoundError(f"No pytorch PDB in {directory}")
    ours_path = ours_pdbs[0]
    stem = Path(ours_path.stem).stem

    if ref_base is not None:
        ref_path = ref_base / f"{stem}.done" / "FrustrationData" / f"{stem}.pdb"
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference PDB not found: {ref_path}")
    else:
        ref_pdbs = [p for p in pdbs if "frustratometer_corrected" in p.name]
        if not ref_pdbs:
            raise FileNotFoundError(
                f"No *frustratometer_corrected.pdb in {directory} "
                f"and no --ref-base given"
            )
        ref_path = ref_pdbs[0]

    return ours_path, ref_path


def run_one(ours_path: Path, ref_path: Path, label: str = "",
            frust_mode: str | None = None):
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
    print(f"Ours : {ours_path}")
    print(f"Ref  : {ref_path}")
    try:
        ours_rec = parse_pdb(ours_path)
        ref_rec  = parse_pdb(ref_path)
    except Exception as e:
        print(f"  ERROR reading files: {e}")
        return

    section_bfactors(ours_rec, ref_rec)
    R, t = section_superimpose(ours_rec, ref_rec)
    section_atom_positions(ours_rec, ref_rec, R, t)
    section_density_diagnostics(ours_rec, ref_rec, R, t)

    if frust_mode:
        # Frustration txt files sit alongside the PDB with _{mode} suffix
        ours_frust = ours_path.parent / f"{ours_path.name}_{frust_mode}"
        ref_frust  = ref_path.parent  / f"{ref_path.name}_{frust_mode}"
        section_frust_pairs(ours_frust, ref_frust, ours_rec, ref_rec,
                            mode=frust_mode)

    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", nargs="?", type=Path,
                    help="FrustrationData directory for a single protein")
    ap.add_argument("--ours", type=Path, help="PyTorch PDB file (single protein)")
    ap.add_argument("--ref",  type=Path, help="Reference PDB file (single protein)")
    ap.add_argument("--domains", type=Path,
                    help="domains.txt listing full PDB paths (one per line) for bulk mode")
    ap.add_argument("--results-dir", type=Path,
                    help="Root results directory containing <stem>.done/ subdirs (bulk mode)")
    ap.add_argument("--ref-base", type=Path, default=None,
                    help="Base directory for reference PDBs: "
                         "<ref-base>/<stem>.done/FrustrationData/<stem>.pdb "
                         "(e.g. ../dompdb_v4.4)")
    ap.add_argument("--frust-mode", default=None,
                    choices=["configurational", "mutational", "singleresidue"],
                    help="If given, read <stem>.pdb_<mode> frustration txt files and "
                         "run section 5 (pair-set comparison + NativeEnergy diff)")
    args = ap.parse_args()

    ref_base   = args.ref_base
    frust_mode = args.frust_mode

    if args.domains and args.results_dir:
        # Bulk mode: iterate over all proteins in domains file
        stems = []
        with open(args.domains) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    stems.append(Path(line).stem)  # e.g. "4l1gA00" from full path

        missing = []
        for stem in stems:
            frust_dir = args.results_dir / f"{stem}.done" / "FrustrationData"
            if not frust_dir.exists():
                missing.append(stem)
                continue
            try:
                ours_path, ref_path = find_pair(frust_dir, ref_base=ref_base)
            except FileNotFoundError as e:
                print(f"\n{'='*70}\n  {stem}\n{'='*70}")
                print(f"  SKIP: {e}")
                sys.stdout.flush()
                continue
            run_one(ours_path, ref_path, label=stem, frust_mode=frust_mode)

        if missing:
            print(f"\n{'='*70}")
            print(f"Missing .done dirs ({len(missing)}): {missing}")

    elif args.ours and args.ref:
        run_one(args.ours, args.ref, frust_mode=frust_mode)

    elif args.directory:
        ours_path, ref_path = find_pair(args.directory, ref_base=ref_base)
        run_one(ours_path, ref_path, frust_mode=frust_mode)

    else:
        ap.print_help(); sys.exit(1)


if __name__ == "__main__":
    main()
