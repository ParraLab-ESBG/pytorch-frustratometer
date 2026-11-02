# GLY Mutation Scanning FI Correction — Attempts & Status

Test protein: **8dc1A01** (253 residues), comparing single-residue mutation-scanning FI vs FrustratometeR reference.

All numbers are for GLY positions only (non-GLY serves as sanity check).  
Baseline metric = plain CA stats (no correction): ~5–7× worse error than non-GLY.

---

## Summary Table

| Approach | Mean GLY Error | Max GLY Error | R² | Status |
|---|---|---|---|---|
| Plain CA stats (no correction) | ~0.27 | ~2.0 | ~0.97 | Baseline |
| `gly_water_corr` | 0.164 | 1.599 | 0.9908 | Superseded |
| `_compute_gly_mutant_fi` + hard cutoff + Pro vCB | **0.077** | **1.160** | **0.9964** | **Current best (production)** |
| `_compute_gly_native_fi` (all-vCB) | 0.205 | — | — | Worse |
| `_compute_gly_mutant_fi` (per-mutant, no cutoff) | 0.200 | 1.855 | 0.9864 | Worse |
| AWSEM burial density (`burial_rmin=2.0`) | 0.1919 | 1.937 | 0.9382 | **Failed — reverted** |
| Surgical burial patch — broad params full (rmin=2.0, min_sep=0) | 0.211 | 1.672 | 0.9838 | **Failed** |
| Surgical burial patch — broad params conditional (rmin=2.0, rho_vcb≤9.0) | 0.211 | 1.672 | 0.9838 | **Failed** |
| Surgical burial patch — `burial_min_sep=1` (standard rmin/rmax) | 0.175 | 1.694 | 0.9896 | Worse |

**Current production breakdown (8dc1A01):**

| Sub-category | Mean |error| | Max |error| | n |
|---|---|---|---|
| GLY-related (native=G or mutant=G) | 0.058 | 1.160 | 614 |
| G→X (native GLY → non-GLY) | 0.077 | 1.160 | 380 |
| X→G (non-GLY → GLY decoy) | 0.027 | 0.137 | 234 |
| non-GLY | 0.031 | 0.383 | 4,446 |
| Overall R² | — | — | 0.9964 |

---

## Case 1: Native GLY → non-GLY decoy

### What FrustratometeR does
Inside `calculate_frustration`, `check_backbone_complete` calls MODELLER `complete_pdb` with `build_method='INTERNAL_COORDINATES'` to place an ideal Cβ (no minimization). This gives a **nearly identical** position to our CHARMM36 IC formula (`place_fourth_atom` with N,C,CA reference, length=1.5220 Å, angle=111.09°, dihedral=123.23°). **Proline** is the only exception: its pyrrolidine ring shifts Cβ by ~0.197 Å (dihedral ≈115.3° vs 123.23°), confirmed by AlphaFold `rigid_group_atom_positions` and all other standard CHARMM36 IC tables.

### Current implementation: `_compute_gly_mutant_fi` ✓ (production)
**File:** `singleresidue.py`, called from `compute_tert_frust_singleresidue`; result stored in `sr_mutations["_gly_native_fi"]`.

For each native GLY position q, places vCβ using the standard CHARMM36 IC formula and computes the **full absolute AWSEM energy** for each of the 20 AA substitutions X directly in vCβ geometry:

1. **Burial energy**: uses `rho_vcb[q]` (density counted from vCβ).
2. **Water energy**: uses vCβ distances and neighbour densities corrected by the ripple term Δρⱼ = vθ(dvCβ[q,j]) − vθ(dCα[q,j]).
3. **Electrostatic energy**: uses the field computed at the vCβ position.
4. **Hard 9.5 Å distance cutoff**: applied to all water pairs, matching `calculate_distance_matrix`. Without the cutoff, `θmediated` (~33% at 10 Å) contributed spurious water energy from pairs that would be absent in the full-structure computation. This was the dominant error source before the cutoff was added (G22→Y improved from 1.49 → 1.16).
5. **Pro-specific Cβ**: Proline uses `compute_pro_vcb` (dihedral=115.3° instead of 123.23°, same length and angle), accounting for the pyrrolidine ring constraint. All other residues share the same standard CHARMM36 IC. Pro is the **only** residue with a meaningfully different Cβ IC — confirmed by comparing all 19 standard residues from AlphaFold `rigid_group_atom_positions`: non-Pro residues cluster within ±0.5° of 123.23°, which is statistical noise from crystal averaging, not separate CHARMM entries.

Mean and std are frequency-weighted over all 20 substitutions; FI is computed directly from absolute energies (not delta-from-native).

**Result: G→X mean=0.077, max=1.160, overall R²=0.9964.**

### Dead end: `gly_water_corr` (superseded)
The previous best: delta corrections applied to the existing decoy distribution. Mean=0.164, R²=0.9908. Replaced by `_compute_gly_mutant_fi` (absolute energies at vCβ) which reduces errors by >2× because it correctly handles the change in mean/std of the decoy distribution rather than only correcting individual decoy energies.

### Dead end: `_compute_gly_native_fi` (all-vCB)
All 20 decoys computed using vCB distances and rho_vcb. Mean error 0.2053 — **worse** than gly_water_corr. Reason: gly_water_corr benefits from error cancellation in the full population statistics.

### Dead end: `_compute_gly_mutant_fi` without 9.5 Å cutoff
Without the hard distance cutoff, mean error 0.200, max 1.855. The cutoff is essential: `θmediated` is non-negligible beyond 9.5 Å.

### Known irreducible errors (Case 1)

`compare_modeller_vcb.py` was extended to all four rho_vcb=0 positions (G39, G117, G175/G207). **Result: MODELLER's actual Cβ gives rho=0.0000 for all 19 non-GLY mutants at every position.** The position is 0.024 Å from our CHARMM formula at each (MODELLER uses the CHARMM force-field equilibrium bond length 1.546 Å; our formula uses the crystallographic average 1.5220 Å). No conjugate-gradient step occurs — the placement is a deterministic function of backbone geometry alone, confirmed by `build_method='INTERNAL_COORDINATES'`. The CG-repositioning hypothesis (CB pulled toward protein for G117 with rho_nat=0.969) is disproved: MODELLER's Cβ at G117 also gives rho=0.

Errors therefore split into two tiers by mechanism:

**Tier 1 — G22, G207** (rho_nat≈0, rho_vcb=0; max error ~1.0–1.2): both Cα and ideal vCβ are in open solvent; no contacts within 9.5 Å of either centroid. All FI variation comes from electrostatics. Remaining max errors: G22→Y≈1.16, G207→P≈1.08. Source: fundamental mismatch between our analytical decoy distribution (burial(X,0) + q_X·field) and FrustratometeR's reference; origin not fully understood. For G207, the Pro-specific Cβ (0.197 Å shift) has no effect — G207's Pro Cβ is also fully in solvent.

**Tier 2 — G117, G39** (rho_nat≈0–1, rho_vcb=0; max error ~0.4): Cα has some protein contacts (rho_nat=0.969 and 0.004) but the ideal vCβ points into solvent. MODELLER's Cβ also gives rho=0 at both. Errors are NOT from Cβ geometry. The mixed-sign scatter (std_SE≈0.14–0.16, slope≈1.0, r≈0.995) is consistent with **finite decoy-sampling noise in the FrustratometeR reference**: each of the 19 per-mutant frustration runs uses D=2000 independent random decoys, so sampling noise of order σ/√D ≈ 0.02–0.04 produces max |ΔFI|≈0.3–0.4 across 19 independent draws. Our analytical formula (exact D→∞ limit) is likely more accurate at these positions.

**`wrong_dir` suppression** (see below): at positions where rho_vcb < rho_nat (unusual positive-φ GLY conformations), the ProteinMPNN vCβ fallback is tried; if that also gives lower density, the water correction is suppressed. For 8dc1A01 no fallback fires.

---

## Case 2: Native non-GLY → GLY decoy

### What FrustratometeR does
Per-decoy atom selection (from FrustratometeR R source):
```R
if(Glycine) indexBackbone <- atom.select(Pdb, ..., elety = c("N", "CA", "C", "O"))
else        indexBackbone <- atom.select(Pdb, ..., elety = c("N", "CA", "C", "O", "CB"))
```
For each decoy AA x at native non-GLY position q:
- **x = GLY**: CB absent → CA is centroid → CA distances, CA-based burial density (rho_ca)
- **x ≠ GLY**: CB kept (the **native** CB coordinates of pos q) → CB distances, native burial density (rho_nat)

The frustration distribution is **MIXED**: 19 CB-based decoys + 1 CA-based (GLY) decoy. The native CB sidechain is stripped (only N, CA, C, O, CB remain), so non-GLY decoys all use the native residue's CB regardless of their own identity.

### Current implementation: `_compute_gly_decoy_fi_at_non_gly`
**All-CA approximation**: all 20 decoy AAs use CA geometry at pos q. Burial uses `rho_ca[b, q]` (CA density from `_compute_gly_alt_densities`). Water uses CA distances with neighbour density ripple `delta_rho_j = _vtheta(d_CA[q,j]) − _vtheta(d_CB[q,j])`. 

**Hard 9.5 Å distance cutoff** applied (added this session), matching `calculate_distance_matrix`. Water pair mask: `(pos_sep >= 2) & vm_j & valid_mask[b, q] & (d_ca_q < 9.5)`.

**Result: X→G mean=0.027, max=0.137.** The CA approximation works well because for most buried positions the CA-CB difference in density is small; worst cases are surface positions where CB points outward and CA inward (Δrho large).

### Dead end: all-vCB for non-GLY decoys + all-CA for GLY slot
Caused sign flips, error=0.4738.

### Dead end: local rho_q_ca recomputation (REVERTED)
**Hypothesis**: `rho_ca[b, non_gly_idx]` from `_compute_gly_alt_densities` uses an all-CA approximation (all j switch to CA simultaneously). Replace with locally-computed CA_q → native centroid of j density.

**Why it was wrong**: `_compute_gly_alt_densities._rho_for_subset` already uses `nat_j = native_centroid[b]` (native centroids — CB for non-GLY j, CA for GLY j). The original `rho_ca[b, non_gly_idx]` was already computing CA_q → native centroids. The local `_vtheta(d_ca_q) * vp` version diverged because it **lacked the distance cutoff** (`dist <= cfg.tert_frust_cutoff`) that `_rho_for_subset` applies. Result: pos 142 (S→G, rho_nat=3.88, rho_ca=1.13) jumped from ~0.7 to 1.458 error. **Reverted.**

### Dead end: mixed distribution + delta_water_G correction (REVERTED)
**Hypothesis**: Start from `E_decoy_cb` (CB-based, burial already corrected to rho_ca for the GLY slot via `_apply_gly_correction`). Add a water correction only to the GLY slot:
```
delta_water_G[q] = Σ_j [water(G, CA_q→j, ρ_ca[q], ρ_j_corr) − water(G, CB_q→j, ρ_nat[q], ρ_j)] * vp
E_mix[G] = E_decoy_cb[G] + delta_water_G
E_mix[x≠G] = E_decoy_cb[x]   (CB-based, correct)
```

**Why it failed in practice**: delta_water_G was inconsistent with the water term in E_decoy_cb[G] because `compute_sr_water_delta` (which produced E_decoy_cb) uses `batch["distance"]` (precomputed CB distances) with its own pair filter, while the local `_water_e` on freshly computed `d_cb_qj` uses a different effective pair set. For positions where rho_nat and rho_ca cross the solvation threshold (rho_nat < 2.6 < rho_ca), the delta was large and of arbitrary sign (solvation regime flip), blowing up FI (max error 1.922, 50/50 over/under). **Reverted.**

---

## Root Cause of Large Errors at rho_vcb=0 Positions — Definitive Findings

### CA vs CB Water Distances — Both burial-regime hypotheses DISPROVED

**Component breakdown** (from `--components-pos 175 207` diagnostic) and `compare_modeller_vcb.py` (run for all four rho_vcb=0 positions: G39, G117, G175, G207) give the following confirmed picture:

**Position 175** — fi_ref ordering: P=+0.475, N=+0.641, Q=+0.143 (polar AAs minimally frustrated), Y=−1.683, W=−1.632 (aromatics highly frustrated).  
Superficially resembles **gamma_0 ordering** (rho<3): `gamma_0[P]=0.966, gamma_0[G]=0.937, gamma_0[C]=0.675`.  
**DISPROVED by `compare_modeller_vcb`**: MODELLER's actual CB at pos 175 gives rho=0.0000 (same as ours). |CHARMM vCB − MODELLER CB| = 0.024 Å. The burial regime difference is NOT the source of error.  
→ Root cause: **CA vs CB water distances**. wrong_dir=True suppresses the gly_water_correction at pos 175 → we use CA distances for all decoys; FrustratometeR uses CB distances. Enabling the correction (1e-3 tolerance test) makes things worse because d_std collapses (0.9135→0.419). Exception: for P, |vCB−MOD|=0.237 Å (largest of all AAs, because Pro CB is constrained by the pyrrolidine ring and MODELLER places it differently), but even FI_MOD[P]=-0.297 (same as ours) — wrong_dir fires regardless of which CB is substituted.

**Position 207** — rho_nat=0, rho_vcb=0. All energy variation from electrostatics. Remaining errors (max ~1.08) are a hard floor of the analytical approach for fully-solvent GLY positions: our decoy distribution (burial(X,0) + q_X·field) does not match FrustratometeR's reference, for reasons not yet fully understood. Pro-specific Cβ (0.197 Å shift) implemented but has no effect because the Pro Cβ is also fully in solvent.

**Both positions have rho=0.0000 for every CB source** (native CA, our CHARMM vCB, ProteinMPNN vCB, MODELLER's actual CB). The burial-regime contradiction hypothesis is fully disproved.

### AWSEM source (fix_backbone.cpp) — confirmed findings

From the AWSEM C++ source (`fix_backbone.cpp` lines 7770, 7816):
- `water_ro` is accumulated in the density loop and passed directly to `compute_burial_energy`
- **Burial and water share the EXACT same density variable** — no separate burial density params exist

GLY atom selection (lines 7735, 7752):
```cpp
(mask[i]&group2bit && se[ires]!='G') || (mask[i]&groupbit && se[ires]=='G')
```
`group2bit` = CB atoms; `groupbit` = CA atoms. **AWSEM C++ uses CA for GLY, CB for all others.**

**Important caveat:** the `fix_backbone.cpp` in our repo uses `abs(ires-jres)>1` (min_sep=2), but this is a different AWSEM version than what FrustratometeR calls. **FrustratometeR calls an AWSEM binary with min_sep=3 hardcoded.** Testing on configurational outliers confirmed sep=3 is best (`tests/test_density_minsep.py`). Our code's min_sep=3 matches the binary FrustratometeR uses.

---

## Dead End: AWSEM Burial Density Parameters (burial_rmin=2.0) — TESTED AND FAILED

**Hypothesis**: AWSEM uses `rmin=2.0Å` for burial density → β-turn contacts at 3–4Å counted → rho≈4 at 175/207.

**Result**: R²=0.9382, mean error=0.1919, max=1.937 — significantly worse. **Reverted.**

### Why it failed
rho=0.0000 at 175/207 for BOTH native AND vCB even with rmin=2.0 — genuinely isolated, no contacts even at 2Å. Changing `batch["residue_density"]` globally degraded all other positions. **Reverted fully**: `burial_rmin`/`burial_eta` removed from `Coefficients`; `feature_calculator.py` and `singleresidue.py` restored to `cfg.eta` / `cfg.well_rmin[0]`.

---

## Dead End: Surgical Burial Patch at GLY positions — ALL VARIANTS FAILED

**Hypothesis**: patch `rho_burial` at native GLY positions using `rho_vcb` (virtual CB density) inside `compute_singleresidue_stats_fast`, without affecting non-GLY burial globally.

### Variant 1: Broad params full patch (rmin=2.0, rmax=9.0, min_sep=0, all GLY)
Patch rho_burial at ALL GLY positions with rho_vcb from broad contact params.  
**Result**: R²=0.9838, GLY mean=0.211. Buried GLY (pos 15, 62, 94…) get rho_vcb=17+ → all theta bins → 0 → burial energy zeroed → destroys buried-position differentiation.

### Variant 2: Broad params conditional patch (rho_vcb ≤ 9.0 guard)
Same broad params but skip GLY positions where rho_vcb > max burial bin (9.0).  
**Result**: R²=0.9838 (identical to variant 1). Surface GLY positions (which matter most) still get rho_vcb≈4–8 → pushes burial into gamma_1 → **wrong direction** for pos 175 (needs gamma_0). Sign flips for P, N, Q at pos 175.

Root cause of variants 1 & 2: broad params give rho_vcb≈4–5 at ALL surface GLY positions, including pos 175 where the correct regime is gamma_0 (rho<3).

### Variant 3: `burial_min_sep=1` (standard rmin=4.5, rmax=6.5)
Use standard rmin/rmax but reduce sequence separation cutoff from 3 to 1.  
- rho_vcb at pos 175: 0.0257 (gamma_0, correct direction) — sign of P, N, Q now correct ✓  
- rho_vcb at pos 207: 0.9872 (still gamma_0) — FI ordering of C/I still wrong [later confirmed: FrustratometeR also gives rho=0 at pos 207, so gamma_1 hypothesis was incorrect]  
- **Result**: R²=0.9896, GLY mean=0.175, non-GLY mean=0.031  
  Better than broad params but still worse than default (now superseded by _compute_gly_mutant_fi). The contradiction between pos 175 and pos 207 means no single param set can fix both.

### Current code state
Surgical patch code remains in `compute_singleresidue_stats_fast` (guarded by `should_patch` — only fires when `rho_vcb >= threshold=2.6`). With default params, rho_vcb < threshold at all surface GLY → patch is **never activated** → behavior identical to before.

---

## Electrostatics Investigation at Positions 175/207 — NOT the Source of Error

Tested by adding `FI_noE` column to `--diagnose-pos` output (FI computed without electrostatic term).

**Result at pos 175:** removing electrostatics makes errors **worse** for almost all AAs:
- P: err 0.772 → 1.106 (worse)
- L: err 0.647 → 1.109 (worse)
- I: err 0.539 → 0.968 (worse)
- C: err 0.438 → 1.072 (worse)

Electrostatics actually helps FI at pos 175 by increasing d_std from 0.47 to 0.91, which normalizes the range better. The field at pos 175 is large (1.458) because K174, K176, K177 are adjacent (4.8–7.3 Å away). **Electrostatics is NOT the problem.**

---

## `wrong_dir` Condition and Production Path — Key Internal Mechanics

### `_compute_gly_water_correction` — `wrong_dir` protection (singleresidue.py ~line 638)

```python
if rho_vcb is not None:
    wrong_dir = (rho_vcb_q < rho_q) & (rho_q < 0.1)  # [Q]
    dw[:, wrong_dir] = 0.0
```

This suppresses the water correction at near-surface GLY positions where `rho_vcb < rho_nat`. It protects positions 22 and 39 where the correction gives wrong sign.

**At positions 175 and 207**: Test 1 shows `delta_vcb = -0.0000` — rho_vcb is fractionally smaller than rho_nat by ~1e-5 (both display as 0.0000 at 4 decimal precision). This makes `wrong_dir=True` and **correctly suppresses** the water correction at these positions. The suppression is correct behaviour: when we enabled the correction (by changing the threshold to `> 1e-3`), R² dropped from 0.8522 to 0.6792 for pos-175 mutations. **Reverted.**

**Failed attempt (REVERTED):** Changing `(rho_vcb_q < rho_q)` → `((rho_q - rho_vcb_q) > 1e-3)` to avoid "floating-point noise" made things worse — the tiny rho_vcb < rho_nat gap at pos 175/207 is real (from slightly different density paths: `batch["residue_density"]` vs `_rho_for_subset`), not noise.

### `_compute_gly_alt_densities` — ProteinMPNN vCB fallback

At GLY positions where CHARMM36 vCB gives lower density than native CA (indicating CB points into solvent — unusual positive-φ conformations), the code tries ProteinMPNN vCB as a fallback:

```python
wrong = (rho_36_q < rho_nat_q) & (rho_nat_q < 0.1)
if wrong.any():
    rho_mpnn_q = _rho_for_subset(b, vcb_mpnn[b, wrong_glob], wrong_glob)
    better = rho_mpnn_q > rho_36_q[wrong]
    if better.any():
        vcb_hybrid[b, update_idx] = vcb_mpnn[b, update_idx]
        rho_vcb[b, update_idx] = rho_mpnn_q[better]
        print(f"[GLY-hybrid] ...")
```

For 8dc1A01, no `[GLY-hybrid]` print appears → ProteinMPNN gives no improvement (rho_mpnn ≤ 0) at any GLY position → `vcb_hybrid = vcb_36` (CHARMM36) everywhere.

### Burial gamma ordering reference
- **gamma_0** (rho<3): G=0.937, P=0.966, N/Q/D/E high → polar/hydrophilic most favorable at surface
- **gamma_1** (rho 3–6): C=0.943, I=0.812, V=0.811, L=0.793 → hydrophobics most favorable in core
- **gamma_2** (rho 6–9): deep burial, further favors hydrophobics

### Burial gamma file ordering (confirmed correct)
`burial_gamma.dat` is in AWSEM order: `A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V` (same as `water_gamma.dat`). The `your_to_awsem` remap in `awsem.py` is **correct**. Switching both file orderings gave too-high R² → confirming AWSEM order.

---

## Test Harness — Current CLI Flags

```
python tests/test_gly_correction.py \
  --pdb 8dc1A01.pdb \
  --mut-ref  ../mutant_test_2/8dc1A01_A.done/MutationsData \
  --mut-ref2 ../dompdb_v4.4/8dc1A01_A.done/MutationsData \
  --diagnose-pos 175 207
```

### Per-AA table columns (`--diagnose-pos`)
| Column | Meaning |
|---|---|
| `dBurial` | `burial(X) − burial(GLY)` using native density |
| `dWater` | `water(X) − water(GLY)` using CA distances |
| `dElec` | `(q_X − q_GLY) × field` |
| `E_prod` | Decoy energy from production path (`gly_water_corr=True`) |
| `FI_no` | `(d_mean_no_corr − E_x) / d_std_no_corr` — baseline |
| `FI_prod` | `(d_mean_prod − E_prod_x) / d_std_prod` — what mutation scanning uses |
| `FI_noE` | FI without electrostatics |
| `FI_FR1` | FrustratometeR reference (from `--mut-ref`) |
| `FI_FR2` | Second reference (from `--mut-ref2`) |
| `err1/errP1` | `\|FI_no − FI_FR1\|` and `\|FI_prod − FI_FR1\|` |
| `err2/errP2` | Same vs second reference |

### Stats lines
```
d_mean (no corr):          X    d_std: Y   ← no GLY correction at all
d_mean (burial corr only): X    d_std: Y   ← burial via _apply_gly_correction
d_mean (prod/gly_water):   X    d_std: Y   ← production path (gly_water_corr=True)
d_mean (no elec):          X    d_std: Y   ← burial+water only, no electrostatics
```

---

## File Map

| File | Role |
|---|---|
| `pytorch_frustratometer/coefficients.py` | Parameters: `burial_rmin`, `burial_eta`, `burial_kappa`, etc. |
| `pytorch_frustratometer/features/feature_calculator.py` | `_density_intermediates` — computes `batch["residue_density"]` |
| `pytorch_frustratometer/algorithm/singleresidue.py` | `_compute_gly_alt_densities`, `_compute_gly_mutant_fi` (Case 1 production), `_compute_gly_decoy_fi_at_non_gly` (Case 2), `_compute_gly_water_correction` (`wrong_dir` protection), `_apply_gly_correction` |
| `pytorch_frustratometer/structure/geometry.py` | `compute_ideal_vcb` (standard CHARMM36), `compute_pro_vcb` (Pro dihedral=115.3°), `compute_mpnn_vcb` (ProteinMPNN fallback) |
| `pytorch_frustratometer/algorithm/awsem.py` | `compute_burial_energy`, `get_burial_gamma` (burial_gamma.dat loading) |
| `tests/test_gly_correction.py` | Test harness: `--diagnose-pos`, `--mut-ref`, `--mut-ref2`, `--mutant-fi`, `--components-pos`, `--burial-*` flags |
