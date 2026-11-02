"""Single-residue frustration — all SR logic in one class."""
from __future__ import annotations
import dataclasses
from pathlib import Path

import click
import numpy as np
import torch

from ..utils.utils import frustration_loss as _frustration_loss

from ..utils.utils import index_to_pseudo_logits, compute_frustration_metrics
from ..structure.geometry import get_CB_coords, compute_ideal_vcb, compute_pro_vcb, compute_mpnn_vcb
from ..coefficients import Coefficients
from .decoy_statistics import decoy_aa_frequencies, decoy_aa_weights, decoy_statistics


# ---------------------------------------------------------------------------
# Helpers for mutation comparison
# ---------------------------------------------------------------------------

def _pearson_r2(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan"), 0
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    return float(r ** 2), int(ok.sum())


# ---------------------------------------------------------------------------
# Free function (was in awsem.py)
# ---------------------------------------------------------------------------

def sample_decoy_aa_from_sequence(
    aa_type: torch.Tensor,   # [B, L]
    n_decoys: int,
    seq_mask: torch.Tensor,  # [B, L]
    use_c_rand: bool = False,
    exclude_native_aa: bool = False,  # deprecated: use decoy_distribution
    decoy_distribution: str = "sequence",
) -> torch.Tensor:           # [n_decoys, B, L]
    """
    Sample n_decoys AA sequences per position from one of four distributions:
      "sequence"                 — uniform draw from the protein's own sequence composition
      "sequence_exclude_native"  — same, but reject a'=native at each position
      "pseudocount"              — sequence composition + 1-count pseudocount over all 20 AAs
      "pseudocount_exclude_native" — pseudocount, then reject a'=native
    The legacy exclude_native_aa=True is equivalent to "sequence_exclude_native".
    """
    # Back-compat: honour old bool if new string is still the default
    if exclude_native_aa and decoy_distribution == "sequence":
        decoy_distribution = "sequence_exclude_native"

    exclude_native = decoy_distribution.endswith("_exclude_native")
    use_pseudocount = decoy_distribution.startswith("pseudocount")

    B, L = aa_type.shape
    device = aa_type.device

    valid_mask = seq_mask & (aa_type < 20)
    seq_lens = valid_mask.sum(dim=1)
    valid_indices = [torch.where(valid_mask[b])[0] for b in range(B)]

    if use_pseudocount:
        aa_probs = decoy_aa_frequencies(aa_type, seq_mask, "pseudocount", torch.float32)  # [B, 20]
        # Sample n_decoys * L AA indices per protein, then reshape.
        flat = torch.stack([
            torch.multinomial(aa_probs[b], n_decoys * L, replacement=True)
            for b in range(B)
        ], dim=0)  # [B, n_decoys*L]
        samples = flat.view(B, n_decoys, L).permute(1, 0, 2)  # [n_decoys, B, L]

        if exclude_native:
            native_exp = aa_type.unsqueeze(0).expand_as(samples)
            is_same = samples == native_exp
            for _ in range(20):
                if not is_same.any():
                    break
                new_flat = torch.stack([
                    torch.multinomial(aa_probs[b], n_decoys * L, replacement=True)
                    for b in range(B)
                ], dim=0)
                new_samples = new_flat.view(B, n_decoys, L).permute(1, 0, 2)
                samples = torch.where(is_same, new_samples, samples)
                is_same = (samples == native_exp) & is_same
        return samples

    # ── sequence distribution ────────────────────────────────────────────────
    max_n = int(seq_lens.max().item())
    vi_table = (~valid_mask.bool()).long().argsort(dim=1, stable=True)[:, :max_n]

    if use_c_rand:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.srand(1)
        rand_pos = torch.zeros(n_decoys, B, L, dtype=torch.long)
        for pos in range(L):
            for d in range(n_decoys):
                for b in range(B):
                    n = len(valid_indices[b])
                    rand_pos[d, b, pos] = valid_indices[b][libc.rand() % n]
        rand_pos = rand_pos.to(device)
    else:
        n_valid_t = seq_lens.view(1, B, 1)
        u = torch.rand(n_decoys, B, L, device=device)
        slot = (u * n_valid_t).long().clamp(max=n_valid_t - 1)
        slot_t   = slot.permute(1, 0, 2).reshape(B, -1)
        pos_t    = vi_table.gather(1, slot_t)
        rand_pos = pos_t.view(B, n_decoys, L).permute(1, 0, 2)

    samples = aa_type.unsqueeze(0).expand(n_decoys, -1, -1).gather(dim=2, index=rand_pos)

    if exclude_native:
        native_exp = aa_type.unsqueeze(0).expand_as(samples)
        is_same = samples == native_exp
        for _ in range(20):
            if not is_same.any():
                break
            new_u = torch.rand(n_decoys, B, L, device=device)
            new_slot    = (new_u * n_valid_t).long().clamp(max=n_valid_t - 1)
            new_slot_t  = new_slot.permute(1, 0, 2).reshape(B, -1)
            new_pos_t   = vi_table.gather(1, new_slot_t)
            new_rand_pos = new_pos_t.view(B, n_decoys, L).permute(1, 0, 2)
            new_samples  = aa_type.unsqueeze(0).expand(n_decoys, -1, -1).gather(dim=2, index=new_rand_pos)
            samples  = torch.where(is_same, new_samples, samples)
            is_same  = (samples == native_exp) & is_same

    return samples


# ---------------------------------------------------------------------------
# Single-residue Mixin — all SR computation methods
# ---------------------------------------------------------------------------

class SingleResidueMixin:
    """
    Mixin providing all single-residue frustration computation methods.
    Injected into AWSEM via __bases__ reassignment in awsem.py to avoid
    circular imports (mode files → base.py → awsem.py → mode files).
    """

    def compute_tert_frust_singleresidue(self, batch):
        native_energy = self.compute_singleresidue_native_ixn(
            batch,
            batch["aa_type"],
        )

        results = self.compute_singleresidue_decoy_ixns(batch, native_energy=native_energy)

        frustration_index = self.compute_frustration_index(native_energy, results["stats"])

        sr_metrics = None
        sr_losses  = None
        if "single_frust_true" in batch:
            true_fi  = batch["single_frust_true"].to(frustration_index.device)
            mask     = batch.get("single_frust_mask", batch["seq_mask"]).bool().to(frustration_index.device)
            sr_metrics = compute_frustration_metrics(
                frustration_index, true_fi, mask,
                threshold=self.coefficients.frustration_thresholds[0],
            )
            pseudo_logits = index_to_pseudo_logits(
                frustration_index, self.coefficients.frustration_thresholds[0]
            )
            idx_loss, cls_loss = _frustration_loss(
                mode=0,
                pred_frustration_index=frustration_index.unsqueeze(-1),
                pred_frustration_class=pseudo_logits,
                true_frustration_index=true_fi,
                seq_mask=mask,
                stage="test",
            )
            sr_losses = (idx_loss.item(), cls_loss.item())

        sr_mutations = None
        if self.run_dmfs and "mutation_energies" in results:
            sr_mutations = {
                "energies":         results["mutation_energies"],
                "aas":              results["mutation_aas"],
                "stats":            results.get("_mutation_stats"),
                "_gly_mutation_fi": results.get("_gly_mutation_fi"),   # Case 2 [B, L]
                "_gly_native_fi":   results.get("_gly_native_fi"),     # Case 1 [B, L, 20]
            }

        return frustration_index, native_energy, results["stats"], sr_metrics, sr_losses, sr_mutations

    def compute_singleresidue_decoy_ixns(
        self,
        batch,
        return_decoy_energies: bool = False,
        return_stats: bool = True,
        native_energy=False,
    ):
        """
        Fully vectorized single-residue decoy computation.

        Fast path (default): exhaustive 20-AA delta formulation — exact at D→∞
        with sequence-composition weights, no large allocation.

        Slow path (use_c_rand=True): n_decoys (tert_frust_ndecoys) random samples for C++ stream match.
        """
        device = batch["aa_type"].device
        B, L = batch["aa_type"].shape
        D = self.n_decoys

        def _run_decoy_pass(decoy_samples):
            _D = decoy_samples.shape[0]
            decoys = batch["aa_type"].unsqueeze(0).unsqueeze(0).expand(L, _D, B, L).clone()
            pos_indices = torch.arange(L, device=device)
            decoys[pos_indices, :, :, pos_indices] = decoy_samples.permute(2, 0, 1)
            energies_per_pos = []
            for pos_i in range(L):
                e = self.compute_singleresidue_native_ixn(batch, decoys[pos_i])
                energies_per_pos.append(e[:, :, pos_i])
            return torch.stack(energies_per_pos, dim=0).permute(1, 2, 0)

        if not self.use_c_rand and native_energy is not False:
            stats, E_decoy_all, _ = self.compute_singleresidue_stats_fast(
                batch, native_energy
            )
            results = {"stats": stats}
            if native_energy is not None:
                results["frustration"] = self.compute_frustration_index(native_energy, stats)
            if self.run_dmfs:
                # GLY geometry corrections only apply in mutation scanning: FrustratometeR
                # creates a separate mutant structure for each G→X (MODELLER adds CB) and
                # X→G (CB removed). Regular single-residue mode uses native geometry for
                # all decoys (threading) so no correction is needed there.
                rho_vcb, rho_ca, vcb_hybrid = self._compute_gly_alt_densities(batch)
                # Case 1 (native GLY → X): the mutation loop just renames the residue,
                # but check_backbone_complete inside calculate_frustration detects the
                # missing CB and adds it with ideal geometry (vCB-like). gly_water_corr
                # corrects water distances from CA to vCB; _apply_gly_correction fixes
                # burial density from rho_nat to rho_vcb at native GLY positions.
                # Case 1 (native GLY → any decoy): self-consistent vCB geometry via
                # _compute_gly_native_fi (overrides gly_water_corr stats for Case 1).
                # Case 2 (native X → GLY): use native-CB FI — equivalent to running
                # each GLY mutant as a separate batch item, since the GLY mutant applies
                # Case 1 (all-vCB at the newly-GLY position), and vCB ≈ native CB.
                stats_native, E_decoy_native, _ = self.compute_singleresidue_stats_fast(
                    batch, native_energy, rho_vcb=rho_vcb, rho_ca=rho_ca,
                    gly_water_corr=True, vcb=vcb_hybrid,
                )
                # Case 2 (X→G): all-CA distribution with CA-based electrostatic field.
                # Matches FrustratometeR's behaviour of running each GLY mutant as a
                # separate structure (native at pos q = GLY → all 20 decoys use CA geometry).
                fi_gly_case2 = self._compute_gly_decoy_fi_at_non_gly(
                    batch, native_energy, rho_vcb, rho_ca
                )
                # Case 1 (G→X): per-GLY vCB mini-batch. Matches FrustratometeR's behaviour of
                # running each non-GLY mutant as a separate structure where MODELLER adds a CB.
                # Uses position-based sep >= 2 for water pairs (matching calculate_distance_matrix).
                fi_gly_native = self._compute_gly_mutant_fi(batch, native_energy, rho_vcb, rho_ca)
                results["mutation_energies"] = E_decoy_native.permute(1, 2, 0)
                results["mutation_aas"]      = torch.arange(20, device=device)
                results["_mutation_stats"]   = stats_native
                results["_gly_mutation_fi"]  = fi_gly_case2   # [B, L] — Case 2
                results["_gly_native_fi"]    = fi_gly_native  # [B, L, 20] — Case 1
            return results

        random_samples = sample_decoy_aa_from_sequence(
            batch["aa_type"], D, batch["seq_mask"], use_c_rand=self.use_c_rand,
            exclude_native_aa=getattr(self.coefficients, 'tert_frust_exclude_native', False),
            decoy_distribution=getattr(self.coefficients, 'tert_frust_decoy_distribution', 'sequence'),
        )

        total_energy = _run_decoy_pass(random_samples)

        if return_decoy_energies and not return_stats:
            return total_energy

        results = {"energies": total_energy}

        if return_stats:
            finite_mask = torch.isfinite(total_energy)
            safe_energy = total_energy.masked_fill(~finite_mask, 0.0)
            counts = finite_mask.sum(dim=0).clamp(min=1)

            mean = safe_energy.sum(dim=0) / counts
            var  = ((safe_energy - mean.unsqueeze(0)) ** 2 * finite_mask).sum(dim=0) / counts
            std  = torch.sqrt(var + 1e-8)

            stats = torch.stack([mean, std], dim=-1)
            results["stats"] = stats
            if native_energy is not None:
                results["frustration"] = self.compute_frustration_index(native_energy, stats)

        if self.run_dmfs:
            exhaustive_samples = (
                torch.arange(20, device=device)
                .view(20, 1, 1)
                .expand(20, B, L)
                .clone()
            )
            mut_energies = _run_decoy_pass(exhaustive_samples)
            # Use native density (no GLY correction) so mutation energies are consistent
            # with the random-sample decoy stats above. FrustratometeR threading also uses
            # fixed structural density for all decoys.
            print("[GLY-FIX] mutation scanning using native density (no correction)")
            results["mutation_energies"] = mut_energies
            results["mutation_aas"] = torch.arange(20, device=device)

        return results

    def compute_singleresidue_native_ixn(self, batch, res_type):
        """
        Compute per-residue energy for a batch of proteins.

        Args:
            batch:    dict with keys aa_type, distance, residue_density, edge_index
            res_type: [B, L] or [n_decoys, B, L]

        Returns:
            total_energy: same leading dims as res_type, trailing [B, L]
        """
        B, L = batch["aa_type"].shape
        device = batch["aa_type"].device
        energy_dtype = self.dtype

        if res_type.dim() == 2:
            n_decoys = 1
            res_type_expanded = res_type.unsqueeze(0)
        else:
            n_decoys = res_type.shape[0]
            res_type_expanded = res_type

        burial_energy = self.compute_burial_energy(batch, res_type, batch.get("burial_density", batch["residue_density"]))

        edges_sym = torch.cat([batch["edge_index"], batch["edge_index"].flip(1)], dim=-1)
        edges_expanded = edges_sym.unsqueeze(0).expand(n_decoys, -1, -1, -1)
        edges_flat = edges_expanded.reshape(n_decoys * B, 2, -1)
        res_type_flat = res_type_expanded.reshape(n_decoys * B, L)

        aa_src_sym, aa_dst_sym = self._gather_node(res_type_flat, edges_flat)

        density_expanded = batch["residue_density"].unsqueeze(0).expand(n_decoys, -1, -1).reshape(n_decoys * B, L)
        rho_src_sym, rho_dst_sym = self._gather_node(density_expanded, edges_flat)

        distances_expanded = batch["distance"].unsqueeze(0).expand(n_decoys, -1, -1, -1).reshape(n_decoys * B, L, L)
        edge_distances_sym = self._gather_edge_distances(distances_expanded, edges_flat)

        water_energy_edge = self.compute_water_energy(
            batch, edge_distances_sym, rho_src_sym, rho_dst_sym, aa_src_sym, aa_dst_sym
        ).to(energy_dtype)

        E = water_energy_edge.shape[-1]
        src_sym = edges_flat[:, 0, :]

        valid = (src_sym >= 0).view(-1)
        flat_src = (torch.arange(n_decoys * B, device=device)[:, None] * L + src_sym.clamp(min=0)).view(-1)
        flat_edge = water_energy_edge.view(-1)
        water_energy_flat = torch.zeros(n_decoys * B * L, device=device, dtype=energy_dtype)

        water_energy_flat.scatter_add_(0, flat_src[valid], flat_edge[valid])
        water_energy = water_energy_flat.view(n_decoys, B, L)

        dist_all = batch["distance"]

        if L > 300:
            # Charge-sparse electrostatics: avoids materialising [n_decoys, B, L, L]
            # (8+ GiB for large L).  For each (decoy, batch) element, only the
            # charged×charged sub-matrix [N_c, N_c] is computed and summed.
            electrostatics_energy = torch.zeros(n_decoys, B, L, device=device, dtype=energy_dtype)
            if self.coefficients.use_electrostatics:
                for d_idx in range(n_decoys):
                    for b_idx in range(B):
                        aa_db = res_type_expanded[d_idx, b_idx]              # [L]
                        q_db  = self.charge_lookup[aa_db]                     # [L]
                        ci    = (q_db != 0).nonzero(as_tuple=False).view(-1)  # [N_c]
                        if ci.numel() == 0:
                            continue
                        aa_c     = aa_db[ci]                                         # [N_c]
                        d_cc     = dist_all[b_idx, ci.unsqueeze(1), ci.unsqueeze(0)] # [N_c, N_c]
                        aa_src_c = aa_c.unsqueeze(1).expand(-1, ci.numel())          # [N_c, N_c]
                        aa_dst_c = aa_c.unsqueeze(0).expand(ci.numel(), -1)          # [N_c, N_c]
                        V_cc     = self.calculate_electrostatics(aa_src_c, aa_dst_c, d_cc)  # [N_c, N_c]
                        V_cc.diagonal().zero_()   # remove self-interactions
                        electrostatics_energy[d_idx, b_idx, ci] = V_cc.sum(dim=-1).to(energy_dtype)
        else:
            aa_src_all = res_type_expanded.unsqueeze(-1)
            aa_dst_all = res_type_expanded.unsqueeze(-2)
            dist_all_expanded = dist_all.unsqueeze(0).expand(n_decoys, -1, -1, -1)
            aa_src_all = aa_src_all.expand(n_decoys, B, L, L)
            aa_dst_all = aa_dst_all.expand(n_decoys, B, L, L)
            elec_all = self.calculate_electrostatics(aa_src_all, aa_dst_all, dist_all_expanded).to(energy_dtype)
            mask = torch.eye(L, device=device).bool().unsqueeze(0).unsqueeze(0)
            elec_all = elec_all.masked_fill(mask, 0.0)
            electrostatics_energy = elec_all.sum(dim=-1)

        if burial_energy.dim() == 2:
            burial_energy = burial_energy.unsqueeze(0)

        total_energy = burial_energy + water_energy + electrostatics_energy

        if res_type.dim() == 2:
            return total_energy.squeeze(0)
        else:
            return total_energy

    def compute_sr_elec_field(self, batch: dict) -> torch.Tensor:
        """
        Electrostatic field at every residue from all OTHER native residues.
        Returns [B, L] float64.
        """
        B, L   = batch["aa_type"].shape
        device = batch["aa_type"].device

        if not self.coefficients.use_electrostatics:
            return torch.zeros(B, L, device=device, dtype=self.dtype)

        q        = self.charge_lookup[batch["aa_type"]].to(self.dtype)  # [B, L]
        seq_mask = batch["seq_mask"].bool()                              # [B, L]

        if L <= 300:
            d     = batch["distance"].to(self.dtype)                    # [B, L, L]
            eye   = torch.eye(L, device=device, dtype=torch.bool).unsqueeze(0)
            valid = (~eye) & seq_mask.unsqueeze(1) & (q != 0).unsqueeze(1)
            safe_d = d.clamp(min=1e-6)
            kernel = (q.unsqueeze(1) / safe_d
                      * torch.exp(-safe_d / self.coefficients.screening_length))
            kernel = kernel * valid.to(self.dtype)
            return self.coefficients.k_PlusMinus * kernel.sum(-1)

        # Large-structure path: restrict columns to charged j-positions and
        # chunk over i-rows so no [L, L] intermediate is ever allocated.
        result     = torch.zeros(B, L, device=device, dtype=self.dtype)
        chunk_size = 2048
        d_full     = batch["distance"]   # [B, L, L]; stay in original dtype, cast per-chunk
        sl         = self.coefficients.screening_length
        for b_idx in range(B):
            q_b    = q[b_idx]            # [L]
            mask_b = seq_mask[b_idx]     # [L]
            ci     = (q_b != 0).nonzero(as_tuple=False).view(-1)   # [N_c]
            if ci.numel() == 0:
                continue
            q_ci   = q_b[ci]             # [N_c] non-zero charges
            ci_set = ci                  # for self-interaction lookup
            d_b    = d_full[b_idx]       # [L, L] — no copy, just a view slice

            for i_start in range(0, L, chunk_size):
                i_end    = min(i_start + chunk_size, L)
                d_chunk  = d_b[i_start:i_end, ci].to(self.dtype)   # [chunk, N_c]
                safe_d   = d_chunk.clamp_(min=1e-6)
                kernel   = q_ci.unsqueeze(0) / safe_d * torch.exp(-safe_d / sl)  # [chunk, N_c]
                # Zero self-interactions (i == j) within this chunk
                ci_in_chunk = (ci_set >= i_start) & (ci_set < i_end)
                if ci_in_chunk.any():
                    row_local = ci_set[ci_in_chunk] - i_start   # row offset in chunk
                    col_local = ci_in_chunk.nonzero(as_tuple=False).view(-1)
                    kernel[row_local, col_local] = 0.0
                # Zero rows where seq_mask is False
                kernel[~mask_b[i_start:i_end]] = 0.0
                result[b_idx, i_start:i_end] = kernel.sum(-1)

        return self.coefficients.k_PlusMinus * result

    def compute_sr_water_delta(
        self,
        batch: dict,
        aa_native: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each position i and each of the 20 possible AA substitutions,
        compute the change in i's water-energy contribution.
        Returns [20, B, L] float64.
        """
        B, L   = aa_native.shape
        device = aa_native.device
        all20  = torch.arange(20, device=device)

        edge_sym = torch.cat(
            [batch["edge_index"], batch["edge_index"].flip(1)], dim=-1
        )
        E2    = edge_sym.shape[-1]
        b_idx = torch.arange(B, device=device)[:, None].expand(B, E2)
        src   = edge_sym[:, 0, :]
        dst   = edge_sym[:, 1, :]

        valid_e   = (src >= 0) & (dst >= 0)
        src_safe  = src.clamp(min=0)
        dst_safe  = dst.clamp(min=0)

        rho_src   = rho[b_idx, src_safe]
        rho_dst   = rho[b_idx, dst_safe]
        dist_e    = batch["distance"][b_idx, src_safe, dst_safe]
        aa_dst    = aa_native[b_idx, dst_safe]
        aa_src_n  = aa_native[b_idx, src_safe]

        aa_all_e  = all20.view(20, 1, 1).expand(20, B, E2)
        aa_dst_e  = aa_dst.unsqueeze(0).expand(20, B, E2)
        rho_s_e   = rho_src.unsqueeze(0).expand(20, B, E2)
        rho_d_e   = rho_dst.unsqueeze(0).expand(20, B, E2)
        dist_e_x  = dist_e.unsqueeze(0).expand(20, B, E2)

        w_all20 = self.compute_water_energy(
            batch, dist_e_x, rho_s_e, rho_d_e, aa_all_e, aa_dst_e
        )
        w_native = self.compute_water_energy(
            batch, dist_e, rho_src, rho_dst, aa_src_n, aa_dst
        )

        delta_w = (w_all20 - w_native.unsqueeze(0)) * valid_e.unsqueeze(0).to(self.dtype)

        out = torch.zeros(20, B, L, device=device, dtype=delta_w.dtype)
        src_exp = src_safe.unsqueeze(0).expand(20, B, E2)
        out.scatter_add_(2, src_exp, delta_w)
        return out

    def _compute_gly_alt_densities(self, batch):
        """
        Compute two density variants needed for GLY ↔ non-GLY mutation corrections.

        The native density uses CA as centroid for GLY and CB for all others.
        When GLY mutates to non-GLY the correct centroid is the ideal virtual CB;
        when non-GLY mutates to GLY the correct centroid is CA.

        For GLY positions where the CHARMM36 ideal CB points INTO solvent
        (rho_vcb_36 < rho_nat AND rho_nat < 0.1 — unusual positive-φ conformations),
        falls back to the ProteinMPNN vCB formula when it gives higher burial density.

        Returns
        -------
        rho_vcb    : [B, L]  density using best-choice vCB for GLY; native elsewhere
        rho_ca     : [B, L]  density using CA for non-GLY; native elsewhere
        vcb_hybrid : [B, L, 3]  vCB coordinates used for rho_vcb (CHARMM36 or ProteinMPNN)
        """
        atom_pos   = batch["atom_positions"]  # [B, L, 6, 3]
        aatype_int = batch["aa_type"]         # [B, L]
        B, L = aatype_int.shape
        device = atom_pos.device
        dt = self.dtype

        CA = atom_pos[:, :, 1].to(dt)

        vcb_36   = compute_ideal_vcb(batch, dtype=dt)   # CHARMM36 IC [B, L, 3]
        vcb_mpnn = compute_mpnn_vcb(batch, dtype=dt)    # ProteinMPNN [B, L, 3]

        # Start with CHARMM36 as default; selectively replaced below
        vcb_hybrid = vcb_36.clone()

        is_gly = (aatype_int == 5)  # [B, L]
        native_centroid = get_CB_coords(atom_pos[:, :, :5], aatype_int, gly_hbeta=getattr(self.coefficients, 'gly_hbeta', False)).to(dt)  # [B, L, 3]

        cfg     = self.coefficients
        eta     = float(cfg.eta)
        rmin    = float(getattr(cfg, 'burial_rmin',    cfg.well_rmin[0]))
        rmax    = float(getattr(cfg, 'burial_rmax',    cfg.well_rmax[0]))
        burial_min_sep = cfg.burial_min_sep
        cutoff  = float(cfg.tert_frust_cutoff)

        residue_index = batch["residue_index"]                           # [B, L]
        edge_mask     = batch["seq_mask"].bool() & ~batch["hetatm"].bool()  # [B, L]

        def _rho_for_subset(b_idx, query_xyz, query_pos_idx, _max_elems: int = 50_000_000):
            """
            Density for a subset of positions in batch element b_idx.

            query_xyz      [Q, 3]  centroids to use for these positions
            query_pos_idx  [Q]     their sequence positions in [0, L)

            Returns [Q].  A position never counts itself (sep=0).
            Uses vectorized [Q, L] path for small proteins; chunks over Q for
            large ones to keep peak allocation at O(chunk_size × L).
            """
            ri_q  = residue_index[b_idx, query_pos_idx]       # [Q]
            em_q  = edge_mask[b_idx, query_pos_idx]           # [Q]
            ri_j  = residue_index[b_idx]                      # [L]
            em_j  = edge_mask[b_idx]                          # [L]
            nat_j = native_centroid[b_idx]                    # [L, 3]

            Q, L_full = query_xyz.shape[0], nat_j.shape[0]

            def _compute_chunk(xyz_c, ri_c, em_c):
                dist    = torch.cdist(xyz_c.float(), nat_j.float()).to(dt)
                t_min   = torch.tanh(eta * (dist - rmin))
                t_max   = torch.tanh(eta * (rmax - dist))
                v_theta = 0.25 * (1.0 + t_min) * (1.0 + t_max)
                del t_min, t_max
                sep    = (ri_c.to(torch.int32).unsqueeze(1) -
                          ri_j.to(torch.int32).unsqueeze(0)).abs()
                excl   = (sep <= burial_min_sep) | (sep == 0)
                del sep
                pair_m = em_c.unsqueeze(1) & em_j.unsqueeze(0)
                within = dist <= cutoff
                return (v_theta * (~excl & pair_m & within).to(dt)).sum(dim=-1)

            if Q * L_full <= _max_elems:
                return _compute_chunk(query_xyz, ri_q, em_q)

            chunk_size = max(1, _max_elems // L_full)
            result = torch.zeros(Q, device=query_xyz.device, dtype=dt)
            for q0 in range(0, Q, chunk_size):
                q1 = min(q0 + chunk_size, Q)
                result[q0:q1] = _compute_chunk(
                    query_xyz[q0:q1], ri_q[q0:q1], em_q[q0:q1])
            return result

        rho_native = batch.get("burial_density", batch["residue_density"])
        rho_vcb    = rho_native.clone()
        rho_ca     = rho_native.clone()
        store_dt   = rho_native.dtype  # match destination dtype on assignment

        for b in range(B):
            gly_idx = torch.where(is_gly[b])[0]
            if gly_idx.numel() > 0:
                rho_36_q = _rho_for_subset(b, vcb_36[b, gly_idx], gly_idx)
                rho_vcb[b, gly_idx] = rho_36_q.to(store_dt)

                # For near-zero-rho GLY positions where CHARMM36 places CB into solvent,
                # try ProteinMPNN as fallback — it uses average protein geometry and may
                # give CB in the correct direction for unusual positive-φ conformations.
                rho_nat_q = rho_native[b, gly_idx]
                wrong = (rho_36_q < rho_nat_q) & (rho_nat_q < 0.1)
                if wrong.any():
                    wrong_glob = gly_idx[wrong]
                    rho_mpnn_q = _rho_for_subset(b, vcb_mpnn[b, wrong_glob], wrong_glob)
                    better = rho_mpnn_q > rho_36_q[wrong]
                    if better.any():
                        update_idx = wrong_glob[better]
                        vcb_hybrid[b, update_idx] = vcb_mpnn[b, update_idx]
                        rho_vcb[b, update_idx] = rho_mpnn_q[better].to(store_dt)
                        ri = batch["residue_index"][b, update_idx].tolist()
                        print(f"[GLY-hybrid] batch {b}: switched to ProteinMPNN vCB at "
                              f"res {ri}, rho36={rho_36_q[wrong][better].tolist()} → "
                              f"rhoMPNN={rho_mpnn_q[better].tolist()}")

            non_gly_idx = torch.where(~is_gly[b])[0]
            if non_gly_idx.numel() > 0:
                rho_ca[b, non_gly_idx] = _rho_for_subset(b, CA[b, non_gly_idx], non_gly_idx).to(store_dt)

        return rho_vcb, rho_ca, vcb_hybrid

    def _compute_gly_water_correction(
        self,
        batch,
        rho_vcb: torch.Tensor | None = None,
        vcb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Case 1 — native GLY → non-GLY decoy X:
            batch["distance"] uses CA at GLY positions; FrustratometeR places an ideal
            vCB via MODELLER, giving different water contacts.
            Correction = W_vCB(X) − W_CA(X) summed over neighbors, zero for X=GLY.

        rho_vcb: if provided, uses vCB density for the solvation weight (sw) in W_vCB
            at positions where rho_nat and rho_vcb are on the same side of cfg.threshold.
            At crossing positions, keeps rho_nat to avoid the gamma-regime flip that
            causes sign errors.
        vcb: optional pre-computed vCB coordinates [B, L, 3] (e.g. hybrid CHARMM36 /
            ProteinMPNN from _compute_gly_alt_densities).  Recomputed from CHARMM36 IC
            if not provided.

        Returns [20, B, L] float correction to ADD to delta_water.
        Non-zero only at native GLY positions for non-GLY decoy AAs.
        """
        B, L = batch["aa_type"].shape
        device = batch["aa_type"].device
        GLY_IDX = 5

        is_gly = (batch["aa_type"] == GLY_IDX)

        cfg = self.coefficients
        all20 = torch.arange(20, device=device)

        atom_pos = batch["atom_positions"]

        if vcb is None:
            vcb = compute_ideal_vcb(batch, dtype=self.dtype)  # [B, L, 3]

        native_centroid = get_CB_coords(atom_pos[:, :, :5], batch["aa_type"], gly_hbeta=getattr(self.coefficients, 'gly_hbeta', False)).to(self.dtype)

        rho_nat = batch["residue_density"]
        residue_index = batch["residue_index"]
        valid_mask = batch["seq_mask"].bool() & ~batch["hetatm"].bool()
        min_seq_sep = int(cfg.min_seq_distance)

        water_gamma = self.get_water_gamma(batch)
        kw = float(cfg.water_kappa)
        rmin0 = cfg.well_rmin[0].to(device, dtype=self.dtype)
        rmax0 = cfg.well_rmax[0].to(device, dtype=self.dtype)
        rmin1 = cfg.well_rmin[1].to(device, dtype=self.dtype)
        rmax1 = cfg.well_rmax[1].to(device, dtype=self.dtype)

        # Burial density parameters for the neighbor ripple correction
        eta_b  = float(cfg.eta)
        rmin_b = float(cfg.well_rmin[0])
        rmax_b = float(cfg.well_rmax[0])

        def _vtheta(d):
            t_min = torch.tanh(eta_b * (d - rmin_b))
            t_max = torch.tanh(eta_b * (rmax_b - d))
            return 0.25 * (1.0 + t_min) * (1.0 + t_max)

        correction = torch.zeros(20, B, L, device=device, dtype=self.dtype)
        # Budget for [Q, L] elements per chunk; [20, chunk, L] float32 ≈ 80 MB
        _CHUNK_QL = 1_000_000

        def _water_e_ql(gd_kl, gpm_kl, gwm_kl, dist_ql, rho_src_q, rho_dst_ql):
            # Avoids [20*Q, L] int64 index materialization of the old _water_e.
            # gd_kl, gpm_kl, gwm_kl: [20, L] — precomputed gamma, no Q dim.
            # dist_ql: [Q, L], rho_src_q: [Q], rho_dst_ql: [Q, L].
            # Returns [20, Q, L].
            sigma  = float(cfg.water_kappa_sigma)
            thresh = float(cfg.threshold)
            sw = 0.25 * (1 - torch.tanh(sigma * (rho_src_q.unsqueeze(1) - thresh))) \
                      * (1 - torch.tanh(sigma * (rho_dst_ql - thresh)))  # [Q, L]
            sp = 1.0 - sw  # [Q, L]
            td = 0.25*(1+torch.tanh(kw*(dist_ql-rmin0)))*(1+torch.tanh(kw*(rmax0-dist_ql)))
            tm = 0.25*(1+torch.tanh(kw*(dist_ql-rmin1)))*(1+torch.tanh(kw*(rmax1-dist_ql)))
            # [20, 1, L] * [Q, L] → [20, Q, L] via broadcast
            return -(gd_kl[:, None, :] * td
                     + (sp * gpm_kl[:, None, :] + sw * gwm_kl[:, None, :]) * tm)

        for b in range(B):
            gly_idx = torch.where(is_gly[b] & valid_mask[b])[0]
            if gly_idx.numel() == 0:
                continue
            Q   = gly_idx.numel()
            ri_j = residue_index[b]; vm_j = valid_mask[b]
            aa_j = batch["aa_type"][b]; rho_j = rho_nat[b]

            # Precompute gamma lookups as [20, L] — avoids [20*Q, L] int64 materialization.
            # Use all20[:, None] (not [:]) so we index only the 20 canonical AA rows,
            # matching the original aa_x_e = all20.expand(20, Q, L) indexing.
            gd_kl  = water_gamma[0][all20[:, None], aa_j[None, :], :].mean(-1)  # [20, L]
            gpm_kl = water_gamma[1][all20[:, None], aa_j[None, :], 0]           # [20, L]
            gwm_kl = water_gamma[1][all20[:, None], aa_j[None, :], 1]           # [20, L]

            d_vcb = torch.cdist(vcb[b, gly_idx].float(), native_centroid[b].float()).to(self.dtype)
            d_ca  = batch["distance"][b][gly_idx, :]
            ri_q  = residue_index[b, gly_idx].to(torch.int32)
            sep   = (ri_q.unsqueeze(1) - ri_j.to(torch.int32).unsqueeze(0)).abs()  # [Q, L] int32
            vp    = (sep > min_seq_sep) & vm_j.unsqueeze(0) & valid_mask[b, gly_idx].unsqueeze(1)
            del sep
            rho_q = rho_nat[b, gly_idx]   # [Q] native CA density (baseline)

            # Selective solvation weight for source (position q) in W_vcb:
            # Use rho_vcb where rho_nat and rho_vcb are on the same side of threshold.
            # Where they cross, keep rho_nat to avoid flipping the gamma regime.
            if rho_vcb is not None:
                rho_q_vcb = rho_vcb[b, gly_idx]  # [Q]
                thresh = float(cfg.threshold)
                crosses  = (rho_q < thresh) != (rho_q_vcb < thresh)  # [Q]
                rho_q_sw = torch.where(crosses, rho_q, rho_q_vcb)
            else:
                rho_q_sw = rho_q

            # Neighbor density ripple: adding CB at GLY position q changes neighbor j's
            # density by v_theta(d_vCB[q,j]) - v_theta(d_CA[q,j]).
            # W_vcb uses the ripple-corrected neighbor density; W_ca uses native density.
            delta_rho_j = _vtheta(d_vcb) - _vtheta(d_ca)               # [Q, L]
            rho_j_vcb   = (rho_j.unsqueeze(0) + delta_rho_j).clamp(min=0)  # [Q, L]
            del delta_rho_j
            rho_j_row   = rho_j.unsqueeze(0)                            # [1, L]

            if Q * L <= _CHUNK_QL:
                W_vcb = _water_e_ql(gd_kl, gpm_kl, gwm_kl, d_vcb, rho_q_sw, rho_j_vcb)
                W_ca  = _water_e_ql(gd_kl, gpm_kl, gwm_kl, d_ca,  rho_q,    rho_j_row.expand(Q, L))
                dw = ((W_vcb - W_ca) * vp.unsqueeze(0).to(self.dtype)).sum(-1)  # [20, Q]
            else:
                chunk_q = max(1, _CHUNK_QL // L)
                dw = torch.zeros(20, Q, device=device, dtype=self.dtype)
                for q0 in range(0, Q, chunk_q):
                    q1 = min(q0 + chunk_q, Q)
                    c = q1 - q0
                    W_vcb_c = _water_e_ql(gd_kl, gpm_kl, gwm_kl,
                                          d_vcb[q0:q1], rho_q_sw[q0:q1], rho_j_vcb[q0:q1])
                    W_ca_c  = _water_e_ql(gd_kl, gpm_kl, gwm_kl,
                                          d_ca[q0:q1],  rho_q[q0:q1],    rho_j_row.expand(c, L))
                    dw[:, q0:q1] = ((W_vcb_c - W_ca_c)
                                    * vp[q0:q1].unsqueeze(0).to(self.dtype)).sum(-1)

            dw[GLY_IDX, :] = 0.0

            # At truly surface-exposed GLY positions (rho_nat < 0.1, essentially no
            # burial contacts) where the ideal CB is less buried than CA, the CHARMM36
            # IC places CB in an unusual backbone conformation direction that gives the
            # wrong sign for the water correction. Skip it; threading value is closer.
            # We only apply this when rho_nat itself is near-zero; at moderately-buried
            # positions (e.g. rho_nat ~ 1) the correction remains valid even if
            # rho_vcb < rho_nat due to different burial vs water distance ranges.
            if rho_vcb is not None:
                wrong_dir = (rho_q_vcb < rho_q) & (rho_q < 0.1)  # [Q]
                dw[:, wrong_dir] = 0.0

            correction[:, b, gly_idx] = dw.to(self.dtype)

        return correction

    def _compute_gly_decoy_fi_at_non_gly(
        self,
        batch,
        native_energy: torch.Tensor,  # [B, L]
        rho_vcb: torch.Tensor,         # [B, L] virtual-CB density
        rho_ca:  torch.Tensor,         # [B, L] CA-based density
        E_decoy_native: torch.Tensor = None,  # unused, kept for API compat
    ) -> torch.Tensor:                 # [B, L] fi_gly, NaN at GLY positions
        """
        Case 2 — native non-GLY → GLY decoy.

        Uses all-CA distribution (all 20 decoys evaluated with CA geometry at pos q),
        matching FrustratometeR's behaviour of running each GLY mutant as a separate
        structure where the native at pos q IS GLY (so all decoys use CA centroid).

        Key fix vs. the naive all-CA: the electrostatic field uses CA-based distances
        (d_CA_qj) instead of native CB distances from batch["distance"].  This is the
        dominant error source in the original approximation.

        Returns [B, L] FI for the GLY decoy at each non-GLY position, NaN at GLY positions.
        """
        B, L = batch["aa_type"].shape
        device = batch["aa_type"].device
        GLY_IDX = 5
        all20 = torch.arange(20, device=device)

        is_gly = (batch["aa_type"] == GLY_IDX)

        cfg = self.coefficients
        atom_pos = batch["atom_positions"]
        CA = atom_pos[:, :, 1].to(self.dtype)

        native_centroid = get_CB_coords(atom_pos[:, :, :5], batch["aa_type"], gly_hbeta=getattr(self.coefficients, 'gly_hbeta', False)).to(self.dtype)

        rho_nat        = batch["residue_density"]
        rho_nat_burial = batch.get("burial_density", batch["residue_density"])
        residue_index = batch["residue_index"]
        valid_mask = batch["seq_mask"].bool() & ~batch["hetatm"].bool()
        min_seq_sep = int(cfg.min_seq_distance)
        cutoff  = float(getattr(cfg, 'tert_frust_cutoff', 9.5))

        water_gamma = self.get_water_gamma(batch)
        kw = float(cfg.water_kappa)
        rmin0 = cfg.well_rmin[0].to(device, dtype=self.dtype)
        rmax0 = cfg.well_rmax[0].to(device, dtype=self.dtype)
        rmin1 = cfg.well_rmin[1].to(device, dtype=self.dtype)
        rmax1 = cfg.well_rmax[1].to(device, dtype=self.dtype)

        eta_b  = float(cfg.eta)
        rmin_b = float(cfg.well_rmin[0])
        rmax_b = float(cfg.well_rmax[0])

        def _vtheta(d):
            t_min = torch.tanh(eta_b * (d - rmin_b))
            t_max = torch.tanh(eta_b * (rmax_b - d))
            return 0.25 * (1.0 + t_min) * (1.0 + t_max)

        def _water_e(aa_src, aa_dst, dist, rho_src, rho_dst):
            n = aa_src.shape[-1]
            sw = 0.25 * (1 - torch.tanh(cfg.water_kappa_sigma * (rho_src - cfg.threshold))) \
                      * (1 - torch.tanh(cfg.water_kappa_sigma * (rho_dst - cfg.threshold)))
            sp = 1.0 - sw
            asf = aa_src.reshape(-1, n); adf = aa_dst.reshape(-1, n)
            gd  = water_gamma[0, asf, adf, :].mean(-1).reshape(aa_src.shape)
            gpm = water_gamma[1, asf, adf, 0].reshape(aa_src.shape)
            gwm = water_gamma[1, asf, adf, 1].reshape(aa_src.shape)
            td  = 0.25*(1+torch.tanh(kw*(dist-rmin0)))*(1+torch.tanh(kw*(rmax0-dist)))
            tm  = 0.25*(1+torch.tanh(kw*(dist-rmin1)))*(1+torch.tanh(kw*(rmax1-dist)))
            return -(gd*td + (sp*gpm + sw*gwm)*tm)

        all20_exp  = all20.view(20,1,1).expand(20,B,L)
        rho_ca_exp = rho_ca.unsqueeze(0).expand(20,B,L)
        burial_ca_all = self.compute_burial_energy(batch, all20_exp, rho_ca_exp)  # [20, B, L]
        burial_nat_native = self.compute_burial_energy(
            batch, batch["aa_type"].unsqueeze(0), rho_nat_burial.unsqueeze(0)
        )  # [1, B, L]
        delta_burial_ca = burial_ca_all - burial_nat_native  # [20, B, L]

        # Electrostatic charges; field computed per-position inside the loop using CA distances
        q_all  = self.charge_lookup[all20].to(self.dtype)
        q_nat  = self.charge_lookup[batch["aa_type"]].to(self.dtype)

        use_elec = self.coefficients.use_electrostatics
        λ = float(cfg.screening_length)
        k_pm = float(cfg.k_PlusMinus)

        vm = batch["seq_mask"].bool()
        seq_len   = vm.float().sum(-1).clamp(min=1)
        aa_counts = torch.stack(
            [((batch["aa_type"] == a) & vm).float().sum(-1) for a in range(20)], dim=-1
        ).to(self.dtype)
        freq = aa_counts / seq_len.unsqueeze(-1)  # [B, 20]

        fi_gly = torch.full((B, L), float("nan"), device=device, dtype=self.dtype)
        _CHUNK_QL = 5_000_000  # switch to chunked path above this many [Q×L] elements

        for b in range(B):
            non_gly_idx = torch.where(~is_gly[b] & valid_mask[b])[0]
            if non_gly_idx.numel() == 0:
                continue
            Q  = non_gly_idx.numel()
            vm_j = valid_mask[b]
            aa_j = batch["aa_type"][b]; rho_j = rho_nat[b]
            nat_aa_q = batch["aa_type"][b, non_gly_idx]  # [Q]
            db_ca = delta_burial_ca[:, b, non_gly_idx]   # [20, Q]
            ne_q  = native_energy[b, non_gly_idx]         # [Q]
            q_res = self.charge_lookup[aa_j].to(self.dtype) if use_elec else None  # [L]

            if Q * L <= _CHUNK_QL:
                # ── Small protein: original vectorized path ───────────────────
                d_ca_q  = torch.cdist(CA[b, non_gly_idx].float(), native_centroid[b].float()).to(self.dtype)

                pos_j   = torch.arange(L, device=device, dtype=torch.int32)
                pos_sep = (non_gly_idx.to(torch.int32).unsqueeze(1) - pos_j.unsqueeze(0)).abs()
                vp   = ((pos_sep >= 2)
                        & vm_j.unsqueeze(0)
                        & valid_mask[b, non_gly_idx].unsqueeze(1)
                        & (d_ca_q < cutoff))
                del pos_sep

                rho_q_ca  = rho_ca[b, non_gly_idx]
                aa_x_e    = all20.view(20,1,1).expand(20,Q,L)
                aa_j_e    = aa_j.view(1,1,L).expand(20,Q,L)
                aa_nat_e  = nat_aa_q.view(1,Q,1).expand(1,Q,L)
                rho_ca_e  = rho_q_ca.view(1,Q,1).expand(20,Q,L)
                rho_ca_e1 = rho_q_ca.view(1,Q,1).expand(1,Q,L)

                d_cb_qj = torch.cdist(native_centroid[b, non_gly_idx].float(),
                                      native_centroid[b].float()).to(self.dtype)
                delta_rho_j = _vtheta(d_ca_q) - _vtheta(d_cb_qj)
                del d_cb_qj
                rho_j_corr  = (rho_j.unsqueeze(0) + delta_rho_j).clamp(min=0)
                del delta_rho_j
                rho_je = rho_j_corr.unsqueeze(0).expand(20, Q, L)

                W_ca_all = _water_e(aa_x_e, aa_j_e,
                                    d_ca_q.unsqueeze(0).expand(20, Q, L),
                                    rho_ca_e, rho_je)
                W_ca_nat = _water_e(aa_nat_e, aa_j_e[:1], d_ca_q.unsqueeze(0),
                                    rho_ca_e1, rho_j_corr.unsqueeze(0)).squeeze(0)
                # [1, Q, L] broadcast with [20, Q, L] — no 20× expand needed
                dw_ca = ((W_ca_all - W_ca_nat.unsqueeze(0))
                         * vp.unsqueeze(0).to(self.dtype)).sum(-1)

                if use_elec:
                    d_ca_safe = d_ca_q.clamp(min=1e-6)
                    self_mask = torch.zeros(Q, L, dtype=torch.bool, device=device)
                    self_mask[torch.arange(Q, device=device), non_gly_idx] = True
                    valid_e   = (q_res != 0).unsqueeze(0) & vm_j.unsqueeze(0) & ~self_mask
                    kernel_ca = q_res.unsqueeze(0) / d_ca_safe * torch.exp(-d_ca_safe / λ)
                    field_ca_q = k_pm * (kernel_ca * valid_e.to(self.dtype)).sum(-1)
                    q_nat_q   = q_nat[b, non_gly_idx]
                    de_q = (q_all.unsqueeze(1) - q_nat_q.unsqueeze(0)) * field_ca_q.unsqueeze(0)
                else:
                    de_q = torch.zeros(20, Q, device=device, dtype=self.dtype)

            else:
                # ── Large protein: chunk over Q to avoid [20, Q, L] allocations ──
                chunk_q = max(1, _CHUNK_QL // L)
                dw_ca   = torch.zeros(20, Q, device=device, dtype=self.dtype)
                de_q    = torch.zeros(20, Q, device=device, dtype=self.dtype)
                pos_j_i32 = torch.arange(L, device=device, dtype=torch.int32)

                for q0 in range(0, Q, chunk_q):
                    q1 = min(q0 + chunk_q, Q)
                    C  = q1 - q0
                    ngi_c = non_gly_idx[q0:q1]

                    d_ca_c = torch.cdist(CA[b, ngi_c].float(),
                                         native_centroid[b].float()).to(self.dtype)  # [C, L]

                    pos_sep_c = (ngi_c.to(torch.int32).unsqueeze(1)
                                 - pos_j_i32.unsqueeze(0)).abs()
                    vp_c = ((pos_sep_c >= 2) & vm_j.unsqueeze(0)
                            & valid_mask[b, ngi_c].unsqueeze(1) & (d_ca_c < cutoff))
                    del pos_sep_c

                    rho_q_ca_c = rho_ca[b, ngi_c]
                    d_cb_qj_c  = torch.cdist(native_centroid[b, ngi_c].float(),
                                              native_centroid[b].float()).to(self.dtype)
                    delta_rho_c = _vtheta(d_ca_c) - _vtheta(d_cb_qj_c)
                    del d_cb_qj_c
                    rho_j_corr_c = (rho_j.unsqueeze(0) + delta_rho_c).clamp(min=0)
                    del delta_rho_c

                    W_ca_all_c = _water_e(
                        all20.view(20,1,1).expand(20,C,L),
                        aa_j.view(1,1,L).expand(20,C,L),
                        d_ca_c.unsqueeze(0).expand(20,C,L),
                        rho_q_ca_c.view(1,C,1).expand(20,C,L),
                        rho_j_corr_c.unsqueeze(0).expand(20,C,L),
                    )
                    W_ca_nat_c = _water_e(
                        nat_aa_q[q0:q1].view(1,C,1).expand(1,C,L),
                        aa_j.view(1,1,L).expand(1,C,L),
                        d_ca_c.unsqueeze(0),
                        rho_q_ca_c.view(1,C,1).expand(1,C,L),
                        rho_j_corr_c.unsqueeze(0),
                    ).squeeze(0)
                    dw_ca[:, q0:q1] = ((W_ca_all_c - W_ca_nat_c.unsqueeze(0))
                                       * vp_c.unsqueeze(0).to(self.dtype)).sum(-1)

                    if use_elec:
                        d_ca_safe_c = d_ca_c.clamp(min=1e-6)
                        self_mask_c = torch.zeros(C, L, dtype=torch.bool, device=device)
                        self_mask_c[torch.arange(C, device=device), ngi_c] = True
                        valid_e_c   = ((q_res != 0).unsqueeze(0) & vm_j.unsqueeze(0)
                                       & ~self_mask_c)
                        kernel_c    = q_res.unsqueeze(0) / d_ca_safe_c * torch.exp(-d_ca_safe_c / λ)
                        field_c     = k_pm * (kernel_c * valid_e_c.to(self.dtype)).sum(-1)
                        q_nat_q_c   = q_nat[b, ngi_c]
                        de_q[:, q0:q1] = ((q_all.unsqueeze(1) - q_nat_q_c.unsqueeze(0))
                                          * field_c.unsqueeze(0))

            E_mix = ne_q.unsqueeze(0) + db_ca + dw_ca + de_q  # [20, Q]

            freq_b   = freq[b].view(20, 1)
            mean_mix = (freq_b * E_mix).sum(0)
            std_mix  = (freq_b * (E_mix - mean_mix.unsqueeze(0))**2).sum(0).clamp(min=0).sqrt()

            fi_q = (mean_mix - E_mix[GLY_IDX]) / std_mix.clamp(min=1e-8)
            fi_q = torch.where(std_mix > 0, fi_q, torch.zeros_like(fi_q))

            fi_gly[b, non_gly_idx] = fi_q.to(self.dtype)

        return fi_gly

    def _compute_gly_native_fi(
        self,
        batch,
        native_energy: torch.Tensor,
        rho_vcb: torch.Tensor,
        rho_ca:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Case 1 — native GLY → any decoy AA.

        FrustratometeR creates non-GLY mutants at GLY positions by adding an ideal CB
        (via check_backbone_complete inside calculate_frustration), then runs frustration
        on that mutant structure with all decoys using CB geometry.

        NOTE: FrustratometeR actually uses a per-mutation geometry (CB for G→X, CA for G→G),
        but sharing one all-vCB distribution is still better than gly_water_corr for
        individual non-GLY mutation slots. Currently NOT called from the fast path because
        gly_water_corr gives lower average error (0.1792 vs 0.2053) due to error cancellation.

        Returns [B, L, 20] FI for all 20 decoy AAs at native GLY positions,
        NaN at non-GLY positions.
        """
        B, L = batch["aa_type"].shape
        device = batch["aa_type"].device
        GLY_IDX = 5
        all20 = torch.arange(20, device=device)

        is_gly = (batch["aa_type"] == GLY_IDX)

        cfg = self.coefficients
        atom_pos = batch["atom_positions"]

        # CHARMM22 IC for interior residues; CHARMM36 at chain starts. Falls back to CA.
        vcb = compute_ideal_vcb(batch, dtype=self.dtype)  # [B, L, 3]

        native_centroid = get_CB_coords(atom_pos[:, :, :5], batch["aa_type"], gly_hbeta=getattr(self.coefficients, 'gly_hbeta', False)).to(self.dtype)

        rho_nat        = batch["residue_density"]                           # water density (sigma_wat)
        rho_nat_burial = batch.get("burial_density", batch["residue_density"])  # burial density
        residue_index = batch["residue_index"]
        valid_mask = batch["seq_mask"].bool() & ~batch["hetatm"].bool()
        min_seq_sep = int(cfg.min_seq_distance)

        water_gamma = self.get_water_gamma(batch)
        kw = float(cfg.water_kappa)
        rmin0 = cfg.well_rmin[0].to(device, dtype=self.dtype)
        rmax0 = cfg.well_rmax[0].to(device, dtype=self.dtype)
        rmin1 = cfg.well_rmin[1].to(device, dtype=self.dtype)
        rmax1 = cfg.well_rmax[1].to(device, dtype=self.dtype)

        # Neighbor ripple correction uses water-contact well params (affects sigma_wat)
        eta_b  = float(cfg.eta)
        rmin_b = float(cfg.well_rmin[0])
        rmax_b = float(cfg.well_rmax[0])

        def _vtheta(d):
            t_min = torch.tanh(eta_b * (d - rmin_b))
            t_max = torch.tanh(eta_b * (rmax_b - d))
            return 0.25 * (1.0 + t_min) * (1.0 + t_max)

        def _water_e(aa_src, aa_dst, dist, rho_src, rho_dst):
            n = aa_src.shape[-1]
            sw = 0.25 * (1 - torch.tanh(cfg.water_kappa_sigma * (rho_src - cfg.threshold))) \
                      * (1 - torch.tanh(cfg.water_kappa_sigma * (rho_dst - cfg.threshold)))
            sp = 1.0 - sw
            asf = aa_src.reshape(-1, n); adf = aa_dst.reshape(-1, n)
            gd  = water_gamma[0, asf, adf, :].mean(-1).reshape(aa_src.shape)
            gpm = water_gamma[1, asf, adf, 0].reshape(aa_src.shape)
            gwm = water_gamma[1, asf, adf, 1].reshape(aa_src.shape)
            td  = 0.25*(1+torch.tanh(kw*(dist-rmin0)))*(1+torch.tanh(kw*(rmax0-dist)))
            tm  = 0.25*(1+torch.tanh(kw*(dist-rmin1)))*(1+torch.tanh(kw*(rmax1-dist)))
            return -(gd*td + (sp*gpm + sw*gwm)*tm)

        # Burial for all 20 decoys at GLY positions using vCB density
        all20_exp   = all20.view(20, 1, 1).expand(20, B, L)
        rho_vcb_exp = rho_vcb.unsqueeze(0).expand(20, B, L)
        burial_vcb_all = self.compute_burial_energy(batch, all20_exp, rho_vcb_exp)  # [20, B, L]
        burial_nat_native = self.compute_burial_energy(
            batch, batch["aa_type"].unsqueeze(0), rho_nat_burial.unsqueeze(0)
        )  # [1, B, L]
        delta_burial_vcb = burial_vcb_all - burial_nat_native  # [20, B, L]

        # Electrostatics (same as main path — uses native CA-based field)
        field  = self.compute_sr_elec_field(batch)
        q_all  = self.charge_lookup[all20].to(self.dtype)
        q_nat  = self.charge_lookup[batch["aa_type"]].to(self.dtype)
        delta_elec = (q_all.view(20, 1, 1) - q_nat.unsqueeze(0)) * field.unsqueeze(0)

        # Sequence composition weights [B, 20]
        vm = valid_mask
        seq_len   = vm.float().sum(-1).clamp(min=1)
        aa_counts = torch.stack(
            [((batch["aa_type"] == a) & vm).float().sum(-1) for a in range(20)], dim=-1
        ).to(self.dtype)
        freq = aa_counts / seq_len.unsqueeze(-1)

        fi_gly_native = torch.full((B, L, 20), float("nan"), device=device, dtype=self.dtype)

        for b in range(B):
            gly_idx = torch.where(is_gly[b] & valid_mask[b])[0]
            if gly_idx.numel() == 0:
                continue

            Q = gly_idx.numel()
            ri_j = residue_index[b]; vm_j = valid_mask[b]
            aa_j = batch["aa_type"][b]; rho_j = rho_nat[b]

            # vCB and CA distances from GLY positions to all native centroids [Q, L]
            d_vcb_q = torch.cdist(vcb[b, gly_idx].float(), native_centroid[b].float()).to(self.dtype)
            d_ca_q  = batch["distance"][b][gly_idx, :]

            ri_q = residue_index[b, gly_idx]
            sep  = (ri_q.unsqueeze(1) - ri_j.unsqueeze(0)).abs()
            vp   = (sep > min_seq_sep) & vm_j.unsqueeze(0) & valid_mask[b, gly_idx].unsqueeze(1)
            vp_e = vp.unsqueeze(0).expand(20, Q, L).to(self.dtype)

            rho_q_vcb = rho_vcb[b, gly_idx]   # [Q] — vCB density for source
            rho_q_nat = rho_nat[b, gly_idx]   # [Q] — CA density for GLY native baseline

            aa_x_e    = all20.view(20, 1, 1).expand(20, Q, L)
            aa_j_e    = aa_j.view(1, 1, L).expand(20, Q, L)
            rho_vcb_e = rho_q_vcb.view(1, Q, 1).expand(20, Q, L)

            # Neighbor density ripple: adding vCB at GLY position r shifts neighbor
            # j's density by delta_rho_j[r,j] = v_theta(d_vCB[r,j]) - v_theta(d_CA[r,j]).
            # Use corrected neighbor density for the vCB-based decoy water energy.
            delta_rho_j = _vtheta(d_vcb_q) - _vtheta(d_ca_q)  # [Q, L]
            rho_j_corrected = (rho_j.unsqueeze(0) + delta_rho_j).clamp(min=0)  # [Q, L]
            rho_je    = rho_j_corrected.unsqueeze(0).expand(20, Q, L)  # [20, Q, L]

            # ── All 20 decoys: vCB distances + vCB source density + corrected neighbor density
            W_vcb_x = _water_e(
                aa_x_e, aa_j_e,
                d_vcb_q.unsqueeze(0).expand(20, Q, L),
                rho_vcb_e, rho_je,
            )  # [20, Q, L]

            # Native GLY water baseline (CA distances, CA density, native neighbor density)
            gly_aa_e   = torch.full((1, Q, L), GLY_IDX, device=device, dtype=torch.long)
            rho_nat_e1 = rho_q_nat.view(1, Q, 1).expand(1, Q, L)
            W_ca_nat = _water_e(
                gly_aa_e, aa_j_e[:1],
                d_ca_q.unsqueeze(0),
                rho_nat_e1,
                rho_j.view(1, 1, L).expand(1, Q, L),  # uncorrected: native structure
            ).squeeze(0)  # [Q, L]

            # All 20 decoys use vCB geometry (CB added by check_backbone_complete for non-GLY
            # mutations; GLY "mutation" gets same vCB treatment since we share one distribution).
            dw = ((W_vcb_x - W_ca_nat.unsqueeze(0)) * vp_e).sum(-1)  # [20, Q]

            # ── Build E_decoy [20, Q] — all use vCB geometry ──────────────────
            db_vcb = delta_burial_vcb[:, b, gly_idx]
            de_q   = delta_elec[:, b, gly_idx]
            ne_q   = native_energy[b, gly_idx]

            E_mix = ne_q.unsqueeze(0) + db_vcb + dw + de_q

            # ── Stats and FI ─────────────────────────────────────────────────
            freq_b   = freq[b].view(20, 1)
            mean_mix = (freq_b * E_mix).sum(0)
            std_mix  = (freq_b * (E_mix - mean_mix.unsqueeze(0))**2).sum(0).clamp(min=0).sqrt()

            fi_q = (mean_mix.unsqueeze(0) - E_mix) / std_mix.unsqueeze(0).clamp(min=1e-8)
            fi_q = torch.where(std_mix.unsqueeze(0) > 0, fi_q, torch.zeros_like(fi_q))

            fi_gly_native[b, gly_idx] = fi_q.T.to(self.dtype)  # [Q, 20]

        return fi_gly_native

    def _compute_gly_mutant_fi(
        self,
        batch,
        native_energy: torch.Tensor,
        rho_vcb: torch.Tensor,
        rho_ca:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Case 1 — native GLY → any decoy AA, computed in the mutant's geometry.

        For each GLY position q computes the ABSOLUTE single-residue energy for
        all 20 decoy amino acids using the virtual CB (vCB) centroid:

            E(X, q) = burial(X, rho_vcb_q)
                    + Σ_j water(X, j, d_vCB[q,j], rho_vcb_q, ρ_j_corrected) · vp[q,j]
                    + charge(X) · field_vCB(q)

        field_vCB(q) is the electrostatic field at vCB (using vCB distances), unlike
        _compute_gly_native_fi which uses the CA-based field from batch["distance"].

        This mirrors FrustratometeR: create a structural mutant with CB at the ideal
        position, then evaluate the full AWSEM single-residue energy there.

        Compared to _compute_gly_native_fi:
        ─ Burial / water / neighbor-ripple: identical
        ─ Electrostatics:  corrected (d_vCB for source position, not d_CA)
        ─ Baseline:        absolute energy (not delta from native; equivalent analytically
                           for neutral AAs, differs only for charged AAs)
        ─ Pair filter:     position-based sep >= 2, matching calculate_distance_matrix

        NOTE: G→G (GLY decoy at GLY native position) uses the vCB distribution, which
        differs from the native CA-based FI.  Callers should substitute the native FI
        for that slot (see run_mutations_from_pdb).

        Returns [B, L, 20] FI (NaN at non-GLY positions).
        """
        B, L = batch["aa_type"].shape
        device = batch["aa_type"].device
        GLY_IDX = 5
        all20 = torch.arange(20, device=device)

        is_gly = (batch["aa_type"] == GLY_IDX)
        valid_mask = batch["seq_mask"].bool() & ~batch["hetatm"].bool()

        cfg = self.coefficients
        atom_pos = batch["atom_positions"]

        vcb     = compute_ideal_vcb(batch, dtype=self.dtype)
        vcb_pro = compute_pro_vcb(batch, dtype=self.dtype)   # Pro-specific IC dihedral (~115.3°)
        native_centroid = get_CB_coords(atom_pos[:, :, :5], batch["aa_type"], gly_hbeta=getattr(self.coefficients, 'gly_hbeta', False)).to(self.dtype)

        rho_nat        = batch["residue_density"]                           # water density (sigma_wat)
        residue_index = batch["residue_index"]
        min_seq_sep = int(cfg.min_seq_distance)
        cutoff  = float(getattr(cfg, 'tert_frust_cutoff', 9.5))
        PRO_IDX = 12

        water_gamma = self.get_water_gamma(batch)
        kw = float(cfg.water_kappa)
        rmin0 = cfg.well_rmin[0].to(device, dtype=self.dtype)
        rmax0 = cfg.well_rmax[0].to(device, dtype=self.dtype)
        rmin1 = cfg.well_rmin[1].to(device, dtype=self.dtype)
        rmax1 = cfg.well_rmax[1].to(device, dtype=self.dtype)

        eta_b  = float(cfg.eta)
        rmin_b = float(cfg.well_rmin[0])
        rmax_b = float(cfg.well_rmax[0])

        def _vtheta(d):
            t_min = torch.tanh(eta_b * (d - rmin_b))
            t_max = torch.tanh(eta_b * (rmax_b - d))
            return 0.25 * (1.0 + t_min) * (1.0 + t_max)

        def _water_e(aa_src, aa_dst, dist, rho_src, rho_dst):
            n = aa_src.shape[-1]
            sw = 0.25 * (1 - torch.tanh(cfg.water_kappa_sigma * (rho_src - cfg.threshold))) \
                      * (1 - torch.tanh(cfg.water_kappa_sigma * (rho_dst - cfg.threshold)))
            sp = 1.0 - sw
            asf = aa_src.reshape(-1, n); adf = aa_dst.reshape(-1, n)
            gd  = water_gamma[0, asf, adf, :].mean(-1).reshape(aa_src.shape)
            gpm = water_gamma[1, asf, adf, 0].reshape(aa_src.shape)
            gwm = water_gamma[1, asf, adf, 1].reshape(aa_src.shape)
            td  = 0.25*(1+torch.tanh(kw*(dist-rmin0)))*(1+torch.tanh(kw*(rmax0-dist)))
            tm  = 0.25*(1+torch.tanh(kw*(dist-rmin1)))*(1+torch.tanh(kw*(rmax1-dist)))
            return -(gd*td + (sp*gpm + sw*gwm)*tm)

        # Burial for all 20 decoys using vCB density
        all20_exp   = all20.view(20, 1, 1).expand(20, B, L)
        rho_vcb_exp = rho_vcb.unsqueeze(0).expand(20, B, L)
        burial_vcb_all = self.compute_burial_energy(batch, all20_exp, rho_vcb_exp)  # [20, B, L]

        # Sequence frequency weights [B, 20]
        vm = valid_mask
        seq_len   = vm.float().sum(-1).clamp(min=1)
        aa_counts = torch.stack(
            [((batch["aa_type"] == a) & vm).float().sum(-1) for a in range(20)], dim=-1
        ).to(self.dtype)
        freq = aa_counts / seq_len.unsqueeze(-1)

        use_elec = bool(cfg.use_electrostatics)
        if use_elec:
            k_pm = float(cfg.k_PlusMinus)
            screening_len = float(cfg.screening_length)
        q_lookup = self.charge_lookup.to(dtype=self.dtype)

        fi_gly_mutant = torch.full((B, L, 20), float("nan"), device=device, dtype=self.dtype)
        _CHUNK_QL = 5_000_000  # switch to chunked path above this many [Q×L] elements

        for b in range(B):
            gly_idx = torch.where(is_gly[b] & valid_mask[b])[0]
            if gly_idx.numel() == 0:
                continue
            Q = gly_idx.numel()

            vm_j = valid_mask[b]
            aa_j = batch["aa_type"][b]; rho_j = rho_nat[b]
            burial_x = burial_vcb_all[:, b, gly_idx]  # [20, Q] — precomputed

            if Q * L <= _CHUNK_QL:
                # ── Small protein: original vectorized path ───────────────────
                d_vcb_q = torch.cdist(vcb[b, gly_idx].float(),
                                      native_centroid[b].float()).to(self.dtype)
                d_pro_q = torch.cdist(vcb_pro[b, gly_idx].float(),
                                      native_centroid[b].float()).to(self.dtype)
                d_ca_q  = batch["distance"][b][gly_idx, :]

                pos_j   = torch.arange(L, device=device, dtype=torch.int32)
                pos_sep = (gly_idx.to(torch.int32).unsqueeze(1) - pos_j.unsqueeze(0)).abs()
                base_mask = (pos_sep >= 2) & vm_j.unsqueeze(0) & valid_mask[b, gly_idx].unsqueeze(1)
                del pos_sep
                vp     = base_mask & (d_vcb_q < cutoff)
                vp_pro = base_mask & (d_pro_q < cutoff)
                # [20, Q, L] mask: Pro row uses vp_pro
                vp_e = vp.unsqueeze(0).expand(20, Q, L).clone().to(self.dtype)
                vp_e[PRO_IDX] = vp_pro.to(self.dtype)

                rho_q_vcb = rho_vcb[b, gly_idx]
                aa_x_e    = all20.view(20, 1, 1).expand(20, Q, L)
                aa_j_e    = aa_j.view(1, 1, L).expand(20, Q, L)
                rho_vcb_e = rho_q_vcb.view(1, Q, 1).expand(20, Q, L)

                delta_rho_j = _vtheta(d_vcb_q) - _vtheta(d_ca_q)
                rho_j_corr  = (rho_j.unsqueeze(0) + delta_rho_j).clamp(min=0)
                del delta_rho_j
                rho_je = rho_j_corr.unsqueeze(0).expand(20, Q, L)

                d_all_q = d_vcb_q.unsqueeze(0).expand(20, Q, L).clone()
                d_all_q[PRO_IDX] = d_pro_q

                W_vcb_x = _water_e(aa_x_e, aa_j_e, d_all_q, rho_vcb_e, rho_je)
                water_x = (W_vcb_x * vp_e).sum(-1)

                if use_elec:
                    q_j_b   = q_lookup[aa_j]
                    safe_d  = d_vcb_q.clamp(min=1e-6)
                    self_mask = torch.zeros(Q, L, device=device, dtype=torch.bool)
                    for qi, ri_val in enumerate(gly_idx.tolist()):
                        self_mask[qi, ri_val] = True
                    elec_valid  = (~self_mask) & vm_j.unsqueeze(0) & (q_j_b.unsqueeze(0) != 0)
                    kernel      = (q_j_b.unsqueeze(0) / safe_d) * torch.exp(-safe_d / screening_len)
                    field_vcb_q = k_pm * (kernel * elec_valid.to(self.dtype)).sum(-1)
                    q_all_local = q_lookup[all20]
                    elec_x      = q_all_local.view(20, 1) * field_vcb_q.view(1, Q)
                else:
                    elec_x = torch.zeros(20, Q, device=device, dtype=self.dtype)

            else:
                # ── Large protein: chunk over Q to avoid [20, Q, L] allocations ──
                chunk_q = max(1, _CHUNK_QL // L)
                water_x = torch.zeros(20, Q, device=device, dtype=self.dtype)
                elec_x  = torch.zeros(20, Q, device=device, dtype=self.dtype)
                pos_j_i32   = torch.arange(L, device=device, dtype=torch.int32)
                q_j_b       = q_lookup[aa_j] if use_elec else None
                q_all_local = q_lookup[all20] if use_elec else None

                for q0 in range(0, Q, chunk_q):
                    q1 = min(q0 + chunk_q, Q)
                    C  = q1 - q0
                    gly_c = gly_idx[q0:q1]

                    d_vcb_c = torch.cdist(vcb[b, gly_c].float(),
                                          native_centroid[b].float()).to(self.dtype)
                    d_pro_c = torch.cdist(vcb_pro[b, gly_c].float(),
                                          native_centroid[b].float()).to(self.dtype)
                    d_ca_c  = batch["distance"][b][gly_c, :]

                    pos_sep_c = (gly_c.to(torch.int32).unsqueeze(1)
                                 - pos_j_i32.unsqueeze(0)).abs()
                    base_c = (pos_sep_c >= 2) & vm_j.unsqueeze(0) & valid_mask[b, gly_c].unsqueeze(1)
                    del pos_sep_c
                    vp_c     = base_c & (d_vcb_c < cutoff)
                    vp_pro_c = base_c & (d_pro_c < cutoff)

                    rho_q_vcb_c = rho_vcb[b, gly_c]
                    delta_rho_c = _vtheta(d_vcb_c) - _vtheta(d_ca_c)
                    rho_j_corr_c = (rho_j.unsqueeze(0) + delta_rho_c).clamp(min=0)
                    del delta_rho_c

                    # [20, C, L] Pro-aware distance and mask
                    d_all_c = d_vcb_c.unsqueeze(0).expand(20, C, L).clone()
                    d_all_c[PRO_IDX] = d_pro_c
                    vp_e_c = vp_c.unsqueeze(0).expand(20, C, L).clone().to(self.dtype)
                    vp_e_c[PRO_IDX] = vp_pro_c.to(self.dtype)

                    W_c = _water_e(
                        all20.view(20,1,1).expand(20,C,L),
                        aa_j.view(1,1,L).expand(20,C,L),
                        d_all_c,
                        rho_q_vcb_c.view(1,C,1).expand(20,C,L),
                        rho_j_corr_c.unsqueeze(0).expand(20,C,L),
                    )
                    water_x[:, q0:q1] = (W_c * vp_e_c).sum(-1)

                    if use_elec:
                        safe_d_c    = d_vcb_c.clamp(min=1e-6)
                        self_mask_c = torch.zeros(C, L, device=device, dtype=torch.bool)
                        for ci, ri_val in enumerate(gly_c.tolist()):
                            self_mask_c[ci, ri_val] = True
                        elec_valid_c = ((~self_mask_c) & vm_j.unsqueeze(0)
                                        & (q_j_b.unsqueeze(0) != 0))
                        kernel_c    = (q_j_b.unsqueeze(0) / safe_d_c) * torch.exp(-safe_d_c / screening_len)
                        field_c     = k_pm * (kernel_c * elec_valid_c.to(self.dtype)).sum(-1)
                        elec_x[:, q0:q1] = q_all_local.view(20, 1) * field_c.view(1, C)

            E_abs = burial_x + water_x + elec_x  # [20, Q]

            freq_b   = freq[b].view(20, 1)
            mean_abs = (freq_b * E_abs).sum(0)
            std_abs  = (freq_b * (E_abs - mean_abs.unsqueeze(0))**2).sum(0).clamp(min=0).sqrt()

            fi_q = (mean_abs.unsqueeze(0) - E_abs) / std_abs.unsqueeze(0).clamp(min=1e-8)
            fi_q = torch.where(std_abs.unsqueeze(0) > 0, fi_q, torch.zeros_like(fi_q))

            fi_gly_mutant[b, gly_idx] = fi_q.T.to(self.dtype)  # [Q, 20]

        return fi_gly_mutant

    def _apply_gly_correction(self, batch, E_decoy, rho_vcb, rho_ca,
                              gly_burial_patched: bool = False):
        """
        Correct burial energies in the exhaustive mutation scan for the density
        mismatch that occurs when GLY ↔ non-GLY mutations change the effective
        centroid (CA ↔ virtual CB).

        Parameters
        ----------
        E_decoy            : [20, B, L]  energies computed with native density
        rho_vcb            : [B, L]      ideal-CB density for GLY positions
        rho_ca             : [B, L]      CA density for non-GLY positions
        gly_burial_patched : bool        when True, GLY-native burial is already
                             correct in E_decoy (patched in compute_singleresidue_stats_fast);
                             only Case-2 (non-GLY native + GLY decoy) correction is applied.

        Returns
        -------
        [20, B, L]  corrected energies
        """
        B, L = batch["aa_type"].shape
        device = batch["aa_type"].device
        all20  = torch.arange(20, device=device)
        GLY_IDX = 5

        is_gly_pos = (batch["aa_type"] == GLY_IDX)  # [B, L]
        rho_nat    = batch["residue_density"]         # water density [B, L]

        # Build per-(decoy, batch, position) corrected density
        is_gly_p  = is_gly_pos.unsqueeze(0).expand(20, B, L)
        is_gly_d  = (all20 == GLY_IDX).view(20, 1, 1).expand(20, B, L)
        rho_nat_e = rho_nat.unsqueeze(0).expand(20, B, L)
        rho_vcb_e = rho_vcb.unsqueeze(0).expand(20, B, L)
        rho_ca_e  = rho_ca.unsqueeze(0).expand(20, B, L)

        rho_corr = rho_nat_e.clone()

        if not gly_burial_patched:
            # Case 1: GLY native + non-GLY decoy — correct from rho_nat to rho_vcb.
            # Skip threshold-crossing positions (rho_nat and rho_vcb straddle the
            # solvation threshold): burial correction would flip the solvation regime.
            thresh = float(self.coefficients.threshold)
            gly_thresh_cross = is_gly_pos & ((rho_nat < thresh) != (rho_vcb < thresh))
            gly_cross_e = gly_thresh_cross.unsqueeze(0).expand(20, B, L)
            rho_corr = torch.where(is_gly_p & ~is_gly_d & ~gly_cross_e, rho_vcb_e, rho_corr)

        # Case 2: non-GLY native + GLY decoy → use CA-based density (GLY centroid is CA)
        rho_corr = torch.where(~is_gly_p & is_gly_d, rho_ca_e, rho_corr)

        all20_exp = all20.view(20, 1, 1).expand(20, B, L)

        burial_corr  = self.compute_burial_energy(batch, all20_exp, rho_corr)   # [20, B, L]
        burial_nat   = self.compute_burial_energy(batch, all20_exp, rho_nat_e)  # [20, B, L]
        delta_burial = burial_corr - burial_nat

        return E_decoy + delta_burial

    def compute_singleresidue_stats_fast(
        self,
        batch: dict,
        native_energy: torch.Tensor,
        rho_vcb: torch.Tensor | None = None,
        rho_ca:  torch.Tensor | None = None,
        zero_water_at_gly: bool = False,
        gly_water_corr: bool = False,
        vcb: torch.Tensor | None = None,
    ) -> tuple:
        """
        Exhaustive 20-AA single-residue decoy stats via the delta-energy formulation.
        Exact at D→∞ with sequence-composition weights.

        Returns
        -------
        stats  : [B, L, 2]  (weighted mean, weighted std of decoy energies)
        E_decoy: [20, B, L] per-(AA, position) decoy energies
        fi_gly : [B, L] or None  — corrected FI for GLY decoy at non-GLY positions
                 (non-NaN only at native non-GLY positions; None when gly_water_corr=False)
        """
        B, L   = batch["aa_type"].shape
        device = batch["aa_type"].device
        all20  = torch.arange(20, device=device)

        aa_native   = batch["aa_type"]
        rho         = batch["residue_density"]          # water density (sigma_wat)

        # Burial density: start from water density, then patch GLY positions with rho_vcb.
        # This matches FrustratometeR's complete_pdb behaviour (ideal CB placed at ALL GLY
        # positions before computing burial), without perturbing non-GLY burial energies.
        rho_burial = rho.clone()
        gly_burial_patched = False
        if gly_water_corr and rho_vcb is not None:
            is_gly_mask = (aa_native == 5)          # [B, L]
            if is_gly_mask.any():
                thresh = float(self.coefficients.threshold)
                # Maximum meaningful burial density: upper edge of highest burial bin.
                # rho_vcb >> 9 (from broad contact params) gives theta≈0 for all bins,
                # so patching with those values zeroes out burial energy — avoid that.
                max_burial_rho = float(self.coefficients.burial_ro_max[-1])  # 9.0
                # Only patch surface GLY positions (rho_nat < thresh) where the vCB
                # has meaningful burial contact density within the burial bins.
                # Buried/intermediate GLY (rho_nat >= thresh) are already correct.
                should_patch = (
                    is_gly_mask
                    & (rho < thresh)
                    & (rho_vcb >= thresh)
                    & (rho_vcb <= max_burial_rho)
                )
                if should_patch.any():
                    rho_burial[should_patch] = rho_vcb[should_patch]
                    gly_burial_patched = True

        rho_burial_exp = rho_burial.unsqueeze(0).expand(20, B, L)
        all20_exp  = all20.view(20, 1, 1).expand(20, B, L)
        burial_all = self.compute_burial_energy(batch, all20_exp, rho_burial_exp)
        burial_nat = self.compute_burial_energy(
            batch, aa_native.unsqueeze(0), rho_burial.unsqueeze(0)
        )
        delta_burial = burial_all - burial_nat

        field   = self.compute_sr_elec_field(batch)
        q_all   = self.charge_lookup[all20].to(self.dtype)
        q_nat   = self.charge_lookup[aa_native].to(self.dtype)
        delta_q = q_all.view(20, 1, 1) - q_nat.unsqueeze(0)
        delta_elec = delta_q * field.unsqueeze(0)

        delta_water = self.compute_sr_water_delta(batch, aa_native, rho)

        if zero_water_at_gly:
            gly_mask = (aa_native == 5).unsqueeze(0)  # [1, B, L]
            delta_water = delta_water.masked_fill(gly_mask, 0.0)

        fi_gly = None
        if gly_water_corr:
            delta_water = delta_water + self._compute_gly_water_correction(
                batch, rho_vcb=rho_vcb, vcb=vcb
            )

        E_decoy = native_energy.unsqueeze(0) + delta_burial + delta_water + delta_elec

        if rho_vcb is not None and rho_ca is not None:
            E_decoy = self._apply_gly_correction(
                batch, E_decoy, rho_vcb, rho_ca,
                gly_burial_patched=gly_burial_patched,
            )

        decoy_dist = getattr(self.coefficients, 'tert_frust_decoy_distribution', 'sequence')
        # back-compat: old bool → string
        if getattr(self.coefficients, 'tert_frust_exclude_native', False) and decoy_dist == 'sequence':
            decoy_dist = 'sequence_exclude_native'

        weights = decoy_aa_weights(aa_native, batch["seq_mask"], aa_native, decoy_dist, self.dtype)  # [B, L, 20]
        stats = decoy_statistics(E_decoy, weights.permute(2, 0, 1), dim=0)  # [B, L, 2]
        return stats, E_decoy, fi_gly

    # ------------------------------------------------------------------
    # Forward and output methods (used by SingleResidueFrustratometer)
    # ------------------------------------------------------------------

    def forward(self, batch):
        B = batch["aa_type"].shape[0]
        if B == 1:
            return self._forward_single(batch)
        if self.use_c_rand:
            results = [self._forward_single(self._slice_batch(batch, i)) for i in range(B)]
            return self._merge_results(results)
        return self._forward_single(batch)

    def _forward_single(self, batch):
        with torch.no_grad():
            self.compute_densities(batch)
            fi, nat_e, stats, sr_metrics, sr_losses, sr_mutations = \
                self.compute_tert_frust_singleresidue(batch)
        result = {
            "singleresidue":               fi,
            "_singleresidue_native_energy": nat_e,
            "_singleresidue_decoy_stats":   stats,
        }
        if sr_metrics   is not None: result["_singleresidue_metrics"]   = sr_metrics
        if sr_losses    is not None: result["_singleresidue_losses"]    = sr_losses
        if sr_mutations is not None:
            result["_singleresidue_mutations"] = sr_mutations
            mut_stats = sr_mutations.get("_mutation_stats") or sr_mutations.get("stats")
            if mut_stats is not None:
                result["_singleresidue_mutation_stats"] = mut_stats
            if sr_mutations.get("_gly_mutation_fi") is not None:
                result["_singleresidue_gly_fi"] = sr_mutations["_gly_mutation_fi"]
            if sr_mutations.get("_gly_native_fi") is not None:
                result["_singleresidue_gly_native_fi"] = sr_mutations["_gly_native_fi"]
        return result

    def _write_extra_outputs(self, batch, result, frust_bases, names, done_dirs,
                             residue_index, chain_index, output_dir):
        if self.run_dmfs:
            self._write_mutation_files(batch, result, frust_bases, names,
                                       residue_index, chain_index)
            if self.draw_figure:
                draw_mutation_scanning(batch, result, frust_bases)

    def _write_mutation_files(self, batch, result, frust_bases, names,
                              residue_index, chain_index):
        """Write per-residue × per-AA mutation energy tables."""
        from pathlib import Path
        from ..structure.residue_constants import ID_TO_AA
        from ..data.io import read_mutations_data
        from ..frustratometer import _find_mutations_data_dir
        import pandas as pd
        from ..structure.protein import PDB_CHAIN_IDS as _PDB_CHAIN_IDS

        sr_mut = result.get("_singleresidue_mutations")
        if sr_mut is None:
            return

        decoy_stats = result.get("_singleresidue_mutation_stats",
                                 result.get("_singleresidue_decoy_stats"))
        sf_raw = batch.get("structure_file")

        GLY_IDX = 5
        gly_fi_all        = result.get("_singleresidue_gly_fi")
        gly_native_fi_all = result.get("_singleresidue_gly_native_fi")
        _cid_map_list     = batch.get("chain_id_map")

        for i, (frust_base, name) in enumerate(zip(frust_bases, names)):
            mask    = batch["seq_mask"][i].bool()
            aa      = batch["aa_type"][i, mask].cpu()
            res_nums = (residue_index[i, mask].cpu().numpy() if residue_index is not None
                        else np.arange(1, mask.sum().item() + 1))
            _cid_map_i = _cid_map_list[i] if _cid_map_list is not None else None
            chain_ids = ([(_cid_map_i[int(c)] if _cid_map_i else _PDB_CHAIN_IDS[int(c)])
                          for c in chain_index[i, mask].cpu().tolist()]
                         if chain_index is not None else ["A"] * mask.sum().item())

            mut_e    = sr_mut["energies"][i, mask].cpu().float()
            nat_e_sr = result["_singleresidue_native_energy"][i, mask].cpu().float()

            d_mean = d_std = None
            if decoy_stats is not None:
                ds     = decoy_stats[i, mask].cpu().float()
                d_mean = ds[:, 0]
                d_std  = ds[:, 1]

            gly_fi_masked = None
            if gly_fi_all is not None:
                gly_fi_masked = gly_fi_all[i, mask].cpu().float()

            gly_native_fi_masked = None
            if gly_native_fi_all is not None:
                gly_native_fi_masked = gly_native_fi_all[i, mask].cpu().float()

            rows = []
            for r in range(mut_e.shape[0]):
                aa_nat = ID_TO_AA.get(int(aa[r]), "X")
                is_nat_gly = (int(aa[r]) == GLY_IDX)
                for d in range(20):
                    if is_nat_gly and gly_native_fi_masked is not None:
                        fi = float(gly_native_fi_masked[r, d])
                    elif (d == GLY_IDX and not is_nat_gly
                            and gly_fi_masked is not None
                            and not torch.isnan(gly_fi_masked[r])):
                        fi = float(gly_fi_masked[r])
                    else:
                        fi = (float((d_mean[r] - mut_e[r, d]) / d_std[r])
                              if d_mean is not None else float("nan"))
                    rows.append({
                        "Res":          int(res_nums[r]),
                        "ChainRes":     chain_ids[r],
                        "AA_native":    aa_nat,
                        "AA_mutant":    ID_TO_AA.get(d, "X"),
                        "NativeEnergy": float(nat_e_sr[r]),
                        "MutantEnergy": float(mut_e[r, d]),
                        "FrstIndex":    fi,
                    })
            out_path = f"{frust_base}.{name}_singleresidue_mutations"
            pd.DataFrame(rows).to_csv(out_path, sep=" ", index=False)

            if d_mean is None:
                continue
            sf = (sf_raw[i] if isinstance(sf_raw, (list, tuple)) else sf_raw)
            if sf is None:
                continue
            ref_dir = _find_mutations_data_dir(Path(sf))
            if ref_dir is None:
                continue
            try:
                ref_df = read_mutations_data(ref_dir)
                our_df = pd.DataFrame(rows)
                merged = our_df.merge(ref_df, left_on=["Res", "ChainRes", "AA_mutant"],
                                      right_on=["Res", "ChainRes", "AA"])
                if len(merged) >= 2:
                    r2, n = _pearson_r2(merged["FrstIndex_x"].values,
                                        merged["FrstIndex_y"].values)
                    stem = Path(frust_base).name
                    print(f"[mutations R²] {stem}: r²={r2:.4f}  n={n}")
            except Exception as exc:
                print(f"[mutations R²] could not compare {frust_base}: {exc}")


# ---------------------------------------------------------------------------
# Lazy SingleResidueFrustratometer (defers FrustratometerBase / lightning)
# ---------------------------------------------------------------------------

_SingleResidueFrustratometer = None


def _get_SingleResidueFrustratometer():
    global _SingleResidueFrustratometer
    if _SingleResidueFrustratometer is not None:
        return _SingleResidueFrustratometer
    from ..base import FrustratometerBase

    class SingleResidueFrustratometer(FrustratometerBase):
        """All single-residue frustration logic.  Computation via SingleResidueMixin;
        Lightning training wrapper via FrustratometerBase.

        AWSEM already holds every mode mixin, so the methods SingleResidueMixin shares
        with other classes in the MRO are taken from it explicitly."""

        forward               = SingleResidueMixin.forward
        _forward_single       = SingleResidueMixin._forward_single
        _write_extra_outputs  = SingleResidueMixin._write_extra_outputs
        _write_mutation_files = SingleResidueMixin._write_mutation_files

        def __init__(self, coefficients: Coefficients, output_dir=None,
                     draw_figure: bool = False,
                     dtype: torch.dtype = torch.float32):
            coefficients = dataclasses.replace(coefficients, mode="singleresidue")
            super().__init__(coefficients, output_dir=output_dir, dtype=dtype)
            self.draw_figure = draw_figure

        def _check_r_threshold(self, batch, result, r_min: float = 0.99, acc_min: float = 0.9):
            mode_cfg = [
                ("singleresidue", "singleresidue", "single_frust_true",
                 "single_frust_mask", self.coefficients.frustration_thresholds[0]),
            ]
            self._run_mode_check(batch, result, mode_cfg, r_min, acc_min)

    _SingleResidueFrustratometer = SingleResidueFrustratometer
    return _SingleResidueFrustratometer


def __getattr__(name):
    if name == "SingleResidueFrustratometer":
        return _get_SingleResidueFrustratometer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--structure', '-s', required=True, multiple=True,
              help='Structure file(s), directory, or text file of paths [PDB/mmCIF]')
@click.option('--output_dir', '-o', default=None,
              help='Directory for output files.')
@click.option('--ndecoys', '-n', default=None, type=int)
@click.option('--cutoff', default=None, type=float)
@click.option('--min_seq_dist', default=None, type=int)
@click.option('--config', '-c', default=None, type=click.Path(exists=True))
@click.option('--set', 'overrides', multiple=True, metavar='KEY=VALUE')
@click.option('--compare_to_original', is_flag=True, default=False)
@click.option('--run_dmfs', is_flag=True, default=False,
              help='Write per-residue×per-AA mutation energy tables.')
@click.option('--draw_figure', is_flag=True, default=False,
              help='Draw mutation scanning heatmap (requires --run_dmfs).')
@click.option('--use_c_rand', is_flag=True, default=False)
@click.option('--dtype', default='float32',
              type=click.Choice(['float32', 'float64'], case_sensitive=False))
@click.option('--no_electrostatics', is_flag=True, default=False)
@click.option('--wandb', 'use_wandb', is_flag=True, default=False)
@click.option('--wandb_project', default='frustratometer')
@click.option('--wandb_run_name', default='singleresidue_baseline')
@click.option('--wandb_offline', is_flag=True, default=False)
@click.option('--batch_size', '-b', default=1, type=int)
@click.option('--num_workers', default=4, type=int)
@click.option('--use_completed_pdb', is_flag=True, default=False)
def cli(structure, output_dir, ndecoys, cutoff, min_seq_dist,
        config, overrides, compare_to_original, use_completed_pdb, use_c_rand, dtype,
        no_electrostatics, run_dmfs, draw_figure, use_wandb, wandb_project,
        wandb_run_name, wandb_offline, batch_size, num_workers):
    cfg_dict = {"mode": "singleresidue"}
    if config:
        import yaml
        with open(config) as fh:
            cfg_dict.update(yaml.safe_load(fh) or {})
    cfg_dict["mode"] = "singleresidue"
    if ndecoys is not None:          cfg_dict["tert_frust_ndecoys"] = ndecoys
    if cutoff is not None:           cfg_dict["tert_frust_cutoff"]  = cutoff
    if min_seq_dist is not None:     cfg_dict["min_seq_distance"]   = min_seq_dist
    if run_dmfs:               cfg_dict["run_dmfs"]      = True
    if no_electrostatics:            cfg_dict["use_electrostatics"] = False
    for kv in overrides:
        if '=' not in kv:
            raise click.BadParameter(f"KEY=VALUE required, got: {kv!r}")
        k, raw = kv.split('=', 1)
        cfg_dict[k] = Coefficients._parse_cli_value(raw)
    cfg_dict["mode"] = "singleresidue"

    coefficients = Coefficients.from_dict(cfg_dict)
    torch_dtype  = torch.float64 if dtype == 'float64' else torch.float32

    from torch.utils.data import DataLoader
    from lightning.pytorch import Trainer
    from ..data.dataset import FrustratometerDataset, collate_frustratometer
    SingleResidueFrustratometer = _get_SingleResidueFrustratometer()

    dataset = FrustratometerDataset(structure, coefficients,
                                   compare_to_original=compare_to_original,
                                   use_completed_pdb=use_completed_pdb,
                                   dtype=torch_dtype)
    model = SingleResidueFrustratometer(coefficients, output_dir=output_dir,
                                        draw_figure=draw_figure, dtype=torch_dtype)
    model.use_c_rand = use_c_rand
    if torch.cuda.is_available():
        model = model.to("cuda")

    logger = True
    if use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        import os
        if wandb_offline:
            os.environ.setdefault("WANDB_MODE", "offline")
        logger = WandbLogger(project=wandb_project, name=wandb_run_name, offline=wandb_offline)

    trainer = Trainer(logger=logger)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=collate_frustratometer,
                            prefetch_factor=2 if num_workers > 0 else None,
                            persistent_workers=num_workers > 0,
                            pin_memory=torch.cuda.is_available(), shuffle=False)

    if compare_to_original:
        trainer.test(model, dataloader)
    else:
        trainer.predict(model, dataloader)


if __name__ == "__main__":
    cli()
