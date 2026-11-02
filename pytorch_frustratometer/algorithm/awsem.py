from pathlib import Path
from typing import Optional, Union

import click
import torch
import torch.nn.functional as F
import numpy as np

from ..structure.residue_constants import CHARGES as charges
from ..utils.utils import frustration_loss as _frustration_loss

# Shared utilities live in _utils.py; re-exported here for backward compatibility.
from ..utils.utils import (
    index_to_pseudo_logits,
    compute_frustration_metrics,
)


class AWSEM(torch.nn.Module):
    """
    PyTorch implementation of the Frustratometer energy model.

    This class computes four types of frustration metrics for protein
    structures using AWSEM-style statistical potentials:

    - Single-residue frustration
    - Mutational frustration
    - Configurational frustration
    - Smooth frustration

    The implementation is fully vectorized and supports batch
    decoy evaluation for efficient computation.

    Parameters
    ----------
    aa_type : torch.IntTensor
        Residue identity indices for the protein sequence.
        Shape: [N_res]. Values must be in [0, 19].

    edge_index : torch.IntTensor
        Pairwise contact edges used for interaction energies.
        Shape: [2, N_edges].

    distances : torch.FloatTensor
        Pairwise distances corresponding to edges.
        Shape: [N_edges] or [N_res, N_res].

    phospho_sites : Optional[torch.IntTensor]
        Indices of phosphorylated residues.

    water_flag : bool
        If True, use water-mediated interaction potentials.

    n_decoys : int
        Number of sampled decoys: configurational frustration, single-residue with
        use_c_rand and mutational frustration with use_c_rand.

    mode : list[str] | tuple[str] | str
        Frustration modes to compute. Options are:
        "singleresidue", "mutational", "configurational", "smooth".

    run_dmfs : bool
        If True, return decoy amino acid identities together with results.
    """
    def __init__(
        self,
        coefficients,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.coefficients = coefficients
        self._dtype = dtype

        self.all_modes = ["singleresidue", "mutational", "configurational", "configurational_exact", "smooth"]

        mode = self.coefficients.mode
        if isinstance(mode, str):
            if mode == "all":
                mode = ["singleresidue", "mutational", "configurational"]
            else:
                mode = [mode]

        assert isinstance(mode, (list, tuple))
        assert all(m in self.all_modes for m in mode)

        self.mode = mode
        self.run_dmfs = self.coefficients.run_dmfs
        self.n_decoys = self.coefficients.tert_frust_ndecoys
        self.water_flag = True
        self.phospho_flag = False
        self.use_c_rand = False  # set to True to exactly reproduce C++ output (slow); forces float64

        data_dir = Path(__file__).parent.parent / "data"

        AWSEM_ORDER = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        YOUR_ORDER  = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

        # --- Mapping from YOUR order → AWSEM index ---
        your_to_awsem = torch.tensor([AWSEM_ORDER.index(aa) for aa in YOUR_ORDER], dtype=torch.long)

        burial_gamma_raw = np.loadtxt(str(data_dir / "burial_gamma.dat"))
        burial_gamma_raw = torch.from_numpy(burial_gamma_raw).to(dtype)

        # Remap to YOUR alphabetical order
        burial_gamma = torch.ones((21,3)).to(dtype)
        burial_gamma[:20, :] = burial_gamma_raw[your_to_awsem, :]
        self.register_buffer("burial_gamma", burial_gamma)

        # Water interaction gamma
        if self.water_flag:
            water_gamma_raw = np.loadtxt(str(data_dir / "water_gamma.dat"))  # shape (420, 2)

            water_gamma_raw = torch.from_numpy(water_gamma_raw).reshape(self.coefficients.n_wells, 210, 2).to(dtype)
            water_gamma_raw *= self.coefficients.k_water

            # Build upper triangle indices
            i_idx, j_idx = torch.triu_indices(20, 20)  # each length 210

            water_gamma_20 = torch.zeros((self.coefficients.n_wells, 20, 20, 2)).to(dtype)
            water_gamma_20[:, i_idx, j_idx, :] = water_gamma_raw  # upper triangle
            water_gamma_20[:, j_idx, i_idx, :] = water_gamma_raw  # mirror to lower

            # Remap rows and columns to YOUR alphabetical order
            water_gamma_20 = water_gamma_20[:, your_to_awsem, :][:, :, your_to_awsem, :]

            water_gamma = torch.ones((self.coefficients.n_wells, 21, 21, 2)).to(dtype)
            water_gamma[:, :20, :20, :] = water_gamma_20
        else:
            water_gamma = torch.ones((self.coefficients.n_wells, 21, 21, 2)).to(dtype)
        
        self.register_buffer("water_gamma", water_gamma)

        # Register charge lookup as buffer so it moves with .to(device)
        self.register_buffer("charge_lookup", torch.IntTensor(charges))
        
    @property
    def dtype(self) -> torch.dtype:
        """float64 when use_c_rand is set (required for C++ reproducibility), else self._dtype."""
        return torch.float64 if self.use_c_rand else self._dtype

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _b_idx(self, B: int, E: int, device) -> torch.Tensor:
        """[B, E] batch index tensor for advanced indexing."""
        return torch.arange(B, device=device)[:, None].expand(-1, E)

    def _gather_node(
        self,
        node_features: torch.Tensor,  # [B, L] or [B, L, D]
        edge_index: torch.Tensor,      # [B, 2, E]
    ):
        """
        Gather node features for src and dst of each edge.

        Returns:
            src_feats: [B, E] or [B, E, D]
            dst_feats: [B, E] or [B, E, D]
        """
        B, _, E = edge_index.shape
        b_idx = self._b_idx(B, E, edge_index.device)
        src = edge_index[:, 0, :]  # [B, E]
        dst = edge_index[:, 1, :]  # [B, E]
        return node_features[b_idx, src], node_features[b_idx, dst]

    def _gather_edge_distances(
        self,
        distance: torch.Tensor,    # [B, L, L]
        edge_index: torch.Tensor,  # [B, 2, E]
    ) -> torch.Tensor:             # [B, E]
        B, _, E = edge_index.shape
        b_idx = self._b_idx(B, E, edge_index.device)
        return distance[b_idx, edge_index[:, 0, :], edge_index[:, 1, :]]


    def forward(self, batch):
        """
        Compute requested frustration modes.

        Returns
        -------
        dict
            Dictionary containing requested frustration metrics.
            Keys may include:

            - "singleresidue"
            - "mutational"
            - "configurational"
            - "smooth"
        """
        
        result = {}
        with torch.no_grad():
            if "singleresidue" in self.mode or "smooth" in self.mode:
                fi, nat_e, stats, sr_metrics, sr_losses, sr_mutations = \
                    self.compute_tert_frust_singleresidue(batch)
                # smooth is derived from the single-residue FI, so keep it in the result even when
                # only "smooth" was requested: the structure (.pdb with B-factors) and the .pml
                # files that load it are written by the single-residue writer.
                if "singleresidue" in self.mode or "smooth" in self.mode:
                    result["singleresidue"] = fi
                    result["_singleresidue_native_energy"] = nat_e
                    result["_singleresidue_decoy_stats"]   = stats
                    if sr_metrics is not None:
                        result["_singleresidue_metrics"] = sr_metrics
                    if sr_losses is not None:
                        result["_singleresidue_losses"] = sr_losses
                    if sr_mutations is not None:
                        result["_singleresidue_mutations"] = sr_mutations
                if "smooth" in self.mode:
                    # F_ij^smooth = -(Fi + Fj)  (Guan et al. 2024, PNAS doi:10.1073/pnas.2410662121)
                    fi_flat = fi.squeeze(-1) if fi.dim() == 3 else fi  # [B, L]
                    fi_src, fi_dst = self._gather_node(fi_flat, batch["edge_index"])
                    result["smooth"] = -(fi_src + fi_dst)
            if "configurational" in self.mode or "configurational_exact" in self.mode or "mutational" in self.mode:
                result.update(self.compute_tert_frust(batch))
        return result

    # ------------------------------------------------------------------
    # Mutational / configurational frustration
    # ------------------------------------------------------------------
    
    def compute_tert_frust(self, batch):
        """
        Compute pairwise frustration metrics.

        This function evaluates mutational and configurational frustration
        for all interacting residue pairs.

        Returns
        -------
        dict
            Dictionary containing frustration indices for each pair.
        """
        native_energy = self.compute_native_ixn(batch)


        decoy_results = self.compute_decoy_ixns(batch)

        # Native aa types at each edge: [B, E]
        aa_src, aa_dst = self._gather_node(batch["aa_type"], batch["edge_index"])
        base_aa = torch.stack([aa_src, aa_dst], dim=0)  # [2, B, E]

        out = {}
        if "mutational" in self.mode:
            fi_mut, nat_mut, mut_stats, mut_metrics, mut_losses, mut_mutations = \
                self.compute_tert_frust_mutational(batch)
            out["_mutational_native_energy"] = nat_mut
            out["_mutational_decoy_stats"]   = mut_stats
            out["mutational"] = fi_mut
            if mut_mutations is not None:
                out["_mutational_mutations"] = mut_mutations
            if mut_metrics is not None:
                out["_mutational_metrics"] = mut_metrics
            if mut_losses is not None:
                out["_mutational_losses"] = mut_losses

        if "configurational_exact" in decoy_results:
            conf_stats = decoy_results["configurational_exact"]["stats"]  # [B, 2]
            fi_conf_exact = self.compute_frustration_index(
                native_energy["configurational_exact"], conf_stats.unsqueeze(1)
            )
            out["_configurational_exact_native_energy"] = native_energy["configurational_exact"]
            out["_configurational_exact_decoy_stats"]   = conf_stats
            out["configurational_exact"] = fi_conf_exact

        if "configurational" in decoy_results:
            conf_stats = decoy_results["configurational"]["stats"]  # [B, 2]
            # Unsqueeze to [B, 1, 2] so mean/std broadcast over the E edge dimension.
            fi_conf = self.compute_frustration_index(native_energy["configurational"], conf_stats.unsqueeze(1))

            out["_configurational_native_energy"] = native_energy["configurational"]
            out["_configurational_decoy_stats"]   = conf_stats

            out["configurational"] = fi_conf
            if self.run_dmfs:
                conf_mut = decoy_results["configurational"]
                if "energies" in conf_mut:
                    out["_configurational_mutations"] = {
                        "energies": conf_mut["energies"],
                        "res_type": conf_mut["res_type"],
                    }

            if "configurational_frust_true" in batch and "configurational_frust_mask" in batch:
                true_conf  = batch["configurational_frust_true"].to(fi_conf.device)
                mask_conf  = batch["configurational_frust_mask"].to(fi_conf.device).bool()
                conf_metrics = compute_frustration_metrics(
                    fi_conf, true_conf, mask_conf,
                    threshold=self.coefficients.frustration_thresholds[1],
                )
                out["_configurational_metrics"] = conf_metrics
                # FrustrAI-Fold-compatible losses
                pseudo_logits_conf = index_to_pseudo_logits(
                    fi_conf, self.coefficients.frustration_thresholds[1]
                )  # [B, E, 3]
                welltype_conf = batch.get("configurational_welltype")
                idx_loss_conf, cls_loss_conf = _frustration_loss(
                    mode=2,
                    pred_frustration_index=fi_conf.unsqueeze(-1),
                    pred_frustration_class=pseudo_logits_conf,
                    true_frustration_index=true_conf,
                    seq_mask=mask_conf,
                    welltype=welltype_conf,
                    stage="test",
                )
                out["_configurational_losses"] = (idx_loss_conf.item(), cls_loss_conf.item())

        return out

    def compute_native_ixn(self, batch):
        """
        Compute native interaction energies.

        Returns
        -------
        dict
            Dictionary with native energies for each frustration mode.
        """

        result = {}
        B, _, E = batch["edge_index"].shape
        device = batch["edge_index"].device

        # Gather per-edge quantities
        aa_src, aa_dst = self._gather_node(batch["aa_type"], batch["edge_index"])   # [B, E]
        rho_src, rho_dst = self._gather_node(batch["residue_density"], batch["edge_index"])  # [B, E]
        edge_distances = self._gather_edge_distances(batch["distance"], batch["edge_index"])  # [B, E]

        aa_type  = torch.stack([aa_src,  aa_dst],  dim=0)   # [2, B, E]
        density  = torch.stack([rho_src, rho_dst], dim=0)   # [2, B, E]

        src = batch["edge_index"][:, 0, :]  # [B, E]
        dst = batch["edge_index"][:, 1, :]  # [B, E]

        electrostatics = self.calculate_electrostatics(aa_src, aa_dst, edge_distances)  # [B, E]

        if "configurational" in self.mode or "configurational_exact" in self.mode:
            nat_conf = self.compute_configurational_ixn(
                batch,
                aa_type,
                edge_distances,
                density,
                electrostatics,
                return_decoy_energies=True,  # return raw [B, E] energy
                return_stats=False,
            )  # [B, E]
            if "configurational" in self.mode:
                result["configurational"] = nat_conf
            if "configurational_exact" in self.mode:
                result["configurational_exact"] = nat_conf

        return result

    def compute_decoy_ixns(self, batch):
        """
        Generate decoy interactions and compute their energies.

        Returns
        -------
        dict
            Dictionary containing decoy statistics and sampled amino acids.
        """
        from .configurational import sample_configurational_decoys

        results = {}

        if "configurational" in self.mode:
            decoy_distances, decoy_density, decoy_aa = sample_configurational_decoys(
                distance_matrix=batch["distance"],
                residue_density=batch["residue_density"],
                aa_type=batch["aa_type"],
                seq_mask=batch["seq_mask"],
                n_decoys=self.n_decoys,
                cutoff=self.coefficients.tert_frust_cutoff,
                use_c_rand=self.use_c_rand,
            )

            decoy_elec = self.calculate_electrostatics(decoy_aa[0], decoy_aa[1], decoy_distances)

            conf_stats = self._compute_energy(
                batch, decoy_aa, decoy_distances, decoy_density, decoy_elec,
                return_decoy_energies=self.run_dmfs, return_stats=True
            )

            if self.run_dmfs:
                results["configurational"] = {
                    "stats": conf_stats[1], "energies": conf_stats[0], "res_type": decoy_aa
                }
            else:
                results["configurational"] = {"stats": conf_stats}

        if "configurational_exact" in self.mode:
            from .configurational_exact import enumerate_configurational_decoys
            exact_stats = enumerate_configurational_decoys(self, batch)
            results["configurational_exact"] = {"stats": exact_stats}

        return results

    # ------------------------------------------------------------------
    # Energy terms
    # ------------------------------------------------------------------

    def compute_frustration_index(self, energy, stats):
        """stats: [..., 2] where [..., 0]=mean, [..., 1]=std."""
        std = stats[..., 1]
        fi = (stats[..., 0] - energy) / std.clamp(min=1e-8)
        # std==0 means all decoys gave identical energy → no information → FI=0 (neutral)
        return torch.where(std < 1e-8, torch.zeros_like(fi), fi)

    def compute_burial_energy(
        self,
        batch,
        res_type,
        rho
    ):
        """
        Args:
            res_type: [B, L] or [2, B, E] or [B, n_decoys]
            rho:      same shape as res_type
        Returns:
            burial_energy: same shape as res_type (float64)
        """
        device = res_type.device
        rho = rho.to(self.dtype)

        t_min = torch.tanh(self.coefficients.burial_kappa * (rho.unsqueeze(-1) - self.coefficients.burial_ro_min.to(device, dtype=self.dtype)))  # [..., L, 3]
        t_max = torch.tanh(self.coefficients.burial_kappa * (self.coefficients.burial_ro_max.to(device, dtype=self.dtype) - rho.unsqueeze(-1)))  # [..., L, 3]

        # C++ uses (t_min + t_max), NOT 0.25*(1+t_min)*(1+t_max)
        theta = t_min + t_max  # [..., L, 3]

        burial_gamma = self.get_burial_gamma(batch, res_type).to(device, dtype=self.dtype)  # [..., L, 3]
        burial_energy = -0.5 * self.coefficients.k_burial * (burial_gamma * theta).sum(dim=-1)

        return burial_energy

    def compute_water_energy(
        self,
        batch,
        distances, 
        density1,
        density2, 
        aa_type1,
        aa_type2
    ):
        """
        Fully vectorized water energy computation for all edges at once.
        
        edge_index: [2, N_edges] tensor of interacting residue indices
        distances: [N_edges] tensor of distances
        residue_density: [N_res] tensor of per-residue ρ
        aa_type: [N_res] tensor of residue types
        water_gamma: [n_wells, 20, 20, 2] gamma table
        """
        device = distances.device
        distances = distances.to(self.dtype)
        density1 = density1.to(self.dtype)
        density2 = density2.to(self.dtype)

        # Get leading dimensions from aa_type1
        leading_shape = aa_type1.shape[:-1]  # Everything except last dim (edges)
        n_edges = aa_type1.shape[-1]

        sigma_wat_i = (1.0 - torch.tanh(self.coefficients.water_kappa_sigma * (density1 - self.coefficients.threshold)))
        sigma_wat_j = (1.0 - torch.tanh(self.coefficients.water_kappa_sigma * (density2 - self.coefficients.threshold)))
        sigma_wat = 0.25 * sigma_wat_i * sigma_wat_j
        sigma_prot = 1.0 - sigma_wat

        # Water gamma for each pair
        water_gamma = self.get_water_gamma(batch)  # [n_wells, 20, 20, 2]
        
        # Flatten leading dims for indexing, then reshape back
        aa_type1_flat = aa_type1.reshape(-1, n_edges)  # [D*B, N_edges]
        aa_type2_flat = aa_type2.reshape(-1, n_edges)
        
        # Index and reshape to preserve leading dims
        water_gamma_direct_flat = water_gamma[0, aa_type1_flat, aa_type2_flat, :].mean(dim=-1)  # [D*B, N_edges]
        water_gamma_prot_mediated_flat = water_gamma[1, aa_type1_flat, aa_type2_flat, 0]  # [D*B, N_edges]
        water_gamma_wat_mediated_flat = water_gamma[1, aa_type1_flat, aa_type2_flat, 1]  # [D*B, N_edges]
        
        # Reshape back to include leading dims
        water_gamma_direct = water_gamma_direct_flat.reshape(*leading_shape, n_edges)
        water_gamma_prot_mediated = water_gamma_prot_mediated_flat.reshape(*leading_shape, n_edges)
        water_gamma_wat_mediated = water_gamma_wat_mediated_flat.reshape(*leading_shape, n_edges)

        sigma_gamma_direct = water_gamma_direct
        sigma_gamma_mediated = sigma_prot * water_gamma_prot_mediated + sigma_wat * water_gamma_wat_mediated

        # Theta functions for direct and mediated wells
        # Direct well
        t_min_direct = torch.tanh(self.coefficients.water_kappa * (distances - self.coefficients.well_rmin[0].to(device, dtype=self.dtype)))
        t_max_direct = torch.tanh(self.coefficients.water_kappa * (self.coefficients.well_rmax[0].to(device, dtype=self.dtype) - distances))
        theta_direct = 0.25 * (1 + t_min_direct) * (1 + t_max_direct)

        # Mediated well
        t_min_mediated = torch.tanh(self.coefficients.water_kappa * (distances - self.coefficients.well_rmin[1].to(device, dtype=self.dtype)))
        t_max_mediated = torch.tanh(self.coefficients.water_kappa * (self.coefficients.well_rmax[1].to(device, dtype=self.dtype) - distances))
        theta_mediated = 0.25 * (1 + t_min_mediated) * (1 + t_max_mediated)

        # Total water energy
        water_energy = -(sigma_gamma_direct * theta_direct + sigma_gamma_mediated * theta_mediated)

        return water_energy
    
    def calculate_electrostatics(
        self,
        aa_type1,
        aa_type2,
        distances,
    ):
        """
        Args:
            aa_type1/2: int tensors, any shape
            distances:  float tensor, same shape
        Returns:
            V: same shape as distances
        """
        if not self.coefficients.use_electrostatics:
            return torch.zeros_like(distances, dtype=self.dtype)

        q_i = self.charge_lookup[aa_type1]
        q_j = self.charge_lookup[aa_type2]

        charged_mask = (q_i != 0) & (q_j != 0)

        k = self.get_k_ij(q_i, q_j)
        # Clamp to avoid 0/0=NaN for padded entries (distance=0, q=0).
        # charged_mask zeros the result for neutral pairs anyway.
        distances_d = distances.to(self.dtype).clamp(min=1e-6)

        V = k * q_i.to(self.dtype) * q_j.to(self.dtype) / distances_d * \
            torch.exp(-self.coefficients.k_screening * distances_d / self.coefficients.screening_length)

        return V * charged_mask.to(self.dtype)

    def get_k_ij(self, q_i, q_j):
        if (self.coefficients.k_PlusMinus == self.coefficients.k_PlusPlus
                == self.coefficients.k_MinusMinus):
            return self.coefficients.k_PlusMinus

        k = torch.full_like(q_i, self.coefficients.k_PlusMinus)  # default opposite sign
        k = torch.where((q_i > 0) & (q_j > 0), 
                        torch.full_like(k, self.coefficients.k_PlusPlus), k)
        k = torch.where((q_i < 0) & (q_j < 0), 
                        torch.full_like(k, self.coefficients.k_MinusMinus), k)
        return k

    def get_burial_gamma(self, batch, res_type):
        if not self.phospho_flag or "phospho_map" not in batch:
            return self.burial_gamma[res_type.long()]
        else:
            res_type_ = res_type.clone()
            res_type_[batch["phopho_map"]] = 6 # use burial gamma for glutamine instead of serine
            return self.burial_gamma[res_type_]

    def get_water_gamma(self, batch):
        if not self.phospho_flag or "phospho_map" not in batch:
            return self.water_gamma
        else:
            water_gamma = self.water_gamma.clone()
            water_gamma[batch["phopho_map"]] = self.coefficients.phosph_water_gamma[batch["phopho_map"]]
            return water_gamma    


def _register_mode_mixins():
    from .singleresidue import SingleResidueMixin
    from .configurational import ConfigurationalMixin
    from .mutational import MutationalMixin
    import torch.nn as nn
    AWSEM.__bases__ = (SingleResidueMixin, ConfigurationalMixin, MutationalMixin, nn.Module)


_register_mode_mixins()

