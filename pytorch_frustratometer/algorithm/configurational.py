"""Configurational frustration — all configurational logic in one class."""
from __future__ import annotations
import dataclasses
from pathlib import Path

import click
import torch

from ..utils.utils import frustration_loss as _frustration_loss

from ..utils.utils import index_to_pseudo_logits, compute_frustration_metrics
from ..coefficients import Coefficients


# ---------------------------------------------------------------------------
# Free function (was in awsem.py)
# ---------------------------------------------------------------------------

def sample_configurational_decoys(
    distance_matrix: torch.Tensor,  # [B, L, L]
    residue_density: torch.Tensor,  # [B, L]
    aa_type: torch.Tensor,          # [B, L]
    seq_mask: torch.Tensor,         # [B, L]
    n_decoys: int,
    cutoff: float = 9.5,
    use_c_rand: bool = False,
    decoy_distribution: str = "sequence",
) -> tuple:
    """
    Sample configurational decoys — random (distance, density, aa) triples from
    within-cutoff residue pairs, matching C++ ordering.
    """
    B, L = aa_type.shape
    device = aa_type.device

    valid_mask = seq_mask.bool() & (aa_type < 20)
    seq_lens = valid_mask.sum(dim=1).long()

    if use_c_rand:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.srand(1)
        dist_cpu = distance_matrix.cpu()
        rho_cpu  = residue_density.cpu()
        aa_cpu   = aa_type.cpu()
        valid_indices = [torch.where(valid_mask[b])[0].cpu() for b in range(B)]
        lens = [len(vi) for vi in valid_indices]

        dec_dist = torch.zeros(B, n_decoys)
        dec_rho1 = torch.zeros(B, n_decoys)
        dec_rho2 = torch.zeros(B, n_decoys)
        dec_aa1  = torch.zeros(B, n_decoys, dtype=torch.long)
        dec_aa2  = torch.zeros(B, n_decoys, dtype=torch.long)

        for d in range(n_decoys):
            for b in range(B):
                n  = lens[b]
                vi = valid_indices[b]
                while True:
                    ri  = vi[libc.rand() % n].item()
                    rj  = vi[libc.rand() % n].item()
                    rij = dist_cpu[b, ri, rj].item()
                    if rij < cutoff and ri != rj:
                        break
                dec_dist[b, d] = rij
                dec_rho1[b, d] = rho_cpu[b, vi[libc.rand() % n].item()]
                dec_rho2[b, d] = rho_cpu[b, vi[libc.rand() % n].item()]
                dec_aa1[b, d]  = aa_cpu[b, vi[libc.rand() % n].item()]
                dec_aa2[b, d]  = aa_cpu[b, vi[libc.rand() % n].item()]

        decoy_distances = dec_dist.to(device)
        decoy_density   = torch.stack([dec_rho1.to(device), dec_rho2.to(device)], dim=0)
        decoy_aa        = torch.stack([dec_aa1.to(device),  dec_aa2.to(device)],  dim=0)
    else:
        b_idx   = torch.arange(B, device=device)[:, None].expand(-1, n_decoys)
        max_n   = int(seq_lens.max().item())
        vi_table = (~valid_mask).long().argsort(dim=1, stable=True)[:, :max_n]
        n_valid  = seq_lens.view(B, 1)

        def rand_res():
            u    = torch.rand(B, n_decoys, device=device)
            slot = (u * n_valid).long().clamp(max=n_valid - 1)
            return vi_table.gather(1, slot)

        off_diag     = ~torch.eye(L, device=device, dtype=torch.bool)
        valid_pair   = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        contact_mask = (distance_matrix < cutoff) & off_diag.unsqueeze(0) & valid_pair

        ri = rand_res(); rj = rand_res()
        decoy_distances = distance_matrix[b_idx, ri, rj]
        bad = (decoy_distances >= cutoff) | (ri == rj) | decoy_distances.isnan()
        if bad.any():
            for b in range(B):
                bad_b = bad[b]
                if bad_b.any():
                    ci, cj = contact_mask[b].nonzero(as_tuple=True)
                    idx = torch.randint(0, len(ci), (bad_b.sum().item(),), device=device)
                    decoy_distances[b, bad_b] = distance_matrix[b, ci[idx], cj[idx]]

        dec_rho1 = residue_density[b_idx, rand_res()]
        dec_rho2 = residue_density[b_idx, rand_res()]
        decoy_density = torch.stack([dec_rho1, dec_rho2], dim=0)

        if decoy_distribution.startswith("pseudocount"):
            from .decoy_statistics import decoy_aa_frequencies
            aa_probs = decoy_aa_frequencies(aa_type, seq_mask, "pseudocount", torch.float32)
            dec_aa1 = torch.stack([
                torch.multinomial(aa_probs[b], n_decoys, replacement=True)
                for b in range(B)
            ], dim=0)  # [B, n_decoys]
            dec_aa2 = torch.stack([
                torch.multinomial(aa_probs[b], n_decoys, replacement=True)
                for b in range(B)
            ], dim=0)
        else:
            dec_aa1 = aa_type[b_idx, rand_res()]
            dec_aa2 = aa_type[b_idx, rand_res()]
        decoy_aa = torch.stack([dec_aa1, dec_aa2], dim=0)

    return decoy_distances, decoy_density, decoy_aa


# ---------------------------------------------------------------------------
# Configurational Mixin — all configurational computation methods
# ---------------------------------------------------------------------------

class ConfigurationalMixin:
    """
    Mixin providing configurational frustration computation methods.
    Injected into AWSEM via __bases__ reassignment in awsem.py.
    """

    def _compute_energy(
        self,
        batch,
        res_type:       torch.Tensor,
        distances:      torch.Tensor,
        density:        torch.Tensor,
        electrostatics: torch.Tensor,
        return_decoy_energies: bool = False,
        return_stats: bool = True,
        native_edge_aa: tuple | None = None,  # (src_aa [B,E], dst_aa [B,E]) for a≠a' per-edge masking
    ):
        """Shared energy computation for configurational decoy path."""
        water_energy = self.compute_water_energy(
            batch,
            distances=distances,
            density1=density[0],
            density2=density[1],
            aa_type1=res_type[0],
            aa_type2=res_type[1],
        )
        burial_i = self.compute_burial_energy(batch, res_type[0], density[0])
        burial_j = self.compute_burial_energy(batch, res_type[1], density[1])

        total_energy = water_energy + burial_i + burial_j + electrostatics

        if return_decoy_energies and not return_stats:
            return total_energy

        if native_edge_aa is not None:
            # Per-edge exclusion: for each edge (b,e), exclude decoys where
            # decoy_aa[0]==native_src[b,e] OR decoy_aa[1]==native_dst[b,e].
            # res_type[0/1]: [B, n_decoys]; native_edge_aa: ([B,E], [B,E])
            nat_src, nat_dst = native_edge_aa  # [B, E]
            B, E = nat_src.shape
            # valid[b, e, d] = neither AA matches its native
            valid = (
                (res_type[0].unsqueeze(1) != nat_src.unsqueeze(-1)) &  # [B, E, D]
                (res_type[1].unsqueeze(1) != nat_dst.unsqueeze(-1))
            ).to(total_energy.dtype)  # [B, E, D]
            e_exp = total_energy.unsqueeze(1).expand(B, E, -1)  # [B, E, D]
            count = valid.sum(-1).clamp(min=1)
            mean_d = (e_exp * valid).sum(-1) / count
            var_d  = ((e_exp - mean_d.unsqueeze(-1)) ** 2 * valid).sum(-1) / count
            std_d  = var_d.clamp(min=0).sqrt()
            stats  = torch.stack([mean_d, std_d], dim=-1)  # [B, E, 2]
        else:
            stats = torch.stack([total_energy.mean(dim=-1), total_energy.std(dim=-1)], dim=-1)

        if return_decoy_energies:
            return total_energy, stats
        return stats

    def compute_configurational_ixn(
        self,
        batch,
        res_type:       torch.Tensor,
        distances:      torch.Tensor,
        density:        torch.Tensor,
        electrostatics: torch.Tensor,
        return_decoy_energies: bool = False,
        return_stats: bool = False,
    ):
        return self._compute_energy(
            batch, res_type, distances, density, electrostatics,
            return_decoy_energies=return_decoy_energies,
            return_stats=return_stats,
        )

    def forward(self, batch):
        B = batch["aa_type"].shape[0]
        if B == 1:
            return self._forward_single(batch)
        if self.use_c_rand:
            results = [self._forward_single(self._slice_batch(batch, i)) for i in range(B)]
            return self._merge_results(results)
        return self._forward_single(batch)

    def _build_gly_distance_pools(self, batch, cutoff):
        """
        Split within-cutoff pair distances into GLY-involving and non-GLY pools.

        GLY pairs use CA-based distances; non-GLY pairs use CB-based distances.
        When a decoy AA is GLY the energy function should see a CA-range distance,
        so we sample from the correct pool rather than the mixed overall pool.

        Returns:
            gly_dists   : list[B] of 1-D tensors — distances where ≥1 endpoint is GLY
            nongly_dists: list[B] of 1-D tensors — distances where no endpoint is GLY
        """
        GLY_IDX = 5
        B, L    = batch["aa_type"].shape
        device  = batch["aa_type"].device

        gly_dists    = []
        nongly_dists = []

        iu = torch.triu_indices(L, L, offset=1, device=device)  # upper-tri pairs

        for b in range(B):
            aa   = batch["aa_type"][b]
            vm   = batch["seq_mask"][b].bool()
            dist = batch["distance"][b]

            is_gly  = (aa == GLY_IDX) & vm
            pair_ok = vm[iu[0]] & vm[iu[1]] & (dist[iu[0], iu[1]] < cutoff)

            gly_pair  = pair_ok & (is_gly[iu[0]] | is_gly[iu[1]])
            ng_pair   = pair_ok & ~is_gly[iu[0]] & ~is_gly[iu[1]]

            gly_dists.append(dist[iu[0][gly_pair],  iu[1][gly_pair]])
            nongly_dists.append(dist[iu[0][ng_pair], iu[1][ng_pair]])

        return gly_dists, nongly_dists

    def _sample_configurational_decoys_gly_corrected(
        self, batch, n_decoys, cutoff, gly_dists, nongly_dists
    ):
        """
        Sample configurational decoys with GLY-stratified distance sampling.

        Instead of drawing distances from the full mixed pool (CA for GLY pairs,
        CB for others), this samples AAs first, then picks the distance from the
        pool that matches those AAs:
          decoy has ≥1 GLY → draw r from gly_dists   (CA-range distribution)
          decoy has no GLY → draw r from nongly_dists (CB-range distribution)

        Densities are drawn independently as in the original sampler.
        Returns (decoy_distances, decoy_density, decoy_aa) matching the shape
        convention of sample_configurational_decoys.
        """
        GLY_IDX = 5
        B, L    = batch["aa_type"].shape
        device  = batch["aa_type"].device
        dt      = batch["distance"].dtype

        aa_type    = batch["aa_type"]
        valid_mask = batch["seq_mask"].bool() & (aa_type < 20)
        seq_lens   = valid_mask.sum(dim=1).long()

        max_n    = int(seq_lens.max().item())
        vi_table = (~valid_mask).long().argsort(dim=1, stable=True)[:, :max_n]
        n_valid  = seq_lens.view(B, 1)
        b_idx    = torch.arange(B, device=device).view(B, 1).expand(B, n_decoys)

        def _rand_res():
            u    = torch.rand(B, n_decoys, device=device)
            slot = (u * n_valid).long().clamp(max=n_valid - 1)
            return vi_table.gather(1, slot)

        # Step 1: sample AA types first
        dec_aa1 = aa_type[b_idx, _rand_res()]   # [B, n_decoys]
        dec_aa2 = aa_type[b_idx, _rand_res()]

        # Step 2: sample distance from the correct pool per decoy
        decoy_distances = torch.zeros(B, n_decoys, device=device, dtype=dt)
        full_pool_cache = {}  # lazy fallback: whole within-cutoff set per batch item

        for b in range(B):
            involves_gly = (dec_aa1[b] == GLY_IDX) | (dec_aa2[b] == GLY_IDX)

            for flag, mask_b in [(True, involves_gly), (False, ~involves_gly)]:
                idx = mask_b.nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    continue
                pool = gly_dists[b] if flag else nongly_dists[b]
                if len(pool) == 0:   # pool empty (all-GLY or GLY-free protein) → fall back
                    if b not in full_pool_cache:
                        d = batch["distance"][b]
                        vm = batch["seq_mask"][b].bool()
                        full_pool_cache[b] = d[vm.unsqueeze(0) & vm.unsqueeze(1) & (d < cutoff)]
                    pool = full_pool_cache[b]
                ridx = torch.randint(0, len(pool), (len(idx),), device=device)
                decoy_distances[b, idx] = pool[ridx]

        # Step 3: densities sampled independently (same as original)
        rho = batch["residue_density"]
        dec_rho1 = rho[b_idx, _rand_res()]
        dec_rho2 = rho[b_idx, _rand_res()]

        return (decoy_distances,
                torch.stack([dec_rho1, dec_rho2], dim=0),
                torch.stack([dec_aa1,  dec_aa2],  dim=0))

    def _forward_single(self, batch):
        with torch.no_grad():
            self.compute_densities(batch)

            # Native energy
            aa_src, aa_dst   = self._gather_node(batch["aa_type"], batch["edge_index"])
            rho_src, rho_dst = self._gather_node(batch["residue_density"], batch["edge_index"])
            edge_dist        = self._gather_edge_distances(batch["distance"], batch["edge_index"])
            aa_type_pair     = torch.stack([aa_src, aa_dst], dim=0)
            density_pair     = torch.stack([rho_src, rho_dst], dim=0)
            elec             = self.calculate_electrostatics(aa_src, aa_dst, edge_dist)
            native_energy    = self.compute_configurational_ixn(
                batch, aa_type_pair, edge_dist, density_pair, elec,
                return_decoy_energies=True, return_stats=False)

            # Decoy energy — use GLY-stratified sampler when flag is set
            cutoff = self.coefficients.tert_frust_cutoff
            if getattr(self.coefficients, 'gly_pair_correction', False):
                gly_dists, nongly_dists = self._build_gly_distance_pools(batch, cutoff)
                decoy_dist, decoy_dens, decoy_aa = \
                    self._sample_configurational_decoys_gly_corrected(
                        batch, self.n_decoys, cutoff, gly_dists, nongly_dists)
            else:
                _decoy_dist_str = getattr(self.coefficients, 'tert_frust_decoy_distribution', 'sequence')
                if getattr(self.coefficients, 'tert_frust_exclude_native', False) and _decoy_dist_str == 'sequence':
                    _decoy_dist_str = 'sequence_exclude_native'
                decoy_dist, decoy_dens, decoy_aa = sample_configurational_decoys(
                    distance_matrix=batch["distance"],
                    residue_density=batch["residue_density"],
                    aa_type=batch["aa_type"],
                    seq_mask=batch["seq_mask"],
                    n_decoys=self.n_decoys,
                    cutoff=cutoff,
                    use_c_rand=self.use_c_rand,
                    decoy_distribution=_decoy_dist_str,
                )
            decoy_elec  = self.calculate_electrostatics(decoy_aa[0], decoy_aa[1], decoy_dist)
            _decoy_dist_str = getattr(self.coefficients, 'tert_frust_decoy_distribution', 'sequence')
            if getattr(self.coefficients, 'tert_frust_exclude_native', False) and _decoy_dist_str == 'sequence':
                _decoy_dist_str = 'sequence_exclude_native'
            native_edge_aa_excl = None
            if _decoy_dist_str.endswith('_exclude_native'):
                native_edge_aa_excl = (aa_src, aa_dst)  # [B, E] each
            conf_stats  = self._compute_energy(
                batch, decoy_aa, decoy_dist, decoy_dens, decoy_elec,
                return_decoy_energies=False, return_stats=True,
                native_edge_aa=native_edge_aa_excl)

            # conf_stats is [B, E, 2] when per-edge exclusion is active, [B, 2] otherwise
            if native_edge_aa_excl is not None:
                fi_conf = self.compute_frustration_index(native_energy, conf_stats)
            else:
                fi_conf = self.compute_frustration_index(native_energy, conf_stats.unsqueeze(1))

        result = {
            "configurational":                fi_conf,
            "_configurational_native_energy": native_energy,
            "_configurational_decoy_stats":   conf_stats,
        }

        if "configurational_frust_true" in batch and "configurational_frust_mask" in batch:
            true_conf  = batch["configurational_frust_true"].to(fi_conf.device)
            mask_conf  = batch["configurational_frust_mask"].to(fi_conf.device).bool()
            conf_metrics = compute_frustration_metrics(
                fi_conf, true_conf, mask_conf, threshold=self.coefficients.frustration_thresholds[1])
            result["_configurational_metrics"] = conf_metrics

            pseudo_logits = index_to_pseudo_logits(fi_conf, self.coefficients.frustration_thresholds[1])
            idx_loss, cls_loss = _frustration_loss(
                mode=2,
                pred_frustration_index=fi_conf.unsqueeze(-1),
                pred_frustration_class=pseudo_logits,
                true_frustration_index=true_conf,
                seq_mask=mask_conf,
                welltype=batch.get("configurational_welltype"),
                stage="test",
            )
            result["_configurational_losses"] = (idx_loss.item(), cls_loss.item())

        return result



# ---------------------------------------------------------------------------
# Lazy Frustratometer class (defers FrustratometerBase / lightning import)
# ---------------------------------------------------------------------------

_ConfigurationalFrustratometer = None


def _get_ConfigurationalFrustratometer():
    global _ConfigurationalFrustratometer
    if _ConfigurationalFrustratometer is not None:
        return _ConfigurationalFrustratometer
    from ..base import FrustratometerBase

    class ConfigurationalFrustratometer(FrustratometerBase):
        """All configurational frustration logic.  Computation via ConfigurationalMixin;
        Lightning training wrapper via FrustratometerBase.

        AWSEM already holds every mode mixin, so the methods ConfigurationalMixin shares
        with other classes in the MRO are taken from it explicitly."""

        forward         = ConfigurationalMixin.forward
        _forward_single = ConfigurationalMixin._forward_single

        def __init__(self, coefficients: Coefficients, output_dir=None,
                     dtype: torch.dtype = torch.float32):
            coefficients = dataclasses.replace(coefficients, mode="configurational")
            super().__init__(coefficients, output_dir=output_dir, dtype=dtype)

        def _check_r_threshold(self, batch, result, r_min: float = 0.99, acc_min: float = 0.9):
            mode_cfg = [
                ("configurational", "configurational", "configurational_frust_true",
                 "configurational_frust_mask", self.coefficients.frustration_thresholds[1]),
            ]
            self._run_mode_check(batch, result, mode_cfg, r_min, acc_min)

    _ConfigurationalFrustratometer = ConfigurationalFrustratometer
    return _ConfigurationalFrustratometer


def __getattr__(name):
    if name == "ConfigurationalFrustratometer":
        return _get_ConfigurationalFrustratometer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--structure', '-s', required=True, multiple=True,
              help='Structure file(s), directory, or text file of paths [PDB/mmCIF]')
@click.option('--output_dir', '-o', default=None)
@click.option('--ndecoys', '-n', default=None, type=int)
@click.option('--cutoff', default=None, type=float)
@click.option('--min_seq_dist', default=None, type=int)
@click.option('--config', '-c', default=None, type=click.Path(exists=True))
@click.option('--set', 'overrides', multiple=True, metavar='KEY=VALUE')
@click.option('--compare_to_original', is_flag=True, default=False)
@click.option('--use_c_rand', is_flag=True, default=False)
@click.option('--dtype', default='float32',
              type=click.Choice(['float32', 'float64'], case_sensitive=False))
@click.option('--no_electrostatics', is_flag=True, default=False)
@click.option('--wandb', 'use_wandb', is_flag=True, default=False)
@click.option('--wandb_project', default='frustratometer')
@click.option('--wandb_run_name', default='configurational_baseline')
@click.option('--wandb_offline', is_flag=True, default=False)
@click.option('--batch_size', '-b', default=1, type=int)
@click.option('--num_workers', default=4, type=int)
@click.option('--use_completed_pdb', is_flag=True, default=False)
def cli(structure, output_dir, ndecoys, cutoff, min_seq_dist,
        config, overrides, compare_to_original, use_completed_pdb, use_c_rand, dtype,
        no_electrostatics, use_wandb, wandb_project, wandb_run_name,
        wandb_offline, batch_size, num_workers):
    cfg_dict = {"mode": "configurational"}
    if config:
        import yaml
        with open(config) as fh:
            cfg_dict.update(yaml.safe_load(fh) or {})
    cfg_dict["mode"] = "configurational"
    if ndecoys is not None:      cfg_dict["tert_frust_ndecoys"] = ndecoys
    if cutoff is not None:       cfg_dict["tert_frust_cutoff"]  = cutoff
    if min_seq_dist is not None: cfg_dict["min_seq_distance"]   = min_seq_dist
    if no_electrostatics:        cfg_dict["use_electrostatics"] = False
    for kv in overrides:
        if '=' not in kv:
            raise click.BadParameter(f"KEY=VALUE required, got: {kv!r}")
        k, raw = kv.split('=', 1)
        cfg_dict[k] = Coefficients._parse_cli_value(raw)
    cfg_dict["mode"] = "configurational"

    coefficients = Coefficients.from_dict(cfg_dict)
    torch_dtype  = torch.float64 if dtype == 'float64' else torch.float32

    from torch.utils.data import DataLoader
    from lightning.pytorch import Trainer
    from ..data.dataset import FrustratometerDataset, collate_frustratometer
    ConfigurationalFrustratometer = _get_ConfigurationalFrustratometer()

    dataset = FrustratometerDataset(structure, coefficients,
                                    compare_to_original=compare_to_original,
                                    use_completed_pdb=use_completed_pdb,
                                    dtype=torch_dtype)
    model = ConfigurationalFrustratometer(coefficients, output_dir=output_dir, dtype=torch_dtype)
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
