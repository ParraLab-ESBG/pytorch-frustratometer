import os
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from click_option_group import optgroup
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from .frustratometer import Frustratometer3
from .coefficients import Coefficients
from .data.dataset import FrustratometerDataset, collate_frustratometer


_VALID_MODES = ("singleresidue", "mutational", "configurational", "configurational_exact", "smooth", "all")


class _DefaultRunGroup(click.Group):
    """Click Group that forwards bare invocations (no subcommand) to 'run'.

    Preserves the original interface:
        pytorch_frustratometer --structure foo.pdb ...
    while also supporting explicit subcommands:
        pytorch_frustratometer run --structure foo.pdb ...
        pytorch_frustratometer clusters RESULTS_DIR ...
    """

    def parse_args(self, ctx, args):
        _help_flags = {"--help", "-h"}
        if args and args[0] not in self.commands and args[0] not in _help_flags:
            args.insert(0, "run")
        return super().parse_args(ctx, args)


@click.group(cls=_DefaultRunGroup)
def cli():
    """PyTorch Frustratometer — GPU-accelerated protein frustration analysis."""




@cli.command("run")
@optgroup.group('Input / Output')
@optgroup.option('--structure', '-s', required=True, multiple=True,
                 help='Structure file(s), directory, or text file of paths [PDB/mmCIF].')
@optgroup.option('--output-dir', '-o', default=None,
                 help='Directory for output files. Defaults to each input file\'s directory.')
@optgroup.option('--mode', '-m', multiple=True,
                 type=click.Choice(_VALID_MODES, case_sensitive=False),
                 help='Frustration mode(s) to compute. Pass multiple times or use "all" '
                      '(singleresidue + mutational + configurational). '
                      'Add "smooth" explicitly for the derived pairwise mode. Default: all.')
@optgroup.option('--chain', 'chain_id', multiple=True, default=(),
                 help='Chain(s) to extract, e.g. --chain A. Repeat to align with multiple '
                      '--structure values. Omit for all chains.')
@optgroup.option('--force', is_flag=True, default=False,
                 help='Recompute frustration even if a <stem>.done/ directory already exists.')
@optgroup.option('--verbose', '-v', is_flag=True, default=False,
                 help='Print each output file as it is written.')
@optgroup.group('Frustration Parameters')
@optgroup.option('--decoys', '--ndecoys', '-n', default=None, type=int,
                 help='Number of sampled decoys for configurational frustration and --use-c-rand '
                      'runs. Single-residue and mutational frustration are exact. Default: 100000.')
@optgroup.option('--cutoff', default=None, type=float,
                 help='Contact distance cutoff in Å. Default: 9.5.')
@optgroup.option('--min-seq-sep', '--min_seq_dist', default=None, type=int,
                 help='Minimum sequence separation |i-j| >= value for a contact to be considered. Default: 3.')
@optgroup.option('--decoy-distribution', '--decoy_distribution', default=None,
                 type=click.Choice(['sequence', 'sequence_exclude_native',
                                    'pseudocount', 'pseudocount_exclude_native'],
                                   case_sensitive=False),
                 help='AA decoy sampling distribution. "sequence" (default): sample from the '
                      'protein\'s own sequence composition. "pseudocount": adds 1 virtual count '
                      'per AA type so AAs absent from the sequence can still appear as decoys. '
                      'Append "_exclude_native" to either to reject the native AA at each position.')
@optgroup.option('--electrostatics', default=None, metavar='true|false|K|K_PP,K_MM,K_PM',
                 help='Control electrostatic terms. "false" disables all; "true" enables (default, 4.15 K); '
                      'a single number sets k_PlusPlus=k_MinusMinus=k_PlusMinus=K; '
                      'three comma-separated values set them individually.')
@optgroup.group('Output Files')
@optgroup.option('--DMFS', '--deep-mutational-frustration-scanning', '--save-mutations',
                 'run_dmfs', is_flag=True, default=False,
                 help='Write full per-residue / per-contact mutation energy tables '
                      '(20 AA × L for singleresidue; 20×20 × E for mutational). '
                      '--save-mutations is a backwards-compatible alias.')
@optgroup.option('--write-pymol', is_flag=True, default=False,
                 help='Write a PyMOL .pml colouring script alongside each output file.')
@optgroup.option('--write-mvsj', is_flag=True, default=False,
                 help='Write a MolViewSpec .mvsj file for each output.')
@optgroup.option('--no-pdb-write', is_flag=True, default=False,
                 help='Skip writing the per-protein PDB file with frustration in B-factors. '
                      'Also suppresses the repair log (<name>.pdb.repair.log) and residue '
                      'index mapping (<name>.pdb.repair_map.tsv) that record structural '
                      'modifications made during PDB parsing.')
@optgroup.option('--shard-output', default=None, type=click.Path(),
                 help='Write frustration results as shards to this directory instead of '
                      'per-protein .done/ directories. Format is controlled by --shard-format. '
                      'Suitable for million-scale datasets.')
@optgroup.option('--shard-rank', default=0, type=int, show_default=True,
                 help='Rank of this worker in a sharded dataset run. Default: 0.')
@optgroup.option('--shard-world-size', default=1, type=int, show_default=True,
                 help='Total number of parallel shard workers. Default: 1.')
@optgroup.option('--shard-size', default=200, type=int, show_default=True,
                 help='Proteins per shard for normal (≤500 res) proteins. Default: 200.')
@optgroup.option('--shard-size-large', default=16, type=int, show_default=True,
                 help='Proteins per shard for large (>500 res) proteins. Default: 16.')
@optgroup.option('--shard-format', default='parquet', show_default=True,
                 type=click.Choice(['parquet', 'hdf5', 'h5native', 'pt.zst'], case_sensitive=False),
                 help='Output format for --shard-output. "parquet" (default) and "hdf5" write '
                      'tabular per-mode files; "h5native" writes native HDF5 (one group per '
                      'protein, N-D arrays at true shape, no flattening); '
                      '"pt.zst" is the legacy batched-tensor format '
                      'used by FrustrAI-Fold.')
@optgroup.option('--shard-save-structure', is_flag=True, default=False,
                 help='Save backbone atom positions (N, CA, C, O, CB) alongside frustration '
                      'data. Adds a "structure" table (parquet/hdf5) or backbone_positions '
                      'tensor (pt.zst). Useful for reloading structures without original PDB files.')
@optgroup.option('--shard-save-sidechain-centroid', is_flag=True, default=False,
                 help='When --shard-save-structure is set, also save the sidechain centroid '
                      '(average of non-backbone heavy atoms) as x_SC/y_SC/z_SC columns '
                      '(parquet/hdf5) or sidechain_centroid tensor (pt.zst).')
@optgroup.group('Visualization')
@optgroup.option('--draw-figure', is_flag=True, default=False,
                 help='Generate frustration plots and NxN contact maps per protein.')
@optgroup.option('--image-format', 'image_formats', multiple=True, default=('png',),
                 help='Figure format(s); may be repeated. Default: png.')
@optgroup.option('--render-structure', is_flag=True, default=False,
                 help='Add a PyMOL 3-D structure panel to figures (requires pymol in PATH).')
@optgroup.group('Computation')
@optgroup.option('--dtype', default='float32',
                 type=click.Choice(['float32', 'float64'], case_sensitive=False),
                 help='Floating-point precision. Default: float32.')
@optgroup.option('--batch-size', '-b', default=1, type=int,
                 help='Number of structures per GPU batch. Default: 1.')
@optgroup.option('--num-workers', default=4, type=int,
                 help='DataLoader worker processes for parallel structure loading. Default: 4.')
@optgroup.option('--devices', default='auto',
                 help='GPUs to use: "auto", an integer count, or comma-separated IDs. Default: auto.')
@optgroup.option('--accelerator', default='auto',
                 help='PyTorch Lightning accelerator: "auto", "gpu", "cpu", "tpu". Default: auto.')
@optgroup.option('--strategy', default='auto',
                 help='PyTorch Lightning strategy: "auto", "ddp", "ddp_spawn", "dp". Default: auto.')
@optgroup.group('Configuration')
@optgroup.option('--config', '-c', default=None, type=click.Path(exists=True),
                 help='YAML file with Coefficients field overrides (applied before --set).')
@optgroup.option('--set', 'overrides', multiple=True, metavar='KEY=VALUE',
                 help='Override any Coefficients field. Parsed with ast.literal_eval, '
                      'e.g. --set k_water=0.5 --set well_rmax="[6.5,9.5]". Applied last.')
@optgroup.option('--backbone-method', default="auto",
                 help='Backbone completion method: auto, modeller, or none. Default: auto.')
@optgroup.option('--scratch-dir', default=None,
                 help='Scratch directory for intermediate files before copying to --output-dir. '
                      'Pass "" to write directly to --output-dir.')
@optgroup.group('Array Job')
@optgroup.option('--task-index', default=0, type=int,
                 help='0-based index of this task in a job array. '
                      'Splits the structure list so each task processes its own slice; '
                      'also auto-sets --shard-rank when not explicitly provided.')
@optgroup.option('--num-tasks', default=1, type=int,
                 help='Total number of array tasks (default: 1, i.e. no splitting). '
                      'Pass $SLURM_ARRAY_TASK_COUNT or equivalent from your scheduler script.')
@optgroup.group('Developer / Validation')
@optgroup.option('--energy-function', '-e', default='awsem',
                 type=click.Choice(['awsem', 'mj'], case_sensitive=False),
                 help='Energy function. "mj" uses Miyazawa-Jernigan potentials (mutational only). '
                      'Default: awsem.')
@optgroup.option('--use-c-rand', is_flag=True, default=False,
                 help='Reproduce C++ decoy sampling exactly using C rand() seed=1 (slow; forces float64).')
@optgroup.option('--use-completed-pdb', is_flag=True, default=False,
                 help='Load FrustratometeR-processed PDB from <stem>.done/FrustrationData/<stem>.pdb '
                      'instead of the original input file.')
@optgroup.option('--compare-to-original', is_flag=True, default=False,
                 help='Compare against FrustratometeR reference files and report R².')
@optgroup.option('--save-bulk-stats', is_flag=True, default=False,
                 help='Write per-protein bulk comparison stats during --compare-to-original runs.')
@optgroup.option('--cluster', 'run_cluster', is_flag=True, default=False,
                 help='Run dynamic cluster detection on --output-dir after frustration computation. '
                      'Requires --output-dir and pip install pytorch-frustratometer[analysis]. '
                      'WARNING: This has not been run or tested.')
@optgroup.option('--wandb', 'use_wandb', is_flag=True, default=False,
                 help='Enable Weights & Biases logging.')
@optgroup.option('--wandb-project', default='frustratometer',
                 help='WandB project name. Default: frustratometer.')
@optgroup.option('--wandb-run-name', default='frustratometer3_baseline',
                 help='WandB run name. Default: frustratometer3_baseline.')
@optgroup.option('--wandb-offline', is_flag=True, default=False,
                 help='Run WandB in offline mode (sync later with wandb sync).')
@optgroup.group('Deprecated')
@optgroup.option('--reproduce-FrustratometeR', 'reproduce_frustratometer', is_flag=True, default=False,
                 help='Set burial_min_sep = min_seq_distance to reproduce FrustratometeR behaviour '
                      '(burial density uses the same sequence-separation cutoff as the contact network). '
                      'This was the default before v3 and is now known to underestimate burial for '
                      'short loops. Do not use for new analyses.')
@optgroup.option('--burial-min-sep', '--burial_min_sep', default=None, type=int,
                 help='[Deprecated — do not change] Minimum sequence separation |i-j| > value for a '
                      'residue pair to count toward the residue density (rho). Default: 1. '
                      'The correct value is 1 (fixed in coefficients.py). Changing this produces '
                      'incorrect burial density and is strongly discouraged.')
def run_cmd(structure, output_dir, mode, energy_function, decoys, cutoff, min_seq_sep,
            burial_min_sep, reproduce_frustratometer, decoy_distribution,
            config, overrides, compare_to_original, use_completed_pdb, use_c_rand, dtype,
            electrostatics, run_dmfs, draw_figure, write_pymol, write_mvsj, no_pdb_write, use_wandb,
            wandb_project, wandb_run_name, wandb_offline, batch_size, num_workers, devices,
            accelerator, strategy, render_structure, image_formats, verbose, force, run_cluster,
            save_bulk_stats, scratch_dir, backbone_method, chain_id,
            shard_output, shard_rank, shard_world_size, shard_size, shard_size_large, shard_format,
            shard_save_structure, shard_save_sidechain_centroid,
            task_index, num_tasks):
    # Default output dir: frustratometer_YYYYMMDD_HHMMSS/ in cwd
    if output_dir is None:
        from datetime import datetime
        output_dir = f"frustratometer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    # Build coefficients: defaults → YAML → named options → --set overrides
    cfg_dict = {}

    if config:
        import yaml
        with open(config) as fh:
            cfg_dict = yaml.safe_load(fh) or {}

    if mode:
        modes = list(mode)
        if "all" in modes:
            modes = ["singleresidue", "mutational", "configurational"]
        cfg_dict["mode"] = modes[0] if len(modes) == 1 else modes
    if decoys is not None:
        cfg_dict["tert_frust_ndecoys"] = decoys
    if cutoff is not None:
        cfg_dict["tert_frust_cutoff"] = cutoff
    if min_seq_sep is not None:
        cfg_dict["min_seq_distance"] = min_seq_sep
    if reproduce_frustratometer:
        cfg_dict["burial_min_sep"] = cfg_dict.get("min_seq_distance", 3)
    if burial_min_sep is not None:
        cfg_dict["burial_min_sep"] = burial_min_sep
    if decoy_distribution is not None:
        cfg_dict["tert_frust_decoy_distribution"] = decoy_distribution
    if run_dmfs:
        cfg_dict["run_dmfs"] = True
    if electrostatics is not None:
        val = electrostatics.strip().lower()
        if val in ('false', '0', 'off', 'no'):
            cfg_dict['use_electrostatics'] = False
        elif val in ('true', '1', 'on', 'yes'):
            cfg_dict['use_electrostatics'] = True
        else:
            parts = val.replace(',', ' ').split()
            if len(parts) == 1:
                k = float(parts[0])
                cfg_dict['k_PlusPlus'] = k
                cfg_dict['k_MinusMinus'] = k
                cfg_dict['k_PlusMinus'] = k
            elif len(parts) == 3:
                cfg_dict['k_PlusPlus'] = float(parts[0])
                cfg_dict['k_MinusMinus'] = float(parts[1])
                cfg_dict['k_PlusMinus'] = float(parts[2])
            else:
                raise click.BadParameter(
                    'expected true/false, a single number, or three comma-separated numbers',
                    param_hint='--electrostatics'
                )

    for kv in overrides:
        if '=' not in kv:
            raise click.BadParameter(f"--set values must be KEY=VALUE, got: {kv!r}")
        key, raw = kv.split('=', 1)
        cfg_dict[key] = Coefficients._parse_cli_value(raw)

    try:
        coefficients = Coefficients.from_dict(cfg_dict) if cfg_dict else Coefficients()
    except ValueError as error:
        raise click.UsageError(str(error)) from error

    if backbone_method is not None and backbone_method.lower() == "none":
        backbone_method = None

    if render_structure:
        draw_figure = True

    torch_dtype = torch.float64 if dtype == 'float64' else torch.float32

    # Detect if this process was re-launched by Lightning's DDPStrategy subprocess launcher.
    _ddp_rank = int(os.environ.get("RANK", "0"))
    _is_print_rank = _ddp_rank == 0

    dataset = FrustratometerDataset(structure, coefficients, compare_to_original=compare_to_original,
                                    use_completed_pdb=use_completed_pdb, dtype=torch_dtype,
                                    backbone_method=backbone_method, chain_id=chain_id,
                                    write_repair_files=not no_pdb_write and not shard_output)

    # ── Skip already-completed structures (unless --force) ─────────────────
    if not force and not shard_output and dataset.data:
        _all_modes = {"singleresidue", "mutational", "configurational"}
        _mode_files = {
            "singleresidue":   ["singleresidue"],
            "mutational":      ["mutational"],
            "configurational": ["configurational"],
            "smooth":          ["smooth"],
            "all":             sorted(_all_modes),
        }
        _modes = set(mode) if mode else {"singleresidue"}
        if "all" in _modes:
            _modes = _all_modes
        expected_suffixes = sorted(
            {suf for m in _modes for suf in _mode_files.get(m, [m])}
        )

        def _is_done(sf):
            p = Path(sf)
            parent = Path(output_dir) if output_dir else p.parent
            stem = p.stem
            done_dir = parent / f"{stem}.done" / "FrustrationData"
            if not all((done_dir / f"{stem}.pdb_{suf}").exists() for suf in expected_suffixes):
                return False
            if run_dmfs and not (done_dir / f"{stem}.pdb_singleresidue_mutations").exists():
                return False
            if draw_figure:
                vis_dir = parent / f"{stem}.done" / "Visualizations"
                if "singleresidue" in _modes and not (vis_dir / f"{stem}_per_protein.png").exists():
                    return False
            return True

        before = len(dataset.data)
        with ThreadPoolExecutor(max_workers=min(32, before)) as pool:
            done_flags = list(pool.map(_is_done, [sf for sf, _ in dataset.data]))
        dataset.data = [entry for entry, done in zip(dataset.data, done_flags) if not done]
        skipped = before - len(dataset.data)
        if skipped and _is_print_rank:
            _mode_str = ", ".join(expected_suffixes)
            print(f"[skip] {skipped}/{before} structure(s) already have {_mode_str} output "
                  f"— pass --force to recompute.", flush=True)
        if not dataset.data:
            if _is_print_rank:
                print("[skip] All structures already computed. Nothing to do.", flush=True)
            return

    # ── Task splitting for array jobs ─────────────────────────────────────
    if num_tasks > 1:
        dataset.data = dataset.data[task_index::num_tasks]
        # Auto-set shard rank from task index when the user left the defaults
        if shard_rank == 0 and shard_world_size == 1:
            shard_rank = task_index
            shard_world_size = num_tasks
        if _is_print_rank:
            print(f"[Task] {task_index}/{num_tasks}: {len(dataset.data)} structures", flush=True)
        if not dataset.data:
            if _is_print_rank:
                print("[Task] No structures assigned to this task. Exiting.", flush=True)
            return

    # ── Determine effective write directory and I/O strategy ──────────────
    from .base import _precreated_dirs
    import pytorch_frustratometer.base as _base_mod

    effective_output_dir = output_dir
    using_tmpdir = False

    if output_dir:
        force_no_scratch = scratch_dir is not None and scratch_dir.strip() == ""
        explicit_scratch  = scratch_dir.strip() if scratch_dir and scratch_dir.strip() else None

        cand_to_try = []
        if not force_no_scratch:
            if explicit_scratch:
                cand_to_try = [("--scratch-dir", explicit_scratch)]
            else:
                cand_to_try = [(env, os.environ.get(env))
                               for env in ("TMPDIR", "SCRATCH", "LOCAL_SCRATCH", "SLURM_TMPDIR")
                               if os.environ.get(env)]

        for env_var, cand in cand_to_try:
            try:
                cand_stat = os.stat(cand)
                out_stat  = os.stat(output_dir) if os.path.exists(output_dir) else None
                if (os.access(cand, os.W_OK) and
                        (explicit_scratch or out_stat is None or
                         cand_stat.st_dev != out_stat.st_dev)):
                    effective_output_dir = os.path.join(cand, "frustratometer_out")
                    os.makedirs(effective_output_dir, exist_ok=True)
                    using_tmpdir = True
                    if _is_print_rank:
                        print(f"[I/O] Scratch: {env_var}={cand}"
                              f" → writing to {effective_output_dir},"
                              f" copying to {output_dir} after run.", flush=True)
                    break
            except OSError:
                continue

        if not using_tmpdir:
            _base_mod._use_in_progress = True
            if _is_print_rank:
                reason = "disabled via --scratch-dir=''" if force_no_scratch else "none found"
                print(f"[I/O] Scratch {reason} — writing directly to {output_dir}"
                      f" (.in_progress/ → .done/ per batch).", flush=True)

    # ── Pre-create all output directories in parallel ──────────────────────
    if _is_print_rank and effective_output_dir and dataset.data and not shard_output:
        suffix = ".in_progress" if _base_mod._use_in_progress else ".done"
        print(f"[I/O] Pre-creating {len(dataset.data)} output directories ...", flush=True)
        out_root = Path(effective_output_dir)

        def _mkdir(sf):
            d = out_root / f"{Path(sf).stem}{suffix}" / "FrustrationData"
            d.mkdir(parents=True, exist_ok=True)
            return str(d)

        with ThreadPoolExecutor(max_workers=min(32, len(dataset.data))) as pool:
            _precreated_dirs.update(pool.map(_mkdir, [sf for sf, _ in dataset.data]))
        print(f"[I/O] Directories ready.", flush=True)

    if energy_function == "mj":
        from .algorithm.mj import MJFrustratometer
        frustratometer = MJFrustratometer(coefficients, output_dir=effective_output_dir, dtype=torch_dtype)
    else:
        _scratch_arg   = str(effective_output_dir) if using_tmpdir and output_dir else None
        _final_dir_arg = str(output_dir)           if using_tmpdir and output_dir else None
        if _scratch_arg:
            os.environ["FRUST_SCRATCH_ROOT"]     = _scratch_arg
            os.environ["FRUST_FINAL_OUTPUT_DIR"] = _final_dir_arg
        frustratometer = Frustratometer3(
            coefficients, output_dir=effective_output_dir, dtype=torch_dtype,
            draw_figure=False if shard_output else draw_figure,
            write_pymol=False if shard_output else write_pymol,
            write_mvsj=False if shard_output else write_mvsj,
            render_structure=False if shard_output else render_structure,
            image_formats=image_formats,
            verbose=verbose, write_pdb=False if shard_output else not no_pdb_write,
            use_in_progress=not using_tmpdir and bool(output_dir) and not shard_output,
            save_bulk_stats=save_bulk_stats,
            scratch_root=_scratch_arg, final_output_dir=_final_dir_arg,
        )
    if hasattr(frustratometer, 'awsem'):
        frustratometer.awsem.use_c_rand = use_c_rand

    if shard_output:
        from .data.shard_formats import make_shard_writer
        frustratometer._shard_writer = make_shard_writer(
            fmt=shard_format,
            output_dir=shard_output,
            shard_size=shard_size,
            shard_size_large=shard_size_large,
            rank=shard_rank,
            world_size=shard_world_size,
            coefficients=frustratometer.coefficients,
            save_structure=shard_save_structure,
            save_sidechain_centroid=shard_save_sidechain_centroid,
        )

    if devices == 'auto':
        n_devices = 'auto'
    elif ',' in devices:
        n_devices = [int(x) for x in devices.split(',')]
    else:
        n_devices = int(devices)

    effective_strategy = strategy
    if effective_strategy == "auto" and isinstance(n_devices, int) and n_devices > 1:
        effective_strategy = "ddp"

    if _is_print_rank:
        print(f"[Run] strategy={effective_strategy}  devices={n_devices}  accelerator={accelerator}", flush=True)

    if effective_strategy == "ddp_spawn" and isinstance(n_devices, int) and n_devices > 1:
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
            _s.bind(("127.0.0.1", 0))
            _free_port = _s.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port)

    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    if use_wandb:
        from lightning.pytorch.loggers import WandbLogger
        if wandb_offline:
            os.environ.setdefault("WANDB_MODE", "offline")
        logger = WandbLogger(project=wandb_project, name=wandb_run_name, offline=wandb_offline)
    else:
        logger = False

    if _is_print_rank:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            torch.set_float32_matmul_precision("high")
            print(f"[Device] {n_gpus} GPU(s) available: "
                  + ", ".join(f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(n_gpus)),
                  flush=True)
        else:
            print("[Device] No GPU found — running on CPU", flush=True)
    elif torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        accelerator=accelerator, devices=n_devices, strategy=effective_strategy,
        logger=logger, enable_model_summary=False,
    )

    if _is_print_rank:
        print(f"[Run] {trainer.strategy.__class__.__name__}  ×{trainer.num_devices} device(s)", flush=True)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_frustratometer,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
    )

    try:
        if compare_to_original:
            trainer.test(frustratometer, dataloader)
        else:
            trainer.predict(frustratometer, dataloader)
    finally:
        if getattr(frustratometer, '_shard_writer', None) is not None:
            frustratometer._shard_writer.finalize()
        if frustratometer._copy_executor is not None:
            frustratometer._copy_executor.shutdown(wait=True)

        # Copy everything from scratch to the final output_dir so results
        # survive scratch cleanup after the job ends.
        if using_tmpdir and output_dir and effective_output_dir != output_dir and _is_print_rank:
            import shutil
            src_root = Path(effective_output_dir)
            dst_root = Path(output_dir)
            dst_root.mkdir(parents=True, exist_ok=True)
            for item in src_root.iterdir():
                dst = dst_root / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst)
            print(f"[I/O] Copied all scratch results {src_root} → {dst_root}", flush=True)

    if run_cluster:
        cluster_dir = output_dir or effective_output_dir
        if not cluster_dir:
            raise click.UsageError("--cluster requires --output_dir")
        _run_clusters(results_dir=cluster_dir)


def _run_clusters(
    results_dir,
    order_list=None,
    loess_span=0.05,
    min_frst_range=0.7,
    filt_mean=0.15,
    n_components=10,
    min_corr=0.95,
    leiden_resolution=1.0,
    corr_type="spearman",
    p_threshold=0.05,
    device=None,
    seed=42,
    output=None,
):
    """Shared implementation for both --cluster flag and clusters subcommand.

    WARNING: This has not been run or tested.
    """
    try:
        from .analysis.dynamic_clusters import detect_dynamic_clusters, get_clusters
    except ImportError as exc:
        raise click.ClickException(
            "The [analysis] optional dependencies are required for clustering.\n"
            "Install them with:  pip install pytorch-frustratometer[analysis]"
        ) from exc

    results_dir = Path(results_dir)
    result = detect_dynamic_clusters(
        results_dir=results_dir,
        order_list=order_list,
        loess_span=loess_span,
        min_frst_range=min_frst_range,
        filt_mean=filt_mean,
        n_components=n_components,
        min_corr=min_corr,
        leiden_resolution=leiden_resolution,
        corr_type=corr_type,
        p_threshold=p_threshold,
        device=device,
        random_state=seed,
        verbose=True,
    )

    df = get_clusters(result)
    out_path = Path(output) if output else results_dir / "clusters.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[Clusters] {result['n_communities']} communities → {out_path}", flush=True)
    for i, comm in enumerate(result["communities"]):
        print(f"  Community {i}: {len(comm)} residues", flush=True)


@cli.command("clusters")
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--order_list", "-l", default=None, type=click.Path(exists=True),
              help="Text file with one PDB path per line defining frame order. "
                   "Auto-discovers all *.done/ directories when omitted.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output TSV for cluster assignments. Default: <results_dir>/clusters.tsv.")
@click.option("--loess_span", default=0.05, type=float, show_default=True,
              help="LOWESS bandwidth fraction.")
@click.option("--min_frst_range", default=0.7, type=float, show_default=True,
              help="Quantile threshold for per-residue frustration range filter.")
@click.option("--filt_mean", default=0.15, type=float, show_default=True,
              help="Drop residues with |mean frustration| ≤ this value.")
@click.option("--n_components", default=10, type=int, show_default=True,
              help="Number of PCA components to retain.")
@click.option("--min_corr", default=0.95, type=float, show_default=True,
              help="Minimum |correlation| to draw a graph edge.")
@click.option("--leiden_resolution", default=1.0, type=float, show_default=True,
              help="Leiden CPM resolution parameter.")
@click.option("--corr_type", default="spearman", show_default=True,
              type=click.Choice(["spearman", "pearson"], case_sensitive=False),
              help="Correlation type for PCA-space similarity.")
@click.option("--p_threshold", default=0.05, type=float, show_default=True,
              help="Maximum p-value for a graph edge.")
@click.option("--device", default=None,
              help="PyTorch device (e.g. cuda, cpu). Auto-detects CUDA when omitted.")
@click.option("--seed", default=42, type=int, show_default=True,
              help="Random seed for PCA and Leiden.")
def clusters_cmd(results_dir, order_list, output, loess_span, min_frst_range,
                 filt_mean, n_components, min_corr, leiden_resolution, corr_type,
                 p_threshold, device, seed):
    """Detect residue modules with correlated dynamic frustration.

    RESULTS_DIR must contain *.done/FrustrationData/*.pdb_singleresidue files
    produced by the 'run' command.

    WARNING: This has not been run or tested.
    """
    order = None
    if order_list:
        order = Path(order_list).read_text().strip().splitlines()

    _run_clusters(
        results_dir=results_dir,
        order_list=order,
        loess_span=loess_span,
        min_frst_range=min_frst_range,
        filt_mean=filt_mean,
        n_components=n_components,
        min_corr=min_corr,
        leiden_resolution=leiden_resolution,
        corr_type=corr_type,
        p_threshold=p_threshold,
        device=device,
        seed=seed,
        output=output,
    )


if __name__ == "__main__":
    cli()
