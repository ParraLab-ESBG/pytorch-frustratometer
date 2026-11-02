"""Generic shard-writer interface for large-scale frustration precomputation.

Four concrete formats are supported:

    parquet   — per-mode Parquet files; default, best for downstream analysis
    hdf5      — per-mode HDF5 files via pandas/PyTables (requires ``pip install tables``)
    h5native  — native h5py; one file per shard, one group per protein, N-D arrays at true shape
    pt.zst    — batched PyTorch tensors compressed with zstd (FrustrAI-Fold format)

Factory::

    writer = make_shard_writer("parquet", output_dir="/path/to/shards",
                               shard_size=200, rank=0, world_size=4)
    for entry, features, length in stream:
        writer.add(entry, features, length)
    writer.finalize()

Schema (parquet / hdf5) — ASCII column names
---------------------------------------------
singleresidue         : entry, Res, ChainRes, AA, DensityRes, NativeEnergy, DecoyEnergy, SDEnergy, FrstIndex, FrstState
singleresidue_mutations: entry, Res, ChainRes, AA_native, AA_mutant, NativeEnergy, MutantEnergy, ScanFrstIndex, ScanFrstState
mutational            : entry, Res1, Res2, ChainRes1, ChainRes2, AA1, AA2, DensityRes1, DensityRes2, NativeEnergy, DecoyEnergy, SDEnergy, Welltype, FrstIndex, FrstState
configurational       : (same columns as mutational)
structure             : entry, Res, ChainRes, AA, x_N, y_N, z_N, x_CA, y_CA, z_CA, x_C, y_C, z_C, x_CB, y_CB, z_CB, x_O, y_O, z_O
                        [, x_SC, y_SC, z_SC when save_sidechain_centroid=True]
                        (written only when save_structure=True; atom_positions[L, ≥5, 3] expected in features)
repair                : entry, repair_log, repair_map  (one row per protein; empty strings when no repairs were made)

Shard filenames
---------------
shard_{mode}_{bucket}[_rank{r}]_{idx:04d}.{ext}
  mode   = singleresidue | mutational | configurational
  bucket = normal | large
  ext    = parquet | h5 | pt.zst
"""

from __future__ import annotations

import gc
import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np

_PAIRWISE_MODES = ("mutational", "configurational")
_MODES = ("singleresidue",) + _PAIRWISE_MODES
_ALL_MODES = _MODES + ("singleresidue_mutations", "structure", "repair")

_BACKBONE_ATOMS = ("N", "CA", "C", "CB", "O")  # indices 0–4 of atom_positions per residue_constants.atom_types


# ── shared row conversion ─────────────────────────────────────────────────────

def _arr(features: dict, key: str):
    """Return a numpy array for *key*, or None if the key is absent."""
    v = features.get(key)
    if v is None:
        return None
    try:
        return v.cpu().numpy()
    except AttributeError:
        return np.asarray(v)


def _features_to_rows(
    entry: str,
    features: dict,
    save_structure: bool = False,
    save_sidechain_centroid: bool = False,
) -> dict[str, "pd.DataFrame"]:
    """Convert one protein's feature dict to per-mode DataFrames (ASCII naming)."""
    import pandas as pd

    dfs: dict[str, pd.DataFrame] = {}

    # ── singleresidue ──────────────────────────────────────────────────────────
    fi_sr = _arr(features, "FrstIndex")
    if fi_sr is not None:
        L = fi_sr.shape[0]

        def _sr(key, dtype, fill):
            v = _arr(features, key)
            return v.astype(dtype) if v is not None else np.full(L, fill, dtype)

        row: dict = {
            "entry":        entry,
            "Res":          _sr("Res",        np.int32,   -1),
            "ChainRes":     _arr(features, "ChainRes") if _arr(features, "ChainRes") is not None
                            else np.full(L, "", object),
            "AA":           _sr("AA",         np.int8,    -1),
            "DensityRes":   _sr("DensityRes", np.float32, np.nan),
            "NativeEnergy": _sr("NativeEnergy", np.float32, np.nan),
            "DecoyEnergy":  _sr("DecoyEnergy",  np.float32, np.nan),
            "SDEnergy":     _sr("SDEnergy",     np.float32, np.nan),
            "FrstIndex":    fi_sr.astype(np.float32),
            "FrstState":    _sr("FrstState",  np.int8,    -1),
        }
        dfs["singleresidue"] = pd.DataFrame(row)

    # ── mutation scan ([L,20] → L*20 rows) ────────────────────────────────────
    fi_scan = _arr(features, "ScanFrstIndex")
    if fi_scan is not None:
        L = fi_scan.shape[0]

        res     = _arr(features, "Res")
        chain   = _arr(features, "ChainRes")
        aa_nat  = _arr(features, "AA")
        nat_e   = _arr(features, "NativeEnergy")
        mut_e   = _arr(features, "MutantEnergy")
        fs_scan = _arr(features, "ScanFrstState")

        dfs["singleresidue_mutations"] = pd.DataFrame({
            "entry":        np.repeat(entry, L * 20),
            "Res":          np.repeat(res, 20).astype(np.int32)   if res    is not None else np.full(L * 20, -1,     np.int32),
            "ChainRes":     np.repeat(chain, 20)                  if chain  is not None else np.full(L * 20, "",     object),
            "AA_native":    np.repeat(aa_nat, 20).astype(np.int8) if aa_nat is not None else np.full(L * 20, -1,     np.int8),
            "AA_mutant":    np.tile(np.arange(20, dtype=np.int8), L),
            "NativeEnergy": np.repeat(nat_e, 20).astype(np.float32) if nat_e  is not None else np.full(L * 20, np.nan, np.float32),
            "MutantEnergy": mut_e.reshape(-1).astype(np.float32)     if mut_e  is not None else np.full(L * 20, np.nan, np.float32),
            "ScanFrstIndex": fi_scan.reshape(-1).astype(np.float32),
            "ScanFrstState": fs_scan.reshape(-1).astype(np.int8)     if fs_scan is not None else np.full(L * 20, -1, np.int8),
        })

    # ── pairwise modes (mutational / configurational) ─────────────────────────
    for mode in _PAIRWISE_MODES:
        fi_pw = _arr(features, f"{mode}_FrstIndex")
        if fi_pw is None:
            continue
        E = fi_pw.shape[0]

        def _pw(key, dtype, fill):
            v = _arr(features, key)
            return v.astype(dtype) if v is not None else np.full(E, fill, dtype)

        dfs[mode] = pd.DataFrame({
            "entry":        entry,
            "Res1":         _pw(f"{mode}_Res1",        np.int32,   -1),
            "Res2":         _pw(f"{mode}_Res2",        np.int32,   -1),
            "ChainRes1":    _arr(features, f"{mode}_ChainRes1") if _arr(features, f"{mode}_ChainRes1") is not None
                            else np.full(E, "", object),
            "ChainRes2":    _arr(features, f"{mode}_ChainRes2") if _arr(features, f"{mode}_ChainRes2") is not None
                            else np.full(E, "", object),
            "AA1":          _pw(f"{mode}_AA1",         np.int8,    -1),
            "AA2":          _pw(f"{mode}_AA2",         np.int8,    -1),
            "DensityRes1":  _pw(f"{mode}_DensityRes1", np.float32, np.nan),
            "DensityRes2":  _pw(f"{mode}_DensityRes2", np.float32, np.nan),
            "NativeEnergy": _pw(f"{mode}_NativeEnergy", np.float32, np.nan),
            "DecoyEnergy":  _pw(f"{mode}_DecoyEnergy",  np.float32, np.nan),
            "SDEnergy":     _pw(f"{mode}_SDEnergy",     np.float32, np.nan),
            "Welltype":     _pw(f"{mode}_Welltype",    np.int8,    -1),
            "FrstIndex":    fi_pw.astype(np.float32),
            "FrstState":    _pw(f"{mode}_FrstState",   np.int8,    -1),
        })

    # ── repair log ────────────────────────────────────────────────────────────
    repair_log = features.get("repair_log")
    repair_map = features.get("repair_map")
    if repair_log is not None or repair_map is not None:
        dfs["repair"] = pd.DataFrame({
            "entry":      [entry],
            "repair_log": [repair_log if repair_log is not None else ""],
            "repair_map": [repair_map if repair_map is not None else ""],
        })

    # ── structure (backbone atom positions ± sidechain centroid) ─────────────
    if save_structure:
        ap = _arr(features, "atom_positions")
        if ap is not None and ap.ndim == 3:
            L = ap.shape[0]
            bb = ap[:, :5, :]  # [L, 5, 3] — N, CA, C, O, CB
            row: dict = {
                "entry":    entry,
                "Res":      (_arr(features, "Res").astype(np.int32)
                             if _arr(features, "Res") is not None
                             else np.arange(L, dtype=np.int32)),
                "ChainRes": (_arr(features, "ChainRes")
                             if _arr(features, "ChainRes") is not None
                             else np.full(L, "", object)),
                "AA":       (_arr(features, "AA").astype(np.int8)
                             if _arr(features, "AA") is not None
                             else np.full(L, -1, np.int8)),
            }
            for i, atom in enumerate(_BACKBONE_ATOMS):
                row[f"x_{atom}"] = bb[:, i, 0].astype(np.float32)
                row[f"y_{atom}"] = bb[:, i, 1].astype(np.float32)
                row[f"z_{atom}"] = bb[:, i, 2].astype(np.float32)
            if save_sidechain_centroid:
                sc = _arr(features, "sidechain_centroid")
                if sc is None:
                    # average of non-backbone heavy atoms (indices 4+, non-zero)
                    sc_mask = (ap != 0).all(axis=-1)      # [L, N_atoms]
                    sc_mask[:, :4] = False                 # exclude N, CA, C, O
                    count = sc_mask.sum(axis=-1, keepdims=True).astype(np.float32) + 1e-8
                    sc = (ap * sc_mask[..., None]).sum(axis=-2).astype(np.float32) / count
                row["x_SC"] = sc[:, 0].astype(np.float32)
                row["y_SC"] = sc[:, 1].astype(np.float32)
                row["z_SC"] = sc[:, 2].astype(np.float32)
            dfs["structure"] = pd.DataFrame(row)

    return dfs


# ── abstract base ─────────────────────────────────────────────────────────────

class BaseShardWriter(ABC):
    """Common interface for all shard writers."""

    @abstractmethod
    def add(self, entry: str, features: dict, length: int) -> None: ...

    @abstractmethod
    def record_failure(self) -> None: ...

    @abstractmethod
    def finalize(self) -> None: ...


# ── tabular base (shared by parquet + hdf5) ───────────────────────────────────

class TabularShardWriter(BaseShardWriter):
    """Accumulate per-protein DataFrames and flush to per-shard files.

    Subclasses provide ``_ext`` (file extension) and ``_write_shard()``.
    """

    _ext: str = ""  # override in subclass

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 200,
        shard_size_large: int = 16,
        large_threshold: int = 500,
        rank: int = 0,
        world_size: int = 1,
        coefficients=None,
        save_structure: bool = False,
        save_sidechain_centroid: bool = False,
    ):
        self.output_dir               = Path(output_dir)
        self._coefficients            = coefficients
        self._save_structure          = save_structure
        self._save_sidechain_centroid = save_sidechain_centroid
        self.shard_size       = shard_size
        self.shard_size_large = shard_size_large
        self.large_threshold  = large_threshold
        self.rank             = rank
        self.world_size       = world_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        # _buffers[mode][bucket] = list[pd.DataFrame]
        self._buffers: dict[str, dict[str, list]] = {
            m: {"normal": [], "large": []} for m in _ALL_MODES
        }
        self._counts: dict[str, dict[str, int]] = {
            m: {"normal": 0, "large": 0} for m in _ALL_MODES
        }
        self._n_ok    = 0
        self._n_fail  = 0
        self._lengths: list[int] = []

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, entry: str, features: dict, length: int) -> None:
        bucket = "large" if length > self.large_threshold else "normal"
        cap    = self.shard_size_large if bucket == "large" else self.shard_size

        row_dfs = _features_to_rows(
            entry, features,
            save_structure=self._save_structure,
            save_sidechain_centroid=self._save_sidechain_centroid,
        )
        flushes: dict[str, tuple] = {}

        with self._lock:
            self._lengths.append(length)
            self._n_ok += 1
            for mode, df in row_dfs.items():
                self._buffers[mode][bucket].append(df)
                if len(self._buffers[mode][bucket]) >= cap:
                    flushes[mode] = (self._buffers[mode][bucket][:],
                                     self._counts[mode][bucket])
                    self._buffers[mode][bucket].clear()
                    self._counts[mode][bucket] += 1

        for mode, (buf, idx) in flushes.items():
            self._flush(buf, mode, bucket, idx)

    def record_failure(self) -> None:
        with self._lock:
            self._n_fail += 1

    def finalize(self) -> None:
        for mode in _ALL_MODES:
            for bucket in ("normal", "large"):
                with self._lock:
                    buf = self._buffers[mode][bucket][:]
                    idx = self._counts[mode][bucket]
                    self._buffers[mode][bucket].clear()
                    self._counts[mode][bucket] += 1
                if buf:
                    self._flush(buf, mode, bucket, idx)

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump({
                "format":      self._ext.lstrip("."),
                "num_samples": self._n_ok,
                "lengths":     self._lengths,
                "rank":        self.rank,
                "world_size":  self.world_size,
            }, f)
        tag = self._ext.lstrip(".")
        print(f"[Shards/{tag}] rank={self.rank}: {self._n_ok} ok, "
              f"{self._n_fail} failed → {self.output_dir}", flush=True)

    # ── internals ─────────────────────────────────────────────────────────────

    def _shard_path(self, mode: str, bucket: str, idx: int) -> Path:
        suffix = f"_rank{self.rank}" if self.world_size > 1 else ""
        return self.output_dir / f"shard_{mode}_{bucket}{suffix}_{idx:04d}{self._ext}"

    def _flush(self, buf: list, mode: str, bucket: str, idx: int) -> None:
        import pandas as pd
        df   = pd.concat(buf, ignore_index=True)
        path = self._shard_path(mode, bucket, idx)
        n    = df["entry"].nunique()
        self._write_shard(df, path, mode)
        tag = self._ext.lstrip(".")
        print(f"[Shards/{tag}] wrote {path.name}  "
              f"({n} proteins, {len(df):,} rows)", flush=True)
        del df, buf
        gc.collect()

    @abstractmethod
    def _write_shard(self, df: "pd.DataFrame", path: Path, mode: str) -> None: ...


# ── parquet ───────────────────────────────────────────────────────────────────

class ParquetShardWriter(TabularShardWriter):
    """Write per-mode Parquet shards (zstd-compressed)."""

    _ext = ".parquet"

    def __init__(self, *args, compression: str = "zstd", **kwargs):
        super().__init__(*args, **kwargs)
        self.compression = compression

    def _write_shard(self, df: "pd.DataFrame", path: Path, mode: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._coefficients is not None:
            existing = table.schema.metadata or {}
            table = table.replace_schema_metadata(
                {**existing, b"coefficients": json.dumps(self._coefficients.to_dict()).encode()}
            )
        pq.write_table(table, path, compression=self.compression)


# ── hdf5 ──────────────────────────────────────────────────────────────────────

class HDF5ShardWriter(TabularShardWriter):
    """Write per-mode HDF5 shards via pandas / PyTables (requires ``pip install tables``).

    Each shard is an HDF5 file readable with ``pd.read_hdf(path, key=mode)``.
    Columns match the ASCII output files; see module docstring for the full schema.
    """

    _ext = ".h5"

    def __init__(self, *args, compression: str = "zlib", compression_opts: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression      = compression
        self.compression_opts = compression_opts

    def _write_shard(self, df: "pd.DataFrame", path: Path, mode: str) -> None:
        try:
            df.to_hdf(path, key=mode, mode="w",
                      complib=self.compression, complevel=self.compression_opts)
        except ImportError:
            raise ImportError(
                "PyTables is required for HDF5 shard output. "
                "Install it with: pip install tables"
            ) from None
        if self._coefficients is not None:
            import h5py
            with h5py.File(str(path), "a") as hf:
                hf.attrs["coefficients"] = json.dumps(self._coefficients.to_dict())


# ── native h5py writer ───────────────────────────────────────────────────────

class NativeHDF5ShardWriter(BaseShardWriter):
    """Native h5py shard writer — one file per shard, one group per protein.

    Unlike ``HDF5ShardWriter`` (pandas/PyTables), this class writes arrays at
    their true shape with no flattening:

    * ``singleresidue/*``      — [L] per field
    * ``mutational/*``         — [E] per field; ``edge_index`` is [2, E]
    * ``configurational/*``    — same as mutational
    * ``mutation_scan/*``      — [L, 20] per field (no unrolling)
    * ``structure/X``          — [L, 5, 3] backbone coords (N, CA, C, CB, O)

    All modes present for a given protein land in the same shard file under
    ``/<entry>/``.  Coefficients are stored as a root JSON attribute.

    Format name: ``"h5native"``
    """

    _ext = ".h5native"

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 200,
        shard_size_large: int = 16,
        large_threshold: int = 500,
        rank: int = 0,
        world_size: int = 1,
        coefficients=None,
        save_structure: bool = False,
        save_sidechain_centroid: bool = False,
        compression: str = "lzf",
    ):
        self.output_dir               = Path(output_dir)
        self._coefficients            = coefficients
        self._save_structure          = save_structure
        self._save_sidechain_centroid = save_sidechain_centroid
        self.shard_size               = shard_size
        self.shard_size_large         = shard_size_large
        self.large_threshold          = large_threshold
        self.rank                     = rank
        self.world_size               = world_size
        self._compression             = compression

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock   = threading.Lock()
        self._buf_normal: list[tuple[str, dict]] = []
        self._buf_large:  list[tuple[str, dict]] = []
        self._idx_normal  = 0
        self._idx_large   = 0
        self._n_ok   = 0
        self._n_fail = 0
        self._lengths: list[int] = []

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, entry: str, features: dict, length: int) -> None:
        bucket = "large" if length > self.large_threshold else "normal"
        cap    = self.shard_size_large if bucket == "large" else self.shard_size

        with self._lock:
            self._lengths.append(length)
            self._n_ok += 1
            if bucket == "large":
                self._buf_large.append((entry, features))
                if len(self._buf_large) >= cap:
                    buf, idx = self._buf_large[:], self._idx_large
                    self._buf_large.clear()
                    self._idx_large += 1
                else:
                    buf = None
            else:
                self._buf_normal.append((entry, features))
                if len(self._buf_normal) >= cap:
                    buf, idx = self._buf_normal[:], self._idx_normal
                    self._buf_normal.clear()
                    self._idx_normal += 1
                else:
                    buf = None

        if buf is not None:
            self._flush(buf, bucket, idx)

    def record_failure(self) -> None:
        with self._lock:
            self._n_fail += 1

    def finalize(self) -> None:
        for bucket, buf_attr, idx_attr in (
            ("normal", "_buf_normal", "_idx_normal"),
            ("large",  "_buf_large",  "_idx_large"),
        ):
            with self._lock:
                buf = getattr(self, buf_attr)[:]
                idx = getattr(self, idx_attr)
                getattr(self, buf_attr).clear()
            if buf:
                self._flush(buf, bucket, idx)

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump({
                "format":      "h5native",
                "num_samples": self._n_ok,
                "lengths":     self._lengths,
                "rank":        self.rank,
                "world_size":  self.world_size,
            }, f)
        print(f"[Shards/h5native] rank={self.rank}: {self._n_ok} ok, "
              f"{self._n_fail} failed → {self.output_dir}", flush=True)

    # ── internals ─────────────────────────────────────────────────────────────

    def _shard_path(self, bucket: str, idx: int) -> Path:
        suffix = f"_rank{self.rank}" if self.world_size > 1 else ""
        return self.output_dir / f"shard_{bucket}{suffix}_{idx:04d}{self._ext}"

    def _flush(self, buf: list[tuple[str, dict]], bucket: str, idx: int) -> None:
        import h5py
        path = self._shard_path(bucket, idx)
        c    = self._compression

        with h5py.File(path, "w") as hf:
            if self._coefficients is not None:
                hf.attrs["coefficients"] = json.dumps(self._coefficients.to_dict())

            entries = []
            for entry, features in buf:
                grp = hf.require_group(entry)
                entries.append(entry)
                self._write_protein(grp, features, c)

            dt = h5py.string_dtype()
            hf.create_dataset("entries", data=np.array(entries, dtype=object), dtype=dt)

        print(f"[Shards/h5native] wrote {path.name}  ({len(buf)} proteins)", flush=True)
        gc.collect()

    def _ds(self, grp, name, data, compression):
        """Create a compressed dataset; skip compression for tiny arrays."""
        arr = np.asarray(data)
        kw  = {"compression": compression} if arr.size > 16 else {}
        grp.create_dataset(name, data=arr, **kw)

    def _write_protein(self, grp, features: dict, c: str) -> None:
        ds = self._ds

        # ── singleresidue ──────────────────────────────────────────────────────
        fi_sr = _arr(features, "FrstIndex")
        if fi_sr is not None:
            sg = grp.require_group("singleresidue")
            for key, dtype in (
                ("Res",          np.int32),
                ("AA",           np.int8),
                ("DensityRes",   np.float32),
                ("NativeEnergy", np.float32),
                ("DecoyEnergy",  np.float32),
                ("SDEnergy",     np.float32),
                ("FrstState",    np.int8),
            ):
                v = _arr(features, key)
                if v is not None:
                    ds(sg, key, v.astype(dtype), c)
            ds(sg, "FrstIndex", fi_sr.astype(np.float32), c)
            chain = _arr(features, "ChainRes")
            if chain is not None:
                sg.create_dataset("ChainRes", data=chain.astype("S"), compression=c
                                  if chain.size > 16 else None)

        # ── mutation scan [L, 20] — stored at native shape ─────────────────────
        fi_scan = _arr(features, "ScanFrstIndex")
        if fi_scan is not None:
            mg = grp.require_group("mutation_scan")
            ds(mg, "ScanFrstIndex", fi_scan.astype(np.float32), c)
            for key, dtype in (("ScanFrstState", np.int8), ("MutantEnergy", np.float32)):
                v = _arr(features, key)
                if v is not None:
                    ds(mg, key, v.astype(dtype), c)

        # ── pairwise modes (mutational / configurational) ──────────────────────
        for mode in _PAIRWISE_MODES:
            fi_pw = _arr(features, f"{mode}_FrstIndex")
            if fi_pw is None:
                continue
            pg = grp.require_group(mode)
            ds(pg, "FrstIndex", fi_pw.astype(np.float32), c)
            r1 = _arr(features, f"{mode}_Res1")
            r2 = _arr(features, f"{mode}_Res2")
            if r1 is not None and r2 is not None:
                ds(pg, "edge_index",
                   np.stack([r1.astype(np.int32), r2.astype(np.int32)]), c)
            for key, dtype in (
                (f"{mode}_AA1",          np.int8),
                (f"{mode}_AA2",          np.int8),
                (f"{mode}_DensityRes1",  np.float32),
                (f"{mode}_DensityRes2",  np.float32),
                (f"{mode}_NativeEnergy", np.float32),
                (f"{mode}_DecoyEnergy",  np.float32),
                (f"{mode}_SDEnergy",     np.float32),
                (f"{mode}_Welltype",     np.int8),
                (f"{mode}_FrstState",    np.int8),
            ):
                v = _arr(features, key)
                if v is not None:
                    short = key[len(mode) + 1:]  # strip "mode_" prefix
                    ds(pg, short, v.astype(dtype), c)
            for side in ("1", "2"):
                chain = _arr(features, f"{mode}_ChainRes{side}")
                if chain is not None:
                    pg.create_dataset(f"ChainRes{side}", data=chain.astype("S"),
                                      compression=c if chain.size > 16 else None)

        # ── structure [L, 5, 3] ────────────────────────────────────────────────
        if self._save_structure:
            ap = _arr(features, "atom_positions")
            if ap is not None and ap.ndim == 3:
                sg = grp.require_group("structure")
                X = ap[:, :5, :].astype(np.float32)  # N, CA, C, CB, O
                if self._save_sidechain_centroid:
                    sc = _arr(features, "sidechain_centroid")
                    if sc is None:
                        sc_mask = (ap != 0).all(axis=-1)
                        sc_mask[:, :4] = False
                        count = sc_mask.sum(axis=-1, keepdims=True).astype(np.float32) + 1e-8
                        sc = (ap * sc_mask[..., None]).sum(axis=-2).astype(np.float32) / count
                    X = np.concatenate([X, sc[:, None, :].astype(np.float32)], axis=1)
                ds(sg, "X", X, c)
                res = _arr(features, "Res")
                if res is not None:
                    ds(sg, "Res", res.astype(np.int32), c)
                aa = _arr(features, "AA")
                if aa is not None:
                    ds(sg, "AA", aa.astype(np.int8), c)


# ── pt.zst wrapper ────────────────────────────────────────────────────────────

class PtZstShardWriter(BaseShardWriter):
    """Thin wrapper around ShardWriter to satisfy the BaseShardWriter interface."""

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 200,
        shard_size_large: int = 16,
        large_threshold: int = 500,
        rank: int = 0,
        world_size: int = 1,
        zstd_level: int = 3,
        coefficients=None,
        save_structure: bool = False,
        save_sidechain_centroid: bool = False,
    ):
        from .shard_writer import ShardWriter
        self._writer = ShardWriter(
            output_dir=output_dir,
            shard_size=shard_size,
            shard_size_large=shard_size_large,
            large_threshold=large_threshold,
            rank=rank,
            world_size=world_size,
            zstd_level=zstd_level,
        )
        self._output_dir              = Path(output_dir)
        self._coefficients            = coefficients
        self._save_structure          = save_structure
        self._save_sidechain_centroid = save_sidechain_centroid

    def add(self, entry: str, features: dict, length: int) -> None:
        ap = features.get("atom_positions")
        if ap is not None:
            # Rebuild features without atom_positions; add derived tensors if requested
            features = {k: v for k, v in features.items() if k != "atom_positions"}
            if self._save_structure:
                features["backbone_positions"] = ap[:, :5, :]  # [L, 5, 3]
            if self._save_sidechain_centroid:
                sc = features.get("sidechain_centroid")
                if sc is None:
                    ap_np = ap.cpu().numpy() if hasattr(ap, "cpu") else np.asarray(ap)
                    sc_mask = (ap_np != 0).all(axis=-1)        # [L, N_atoms]
                    sc_mask[:, :4] = False                      # exclude N, CA, C, O
                    count = sc_mask.sum(axis=-1, keepdims=True).astype(np.float32) + 1e-8
                    import torch as _torch
                    sc = _torch.from_numpy(
                        (ap_np * sc_mask[..., None]).sum(axis=-2).astype(np.float32) / count
                    )
                features["sidechain_centroid"] = sc            # [L, 3]
        self._writer.add(entry, features, length)

    def record_failure(self) -> None:
        self._writer.record_failure()

    def finalize(self) -> None:
        self._writer.finalize()
        if self._coefficients is not None:
            with open(self._output_dir / "coefficients.json", "w") as f:
                json.dump(self._coefficients.to_dict(), f)


# ── factory ───────────────────────────────────────────────────────────────────

_FORMATS = ("parquet", "hdf5", "h5", "h5native", "pt.zst")

def make_shard_writer(
    fmt: Literal["parquet", "hdf5", "h5", "pt.zst"],
    output_dir: str | Path,
    shard_size: int = 200,
    shard_size_large: int = 16,
    large_threshold: int = 500,
    rank: int = 0,
    world_size: int = 1,
    coefficients=None,
    save_structure: bool = False,
    save_sidechain_centroid: bool = False,
    **fmt_kwargs,
) -> BaseShardWriter:
    """Return a shard writer for the requested format.

    Parameters
    ----------
    fmt:
        Output format: ``"parquet"`` (default), ``"hdf5"`` / ``"h5"``,
        or ``"pt.zst"`` (legacy FrustrAI-Fold format).
    output_dir:
        Directory where shard files will be written.
    shard_size:
        Proteins per shard for normal-length (≤ ``large_threshold``) proteins.
    shard_size_large:
        Proteins per shard for large proteins.
    large_threshold:
        Residue-count threshold separating normal from large buckets.
    rank / world_size:
        For multi-process runs; each rank writes its own shard stream.
    coefficients:
        ``Coefficients`` instance. When provided, embedded natively in each
        shard file at write time — as parquet schema metadata (key
        ``"coefficients"``) for parquet, or as an HDF5 root attribute for
        hdf5, or written to ``coefficients.json`` in the output directory for
        pt.zst.
    save_structure:
        When True, backbone atom positions (N, CA, C, O, CB) are written
        alongside frustration data. For parquet/hdf5 a ``structure`` table is
        added; for pt.zst ``backbone_positions`` (shape ``[L, 5, 3]``) is
        stored as a tensor. ``features`` must contain ``"atom_positions"``
        (shape ``[L, ≥5, 3]``) for this to have any effect.
    save_sidechain_centroid:
        When True (and ``save_structure=True``), the average position of
        non-backbone heavy atoms is stored as ``x_SC, y_SC, z_SC`` columns
        (parquet/hdf5) or a ``sidechain_centroid`` tensor ``[L, 3]`` (pt.zst).
        Computed from ``atom_positions`` when not pre-supplied in ``features``
        as ``"sidechain_centroid"``.
    **fmt_kwargs:
        Format-specific options forwarded to the writer constructor.
        Parquet: ``compression`` (default ``"zstd"``).
        HDF5:    ``compression`` (default ``"zlib"``),
                 ``compression_opts`` (default ``4``) — passed to PyTables.
        pt.zst:  ``zstd_level`` (default ``3``).
    """
    kw = dict(
        output_dir=output_dir,
        shard_size=shard_size,
        shard_size_large=shard_size_large,
        large_threshold=large_threshold,
        rank=rank,
        world_size=world_size,
        coefficients=coefficients,
        save_structure=save_structure,
        save_sidechain_centroid=save_sidechain_centroid,
        **fmt_kwargs,
    )
    if fmt == "parquet":
        return ParquetShardWriter(**kw)
    if fmt in ("hdf5", "h5"):
        return HDF5ShardWriter(**kw)
    if fmt == "h5native":
        return NativeHDF5ShardWriter(**kw)
    if fmt == "pt.zst":
        return PtZstShardWriter(**kw)
    raise ValueError(f"Unknown shard format {fmt!r}. Choose from {_FORMATS}.")
