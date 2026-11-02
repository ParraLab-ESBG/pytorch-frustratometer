"""Thread-safe shard writer for large-scale frustration precomputation.

Buffers per-protein feature dicts and flushes them as compressed .pt.zst shards
when the buffer reaches its target size.  Mirrors the convention used in
FrustrAI-Fold's precompute_features.py:

  * "normal" bucket  — proteins with L ≤ large_threshold (default 500 residues)
  * "large"  bucket  — proteins with L >  large_threshold

Usage::

    writer = ShardWriter(output_dir, rank=0, world_size=1)
    for ...:
        writer.add(entry_id, feature_dict, length)
    writer.finalize()   # flush remaining buffers + write metadata.json
"""

import gc
import json
import threading
from pathlib import Path

import torch

from .shard_io import save_shard_file

_COLLATE_KEYS_PAD0 = {
    # structural / coordinates
    "X", "node_features", "edge_features",
    # singleresidue (ASCII names)
    "FrstIndex", "FrstState", "DensityRes", "NativeEnergy", "DecoyEnergy", "SDEnergy",
    # mutation scan (ASCII: MutantEnergy / ScanFrstIndex / ScanFrstState) — [L, 20]
    "MutantEnergy", "ScanFrstIndex", "ScanFrstState",
    # pairwise — mutational
    "mutational_FrstIndex", "mutational_FrstState",
    "mutational_DensityRes1", "mutational_DensityRes2",
    "mutational_NativeEnergy", "mutational_DecoyEnergy", "mutational_SDEnergy",
    # pairwise — configurational
    "configurational_FrstIndex", "configurational_FrstState",
    "configurational_DensityRes1", "configurational_DensityRes2",
    "configurational_NativeEnergy", "configurational_DecoyEnergy", "configurational_SDEnergy",
}
_COLLATE_KEYS_PAD_1 = {
    # AA indices and residue numbers — pad with -1 so masking works
    "AA", "Res",
    "mutational_AA1", "mutational_AA2", "mutational_Res1", "mutational_Res2",
    "mutational_Welltype",
    "configurational_AA1", "configurational_AA2", "configurational_Res1", "configurational_Res2",
    "configurational_Welltype",
    # edge indices — pad with -1 (matches FrustrAI-Fold convention)
    "mutational_edge_index", "configurational_edge_index",
    # legacy classification targets
    "true_single_frustration_class",
    "true_mutational_frustration_class",
    "true_configurational_frustration_class",
}


def _collate(features: list[dict]) -> dict:
    """Pad variable-length tensors along dim-0 and stack into a batch."""
    if not features:
        return {}

    def _pad(tensors, pad_value):
        max_sizes = [max(t.shape[d] for t in tensors) for d in range(tensors[0].dim())]
        out = []
        for t in tensors:
            for d, max_s in enumerate(max_sizes):
                deficit = max_s - t.shape[d]
                if deficit > 0:
                    pad_shape = list(t.shape)
                    pad_shape[d] = deficit
                    t = torch.cat([t, torch.full(pad_shape, pad_value, dtype=t.dtype)], dim=d)
            out.append(t)
        return torch.stack(out)

    result = {}
    for key in features[0]:
        vals = [f[key] for f in features]
        if not isinstance(vals[0], torch.Tensor):
            result[key] = vals
            continue
        pad = -1 if key in _COLLATE_KEYS_PAD_1 else 0
        result[key] = _pad(vals, pad)
    return result


class ShardWriter:
    """Accumulate per-protein dicts and flush to .pt.zst shards.

    Thread-safe: add() may be called from multiple Lightning DDP workers
    (though in practice each rank writes its own shard stream).
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 200,
        shard_size_large: int = 16,
        large_threshold: int = 500,
        rank: int = 0,
        world_size: int = 1,
        zstd_level: int = 3,
    ):
        self.output_dir     = Path(output_dir)
        self.shard_size     = shard_size
        self.shard_size_large = shard_size_large
        self.large_threshold  = large_threshold
        self.rank           = rank
        self.world_size     = world_size
        self.zstd_level     = zstd_level

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock    = threading.Lock()
        self._buffers = {"normal": [], "large": []}
        self._counts  = {"normal": 0, "large": 0}
        self._n_ok    = 0
        self._n_fail  = 0
        self._lengths: list[int] = []

    # ── public API ────────────────────────────────────────────────────────────

    def add(self, entry: str, features: dict, length: int):
        """Add one protein's feature dict to the appropriate buffer."""
        bucket = "large" if length > self.large_threshold else "normal"
        cap    = self.shard_size_large if bucket == "large" else self.shard_size
        buf = None
        idx = None

        with self._lock:
            self._buffers[bucket].append(features)
            self._lengths.append(length)
            self._n_ok += 1

            if len(self._buffers[bucket]) >= cap:
                buf = self._buffers[bucket][:]
                idx = self._counts[bucket]
                self._buffers[bucket].clear()
                self._counts[bucket] += 1

        if buf is not None:
            self._flush(buf, bucket, idx)

    def record_failure(self):
        with self._lock:
            self._n_fail += 1

    def finalize(self):
        """Flush remaining partial buffers and write metadata.json."""
        for bucket in ("normal", "large"):
            with self._lock:
                buf = self._buffers[bucket][:]
                idx = self._counts[bucket]
                self._buffers[bucket].clear()
                self._counts[bucket] += 1
            if buf:
                self._flush(buf, bucket, idx)

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "num_samples": self._n_ok,
                "lengths":     self._lengths,
                "rank":        self.rank,
                "world_size":  self.world_size,
            }, f)
        print(f"[Shards] rank={self.rank}: {self._n_ok} ok, {self._n_fail} failed "
              f"→ {self.output_dir}", flush=True)

    # ── internals ─────────────────────────────────────────────────────────────

    def _shard_path(self, bucket: str, idx: int) -> Path:
        suffix = f"_rank{self.rank}" if self.world_size > 1 else ""
        return self.output_dir / f"shard_{bucket}{suffix}_{idx:04d}.pt.zst"

    def _flush(self, buf: list[dict], bucket: str, idx: int):
        shard = _collate(buf)
        path  = self._shard_path(bucket, idx)
        save_shard_file(shard, path, level=self.zstd_level)
        print(f"[Shards] wrote {path.name}  ({len(buf)} proteins)", flush=True)
        del shard, buf
        gc.collect()
