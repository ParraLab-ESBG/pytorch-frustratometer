"""Shared read/write helpers for shard files, with transparent zstd compression.

Shards are torch.save's pickle-based zip format, which needs seekable I/O to
locate its central directory -- so a compressed shard is decompressed fully
into an in-memory buffer before torch.load rather than streamed. Shards are
at most ~1GB, trivial for node RAM.

New shards are always written as .pt.zst; .pt (uncompressed, pre-existing)
is still readable so this is backward compatible with shards on disk before
this module existed.
"""
import io
from pathlib import Path

import torch
import zstandard as zstd

ZSTD_SUFFIX = ".pt.zst"


def shard_output_name(name: str) -> str:
    """Normalize a shard filename (any of .pt / .pt.zst) to the .pt.zst form."""
    return name.removesuffix(".zst").removesuffix(".pt") + ".pt.zst"


def glob_shards(shard_dir):
    """Return shard_*.pt and shard_*.pt.zst files in one sorted list."""
    shard_dir = Path(shard_dir)
    return sorted(list(shard_dir.glob("shard_*.pt")) + list(shard_dir.glob("shard_*.pt.zst")))


def load_shard_file(path, map_location="cpu", weights_only=True):
    path = str(path)
    if path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            raw = dctx.decompress(f.read())
        return torch.load(io.BytesIO(raw), map_location=map_location, weights_only=weights_only)
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def save_shard_file(obj, path, level=3):
    """Save `obj` compressed to `path` (should end in .pt.zst)."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    cctx = zstd.ZstdCompressor(level=level)
    with open(path, "wb") as f:
        f.write(cctx.compress(buf.getvalue()))
