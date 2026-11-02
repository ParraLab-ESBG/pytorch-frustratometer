"""Unit tests for pytorch_frustratometer/data/shard_formats.py.

Tests cover:
  - _arr() helper
  - _features_to_rows() for all modes and combinations
  - ParquetShardWriter roundtrip
  - HDF5ShardWriter roundtrip (skipped when PyTables not installed)
  - PtZstShardWriter smoke test
  - make_shard_writer() factory
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sr_features(L: int = 6, seed: int = 0) -> dict:
    """Minimal singleresidue-only feature dict (all keys present)."""
    rng = torch.manual_seed(seed)
    return {
        "Res":          torch.arange(1, L + 1, dtype=torch.int32),
        "ChainRes":     np.array([f"A{i}" for i in range(1, L + 1)]),
        "AA":           torch.randint(0, 20, (L,), dtype=torch.int8),
        "DensityRes":   torch.randn(L).float(),
        "NativeEnergy": torch.randn(L).float(),
        "DecoyEnergy":  torch.randn(L).float(),
        "SDEnergy":     torch.randn(L).abs().float(),
        "FrstIndex":    torch.randn(L).float(),
        "FrstState":    torch.randint(0, 3, (L,), dtype=torch.int8),
    }


def _add_scan(features: dict) -> dict:
    """Attach [L,20] mutation-scan keys to an existing features dict."""
    L = features["FrstIndex"].shape[0]
    features["MutantEnergy"]  = torch.randn(L, 20).float()
    features["ScanFrstIndex"] = torch.randn(L, 20).float()
    features["ScanFrstState"] = torch.randint(0, 3, (L, 20), dtype=torch.int8)
    return features


def _add_pairwise(features: dict, mode: str, E: int = 8) -> dict:
    """Attach pairwise keys for *mode* (mutational / configurational)."""
    L = features["FrstIndex"].shape[0]
    features[f"{mode}_FrstIndex"]    = torch.randn(E).float()
    features[f"{mode}_FrstState"]    = torch.randint(0, 3, (E,), dtype=torch.int8)
    features[f"{mode}_Res1"]         = torch.randint(1, L, (E,), dtype=torch.int32)
    features[f"{mode}_Res2"]         = torch.randint(1, L, (E,), dtype=torch.int32)
    features[f"{mode}_ChainRes1"]    = np.array([f"A{i}" for i in range(E)])
    features[f"{mode}_ChainRes2"]    = np.array([f"A{i}" for i in range(E)])
    features[f"{mode}_AA1"]          = torch.randint(0, 20, (E,), dtype=torch.int8)
    features[f"{mode}_AA2"]          = torch.randint(0, 20, (E,), dtype=torch.int8)
    features[f"{mode}_DensityRes1"]  = torch.randn(E).float()
    features[f"{mode}_DensityRes2"]  = torch.randn(E).float()
    features[f"{mode}_NativeEnergy"] = torch.randn(E).float()
    features[f"{mode}_DecoyEnergy"]  = torch.randn(E).float()
    features[f"{mode}_SDEnergy"]     = torch.randn(E).abs().float()
    features[f"{mode}_Welltype"]     = torch.randint(0, 3, (E,), dtype=torch.int8)
    return features


# ---------------------------------------------------------------------------
# _arr helper
# ---------------------------------------------------------------------------

class TestArr:
    def _arr(self, features, key):
        from pytorch_frustratometer.data.shard_formats import _arr
        return _arr(features, key)

    def test_torch_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        out = self._arr({"x": t}, "x")
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_almost_equal(out, [1.0, 2.0, 3.0])

    def test_numpy_array(self):
        a = np.array([4, 5, 6])
        out = self._arr({"x": a}, "x")
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, a)

    def test_python_list(self):
        out = self._arr({"x": ["A", "B"]}, "x")
        assert isinstance(out, np.ndarray)
        assert list(out) == ["A", "B"]

    def test_absent_key_returns_none(self):
        out = self._arr({}, "missing")
        assert out is None


# ---------------------------------------------------------------------------
# _features_to_rows — singleresidue
# ---------------------------------------------------------------------------

class TestFeaturesToRowsSR:
    @pytest.fixture(autouse=True)
    def _import(self):
        from pytorch_frustratometer.data.shard_formats import _features_to_rows
        self._fn = _features_to_rows

    def test_singleresidue_present(self):
        feat = _make_sr_features(L=5)
        dfs = self._fn("1abc", feat)
        assert "singleresidue" in dfs

    def test_singleresidue_absent_when_no_FrstIndex(self):
        feat = _make_sr_features(L=5)
        del feat["FrstIndex"]
        dfs = self._fn("1abc", feat)
        assert "singleresidue" not in dfs

    def test_singleresidue_columns(self):
        feat = _make_sr_features(L=5)
        df = self._fn("1abc", feat)["singleresidue"]
        expected = {"entry", "Res", "ChainRes", "AA", "DensityRes",
                    "NativeEnergy", "DecoyEnergy", "SDEnergy", "FrstIndex", "FrstState"}
        assert set(df.columns) == expected

    def test_singleresidue_row_count(self):
        L = 7
        df = self._fn("x", _make_sr_features(L=L))["singleresidue"]
        assert len(df) == L

    def test_singleresidue_entry_column(self):
        df = self._fn("myprotein", _make_sr_features(L=3))["singleresidue"]
        assert (df["entry"] == "myprotein").all()

    def test_singleresidue_dtypes(self):
        df = self._fn("x", _make_sr_features(L=4))["singleresidue"]
        assert df["Res"].dtype == np.int32
        assert df["AA"].dtype == np.int8
        assert df["FrstIndex"].dtype == np.float32
        assert df["FrstState"].dtype == np.int8

    def test_singleresidue_values_match(self):
        feat = _make_sr_features(L=5)
        df = self._fn("x", feat)["singleresidue"]
        np.testing.assert_array_equal(df["Res"].values, feat["Res"].numpy())
        np.testing.assert_array_almost_equal(df["FrstIndex"].values,
                                             feat["FrstIndex"].numpy())

    def test_singleresidue_missing_optional_keys_filled(self):
        feat = {"FrstIndex": torch.randn(4).float(), "FrstState": torch.zeros(4, dtype=torch.int8)}
        df = self._fn("x", feat)["singleresidue"]
        assert df["DensityRes"].dtype == np.float32
        assert df["Res"].dtype == np.int32
        assert (df["Res"] == -1).all()


# ---------------------------------------------------------------------------
# _features_to_rows — mutation scan
# ---------------------------------------------------------------------------

class TestFeaturesToRowsScan:
    @pytest.fixture(autouse=True)
    def _import(self):
        from pytorch_frustratometer.data.shard_formats import _features_to_rows
        self._fn = _features_to_rows

    def test_scan_absent_when_no_ScanFrstIndex(self):
        dfs = self._fn("x", _make_sr_features(L=4))
        assert "singleresidue_mutations" not in dfs

    def test_scan_present(self):
        feat = _add_scan(_make_sr_features(L=4))
        dfs = self._fn("x", feat)
        assert "singleresidue_mutations" in dfs

    def test_scan_row_count(self):
        L = 5
        feat = _add_scan(_make_sr_features(L=L))
        df = self._fn("x", feat)["singleresidue_mutations"]
        assert len(df) == L * 20

    def test_scan_columns(self):
        feat = _add_scan(_make_sr_features(L=3))
        df = self._fn("x", feat)["singleresidue_mutations"]
        expected = {"entry", "Res", "ChainRes", "AA_native", "AA_mutant",
                    "NativeEnergy", "MutantEnergy", "ScanFrstIndex", "ScanFrstState"}
        assert set(df.columns) == expected

    def test_scan_aa_mutant_range(self):
        L = 4
        feat = _add_scan(_make_sr_features(L=L))
        df = self._fn("x", feat)["singleresidue_mutations"]
        # Each residue should have all 20 AA mutants
        for r in range(L):
            block = df.iloc[r * 20:(r + 1) * 20]
            assert sorted(block["AA_mutant"].tolist()) == list(range(20))

    def test_scan_res_repeats(self):
        L = 3
        feat = _add_scan(_make_sr_features(L=L))
        df = self._fn("x", feat)["singleresidue_mutations"]
        for r, res_val in enumerate(feat["Res"].numpy()):
            block = df.iloc[r * 20:(r + 1) * 20]
            assert (block["Res"] == res_val).all()

    def test_scan_mutant_energy_flat(self):
        L = 3
        feat = _add_scan(_make_sr_features(L=L))
        df = self._fn("x", feat)["singleresidue_mutations"]
        expected = feat["MutantEnergy"].numpy().reshape(-1)
        np.testing.assert_array_almost_equal(df["MutantEnergy"].values, expected)


# ---------------------------------------------------------------------------
# _features_to_rows — pairwise
# ---------------------------------------------------------------------------

class TestFeaturesToRowsPairwise:
    @pytest.fixture(autouse=True)
    def _import(self):
        from pytorch_frustratometer.data.shard_formats import _features_to_rows
        self._fn = _features_to_rows

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_present(self, mode):
        feat = _add_pairwise(_make_sr_features(L=6), mode, E=10)
        dfs = self._fn("x", feat)
        assert mode in dfs

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_absent_when_no_FrstIndex(self, mode):
        dfs = self._fn("x", _make_sr_features(L=6))
        assert mode not in dfs

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_columns(self, mode):
        E = 8
        feat = _add_pairwise(_make_sr_features(L=6), mode, E=E)
        df = self._fn("x", feat)[mode]
        expected = {"entry", "Res1", "Res2", "ChainRes1", "ChainRes2",
                    "AA1", "AA2", "DensityRes1", "DensityRes2",
                    "NativeEnergy", "DecoyEnergy", "SDEnergy",
                    "Welltype", "FrstIndex", "FrstState"}
        assert set(df.columns) == expected

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_row_count(self, mode):
        E = 9
        feat = _add_pairwise(_make_sr_features(L=6), mode, E=E)
        df = self._fn("x", feat)[mode]
        assert len(df) == E

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_dtypes(self, mode):
        feat = _add_pairwise(_make_sr_features(L=6), mode, E=5)
        df = self._fn("x", feat)[mode]
        assert df["Res1"].dtype == np.int32
        assert df["AA1"].dtype == np.int8
        assert df["FrstIndex"].dtype == np.float32
        assert df["FrstState"].dtype == np.int8

    def test_both_pairwise_modes(self):
        feat = _make_sr_features(L=6)
        _add_pairwise(feat, "mutational", E=5)
        _add_pairwise(feat, "configurational", E=7)
        dfs = self._fn("x", feat)
        assert "mutational" in dfs and "configurational" in dfs
        assert len(dfs["mutational"]) == 5
        assert len(dfs["configurational"]) == 7


# ---------------------------------------------------------------------------
# ParquetShardWriter roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
    reason="pyarrow not installed",
)
class TestParquetShardWriter:
    def test_sr_roundtrip(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        import pandas as pd

        writer = make_shard_writer("parquet", tmp_path, shard_size=1)
        feat = _make_sr_features(L=5)
        writer.add("prot1", feat, 5)
        writer.finalize()

        parquets = list(tmp_path.glob("shard_singleresidue_*.parquet"))
        assert len(parquets) >= 1
        df = pd.read_parquet(parquets[0])
        assert set(df.columns) >= {"entry", "Res", "AA", "FrstIndex", "FrstState"}
        assert (df["entry"] == "prot1").all()
        assert len(df) == 5

    def test_multiple_proteins_accumulate(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        import pandas as pd

        writer = make_shard_writer("parquet", tmp_path, shard_size=10)
        for i in range(3):
            writer.add(f"prot{i}", _make_sr_features(L=4, seed=i), 4)
        writer.finalize()

        parquets = list(tmp_path.glob("shard_singleresidue_*.parquet"))
        combined = pd.concat([pd.read_parquet(p) for p in parquets])
        assert combined["entry"].nunique() == 3
        assert len(combined) == 12

    def test_mutation_scan_shard_written(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        import pandas as pd

        writer = make_shard_writer("parquet", tmp_path, shard_size=1)
        feat = _add_scan(_make_sr_features(L=4))
        writer.add("p1", feat, 4)
        writer.finalize()

        scan_files = list(tmp_path.glob("shard_singleresidue_mutations_*.parquet"))
        assert len(scan_files) >= 1
        df = pd.read_parquet(scan_files[0])
        assert len(df) == 4 * 20
        assert "AA_mutant" in df.columns
        assert "ScanFrstIndex" in df.columns

    def test_pairwise_shard_written(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        import pandas as pd

        writer = make_shard_writer("parquet", tmp_path, shard_size=1)
        feat = _add_pairwise(_make_sr_features(L=5), "mutational", E=6)
        writer.add("p1", feat, 5)
        writer.finalize()

        pw_files = list(tmp_path.glob("shard_mutational_*.parquet"))
        assert len(pw_files) >= 1
        df = pd.read_parquet(pw_files[0])
        assert "Res1" in df.columns and "Welltype" in df.columns
        assert len(df) == 6

    def test_metadata_json_written(self, tmp_path):
        import json
        from pytorch_frustratometer.data.shard_formats import make_shard_writer

        writer = make_shard_writer("parquet", tmp_path, shard_size=10)
        writer.add("p1", _make_sr_features(L=4), 4)
        writer.finalize()

        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["num_samples"] == 1
        assert meta["format"] == "parquet"

    def test_large_bucket_separate_shards(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        import pandas as pd

        writer = make_shard_writer("parquet", tmp_path, shard_size=10,
                                   shard_size_large=1, large_threshold=10)
        writer.add("small", _make_sr_features(L=5), 5)
        writer.add("large", _make_sr_features(L=15, seed=1), 15)
        writer.finalize()

        normal_files = list(tmp_path.glob("shard_singleresidue_normal_*.parquet"))
        large_files  = list(tmp_path.glob("shard_singleresidue_large_*.parquet"))
        assert len(normal_files) >= 1
        assert len(large_files)  >= 1

    def test_record_failure_counts(self, tmp_path):
        import json
        from pytorch_frustratometer.data.shard_formats import make_shard_writer

        writer = make_shard_writer("parquet", tmp_path, shard_size=10)
        writer.add("p1", _make_sr_features(L=3), 3)
        writer.record_failure()
        writer.record_failure()
        writer.finalize()

        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["num_samples"] == 1

    def test_values_preserved_roundtrip(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        import pandas as pd

        feat = _make_sr_features(L=4, seed=7)
        writer = make_shard_writer("parquet", tmp_path, shard_size=1)
        writer.add("p1", feat, 4)
        writer.finalize()

        df = pd.read_parquet(next(tmp_path.glob("shard_singleresidue_*.parquet")))
        np.testing.assert_array_almost_equal(
            df["FrstIndex"].values, feat["FrstIndex"].numpy(), decimal=5
        )
        np.testing.assert_array_equal(df["AA"].values, feat["AA"].numpy())


# ---------------------------------------------------------------------------
# HDF5ShardWriter roundtrip
# ---------------------------------------------------------------------------

class TestHDF5ShardWriter:
    @pytest.fixture(autouse=True)
    def skip_if_no_tables(self):
        pytest.importorskip("tables")

    def test_sr_roundtrip(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        import pandas as pd

        writer = make_shard_writer("hdf5", tmp_path, shard_size=1)
        feat = _make_sr_features(L=5)
        writer.add("prot1", feat, 5)
        writer.finalize()

        h5_files = list(tmp_path.glob("shard_singleresidue_*.h5"))
        assert len(h5_files) >= 1
        df = pd.read_hdf(h5_files[0], key="singleresidue")
        assert "FrstIndex" in df.columns
        assert len(df) == 5

    def test_error_message_without_tables(self, tmp_path, monkeypatch):
        import pandas as pd
        from pytorch_frustratometer.data.shard_formats import HDF5ShardWriter

        monkeypatch.setattr(pd.DataFrame, "to_hdf",
                            lambda *a, **kw: (_ for _ in ()).throw(
                                ImportError("No module named 'tables'")))

        writer = HDF5ShardWriter(tmp_path, shard_size=1)
        feat = _make_sr_features(L=3)

        with pytest.raises(ImportError, match="pip install tables"):
            writer.add("p1", feat, 3)
            writer.finalize()


# ---------------------------------------------------------------------------
# PtZstShardWriter smoke test
# ---------------------------------------------------------------------------

class TestPtZstShardWriter:
    def test_add_and_finalize(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer

        writer = make_shard_writer("pt.zst", tmp_path, shard_size=2)
        for i in range(3):
            feat = _make_sr_features(L=4, seed=i)
            writer.add(f"p{i}", feat, 4)
        writer.finalize()

        assert (tmp_path / "metadata.json").exists()

    def test_shard_files_written(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer

        writer = make_shard_writer("pt.zst", tmp_path, shard_size=2)
        for i in range(4):
            feat = _make_sr_features(L=3, seed=i)
            writer.add(f"p{i}", feat, 3)
        writer.finalize()

        shards = list(tmp_path.glob("*.pt.zst"))
        assert len(shards) >= 1

    def test_save_structure_stores_backbone_positions(self, tmp_path):
        """save_structure=True: atom_positions sliced to backbone_positions [L,5,3]."""
        from pytorch_frustratometer.data.shard_formats import PtZstShardWriter

        L = 5
        writer = PtZstShardWriter(tmp_path, shard_size=10, save_structure=True)
        feat = _make_sr_features(L=L)
        feat["atom_positions"] = torch.randn(L, 37, 3).float()

        captured = {}
        original_add = writer._writer.add
        def _capture(entry, features, length):
            captured.update(features)
            original_add(entry, features, length)
        writer._writer.add = _capture

        writer.add("p1", feat, L)
        writer.finalize()

        assert "atom_positions" not in captured, "full atom_positions should be dropped"
        assert "backbone_positions" in captured
        assert captured["backbone_positions"].shape == (L, 5, 3)

    def test_save_structure_false_drops_atom_positions(self, tmp_path):
        """save_structure=False (default): atom_positions dropped, no backbone_positions."""
        from pytorch_frustratometer.data.shard_formats import PtZstShardWriter

        L = 4
        writer = PtZstShardWriter(tmp_path, shard_size=10)
        feat = _make_sr_features(L=L)
        feat["atom_positions"] = torch.randn(L, 37, 3).float()

        captured = {}
        original_add = writer._writer.add
        def _capture(entry, features, length):
            captured.update(features)
            original_add(entry, features, length)
        writer._writer.add = _capture

        writer.add("p1", feat, L)
        writer.finalize()

        assert "atom_positions"     not in captured
        assert "backbone_positions" not in captured

    def test_save_sidechain_centroid_stores_tensor(self, tmp_path):
        """save_sidechain_centroid=True: sidechain_centroid [L,3] stored."""
        from pytorch_frustratometer.data.shard_formats import PtZstShardWriter

        L = 5
        writer = PtZstShardWriter(tmp_path, shard_size=10,
                                  save_structure=True, save_sidechain_centroid=True)
        feat = _make_sr_features(L=L)
        feat["atom_positions"] = torch.randn(L, 37, 3).float()

        captured = {}
        original_add = writer._writer.add
        def _capture(entry, features, length):
            captured.update(features)
            original_add(entry, features, length)
        writer._writer.add = _capture

        writer.add("p1", feat, L)
        writer.finalize()

        assert "sidechain_centroid" in captured
        assert captured["sidechain_centroid"].shape == (L, 3)


# ---------------------------------------------------------------------------
# make_shard_writer factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_parquet_returns_parquet_writer(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import (
            make_shard_writer, ParquetShardWriter,
        )
        w = make_shard_writer("parquet", tmp_path)
        assert isinstance(w, ParquetShardWriter)

    def test_hdf5_returns_hdf5_writer(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import (
            make_shard_writer, HDF5ShardWriter,
        )
        assert isinstance(make_shard_writer("hdf5", tmp_path), HDF5ShardWriter)
        assert isinstance(make_shard_writer("h5",   tmp_path), HDF5ShardWriter)

    def test_ptzst_returns_ptzst_writer(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import (
            make_shard_writer, PtZstShardWriter,
        )
        assert isinstance(make_shard_writer("pt.zst", tmp_path), PtZstShardWriter)

    def test_h5native_returns_native_writer(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import (
            make_shard_writer, NativeHDF5ShardWriter,
        )
        assert isinstance(make_shard_writer("h5native", tmp_path), NativeHDF5ShardWriter)

    def test_unknown_format_raises(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        with pytest.raises(ValueError, match="Unknown shard format"):
            make_shard_writer("csv", tmp_path)


# ---------------------------------------------------------------------------
# NativeHDF5ShardWriter roundtrip
# ---------------------------------------------------------------------------

class TestNativeHDF5ShardWriter:
    @pytest.fixture(autouse=True)
    def skip_if_no_h5py(self):
        pytest.importorskip("h5py")

    def _writer(self, tmp_path, **kw):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        return make_shard_writer("h5native", tmp_path, shard_size=1, **kw)

    def _open(self, tmp_path):
        import h5py
        files = list(tmp_path.glob("shard_normal_*.h5native"))
        assert files, "no shard_normal_*.h5native files found"
        return h5py.File(files[0], "r")

    # ── basic structure ───────────────────────────────────────────────────────

    def test_file_created(self, tmp_path):
        w = self._writer(tmp_path)
        w.add("prot1", _make_sr_features(L=5), 5)
        w.finalize()
        assert list(tmp_path.glob("shard_normal_*.h5native"))

    def test_entries_dataset_present(self, tmp_path):
        w = self._writer(tmp_path)
        w.add("prot1", _make_sr_features(L=5), 5)
        w.finalize()
        with self._open(tmp_path) as hf:
            assert "entries" in hf
            entries = [e.decode() if isinstance(e, bytes) else e for e in hf["entries"]]
            assert "prot1" in entries

    def test_protein_group_present(self, tmp_path):
        w = self._writer(tmp_path)
        w.add("prot1", _make_sr_features(L=5), 5)
        w.finalize()
        with self._open(tmp_path) as hf:
            assert "prot1" in hf

    # ── singleresidue ─────────────────────────────────────────────────────────

    def test_singleresidue_group(self, tmp_path):
        w = self._writer(tmp_path)
        w.add("p1", _make_sr_features(L=6), 6)
        w.finalize()
        with self._open(tmp_path) as hf:
            assert "singleresidue" in hf["p1"]

    def test_singleresidue_array_shapes(self, tmp_path):
        L = 7
        w = self._writer(tmp_path)
        w.add("p1", _make_sr_features(L=L), L)
        w.finalize()
        with self._open(tmp_path) as hf:
            sg = hf["p1/singleresidue"]
            for key in ("Res", "AA", "FrstIndex", "FrstState", "DensityRes"):
                assert sg[key].shape == (L,), f"{key} shape wrong"

    def test_singleresidue_values_roundtrip(self, tmp_path):
        feat = _make_sr_features(L=5, seed=42)
        w = self._writer(tmp_path)
        w.add("p1", feat, 5)
        w.finalize()
        with self._open(tmp_path) as hf:
            np.testing.assert_array_almost_equal(
                hf["p1/singleresidue/FrstIndex"][:],
                feat["FrstIndex"].numpy(), decimal=5,
            )
            np.testing.assert_array_equal(
                hf["p1/singleresidue/AA"][:],
                feat["AA"].numpy(),
            )

    def test_singleresidue_dtypes(self, tmp_path):
        w = self._writer(tmp_path)
        w.add("p1", _make_sr_features(L=4), 4)
        w.finalize()
        with self._open(tmp_path) as hf:
            sg = hf["p1/singleresidue"]
            assert sg["Res"].dtype == np.int32
            assert sg["AA"].dtype == np.int8
            assert sg["FrstIndex"].dtype == np.float32
            assert sg["FrstState"].dtype == np.int8

    # ── mutation scan stored at [L, 20] — NOT flattened ───────────────────────

    def test_mutation_scan_native_shape(self, tmp_path):
        L = 5
        feat = _add_scan(_make_sr_features(L=L))
        w = self._writer(tmp_path)
        w.add("p1", feat, L)
        w.finalize()
        with self._open(tmp_path) as hf:
            assert "mutation_scan" in hf["p1"]
            assert hf["p1/mutation_scan/ScanFrstIndex"].shape == (L, 20)
            assert hf["p1/mutation_scan/MutantEnergy"].shape == (L, 20)
            assert hf["p1/mutation_scan/ScanFrstState"].shape == (L, 20)

    def test_mutation_scan_values(self, tmp_path):
        L = 4
        feat = _add_scan(_make_sr_features(L=L))
        w = self._writer(tmp_path)
        w.add("p1", feat, L)
        w.finalize()
        with self._open(tmp_path) as hf:
            np.testing.assert_array_almost_equal(
                hf["p1/mutation_scan/ScanFrstIndex"][:],
                feat["ScanFrstIndex"].numpy(), decimal=5,
            )

    # ── pairwise modes ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_group(self, tmp_path, mode):
        feat = _add_pairwise(_make_sr_features(L=6), mode, E=8)
        w = self._writer(tmp_path)
        w.add("p1", feat, 6)
        w.finalize()
        with self._open(tmp_path) as hf:
            assert mode in hf["p1"]

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_edge_index_shape(self, tmp_path, mode):
        E = 10
        feat = _add_pairwise(_make_sr_features(L=6), mode, E=E)
        w = self._writer(tmp_path)
        w.add("p1", feat, 6)
        w.finalize()
        with self._open(tmp_path) as hf:
            ei = hf[f"p1/{mode}/edge_index"]
            assert ei.shape == (2, E)

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_arrays_shape(self, tmp_path, mode):
        E = 8
        feat = _add_pairwise(_make_sr_features(L=6), mode, E=E)
        w = self._writer(tmp_path)
        w.add("p1", feat, 6)
        w.finalize()
        with self._open(tmp_path) as hf:
            pg = hf[f"p1/{mode}"]
            for key in ("FrstIndex", "AA1", "AA2", "Welltype", "NativeEnergy"):
                assert pg[key].shape == (E,), f"{key} shape wrong"

    @pytest.mark.parametrize("mode", ["mutational", "configurational"])
    def test_pairwise_edge_index_values(self, tmp_path, mode):
        E = 6
        feat = _add_pairwise(_make_sr_features(L=6), mode, E=E)
        w = self._writer(tmp_path)
        w.add("p1", feat, 6)
        w.finalize()
        with self._open(tmp_path) as hf:
            ei = hf[f"p1/{mode}/edge_index"][:]
            np.testing.assert_array_equal(ei[0], feat[f"{mode}_Res1"].numpy())
            np.testing.assert_array_equal(ei[1], feat[f"{mode}_Res2"].numpy())

    # ── structure [L, 5, 3] ───────────────────────────────────────────────────

    def test_structure_stored(self, tmp_path):
        L = 5
        feat = _make_sr_features(L=L)
        feat["atom_positions"] = torch.randn(L, 37, 3).float()
        w = self._writer(tmp_path, save_structure=True)
        w.add("p1", feat, L)
        w.finalize()
        with self._open(tmp_path) as hf:
            assert "structure" in hf["p1"]
            assert hf["p1/structure/X"].shape == (L, 5, 3)

    def test_structure_values_match_backbone(self, tmp_path):
        L = 4
        feat = _make_sr_features(L=L)
        ap = torch.randn(L, 37, 3).float()
        feat["atom_positions"] = ap
        w = self._writer(tmp_path, save_structure=True)
        w.add("p1", feat, L)
        w.finalize()
        with self._open(tmp_path) as hf:
            np.testing.assert_array_almost_equal(
                hf["p1/structure/X"][:],
                ap[:, :5, :].numpy(), decimal=5,
            )

    def test_structure_with_sidechain_shape(self, tmp_path):
        L = 5
        feat = _make_sr_features(L=L)
        feat["atom_positions"] = torch.randn(L, 37, 3).float()
        w = self._writer(tmp_path, save_structure=True, save_sidechain_centroid=True)
        w.add("p1", feat, L)
        w.finalize()
        with self._open(tmp_path) as hf:
            assert hf["p1/structure/X"].shape == (L, 6, 3)  # 5 backbone + 1 sidechain

    def test_no_structure_by_default(self, tmp_path):
        L = 4
        feat = _make_sr_features(L=L)
        feat["atom_positions"] = torch.randn(L, 37, 3).float()
        w = self._writer(tmp_path)
        w.add("p1", feat, L)
        w.finalize()
        with self._open(tmp_path) as hf:
            assert "structure" not in hf["p1"]

    # ── multi-protein shard ───────────────────────────────────────────────────

    def test_multiple_proteins_one_file(self, tmp_path):
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        # shard_size=10 so all 3 land in one file
        w = make_shard_writer("h5native", tmp_path, shard_size=10)
        for i in range(3):
            w.add(f"prot{i}", _make_sr_features(L=4+i, seed=i), 4+i)
        w.finalize()
        with self._open(tmp_path) as hf:
            for i in range(3):
                assert f"prot{i}" in hf
                assert hf[f"prot{i}/singleresidue/FrstIndex"].shape == (4+i,)

    def test_large_bucket_separate_file(self, tmp_path):
        import h5py
        from pytorch_frustratometer.data.shard_formats import make_shard_writer
        w = make_shard_writer("h5native", tmp_path, shard_size=10,
                              shard_size_large=1, large_threshold=10)
        w.add("small", _make_sr_features(L=5), 5)
        w.add("large", _make_sr_features(L=15, seed=1), 15)
        w.finalize()
        normal_files = list(tmp_path.glob("shard_normal_*.h5native"))
        large_files  = list(tmp_path.glob("shard_large_*.h5native"))
        assert normal_files
        assert large_files
        with h5py.File(large_files[0], "r") as hf:
            assert "large" in hf

    # ── metadata ─────────────────────────────────────────────────────────────

    def test_metadata_written(self, tmp_path):
        import json
        w = self._writer(tmp_path)
        w.add("p1", _make_sr_features(L=4), 4)
        w.finalize()
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["format"] == "h5native"
        assert meta["num_samples"] == 1

    def test_coefficients_stored_as_attribute(self, tmp_path):
        import json, h5py
        from pytorch_frustratometer.coefficients import Coefficients
        coeffs = Coefficients()
        w = self._writer(tmp_path, coefficients=coeffs)
        w.add("p1", _make_sr_features(L=4), 4)
        w.finalize()
        with self._open(tmp_path) as hf:
            stored = json.loads(hf.attrs["coefficients"])
            assert "tert_frust_ndecoys" in stored
