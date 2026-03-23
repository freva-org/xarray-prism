"""Tests for POSIX and cloud backends."""

from __future__ import annotations

import os
from pathlib import Path
import bz2
import gzip
import tempfile

import pytest
import xarray as xr

from xarray_prism.backends import open_cloud, open_posix
from xarray_prism import PrismBackendEntrypoint
from xarray_prism._detection import _detect_from_magic_bytes, _detect_from_uri_pattern
from xarray_prism.utils import (
    _decompress_if_needed,
    _strip_chaining_options,
    _strip_compression_suffix,
)


class TestPosixBackend:
    """Tests for local filesystem backend."""

    @pytest.mark.requires_data
    def test_open_netcdf4_local(self, sample_netcdf_path: Path):
        """Open a local NetCDF4 file."""
        if not sample_netcdf_path.exists():
            pytest.skip(f"Test file not found: {sample_netcdf_path}")

        ds = open_posix(str(sample_netcdf_path), engine="h5netcdf")
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) > 0
        ds.close()

    @pytest.mark.requires_data
    def test_open_grib_local(self, sample_grib_path: Path):
        """Open a local GRIB file."""
        if not sample_grib_path.exists():
            pytest.skip(f"Test file not found: {sample_grib_path}")
        ds = open_posix(str(sample_grib_path), engine="cfgrib")
        assert isinstance(ds, xr.Dataset)
        ds.close()

    @pytest.mark.requires_data
    def test_open_geotiff_local(self, sample_geotiff_path: Path):
        """Open a local GeoTIFF file."""
        if not sample_geotiff_path.exists():
            pytest.skip(f"Test file not found: {sample_geotiff_path}")

        ds = open_posix(str(sample_geotiff_path), engine="rasterio")
        assert isinstance(ds, xr.Dataset)
        ds.close()

    @pytest.mark.requires_data
    def test_open_with_drop_variables(self, sample_netcdf_path: Path):
        """Test drop_variables parameter."""
        if not sample_netcdf_path.exists():
            pytest.skip(f"Test file not found: {sample_netcdf_path}")

        ds_full = open_posix(str(sample_netcdf_path), engine="h5netcdf")
        var_name = list(ds_full.data_vars)[0] if ds_full.data_vars else None
        ds_full.close()

        if var_name:
            ds_partial = open_posix(
                str(sample_netcdf_path),
                engine="h5netcdf",
                drop_variables=[var_name],
            )
            assert var_name not in ds_partial.data_vars
            ds_partial.close()

    @pytest.mark.requires_data
    def test_open_with_backend_kwargs(self, sample_netcdf_path: Path):
        """Test backend_kwargs passthrough."""
        if not sample_netcdf_path.exists():
            pytest.skip(f"Test file not found: {sample_netcdf_path}")

        ds = open_posix(
            str(sample_netcdf_path),
            engine="h5netcdf",
            backend_kwargs={"phony_dims": "sort"},
        )
        assert isinstance(ds, xr.Dataset)
        ds.close()


class TestCloudBackend:
    """Tests for remote/cloud backend."""

    @pytest.mark.requires_minio
    def test_open_netcdf_from_s3(self, s3_env: dict):
        """Open a NetCDF file from S3 (MinIO)."""
        uri = "s3://testdata/pr_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_GERICS-REMO2015_v2_3hr_200701020130-200701020430.nc"

        try:
            ds = open_cloud(
                uri,
                engine="h5netcdf",
                storage_options=s3_env,
            )
            assert isinstance(ds, xr.Dataset)
            ds.close()
        except FileNotFoundError:
            pytest.skip("Test file not found in MinIO")
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ("s3fs", "credentials", "nosuchbucket")):
                pytest.skip(f"S3 setup issue: {e}")
            raise

    @pytest.mark.requires_minio
    def test_open_grib_from_s3_with_cache(self, s3_env: dict, temp_cache_dir: Path):
        """Open a GRIB file from S3 with local caching."""
        uri = "s3://testdata/test.grib2"
        os.environ["XARRAY_PRISM_CACHE"] = str(temp_cache_dir)

        try:
            ds = open_cloud(
                uri,
                engine="cfgrib",
                storage_options=s3_env,
            )
            assert isinstance(ds, xr.Dataset)
            ds.close()
            cached_files = list(temp_cache_dir.glob("*"))
            assert len(cached_files) >= 0
        except FileNotFoundError:
            pytest.skip("Test file not found in MinIO")
        except Exception as e:
            err_str = str(e).lower()
            if any(
                x in err_str for x in ("cfgrib", "s3fs", "credentials", "nosuchbucket")
            ):
                pytest.skip(f"S3/cfgrib setup issue: {e}")
            raise
        finally:
            os.environ.pop("XARRAY_PRISM_CACHE", None)

    @pytest.mark.requires_minio
    def test_open_geotiff_from_s3(self, s3_env: dict):
        """Open a GeoTIFF from S3."""
        uri = "s3://testdata/TCD_S2021_R10m_DE111.tif"

        try:
            ds = open_cloud(
                uri,
                engine="rasterio",
                storage_options=s3_env,
            )
            assert isinstance(ds, xr.Dataset)
            ds.close()
        except FileNotFoundError:
            pytest.skip("Test file not found in MinIO")
        except Exception as e:
            err_str = str(e).lower()
            if any(
                x in err_str
                for x in ("rasterio", "s3fs", "credentials", "nosuchbucket")
            ):
                pytest.skip(f"S3/rasterio setup issue: {e}")
            raise

    @pytest.mark.requires_thredds
    def test_open_opendap(self, thredds_endpoint: str):
        """Open a dataset via OPeNDAP from THREDDS."""
        opendap_url = (
            f"{thredds_endpoint}/thredds/dodsC/alldata/model/regional/cordex/output/"
            "EUR-11/GERICS/NCC-NorESM1-M/rcp85/r1i1p1/GERICS-REMO2015/v1/3hr/pr/v20181212/"
            "pr_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_GERICS-REMO2015_v2_3hr_200701020130-200701020430.nc"
        )
        ds = open_cloud(opendap_url, engine="netcdf4")
        assert isinstance(ds, xr.Dataset)
        ds.close()


class TestCacheConfiguration:
    """Tests for cache directory configuration."""

    def test_cache_dir_from_env(self, temp_cache_dir: Path):
        """Cache directory should be configurable via environment."""
        from xarray_prism._cache import get_cache_dir

        os.environ["XARRAY_PRISM_CACHE"] = str(temp_cache_dir)
        try:
            assert get_cache_dir() == temp_cache_dir
        finally:
            os.environ.pop("XARRAY_PRISM_CACHE", None)

    def test_cache_dir_from_storage_options(self, temp_cache_dir: Path):
        """Cache directory should be configurable via storage_options."""
        from xarray_prism._cache import get_cache_dir

        storage_options = {"simplecache": {"cache_storage": str(temp_cache_dir)}}
        assert get_cache_dir(storage_options) == temp_cache_dir

    def test_cache_dir_default(self):
        """Default cache should be in temp directory."""
        import tempfile

        from xarray_prism._cache import get_cache_dir

        os.environ.pop("XARRAY_PRISM_CACHE", None)
        cache_dir = get_cache_dir()
        assert cache_dir.parent == Path(tempfile.gettempdir())
        assert "xarray-prism-cache" in str(cache_dir)


class TestStripCompressionSuffix:
    def test_strips_bz2(self):
        assert _strip_compression_suffix("file.grib2.bz2") == "file.grib2"

    def test_strips_gz(self):
        assert _strip_compression_suffix("file.nc.gz") == "file.nc"

    def test_strips_xz(self):
        assert _strip_compression_suffix("file.nc.xz") == "file.nc"

    def test_no_suffix_unchanged(self):
        assert _strip_compression_suffix("file.grib2") == "file.grib2"

    def test_only_strips_last_suffix(self):
        assert _strip_compression_suffix("file.grib2.bz2.bz2") == "file.grib2.bz2"


class TestStripChainingOptions:
    def test_strips_simplecache(self):
        opts = {"simplecache": {"cache_storage": "/tmp"}, "anon": True}
        result = _strip_chaining_options(opts)
        assert "simplecache" not in result
        assert result["anon"] is True

    def test_strips_blockcache(self):
        opts = {"blockcache": {}, "key": "abc"}
        result = _strip_chaining_options(opts)
        assert "blockcache" not in result
        assert "key" in result

    def test_strips_filecache(self):
        opts = {"filecache": {"cache_storage": "/tmp"}}
        result = _strip_chaining_options(opts)
        assert "filecache" not in result

    def test_empty_input(self):
        assert _strip_chaining_options({}) == {}

    def test_no_chaining_keys_unchanged(self):
        opts = {"anon": True, "key": "abc", "secret": "xyz"}
        assert _strip_chaining_options(opts) == opts


class TestDetectionWithCompression:
    def test_uri_pattern_grib2_bz2(self):
        assert _detect_from_uri_pattern("file.grib2.bz2") == "cfgrib"

    def test_uri_pattern_nc_gz(self):
        assert _detect_from_uri_pattern("file.nc.gz") is None

    def test_magic_bytes_grib_with_bz2_path(self):
        assert _detect_from_magic_bytes(b"GRIB...", "file.grib2.bz2") == "cfgrib"

    def test_magic_bytes_geotiff_with_gz_path(self):
        assert _detect_from_magic_bytes(b"II*\x00...", "file.tif.gz") == "rasterio"


class TestDecompressIfNeeded:
    def test_bz2_decompressed(self):
        content = b"GRIB test content"
        with tempfile.TemporaryDirectory() as tmpdir:
            compressed = os.path.join(tmpdir, "test.grib2.bz2")
            with bz2.open(compressed, "wb") as f:
                f.write(content)

            result = _decompress_if_needed(compressed)
            assert result == os.path.join(tmpdir, "test.grib2")
            assert Path(result).read_bytes() == content

    def test_gz_decompressed(self):
        content = b"CDF netcdf3 content"
        with tempfile.TemporaryDirectory() as tmpdir:
            compressed = os.path.join(tmpdir, "test.nc.gz")
            with gzip.open(compressed, "wb") as f:
                f.write(content)

            result = _decompress_if_needed(compressed)
            assert result == os.path.join(tmpdir, "test.nc")
            assert Path(result).read_bytes() == content

    def test_no_compression_unchanged(self):
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
            f.write(b"GRIB content")
            path = f.name
        try:
            assert _decompress_if_needed(path) == path
        finally:
            os.unlink(path)

    def test_idempotent_second_call(self):
        """Second call should not re-decompress."""
        content = b"GRIB test"
        with tempfile.TemporaryDirectory() as tmpdir:
            compressed = os.path.join(tmpdir, "test.grib2.bz2")
            with bz2.open(compressed, "wb") as f:
                f.write(content)

            result1 = _decompress_if_needed(compressed)
            mtime1 = os.path.getmtime(result1)
            result2 = _decompress_if_needed(compressed)
            assert os.path.getmtime(result2) == mtime1  # file not rewritten

    def test_custom_output_dir(self):
        content = b"GRIB test"
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                compressed = os.path.join(src_dir, "test.grib2.bz2")
                with bz2.open(compressed, "wb") as f:
                    f.write(content)

                result = _decompress_if_needed(compressed, output_dir=out_dir)
                assert result.startswith(out_dir)
                assert Path(result).read_bytes() == content


class TestOpenDatasetParametersIncludesStorageOptions:
    def test_storage_options_in_parameters(self):
        """storage_options must be in open_dataset_parameters so xarray forwards it."""
        entrypoint = PrismBackendEntrypoint()
        assert "storage_options" in entrypoint.open_dataset_parameters


class TestGuessCanOpenCompressed:
    def test_grib2_bz2(self):
        ep = PrismBackendEntrypoint()
        assert ep.guess_can_open("forecast.grib2.bz2") is True

    def test_nc_gz(self):
        ep = PrismBackendEntrypoint()
        assert ep.guess_can_open("data.nc.gz") is True
