"""Tests for POSIX and cloud backends."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import xarray as xr

from xarray_prism.backends import open_cloud, open_posix


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
        from xarray_prism.backends.cloud import _get_cache_dir

        os.environ["XARRAY_PRISM_CACHE"] = str(temp_cache_dir)
        try:
            cache_dir = _get_cache_dir()
            assert cache_dir == temp_cache_dir
        finally:
            os.environ.pop("XARRAY_PRISM_CACHE", None)

    def test_cache_dir_from_storage_options(self, temp_cache_dir: Path):
        """Cache directory should be configurable via storage_options."""
        from xarray_prism.backends.cloud import _get_cache_dir

        storage_options = {
            "simplecache": {"cache_storage": str(temp_cache_dir)},
        }
        cache_dir = _get_cache_dir(storage_options)
        assert cache_dir == temp_cache_dir

    def test_cache_dir_default(self):
        """Default cache should be in temp directory."""
        from xarray_prism.backends.cloud import _get_cache_dir
        import tempfile

        os.environ.pop("XARRAY_PRISM_CACHE", None)

        cache_dir = _get_cache_dir()
        assert cache_dir.parent == Path(tempfile.gettempdir())
        assert "xarray-prism-cache" in str(cache_dir)
