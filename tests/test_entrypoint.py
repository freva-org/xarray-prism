"""Tests for the xarray backend entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import xarray as xr

from xarray_prism import PrismBackendEntrypoint
from xarray_prism._registry import registry


class TestPrismBackendEntrypoint:
    """Tests for the xarray backend entrypoint."""

    @pytest.fixture
    def entrypoint(self):
        """Create a backend entrypoint instance."""
        return PrismBackendEntrypoint()

    def test_entrypoint_attributes(self, entrypoint: PrismBackendEntrypoint):
        """Entrypoint should have required attributes."""
        assert hasattr(entrypoint, "open_dataset")
        assert hasattr(entrypoint, "guess_can_open")
        assert entrypoint.description is not None

    def test_guess_can_open_zarr(self, entrypoint: PrismBackendEntrypoint):
        """Should recognize Zarr paths."""
        assert entrypoint.guess_can_open("data.zarr") is True
        assert entrypoint.guess_can_open("s3://bucket/data.zarr") is True
        assert entrypoint.guess_can_open("/path/to/store.zarr/") is True

    def test_guess_can_open_netcdf(self, entrypoint: PrismBackendEntrypoint):
        """Should recognize NetCDF paths."""
        assert entrypoint.guess_can_open("data.nc") is True
        assert entrypoint.guess_can_open("data.nc4") is True
        assert entrypoint.guess_can_open("/path/to/file.nc") is True

    def test_guess_can_open_grib(self, entrypoint: PrismBackendEntrypoint):
        """Should recognize GRIB paths."""
        assert entrypoint.guess_can_open("forecast.grib") is True
        assert entrypoint.guess_can_open("forecast.grib2") is True
        assert entrypoint.guess_can_open("data.grb") is True
        assert entrypoint.guess_can_open("data.grb2") is True

    def test_guess_can_open_geotiff(self, entrypoint: PrismBackendEntrypoint):
        """Should recognize GeoTIFF paths."""
        assert entrypoint.guess_can_open("image.tif") is True
        assert entrypoint.guess_can_open("image.tiff") is True

    def test_guess_can_open_reference(self, entrypoint: PrismBackendEntrypoint):
        """Should recognize reference:// URIs."""
        assert entrypoint.guess_can_open("reference://path/to/refs.json") is True

    def test_guess_can_open_opendap(self, entrypoint: PrismBackendEntrypoint):
        """Should recognize OPeNDAP URLs."""
        assert entrypoint.guess_can_open("http://server/thredds/dodsC/data") is True
        assert entrypoint.guess_can_open("http://server/opendap/dataset") is True

    def test_guess_can_open_rejects_unknown(self, entrypoint: PrismBackendEntrypoint):
        """Should reject unknown formats."""
        assert entrypoint.guess_can_open("document.pdf") is False
        assert entrypoint.guess_can_open("image.jpg") is False
        assert entrypoint.guess_can_open(123) is False  # type: ignore

    @pytest.mark.requires_data
    def test_open_dataset_local_netcdf(
        self, entrypoint: PrismBackendEntrypoint, sample_netcdf_path: Path
    ):
        """Open a local NetCDF file via entrypoint."""
        if not sample_netcdf_path.exists():
            pytest.skip(f"Test file not found: {sample_netcdf_path}")

        ds = entrypoint.open_dataset(str(sample_netcdf_path))
        assert isinstance(ds, xr.Dataset)
        ds.close()

    @pytest.mark.requires_data
    def test_open_dataset_local_grib(
        self, entrypoint: PrismBackendEntrypoint, sample_grib_path: Path
    ):
        """Open a local GRIB file via entrypoint."""
        if not sample_grib_path.exists():
            pytest.skip(f"Test file not found: {sample_grib_path}")

        try:
            ds = entrypoint.open_dataset(str(sample_grib_path))
            assert isinstance(ds, xr.Dataset)
            ds.close()
        except Exception as e:
            if "cfgrib" in str(e).lower() or "eccodes" in str(e).lower():
                pytest.skip("cfgrib/eccodes not installed")
            raise

    @pytest.mark.requires_data
    def test_open_dataset_with_drop_variables(
        self, entrypoint: PrismBackendEntrypoint, sample_netcdf_path: Path
    ):
        """Test drop_variables parameter via entrypoint."""
        if not sample_netcdf_path.exists():
            pytest.skip(f"Test file not found: {sample_netcdf_path}")

        ds_full = entrypoint.open_dataset(str(sample_netcdf_path))
        var_names = list(ds_full.data_vars)
        ds_full.close()

        if var_names:
            ds_partial = entrypoint.open_dataset(
                str(sample_netcdf_path), drop_variables=[var_names[0]]
            )
            assert var_names[0] not in ds_partial.data_vars
            ds_partial.close()

    @pytest.mark.requires_data
    def test_open_dataset_explicit_engine(
        self, entrypoint: PrismBackendEntrypoint, sample_netcdf_path: Path
    ):
        """Test explicit engine override."""
        if not sample_netcdf_path.exists():
            pytest.skip(f"Test file not found: {sample_netcdf_path}")

        ds = entrypoint.open_dataset(str(sample_netcdf_path), xarray_engine="h5netcdf")
        assert isinstance(ds, xr.Dataset)
        ds.close()

    def test_open_dataset_invalid_type(self, entrypoint: PrismBackendEntrypoint):
        """Should raise for invalid input types."""
        with pytest.raises(ValueError, match="file path or URL"):
            entrypoint.open_dataset(123)  # type: ignore

    def test_open_dataset_unknown_format(self, entrypoint: PrismBackendEntrypoint):
        """Should raise for undetectable formats."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"unknown content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError):
                entrypoint.open_dataset(temp_path)
        finally:
            os.unlink(temp_path)

    @pytest.mark.requires_minio
    def test_open_dataset_s3(self, entrypoint: PrismBackendEntrypoint, s3_env: dict):
        """Open a dataset from S3 via entrypoint."""
        uri = "s3://testdata/pr_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_GERICS-REMO2015_v2_3hr_200701020130-200701020430.nc"

        try:
            ds = entrypoint.open_dataset(uri, storage_options=s3_env)
            assert isinstance(ds, xr.Dataset)
            ds.close()
        except FileNotFoundError:
            pytest.skip("Test file not found in MinIO")
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ("s3fs", "credentials", "nosuchbucket")):
                pytest.skip(f"S3 setup issue: {e}")
            raise

    @pytest.mark.requires_thredds
    def test_open_dataset_opendap(
        self, entrypoint: PrismBackendEntrypoint, thredds_endpoint: str
    ):
        """Open a dataset via OPeNDAP through entrypoint."""
        opendap_url = (
            f"{thredds_endpoint}/thredds/dodsC/alldata/model/regional/cordex/output/"
            "EUR-11/GERICS/NCC-NorESM1-M/rcp85/r1i1p1/GERICS-REMO2015/v1/3hr/pr/v20181212/"
            "pr_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_GERICS-REMO2015_v2_3hr_200701020130-200701020430.nc"
        )

        try:
            ds = entrypoint.open_dataset(opendap_url)
            assert isinstance(ds, xr.Dataset)
            ds.close()
        except Exception as e:
            if "netcdf4" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"OPeNDAP access failed: {e}")
            raise

    def test_open_dataset_custom_uri_type_no_handler(
        self, entrypoint: PrismBackendEntrypoint
    ):
        """Should raise helpful error for custom uri_type without handler."""
        from xarray_prism._detection import register_uri_type, unregister_uri_type

        @register_uri_type(priority=100)
        def detect_tape(uri: str):
            if uri.startswith("tape://"):
                return "tape"
            return None

        try:
            with pytest.raises(ValueError, match="No handler registered"):
                entrypoint.open_dataset("tape://archive/data.zarr")
        finally:
            unregister_uri_type(detect_tape)


class TestXarrayIntegration:
    """Tests for xarray.open_dataset integration."""

    @pytest.mark.requires_data
    def test_xarray_open_with_freva_engine(self, sample_netcdf_path: Path):
        """xarray should recognize 'prism' as a valid engine."""
        if not sample_netcdf_path.exists():
            pytest.skip(f"Test file not found: {sample_netcdf_path}")

        ds = xr.open_dataset(str(sample_netcdf_path), engine="prism")
        assert isinstance(ds, xr.Dataset)
        ds.close()


class TestCustomRegistry:
    """Tests for custom backend registration."""

    def test_register_custom_handler(self):
        """Register and use a custom handler."""

        @registry.register("custom_format", uri_type="posix")
        def custom_handler(uri, **kwargs):
            return xr.Dataset({"test": (["x"], [1, 2, 3])})

        assert registry.has("custom_format", "posix")
        handler = registry.get("custom_format", "posix")
        assert handler is not None

        ds = handler("any_uri.custom")
        assert "test" in ds.data_vars
        ds.close()

    def test_register_handler_for_both_uri_types(self):
        """Register handler for both posix and cloud."""

        @registry.register("universal_format", uri_type="both")
        def universal_handler(uri, **kwargs):
            return xr.Dataset({"data": (["y"], [4, 5, 6])})

        assert registry.has("universal_format", "posix")
        assert registry.has("universal_format", "cloud")

    def test_custom_handler_used_by_entrypoint(self):
        """Custom handlers should be used by the entrypoint."""
        from xarray_prism._detection import register_detector, unregister_detector

        @register_detector(priority=200)
        def detect_myformat(uri):
            if uri.endswith(".mydata"):
                return "myformat"
            return None

        @registry.register("myformat", uri_type="posix")
        def myformat_handler(uri, **kwargs):
            return xr.Dataset({"custom": (["z"], [7, 8, 9])})

        try:
            entrypoint = PrismBackendEntrypoint()

            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mydata", delete=False) as f:
                f.write(b"dummy content")
                temp_path = f.name

            try:
                ds = entrypoint.open_dataset(temp_path)
                assert "custom" in ds.data_vars
                ds.close()
            finally:
                os.unlink(temp_path)
        finally:
            unregister_detector(detect_myformat)

    def test_custom_uri_type_with_handler(self):
        """Custom URI type with registered handler should work."""
        from xarray_prism._detection import register_uri_type, unregister_uri_type

        @register_uri_type(priority=100)
        def detect_tape(uri: str):
            if uri.startswith("tape://"):
                return "tape"
            return None

        @registry.register("zarr", uri_type="tape")
        def tape_handler(uri, **kwargs):
            return xr.Dataset({"staged": (["t"], [10, 20, 30])})

        try:
            entrypoint = PrismBackendEntrypoint()
            ds = entrypoint.open_dataset("tape://archive/data.zarr")
            assert "staged" in ds.data_vars
            ds.close()
        finally:
            unregister_uri_type(detect_tape)
