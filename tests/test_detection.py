"""Tests for URI and format detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from freva_xarray._detection import (
    detect_engine,
    detect_uri_type,
    is_http_url,
    is_reference_uri,
    is_remote_uri,
    looks_like_opendap_url,
    register_detector,
    register_uri_type,
    unregister_detector,
    unregister_uri_type,
)


class TestURITypeDetection:
    """Tests for URI type detection."""

    def test_posix_local_path(self):
        """Local paths should be detected as posix."""
        assert detect_uri_type("/path/to/file.nc") == "posix"
        assert detect_uri_type("./relative/path.zarr") == "posix"
        assert detect_uri_type("data.grib2") == "posix"

    def test_posix_file_uri(self):
        """file:// URIs should be detected as posix."""
        assert detect_uri_type("file:///path/to/file.nc") == "posix"
        assert detect_uri_type("file://localhost/data.zarr") == "posix"

    def test_cloud_s3_uri(self):
        """S3 URIs should be detected as cloud."""
        assert detect_uri_type("s3://bucket/path/data.nc") == "cloud"
        assert detect_uri_type("s3://my-bucket/nested/path.zarr") == "cloud"

    def test_cloud_gcs_uri(self):
        """GCS URIs should be detected as cloud."""
        assert detect_uri_type("gs://bucket/data.nc") == "cloud"
        assert detect_uri_type("gcs://bucket/path.zarr") == "cloud"

    def test_cloud_azure_uri(self):
        """Azure URIs should be detected as cloud."""
        assert detect_uri_type("az://container/blob.nc") == "cloud"
        assert detect_uri_type("abfs://container/data.zarr") == "cloud"

    def test_cloud_http_uri(self):
        """HTTP(S) URIs should be detected as cloud."""
        assert detect_uri_type("http://example.com/data.nc") == "cloud"
        assert detect_uri_type("https://server.org/path/file.zarr") == "cloud"

    def test_reference_uri(self):
        """reference:// URIs should be detected as reference."""
        assert detect_uri_type("reference://path/to/refs.json") == "reference"
        assert detect_uri_type("REFERENCE://upper/case.json") == "reference"


class TestHelperFunctions:
    """Tests for helper detection functions."""

    def test_is_http_url(self):
        """Test HTTP URL detection."""
        assert is_http_url("http://example.com/data.nc") is True
        assert is_http_url("https://server.org/file.zarr") is True
        assert is_http_url("HTTP://UPPERCASE.com/file") is True
        assert is_http_url("s3://bucket/path") is False
        assert is_http_url("/local/path.nc") is False

    def test_is_remote_uri(self):
        """Test remote URI detection."""
        assert is_remote_uri("s3://bucket/path") is True
        assert is_remote_uri("https://server.org/file") is True
        assert is_remote_uri("file:///local/path") is False
        assert is_remote_uri("/local/path.nc") is False

    def test_is_reference_uri(self):
        """Test reference URI detection."""
        assert is_reference_uri("reference://path/refs.json") is True
        assert is_reference_uri("REFERENCE://upper.json") is True
        assert is_reference_uri("s3://bucket/path") is False

    def test_looks_like_opendap_url(self):
        """Test OPeNDAP URL heuristics."""
        assert looks_like_opendap_url("http://server/thredds/dodsC/data") is True
        assert looks_like_opendap_url("http://server/opendap/dataset") is True
        assert looks_like_opendap_url("http://server/thredds/dods/data") is True
        assert looks_like_opendap_url("http://example.com/data.nc") is False


class TestEngineDetection:
    """Tests for format/engine detection."""

    def test_zarr_extension(self):
        """Zarr should be detected by extension."""
        assert detect_engine("/path/to/data.zarr") == "zarr"
        assert detect_engine("s3://bucket/nested.zarr/") == "zarr"
        assert detect_engine("http://server/data.zarr/subpath") == "zarr"

    def test_reference_uri_maps_to_zarr(self):
        """Reference URIs should map to zarr engine."""
        assert detect_engine("reference://path/to/refs.json") == "zarr"

    def test_opendap_detection(self):
        """OPeNDAP URLs should detect as netcdf4."""
        assert detect_engine("http://server/thredds/dodsc/dataset") == "netcdf4"
        assert detect_engine("http://server/opendap/data") == "netcdf4"

    @pytest.mark.requires_data
    def test_netcdf4_detection_local(self, sample_netcdf_path: Path):
        """NetCDF4 files should be detected via magic bytes."""
        if sample_netcdf_path.exists():
            engine = detect_engine(str(sample_netcdf_path))
            assert engine in ("h5netcdf", "scipy")

    @pytest.mark.requires_data
    def test_grib_detection_local(self, sample_grib_path: Path):
        """GRIB files should be detected via magic bytes."""
        if sample_grib_path.exists():
            assert detect_engine(str(sample_grib_path)) == "cfgrib"

    @pytest.mark.requires_data
    def test_geotiff_detection_local(self, sample_geotiff_path: Path):
        """GeoTIFF files should be detected via magic bytes."""
        if sample_geotiff_path.exists():
            assert detect_engine(str(sample_geotiff_path)) == "rasterio"


class TestCustomDetectors:
    """Tests for custom detector registration."""

    def test_register_and_use_custom_detector(self):
        """Custom detectors should be called in priority order."""

        @register_detector(priority=100)
        def my_detector(uri: str):
            if uri.endswith(".myformat"):
                return "my_custom_engine"
            return None

        try:
            assert detect_engine("file.myformat") == "my_custom_engine"
            assert detect_engine("data.zarr") == "zarr"
        finally:
            unregister_detector(my_detector)

    def test_unregister_detector(self):
        """Detectors should be removable."""

        @register_detector(priority=100)
        def temp_detector(uri: str):
            if uri.endswith(".temp"):
                return "temp_engine"
            return None

        assert detect_engine("file.temp") == "temp_engine"

        result = unregister_detector(temp_detector)
        assert result is True
        assert detect_engine("file.temp") == "unknown"

    def test_detector_priority_order(self):
        """Higher priority detectors should run first."""

        @register_detector(priority=50)
        def low_priority(uri: str):
            if "test" in uri:
                return "low"
            return None

        @register_detector(priority=150)
        def high_priority(uri: str):
            if "test" in uri:
                return "high"
            return None

        try:
            assert detect_engine("test_file") == "high"
        finally:
            unregister_detector(low_priority)
            unregister_detector(high_priority)

    def test_failing_detector_skipped(self):
        """Detectors that raise exceptions should be skipped."""

        @register_detector(priority=100)
        def failing_detector(uri: str):
            raise ValueError("Intentional failure")

        try:
            result = detect_engine("data.zarr")
            assert result == "zarr"
        finally:
            unregister_detector(failing_detector)

    def test_unknown_engine_warning(self):
        """Custom detector returning unknown engine should warn."""
        from freva_xarray import FrevaBackendEntrypoint
        import warnings

        @register_detector(priority=100)
        def fake_engine_detector(uri: str):
            if uri.endswith(".fake"):
                return "nonexistent_engine"
            return None

        try:
            entrypoint = FrevaBackendEntrypoint()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                engine, uri_type = entrypoint._detect("test.fake")
                assert engine == "nonexistent_engine"
                assert len(w) == 1
                assert "not a built-in engine" in str(w[0].message)
        finally:
            unregister_detector(fake_engine_detector)


class TestCustomURITypeDetectors:
    """Tests for custom URI type detector registration."""

    def test_register_and_use_custom_uri_type(self):
        """Custom URI type detectors should be called."""

        @register_uri_type(priority=100)
        def detect_tape(uri: str):
            if uri.startswith("tape://"):
                return "tape"
            return None

        try:
            assert detect_uri_type("tape://archive/data.nc") == "tape"
            assert detect_uri_type("s3://bucket/data.nc") == "cloud"
        finally:
            unregister_uri_type(detect_tape)

    def test_unregister_uri_type(self):
        """URI type detectors should be removable."""

        @register_uri_type(priority=100)
        def detect_custom(uri: str):
            if uri.startswith("custom://"):
                return "custom"
            return None

        assert detect_uri_type("custom://path/data") == "custom"

        result = unregister_uri_type(detect_custom)
        assert result is True
        assert detect_uri_type("custom://path/data") == "cloud"

    def test_uri_type_priority_order(self):
        """Higher priority URI type detectors should run first."""

        @register_uri_type(priority=50)
        def low_priority(uri: str):
            if "special" in uri:
                return "low_type"
            return None

        @register_uri_type(priority=150)
        def high_priority(uri: str):
            if "special" in uri:
                return "high_type"
            return None

        try:
            assert detect_uri_type("special://data") == "high_type"
        finally:
            unregister_uri_type(low_priority)
            unregister_uri_type(high_priority)

    def test_failing_uri_type_detector_skipped(self):
        """URI type detectors that raise exceptions should be skipped."""

        @register_uri_type(priority=100)
        def failing_detector(uri: str):
            raise ValueError("Intentional failure")

        try:
            assert detect_uri_type("s3://bucket/data") == "cloud"
        finally:
            unregister_uri_type(failing_detector)
