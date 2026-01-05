from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
THREDDS_ENDPOINT = os.environ.get("THREDDS_ENDPOINT", "http://localhost:8088")

S3_ENDPOINT_URL = MINIO_ENDPOINT
S3_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
S3_BUCKET = "testdata"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the path to the test data directory."""
    return DATA_DIR


@pytest.fixture(scope="session")
def s3_storage_options() -> dict:
    """Return S3 storage options for MinIO."""
    return {
        "key": S3_ACCESS_KEY,
        "secret": S3_SECRET_KEY,
        "client_kwargs": {"endpoint_url": S3_ENDPOINT_URL},
    }


@pytest.fixture
def s3_env(s3_storage_options: dict) -> Generator[dict, None, None]:
    """
    Set AWS environment variables for S3 access.

    This is needed because detect_engine uses fsspec without storage_options,
    so credentials must come from environment variables.

    Also sets GDAL-specific variables for rasterio/rioxarray.
    """
    endpoint_url = s3_storage_options["client_kwargs"]["endpoint_url"]
    # Extract host:port from endpoint URL for GDAL
    endpoint_host = endpoint_url.replace("http://", "").replace("https://", "")

    env_vars = {
        # Standard AWS env vars
        "AWS_ACCESS_KEY_ID": s3_storage_options["key"],
        "AWS_SECRET_ACCESS_KEY": s3_storage_options["secret"],
        "AWS_ENDPOINT_URL": endpoint_url,
        # GDAL-specific env vars
        "AWS_S3_ENDPOINT": endpoint_host,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "NO",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    try:
        yield s3_storage_options
    finally:
        for key in env_vars:
            os.environ.pop(key, None)


@pytest.fixture(scope="session")
def s3_endpoint() -> str:
    """Return the S3 endpoint URL."""
    return S3_ENDPOINT_URL


@pytest.fixture(scope="session")
def thredds_endpoint() -> str:
    """Return the THREDDS server endpoint."""
    return THREDDS_ENDPOINT


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for GRIB cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_netcdf_path(data_dir: Path) -> Path:
    """Return path to a sample NetCDF4 file."""
    nc_path = (
        data_dir
        / "model/global/cmip6/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/amip/r1i1p1f1"
        / "Amon/ua/gn/v20191108/ua_Amon_ACCESS-CM2_amip_r1i1p1f1_gn_197001-201512.nc"
    )
    return nc_path


@pytest.fixture
def sample_grib_path(data_dir: Path) -> Path:
    """Return path to a sample GRIB file."""
    return data_dir / "grib_data/gfs/2025/11/25/test.grib2"


@pytest.fixture
def sample_geotiff_path(data_dir: Path) -> Path:
    """Return path to a sample GeoTIFF file."""
    return data_dir / "geodata/TCD/2021/10m/districts/DE111/TCD_S2021_R10m_DE111.tif"


@pytest.fixture
def sample_cordex_path(data_dir: Path) -> Path:
    """Return path to a sample CORDEX NetCDF file."""
    return (
        data_dir
        / "model/regional/cordex/output/EUR-11/GERICS/NCC-NorESM1-M/rcp85/r1i1p1"
        / "GERICS-REMO2015/v1/3hr/pr/v20181212"
        / "pr_EUR-11_NCC-NorESM1-M_rcp85_r1i1p1_GERICS-REMO2015_v2_3hr_200701020130-200701020430.nc"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_minio: mark test as requiring MinIO service"
    )
    config.addinivalue_line(
        "markers", "requires_thredds: mark test as requiring THREDDS service"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring local test data"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available services and data."""
    import socket

    def is_service_available(host: str, port: int, timeout: float = 1.0) -> bool:
        """Check if a service is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except (socket.error, OSError):
            return False

    minio_available = is_service_available("localhost", 9000)
    thredds_available = is_service_available("localhost", 8088)
    data_available = DATA_DIR.exists()

    skip_minio = pytest.mark.skip(reason="MinIO service not available")
    skip_thredds = pytest.mark.skip(reason="THREDDS service not available")
    skip_data = pytest.mark.skip(reason="Test data directory not found")

    for item in items:
        if "requires_minio" in item.keywords and not minio_available:
            item.add_marker(skip_minio)
        if "requires_thredds" in item.keywords and not thredds_available:
            item.add_marker(skip_thredds)
        if "requires_data" in item.keywords and not data_available:
            item.add_marker(skip_data)
