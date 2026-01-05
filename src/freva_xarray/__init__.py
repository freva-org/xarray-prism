# ---------------------------------------------------------------#
# Data format   | Remote backend         | Local FS  | Cache
# ---------------------------------------------------------------#
# GRIB          | cfgrib + fsspec        | cfgrib    | fsspec simplecache (full-file)
# Zarr          | zarr + fsspec          | zarr      | chunked key/value store
# NetCDF3       | scipy + fsspec         | scipy     | fsspec byte cache (full-file)
# NetCDF4/HDF5  | h5netcdf + fsspec      | h5netcdf  | fsspec byte cache (5 MB blocks)
# GeoTIFF       | rasterio + fsspec      | rasterio  | GDAL/rasterio block cache
# OPeNDAP/DODS  | netCDF4                | n/a       | n/a
# ---------------------------------------------------------------#

# Important: GRIB and NetCDF3 files are not chunk-addressable.
# cfgrib and scipy typically must read the entire file (and build
# its index) even when only a small subset is requested.

from _version import __version__  # noqa

from ._detection import (
    detect_engine,
    detect_uri_type,
    register_detector,
    register_uri_type,
)
from ._registry import registry
from .entrypoint import FrevaBackendEntrypoint

__all__ = [
    "FrevaBackendEntrypoint",
    "detect_engine",
    "detect_uri_type",
    "register_detector",
    "register_uri_type",
    "registry",
]
