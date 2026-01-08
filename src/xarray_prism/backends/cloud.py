"""Cloud backend for xarray datasets."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import fsspec
import xarray as xr

from ..utils import ProgressBar, gdal_env

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_cache_dir(storage_options: Optional[Dict] = None) -> Path:
    """Get cache directory."""
    env_cache = os.environ.get("XARRAY_PRISM_CACHE")
    # 1. Environment variable
    if env_cache:
        cache_root = Path(env_cache)
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root
    # 2. User-defined storage option
    if storage_options:
        user_cache = storage_options.get("simplecache", {}).get("cache_storage")
        if user_cache:
            cache_root = Path(user_cache)
            cache_root.mkdir(parents=True, exist_ok=True)
            return cache_root
    # 3. Default temp directory
    cache_root = Path(tempfile.gettempdir()) / "xarray-prism-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _cache_remote_file(
    uri: str,
    engine: str,
    storage_options: Optional[Dict] = None,
    show_progress: bool = True,
    lines_above: int = 0,
) -> str:
    """Cache remote file to local"""
    cache_root = _get_cache_dir(storage_options)
    parsed = urlparse(uri)
    filename = Path(parsed.path).name
    cache_name = md5(uri.encode()).hexdigest() + "_" + filename
    local_path = cache_root / cache_name

    if local_path.exists():
        if show_progress and lines_above > 0:
            for _ in range(lines_above):
                sys.stdout.write("\033[A")
                sys.stdout.write("\033[K")
            sys.stdout.flush()
        return str(local_path)

    extra_lines = 0
    if show_progress:
        fmt = "GRIB" if engine == "cfgrib" else "NetCDF3"
        logger.warning(f"Remote {fmt} requires full file download")
        extra_lines = 2

    fs, path = fsspec.core.url_to_fs(uri, **(storage_options or {}))

    if show_progress:
        size = 0
        try:
            size = fs.size(path) or 0
        except Exception:
            pass

        filename = Path(uri).name
        if len(filename) > 35:
            filename = filename[:32] + "..."
        desc = f" Downloadig {filename}"

        total_lines = lines_above + extra_lines

        with ProgressBar(desc=desc, lines_above=total_lines) as progress:
            progress.set_size(size)
            with fs.open(path, "rb") as src, open(local_path, "wb") as dst:
                while True:
                    chunk = src.read(512 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    progress.update(len(chunk))
    else:
        fs.get(path, str(local_path))

    return str(local_path)


def open_cloud(
    uri: str,
    engine: str,
    drop_variables: Optional[Any] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
    lines_above: int = 0,
    **kwargs,
) -> xr.Dataset:
    """Open remote file with detected engine."""
    storage_options = kwargs.pop("storage_options", None)

    def _clear_lines():
        """Clear the detection message lines."""
        if lines_above > 0:
            for _ in range(lines_above):
                sys.stdout.write("\033[A")
                sys.stdout.write("\033[K")
            sys.stdout.flush()

    # GRIB: cache locally
    if engine == "cfgrib":
        local_path = _cache_remote_file(
            uri, engine, storage_options, show_progress, lines_above
        )
        return xr.open_dataset(
            local_path,
            engine=engine,
            drop_variables=drop_variables,
            backend_kwargs=backend_kwargs or None,
            **kwargs,
        )

    # NetCDF3: cache locally
    if engine == "scipy":
        local_path = _cache_remote_file(
            uri, engine, storage_options, show_progress, lines_above
        )
        return xr.open_dataset(
            local_path,
            engine=engine,
            drop_variables=drop_variables,
            backend_kwargs=backend_kwargs or None,
            **kwargs,
        )

    # NetCDF4 (OPeNDAP)
    if engine == "netcdf4":
        ds = xr.open_dataset(
            uri,
            engine=engine,
            drop_variables=drop_variables,
            backend_kwargs=backend_kwargs or None,
            **kwargs,
        )
        _clear_lines()
        return ds

    # Rasterio: use GDAL env vars (doesn't support storage_options)
    if engine == "rasterio":
        with gdal_env(storage_options):
            ds = xr.open_dataset(
                uri,
                engine=engine,
                drop_variables=drop_variables,
                backend_kwargs=backend_kwargs or None,
                **kwargs,
            )
        _clear_lines()
        return ds

    # Zarr, h5netcdf
    ds = xr.open_dataset(
        uri,
        engine=engine,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs or None,
        storage_options=storage_options,
        **kwargs,
    )
    _clear_lines()
    return ds
