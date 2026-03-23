"""Cloud backend for xarray datasets."""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

from .._cache import cache_remote_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def open_cloud(
    uri: str,
    engine: str,
    drop_variables: Optional[Any] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
    lines_above: int = 0,
    **kwargs,
) -> Any:
    """Open remote file with detected engine."""
    import xarray as xr

    from ..utils import gdal_env

    storage_options = kwargs.pop("storage_options", None)

    def _clear_lines():
        """Clear the detection message lines."""
        if lines_above > 0:
            for _ in range(lines_above):
                sys.stdout.write("\033[A")
                sys.stdout.write("\033[K")
            sys.stdout.flush()

    bk = backend_kwargs or None

    # GRIB / NetCDF3: must download the full file first
    if engine in ("cfgrib", "scipy"):
        local_path = cache_remote_file(
            uri, engine, storage_options, show_progress, lines_above
        )
        return xr.open_dataset(
            local_path,
            engine=engine,
            drop_variables=drop_variables,
            backend_kwargs=bk,
            **kwargs,
        )

    # NetCDF4 (OPeNDAP)
    if engine == "netcdf4":
        ds = xr.open_dataset(
            uri,
            engine=engine,
            drop_variables=drop_variables,
            backend_kwargs=bk,
            **kwargs,
        )
        _clear_lines()
        return ds

    # Rasterio: translate storage_options -> GDAL env vars
    if engine == "rasterio":
        from ..utils import sanitize_rasterio_kwargs

        with gdal_env(storage_options):
            ds = xr.open_dataset(
                uri,
                engine=engine,
                drop_variables=drop_variables,
                backend_kwargs=bk,
                **sanitize_rasterio_kwargs(kwargs),
            )
        _clear_lines()
        return ds

    # Zarr, h5netcdf
    ds = xr.open_dataset(
        uri,
        engine=engine,
        drop_variables=drop_variables,
        backend_kwargs=bk,
        storage_options=storage_options,
        **kwargs,
    )
    if engine == "h5netcdf":
        from ..utils import sanitize_dataset_attrs

        return sanitize_dataset_attrs(ds)
    _clear_lines()
    return ds
