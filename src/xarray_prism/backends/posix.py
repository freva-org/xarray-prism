"""Module to open local files using xarray
with a specified engine."""

from __future__ import annotations

from typing import Any, Dict, Optional


def open_posix(
    uri: str,
    engine: str,
    drop_variables: Optional[Any] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """Open local file with detected engine."""
    import xarray as xr

    return xr.open_dataset(
        uri,
        engine=engine,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs or None,
        **kwargs,
    )
