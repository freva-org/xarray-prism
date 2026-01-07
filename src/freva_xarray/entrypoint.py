"""Freva xarray backend entrypoint with
automatic format detection."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import xarray as xr
from xarray.backends import BackendEntrypoint

from ._detection import (
    detect_engine,
    detect_uri_type,
    is_http_url,
    is_reference_uri,
    looks_like_opendap_url,
)
from ._registry import registry
from .backends import open_cloud, open_posix


def _supports_hyperlinks() -> bool:
    """Check if terminal supports OSC 8 hyperlinks.
    based on https://github.com/Alhadis/OSC8-Adoption
    most of frequently used terminals are covered.
    """
    # Not a TTY, no hyperlinks
    if not sys.stderr.isatty():
        return False
    # Windows Terminal
    if os.environ.get("WT_SESSION"):
        return True
    if os.environ.get("TERM_PROGRAM") in ("vscode", "iTerm.app", "WezTerm"):
        return True
    # Modern terminals with truecolor support
    if os.environ.get("COLORTERM") == "truecolor":
        return True
    # KDE Konsole
    if os.environ.get("KONSOLE_VERSION"):
        return True
    # GNOME Terminal
    if os.environ.get("VTE_VERSION"):
        try:
            return int(os.environ["VTE_VERSION"]) >= 5000
        except ValueError:
            pass
    return False


def _make_link(url: str, text: str) -> str:
    """Create clickable link if supported, otherwise full URL."""
    if _supports_hyperlinks():
        return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"
    return url


class FrevaBackendEntrypoint(BackendEntrypoint):
    """
    Multi-format backend with automatic engine detection for climate data.

    Handling:
    - local files: direct xarray open
    - remote GRIB: cache locally via fsspec simplecache
    - remote NetCDF3: fsspec file object
    - remote Zarr/NetCDF4/OPeNDAP: native remote support
    """

    description = (
        "Freva multi-format/multi-storage engine with auto-detection"
        "and entrypoint registry for new formats and URI types."
    )

    url = "https://github.com/freva-org/xarray-prism"
    open_dataset_parameters = ("filename_or_obj", "drop_variables")

    ENGINE_MAP: Dict[str, str] = {
        "zarr": "zarr",
        "cfgrib": "cfgrib",
        "h5netcdf": "h5netcdf",
        "scipy": "scipy",
        "rasterio": "rasterio",
        "netcdf4": "netcdf4",
    }

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: Optional[Any] = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Xarray Generic function: Open dataset with
        automatic format detection."""
        if not isinstance(filename_or_obj, (str, Path)):
            raise ValueError(
                f"Freva backend requires a file path or URL, "
                f"got {type(filename_or_obj).__name__}"
            )

        uri = str(filename_or_obj)

        is_remote = "://" in uri and not uri.startswith("file://")
        lines_printed = 0

        if is_remote:
            sys.stdout.write("[info] Detecting format...")
            sys.stdout.flush()

        engine, uri_type = self._detect(uri, **kwargs)

        if is_remote:
            sys.stdout.write("\r" + " " * 25 + "\r")
            if engine:
                print(f"[info] Detected: {engine}")
                lines_printed = 1
            sys.stdout.flush()

        if engine is None:
            if is_remote:
                from urllib.parse import urlencode, urlparse

                filename = Path(urlparse(uri).path).name
                if len(filename) > 50:
                    filename = filename[:47] + "..."

                issue_params = urlencode(
                    {
                        "title": f"[Detection] Cannot detect format: {filename}",
                        "body": (
                            f"**File/URL:**\n```\n{uri}\n```\n\n"
                            f"**Expected format:** (e.g., NetCDF4, GRIB, Zarr)\n\n"
                        ),
                        "labels": "bug",
                    }
                )

                issue_url = f"{self.url}/issues/new?{issue_params}"
                link = _make_link(issue_url, "🔗 Click here to report")

                raise ValueError(
                    f"Freva Xarray: cannot detect format for {uri!r}\n\n"
                    f"  💡 Help us improve! This takes 10 seconds:\n"
                    f"     {link}\n\n"
                    f"  Or specify manually if you know the engine already:\n"
                    f"     xarray.open_dataset(uri, engine='ENGINE_NAME')\n"
                )
            else:
                raise ValueError(
                    f"Freva Xarray: cannot detect format for {uri!r}\n"
                    f"  Specify manually: "
                    f"  xarray.open_dataset(uri, engine='ENGINE_NAME')"
                )

        # Pop freva-specific kwargs
        kwargs.pop("xarray_engine", None)
        backend_kwargs = kwargs.pop("backend_kwargs", None) or {}

        # Check custom registry first (handles custom uri_types too)
        custom_handler = registry.get(engine, uri_type)
        if custom_handler:
            return custom_handler(
                uri,
                drop_variables=drop_variables,
                backend_kwargs=backend_kwargs,
                **kwargs,
            )

        # Route to POSIX or cloud handler (built-in uri_types only)
        if uri_type == "posix":
            return open_posix(
                uri,
                engine=engine,
                drop_variables=drop_variables,
                backend_kwargs=backend_kwargs,
                **kwargs,
            )
        elif uri_type in ("cloud", "reference"):
            return open_cloud(
                uri,
                engine=engine,
                drop_variables=drop_variables,
                backend_kwargs=backend_kwargs,
                lines_above=lines_printed,
                **kwargs,
            )
        else:
            # Custom uri_type without registered handler
            raise ValueError(
                f"No handler registered for uri_type={uri_type!r} "
                f"with engine={engine!r}. "
                f"Use @registry.register('{engine}', uri_type='{uri_type}')"
                f" to add one."
            )

    def _detect(self, uri: str, **kwargs: Any) -> Tuple[Optional[str], str]:
        """Detect engine and URI type."""
        uri_type = detect_uri_type(uri)

        # Get storage_options for detection
        storage_options = kwargs.get("storage_options")

        # Allow explicit override
        forced = kwargs.get("xarray_engine")
        if forced and forced in self.ENGINE_MAP:
            return self.ENGINE_MAP[forced], uri_type

        # Reference URIs -> zarr
        if is_reference_uri(uri):
            return "zarr", uri_type

        # OPeNDAP detection (before magic bytes)
        if is_http_url(uri) and looks_like_opendap_url(uri):
            return "netcdf4", uri_type

        # Magic byte detection (with storage_options)
        detected = detect_engine(uri, storage_options=storage_options)
        if detected == "unknown":
            return None, uri_type

        engine = self.ENGINE_MAP.get(detected, detected)

        if detected not in self.ENGINE_MAP:
            import warnings

            warnings.warn(
                f"Detected engine '{detected}' is not a built-in engine. "
                f"Ensure it's registered with xarray or use registry.register() "
                f"to add a custom handler.",
                UserWarning,
                stacklevel=3,
            )

        return engine, uri_type

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        """Xarray Generic Function: cheap check without I/O."""
        if not isinstance(filename_or_obj, (str, Path)):
            return False

        u = str(filename_or_obj).lower()

        # Zarr
        if u.endswith(".zarr") or ".zarr/" in u:
            return True
        if u.startswith("reference://"):
            return True

        # Common extensions
        if u.endswith(
            (".grib", ".grib2", ".grb", ".grb2", ".tif", ".tiff", ".nc", ".nc4")
        ):
            return True

        # OPeNDAP
        if is_http_url(u) and looks_like_opendap_url(u):
            return True

        return False
