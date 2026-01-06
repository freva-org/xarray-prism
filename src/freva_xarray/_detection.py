"""Engine and URI type detection for xarray data sources."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import fsspec

try:
    from aiohttp import ClientResponseError
except ImportError:
    ClientResponseError = Exception  # type: ignore


Detector = Callable[[str], Optional[str]]

_custom_detectors: List[Tuple[int, Detector]] = []

_custom_uri_type_detectors: List[Tuple[int, Detector]] = []


# --------------------------------------------------------------------------- #
# External Detector Registration
# --------------------------------------------------------------------------- #


def register_detector(priority: int = 50):
    """
    Decorator to register a custom format detector.

    Higher priority runs first. Built-in detection runs at priority 0.
    """

    def decorator(func: Detector) -> Detector:
        _custom_detectors.append((priority, func))
        # important: highest first
        _custom_detectors.sort(key=lambda x: -x[0])
        # Clear cache when new detector added
        _detect_engine_cached.cache_clear()
        return func

    return decorator


def register_uri_type(priority: int = 50):
    """
    Decorator to register a custom URI type detector.

    Higher priority runs first. Built-in detection runs at priority 0.
    """

    def decorator(func: Detector) -> Detector:
        _custom_uri_type_detectors.append((priority, func))
        _custom_uri_type_detectors.sort(key=lambda x: -x[0])
        return func

    return decorator


def unregister_detector(func: Detector) -> bool:
    """Remove a registered detector."""
    global _custom_detectors
    original_len = len(_custom_detectors)
    _custom_detectors = [(p, f) for p, f in _custom_detectors if f is not func]
    if len(_custom_detectors) != original_len:
        _detect_engine_cached.cache_clear()
        return True
    return False


def unregister_uri_type(func: Detector) -> bool:
    """Remove a registered URI type detector."""
    global _custom_uri_type_detectors
    original_len = len(_custom_uri_type_detectors)
    _custom_uri_type_detectors = [
        (p, f) for p, f in _custom_uri_type_detectors if f is not func
    ]
    return len(_custom_uri_type_detectors) != original_len


def _run_custom_detectors(uri: str) -> Optional[str]:
    """Run custom detectors in priority order."""
    for _, detector in _custom_detectors:
        try:
            result = detector(uri)
            if result is not None:
                return result
        except Exception:
            pass
    return None


def _run_custom_uri_type_detectors(uri: str) -> Optional[str]:
    """Run custom URI type detectors in priority order."""
    for _, detector in _custom_uri_type_detectors:
        try:
            result = detector(uri)
            if result is not None:
                return result
        except Exception:
            pass
    return None


def detect_uri_type(uri: str) -> str:
    """Detect if URI is local (posix), remote (cloud), reference, or custom."""
    # Run custom URI type detectors first
    custom_result = _run_custom_uri_type_detectors(uri)
    if custom_result is not None:
        return custom_result

    # Built-in detection
    lower = uri.lower()
    if lower.startswith("reference://"):
        return "reference"
    if lower.startswith("file://"):
        return "posix"
    if "://" in uri:
        return "cloud"
    return "posix"


def is_http_url(uri: str) -> bool:
    return uri.lower().startswith(("http://", "https://"))


def is_remote_uri(path: str) -> bool:
    return isinstance(path, str) and "://" in path and not path.startswith("file://")


def is_reference_uri(uri: str) -> bool:
    return uri.lower().startswith("reference://")


def looks_like_opendap_url(uri: str) -> bool:
    """Pure string heuristics for OPeNDAP-style URLs."""
    u = uri.lower()
    return any(
        s in u
        for s in (
            "dods",
            "opendap",
            "thredds/dods",
            "thredds/dodsC",
            "thredds/dap4",
            ".dods?",
            "?dap4",
        )
    )


def _detect_from_uri_pattern(lower_uri: str) -> Optional[str]:
    """Detect engine from URI patterns without I/O."""
    # Reference URIs -> zarr (Kerchunk)
    if is_reference_uri(lower_uri):
        return "zarr"

    # Zarr detection by extension
    if lower_uri.endswith(".zarr") or ".zarr/" in lower_uri:
        return "zarr"

    # THREDDS NCSS with explicit accept format (overrides file extension)
    if "/ncss/" in lower_uri or "/ncss?" in lower_uri:
        if "accept=netcdf3" in lower_uri:
            return "scipy"
        if "accept=netcdf4" in lower_uri or "accept=netcdf" in lower_uri:
            return "h5netcdf"

    # OPeNDAP / DODS URL detection
    opendap_patterns = ("/dodsc/", "/dods/", "/opendap/", "thredds/dodsc")
    if any(t in lower_uri for t in opendap_patterns):
        return "netcdf4"

    return None


def _detect_zarr_directory(fs: fsspec.AbstractFileSystem, path: str) -> Any:
    """Check if path is a Zarr directory store."""
    try:
        if fs.isdir(path):
            base = path.rstrip("/")
            return fs.exists(f"{base}/.zgroup") or fs.exists(f"{base}/.zattrs")
    except (FileNotFoundError, OSError):
        pass
    return False


def _read_magic_bytes(fs: fsspec.AbstractFileSystem, path: str) -> Any:
    """Read magic bytes from file, handling errors."""
    try:
        return fs.cat_file(path, start=0, end=64)
    except ClientResponseError as e:
        content_desc = getattr(e, "headers", {}).get("Content-Description", "").lower()
        if "dods" in content_desc:
            return b"__OPENDAP__"
        return None
    except (FileNotFoundError, IsADirectoryError, OSError):
        return None
    except Exception:
        return None


def _detect_from_magic_bytes(header: bytes, lower_path: str) -> str:
    """Detect engine from magic bytes and file extension."""
    # GRIB detection
    if b"GRIB" in header or lower_path.endswith((".grib", ".grb", ".grb2")):
        return "cfgrib"

    # NetCDF3 (Classic)
    if header.startswith(b"CDF"):
        return "scipy"

    # HDF5 / NetCDF4
    if header.startswith(b"\x89HDF\r\n\x1a\n"):
        return "h5netcdf"

    # GeoTIFF
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "rasterio"
    if lower_path.endswith((".tif", ".tiff")):
        return "rasterio"

    return "unknown"


def detect_engine(uri: str, storage_options: Optional[Dict] = None) -> str:
    """
    Unified detection using fsspec.

    Runs custom detectors first (by priority), then built-in detection.
    """
    # Use cached version if no storage_options
    if not storage_options:
        return _detect_engine_cached(uri)

    # Otherwise, detect without caching
    return _detect_engine_impl(uri, storage_options)


@lru_cache(maxsize=256)
def _detect_engine_cached(uri: str) -> str:
    """Cached version for URIs without storage_options."""
    return _detect_engine_impl(uri, None)


def _detect_engine_impl(uri: str, storage_options: Optional[Dict]) -> str:
    """Actual detection logic."""
    # 1. Run custom detectors first
    custom_result = _run_custom_detectors(uri)
    if custom_result is not None:
        return custom_result

    # 2. Pattern-based detection (no I/O)
    lower_uri = uri.lower()
    pattern_result = _detect_from_uri_pattern(lower_uri)
    if pattern_result is not None:
        return pattern_result

    # 3. Filesystem-based detection
    fs, path = fsspec.core.url_to_fs(uri, **(storage_options or {}))
    lower_path = path.lower()

    # Check for Zarr directory
    if _detect_zarr_directory(fs, path):
        return "zarr"

    # 4. Magic byte detection
    header = _read_magic_bytes(fs, path)
    if header is None:
        return "unknown"
    if header == b"__OPENDAP__":
        return "netcdf4"

    return _detect_from_magic_bytes(header, lower_path)
