"""Utility functions"""

import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

_COMPRESSION_SUFFIXES = (".bz2", ".gz", ".xz", ".zst", ".lz4")

STORAGE_OPTIONS_TO_GDAL: Dict[str, str] = {
    "key": "AWS_ACCESS_KEY_ID",
    "secret": "AWS_SECRET_ACCESS_KEY",
    "token": "AWS_SESSION_TOKEN",
    "aws_access_key_id": "AWS_ACCESS_KEY_ID",
    "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
    "aws_session_token": "AWS_SESSION_TOKEN",
    "region": "AWS_DEFAULT_REGION",
    "region_name": "AWS_DEFAULT_REGION",
    "endpoint_url": "AWS_S3_ENDPOINT",
    "client_kwargs.endpoint_url": "AWS_S3_ENDPOINT",
    "anon": "AWS_NO_SIGN_REQUEST",
    "profile": "AWS_PROFILE",
}


class ProgressBar:
    """Progress bar to display cache download progress."""

    def __init__(
        self, desc: str = "Downloading", width: int = 40, lines_above: int = 0
    ):
        self.desc = desc
        self.width = width
        self._total = 0
        self._current = 0
        self._spinner = 0
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._last_line_len = 0
        self._lines_above = lines_above

    def set_size(self, size: int) -> None:
        self._total = size if size else 0

    def update(self, inc: int) -> None:
        self._current += inc
        self._render()

    def _render(self) -> None:
        mb = self._current / 1024**2

        if self._total == 0:
            spinner = self._spinner_chars[self._spinner % len(self._spinner_chars)]
            self._spinner += 1
            line = f"{self.desc} {spinner} {mb:.1f} MB"
        else:
            pct = min(self._current / self._total, 1.0)
            filled = int(self.width * pct)
            bar = "█" * filled + "░" * (self.width - filled)
            total_mb = self._total / 1024**2
            line = f"{self.desc} |{bar}| {pct * 100:.0f}% ({mb:.1f}/{total_mb:.1f} MB)"

        clear = " " * self._last_line_len
        sys.stdout.write(f"\r{clear}\r{line}")
        sys.stdout.flush()
        self._last_line_len = len(line)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Clear the progress line
        sys.stdout.write("\r" + " " * self._last_line_len + "\r")

        # Clear detection + warning messages
        for _ in range(self._lines_above):
            sys.stdout.write("\033[A")
            sys.stdout.write("\033[K")

        sys.stdout.flush()


def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flatten nested dicts with dot notation."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _convert_value(key: str, value: Any) -> Optional[str]:
    """Convert Python values to GDAL env var format."""
    if value is None:
        return None
    if key == "AWS_NO_SIGN_REQUEST":
        return "YES" if value else "NO"
    if key == "AWS_S3_ENDPOINT":
        # important: GDAL expects host:port, not full URL
        s = str(value)
        if s.startswith("https://"):
            return s[8:]
        if s.startswith("http://"):
            return s[7:]
        return s
    return str(value)


@contextmanager
def gdal_env(storage_options: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    """
    Converts fsspec-style storage_options to GDAL environment variables for
    rasterio S3 access, since rasterio does not accept storage_options directly.
    """
    if not storage_options:
        yield
        return

    flat_opts = _flatten_dict(storage_options)

    original_env: Dict[str, Optional[str]] = {}
    set_vars: list = []

    endpoint_url = flat_opts.get("client_kwargs.endpoint_url") or flat_opts.get(
        "endpoint_url"
    )

    try:
        for opt_key, gdal_key in STORAGE_OPTIONS_TO_GDAL.items():
            if opt_key in flat_opts:
                value = _convert_value(gdal_key, flat_opts[opt_key])
                if value is not None:
                    original_env[gdal_key] = os.environ.get(gdal_key)
                    os.environ[gdal_key] = value
                    set_vars.append(gdal_key)

        if endpoint_url:
            if "AWS_HTTPS" not in set_vars:
                original_env["AWS_HTTPS"] = os.environ.get("AWS_HTTPS")
                os.environ["AWS_HTTPS"] = (
                    "YES" if endpoint_url.startswith("https://") else "NO"
                )
                set_vars.append("AWS_HTTPS")
            if "AWS_VIRTUAL_HOSTING" not in set_vars:
                original_env["AWS_VIRTUAL_HOSTING"] = os.environ.get(
                    "AWS_VIRTUAL_HOSTING"
                )
                os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
                set_vars.append("AWS_VIRTUAL_HOSTING")

        yield

    finally:
        for gdal_key in set_vars:
            original = original_env.get(gdal_key)
            if original is None:
                os.environ.pop(gdal_key, None)
            else:
                os.environ[gdal_key] = original


# kwargs that rasterio/rioxarray does not accept at all
_RASTERIO_UNSUPPORTED_KWARGS = frozenset(
    [
        "use_cftime",
        "decode_cf",
        "decode_times",
        "decode_timedelta",
        "use_default_fill_value",
        "cftime_variables",
    ]
)

# kwargs that rasterio/rioxarray accepts but with restricted values;
# maps kwarg name -> (allowed_value, human-readable reason)
_RASTERIO_RESTRICTED_KWARGS: Dict[str, Any] = {
    "decode_coords": (
        "all",
        "rioxarray only supports decode_coords='all'; overriding your value",
    ),
}


def sanitize_rasterio_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove or adjust kwargs that are incompatible with the rasterio/rioxarray
    backend.
    """
    sanitized = dict(kwargs)

    for key in _RASTERIO_UNSUPPORTED_KWARGS:
        if key in sanitized:
            logger.warning(
                "dropping unsupported kwarg "
                "'%s=%r' (rioxarray does not accept this parameter).",
                key,
                sanitized.pop(key),
            )

    for key, (allowed, reason) in _RASTERIO_RESTRICTED_KWARGS.items():
        if key in sanitized and sanitized[key] != allowed:
            logger.warning(
                "'%s=%r' is not supported — " "%s (using '%s' instead).",
                key,
                sanitized[key],
                reason,
                allowed,
            )
            sanitized[key] = allowed

    return sanitized


def _clean_surrogates_str(s: str) -> str:
    return s.encode("utf-8", "replace").decode("utf-8")


def _clean_attr_obj(obj):
    if isinstance(obj, str):
        return _clean_surrogates_str(obj)
    if isinstance(obj, dict):
        return {k: _clean_attr_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_attr_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clean_attr_obj(v) for v in obj)
    return obj


def sanitize_dataset_attrs(ds):
    # global attrs
    if ds.attrs:
        ds.attrs = _clean_attr_obj(dict(ds.attrs))

    # variable attrs only (cheap, no data touched)
    for var in ds.variables.values():
        if var.attrs:
            var.attrs = _clean_attr_obj(dict(var.attrs))

    return ds


def _strip_chaining_options(storage_options: dict) -> dict:
    """Strip fsspec chaining/wrapper protocol keys from storage_options.
    These are keys like 'simplecache', 'blockcache', 'filecache' that are
    fsspec protocol names used for URL chaining, not valid HTTP/remote FS kwargs.
    """
    if not storage_options:
        return {}
    from fsspec.registry import known_implementations

    return {k: v for k, v in storage_options.items() if k not in known_implementations}


def _strip_compression_suffix(uri: str) -> str:
    """Remove common compression suffixes from URI for
    more accurate pattern detection."""
    for suffix in _COMPRESSION_SUFFIXES:
        if uri.endswith(suffix):
            return uri[: -len(suffix)]
    return uri


def _decompress_if_needed(path: str, output_dir: Optional[str] = None) -> str:
    """Decompress a file if it has a known compression suffix.
    Returns the path to the decompressed file (or original if not compressed).
    Uses output_dir if provided, otherwise decompresses alongside the source file.
    """
    import bz2
    import gzip
    import lzma

    # Dict[str, Any] avoids mypy errors from overloaded open() signatures
    _DECOMPRESSORS: Dict[str, Any] = {
        ".bz2": bz2.open,
        ".gz": gzip.open,
        ".xz": lzma.open,
    }

    for suffix, opener in _DECOMPRESSORS.items():
        if path.endswith(suffix):
            bare_name = os.path.basename(path)[: -len(suffix)]
            out_dir = output_dir or os.path.dirname(path) or tempfile.gettempdir()
            decompressed = os.path.join(out_dir, bare_name)
            if not os.path.exists(decompressed):
                with opener(path, "rb") as src, open(decompressed, "wb") as dst:
                    while chunk := src.read(512 * 1024):
                        dst.write(chunk)
            return decompressed

    return path
