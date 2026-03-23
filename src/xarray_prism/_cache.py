"""Cache management for xarray-prism remote file cache."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from hashlib import md5
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

from .utils import _decompress_if_needed, _strip_chaining_options

logger = logging.getLogger(__name__)

MAX_AGE_DAYS = float(os.environ.get("XARRAY_PRISM_MAX_AGE_DAYS", 7))
MAX_SIZE_GB = float(os.environ.get("XARRAY_PRISM_MAX_SIZE_GB", 10))


def get_cache_dir(storage_options: Optional[Dict] = None) -> Path:
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


def cache_remote_file(
    uri: str,
    engine: str,
    storage_options: Optional[Dict] = None,
    show_progress: bool = True,
    lines_above: int = 0,
) -> str:
    """Cache remote file to local."""
    import fsspec

    from .utils import ProgressBar

    cache_root = get_cache_dir(storage_options)
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
        return _decompress_if_needed(str(local_path))

    extra_lines = 0
    if show_progress:
        fmt = "GRIB" if engine == "cfgrib" else "NetCDF3"
        logger.warning(f"Remote {fmt} requires full file download")
        extra_lines = 2

    fs, path = fsspec.core.url_to_fs(
        uri, **_strip_chaining_options(storage_options or {})
    )

    if show_progress:
        size = 0
        try:
            size = fs.size(path) or 0
        except Exception:
            pass

        display_name = Path(parsed.path).name
        if len(display_name) > 35:
            display_name = display_name[:32] + "..."
        desc = f" Downloading {display_name}"

        with ProgressBar(desc=desc, lines_above=lines_above + extra_lines) as progress:
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

    return _decompress_if_needed(str(local_path))


def clear_cache(
    max_age_days: Optional[float] = MAX_AGE_DAYS,
    max_size_gb: Optional[float] = MAX_SIZE_GB,
    dry_run: bool = False,
) -> dict:
    """
    Evict cached files using two independent policies (both run each call):

    1. TTL: remove files not accessed in ``max_age_days`` days.
    2. Size cap: if total cache exceeds ``max_size_gb``, remove
       least-recently-used files until under the limit.

    Parameters
    ----------
    max_age_days : float | None
        Files older than this (by last-access time) are removed.
        Pass ``None`` to skip TTL eviction.
    max_size_gb : float | None
        Target maximum cache size in GiB.
        Pass ``None`` to skip size-cap eviction.
    dry_run : bool
        If True, report what would be deleted without deleting.

    """
    cache = get_cache_dir()
    if not cache.exists():
        return {"removed": [], "freed_bytes": 0}

    files = sorted(cache.iterdir(), key=lambda p: p.stat().st_atime)
    now = time.time()
    removed: list[Path] = []
    freed = 0

    # Policy 1: TTL
    if max_age_days is not None:
        cutoff = now - max_age_days * 86_400
        for f in list(files):
            if f.stat().st_atime < cutoff:
                freed += f.stat().st_size
                if not dry_run:
                    f.unlink(missing_ok=True)
                removed.append(f)
                files.remove(f)

    # Policy 2: Size cap
    if max_size_gb is not None:
        limit = int(max_size_gb * 1024**3)
        total = sum(f.stat().st_size for f in files)
        for f in files:
            if total <= limit:
                break
            size = f.stat().st_size
            freed += size
            total -= size
            if not dry_run:
                f.unlink(missing_ok=True)
            removed.append(f)

    return {"removed": removed, "freed_bytes": freed}


def cache_info() -> dict:
    """Return current cache size and file count"""
    cache = get_cache_dir()
    if not cache.exists():
        return {"files": 0, "size_bytes": 0, "path": str(cache)}
    files = list(cache.iterdir())
    return {
        "files": len(files),
        "size_bytes": sum(f.stat().st_size for f in files),
        "path": str(cache),
    }
