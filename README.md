# Freva xarray Engine

A multi-format and multi-storage xarray engine with automatic engine detection,
and ability to register new data format and uri type for climate data.

> [!Important]
> If you deal with a data that `freva` engine is not able to open that, please
> report the data [here](https://github.com/freva-org/freva-xarray/issues/new)
> to let us improve this engine to be able to be versitile and work with all
> sort of climate data.


## Installation

### Install via PyPI

```bash
pip install freva-xarray
```

### Install via Conda

```bash
conda install freva-xarray
```

## Quick Start

### Using with xarray

```python
import xarray as xr

# Auto-detect format
ds = xr.open_dataset("my_data.unknown_fmt", engine="freva")

# Remote Zarr on S3
ds = xr.open_dataset(
    "s3://freva/workshop/tas.zarr",
    engine="freva",
    storage_options={
        "anon": True,
        "client_kwargs": {
            "endpoint_url": "https://s3.eu-dkrz-1.dkrz.cloud"
        }
    }
)

# Remote NetCDF3 on S3
ds = xr.open_dataset(
    "s3://freva/workshop/tas.nc",
    engine="freva",
    storage_options={
        "anon": True,
        "client_kwargs": {
            "endpoint_url": "https://s3.eu-dkrz-1.dkrz.cloud"
        }
    }
)

# Remote NetCDF4 on S3
ds = xr.open_dataset(
    "s3://freva/workshop/tas.nc4",
    engine="freva",
    storage_options={
        "anon": True,
        "client_kwargs": {
            "endpoint_url": "https://s3.eu-dkrz-1.dkrz.cloud"
        }
    }
)

# Remote Zarr on S3 - non-anon
ds = xr.open_dataset(
    "s3://bucket/data.zarr",
    engine="freva",
    storage_options={
        "key": "YOUR_KEY",
        "secret": "YOUR_SECRET",
        "client_kwargs": {
            "endpoint_url": "S3_ENDPOINT"
        }
    }
)

# OPeNDAP from THREDDS
ds = xr.open_dataset(
    "https://icdc.cen.uni-hamburg.de/thredds/dodsC/ftpthredds/ar5_sea_level_rise/gia_mean.nc",
    engine="freva"
)

# Local GRIB file
ds = xr.open_dataset("forecast.grib2", engine="freva")

# GeoTIFF
ds = xr.open_dataset("satellite.tif", engine="freva")

# tip: Handle the cache manually by yourself
xr.open_dataset(
    "simplecache::s3://bucket/file.nc3",
    engine="freva",
    storage_options={
        "s3": {"anon": True, "client_kwargs": {"endpoint_url": "..."}},
        "simplecache": {"cache_storage": "/path/to/cache"}
    }
)

# Even for the tif format on the S3 you can pass the credential through
# storage_options which is not supported by rasterio:
xr.open_dataset(
    "s3://bucket/file.tif",
    engine="freva",
    storage_options={
        "key": "YOUR_KEY",
        "secret": "YOUR_SECRET",
        "client_kwargs": {
            "endpoint_url": "S3_ENDPOINT"
        }
    }
)
```

## Supported Formats


|Data format   | Remote backend         | Local FS  | Cache|
|--------------|------------------------|-----------|-----------|
|GRIB          | cfgrib + fsspec        | cfgrib    | fsspec simplecache (full-file)|
|Zarr          | zarr + fsspec          | zarr      | chunked key/value store|
|NetCDF3       | scipy + fsspec         | scipy     | fsspec byte cache (5 MB blocks but full dowload)|
|NetCDF4/HDF5  | h5netcdf + fsspec      | h5netcdf  | fsspec byte cache (5 MB block)|
|GeoTIFF       | rasterio + fsspec      | rasterio  | GDAL/rasterio block cache (5 MB block)|
|OPeNDAP/DODS  | netCDF4                | n/a       | n/a|


> [!WARNING]
> **Remote GRIB & NetCDF3 require full file download**
> 
> Unlike Zarr or HDF5, these formats don't support partial/chunk reads over the network.
> 
> By default, freva-xarray caches files in the system temp directory. 
> This works well for most cases. 
> If temp storage is a concern (e.g., limited space or cleared on reboot), 
> you can specify a persistent cache:
> 
> | Option | How |
> |--------|-----|
> | Environment variable | `export FREVA_XARRAY_CACHE=/path/to/cache` |
> | Per-call | `storage_options={"simplecache": {"cache_storage": "/path"}}` |
> | Default | System temp directory |


## Customization

### Custom Format Detectors and URI Types

You can extend **freva-xarray** with custom *format detectors*, *URI types*, and *open handlers* by providing a small plugin package.
Registration happens **at import time**, so importing the plugin activates it.

### Plugin structure

```text
freva_xarray_myplugin/
  __init__.py   # imports the plugin module (triggers registration)
  plugin.py     # detectors, URI types, and open handlers
pyproject.toml
```

### Plugin implementation

`freva_xarray_myplugin/__init__.py`

```python
from .plugin import *  # noqa: F401,F403
```

`freva_xarray_myplugin/plugin.py`

```python
import xarray as xr
from freva_xarray import register_detector, register_uri_type, registry


@register_uri_type(priority=100)
def detect_myfs_uri(uri: str):
    """Detect a custom filesystem URI."""
    if uri.lower().startswith("myfs://"):
        return "myfs"
    return None


@register_detector(priority=100)
def detect_foo_format(uri: str):
    """Detect a custom file format."""
    if uri.lower().endswith(".foo"):
        return "foo"
    return None


@registry.register("foo", uri_type="myfs")
def open_foo_from_myfs(uri: str, **kwargs):
    """Open .foo files from myfs:// URIs."""
    translated = uri.replace("myfs://", "https://my-gateway.example/")
    return xr.open_dataset(translated, engine="h5netcdf", **kwargs)
```

### Plugin installation

`pyproject.toml`

```toml
[project]
name = "freva-xarray-myplugin"
version = "0.1.0"
dependencies = ["freva-xarray"]

[project.entry-points."freva_xarray.plugins"]
myplugin = "freva_xarray_myplugin"
```

### Using the plugin

After installing the plugin package, **import it once** to activate the registrations:

```python
import freva_xarray_myplugin  # activates detectors and handlers

import xarray as xr
ds = xr.open_dataset("myfs://bucket/path/data.foo", engine="freva")
```


## Development

### Setup Development Environment

```bash
# Start test services (MinIO, THREDDS)
docker-compose -f dev-env/docker-compose.yaml up -d --remove-orphans

# Create conda environment
conda create -n freva-xarray python=3.12 -y
conda activate freva-xarray

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run tests
tox -e test

# Run with coverage
tox -e test-cov

# Lint
tox -e lint

# Type checking
tox -e types

# Auto-format code
tox -e format
```

### Creating a Release

Releases are managed via GitHub Actions and tox:

```bash
# Tag a new release (creates git tag)
tox -e release
```

The release workflow is triggered automatically when:
- A version tag (`v*.*.*`) is pushed -> Full release to PyPI
- Manual workflow dispatch with RC number -> Pre-release to PyPI
