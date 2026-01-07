"""Registry for custom backends for xarray_prism."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    import xarray as xr

OpenFunc = Callable[..., "xr.Dataset"]


class BackendRegistry:
    """Registry for custom backends."""

    def __init__(self):
        self._handlers: Dict[tuple, OpenFunc] = {}

    def register(
        self,
        engine: str,
        uri_type: str = "both",
    ) -> Callable[[OpenFunc], OpenFunc]:
        """Decorator to register a custom open function."""

        def decorator(func: OpenFunc) -> OpenFunc:
            if uri_type == "both":
                self._handlers[(engine, "posix")] = func
                self._handlers[(engine, "cloud")] = func
            else:
                self._handlers[(engine, uri_type)] = func
            return func

        return decorator

    def get(self, engine: str, uri_type: str) -> Optional[OpenFunc]:
        """Get handler for engine + uri_type combo."""
        return self._handlers.get((engine, uri_type))

    def has(self, engine: str, uri_type: str) -> bool:
        """Check if handler exists."""
        return (engine, uri_type) in self._handlers


registry = BackendRegistry()
