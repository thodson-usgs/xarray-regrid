from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True)
class RegridSpec:
    """Canonical metadata describing a source->target regridding layout."""

    src_dims: tuple[Hashable, ...]
    dst_dims: tuple[Hashable, ...]
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    x_coord: str
    y_coord: str
    spherical: bool
