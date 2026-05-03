from collections.abc import Hashable
from dataclasses import dataclass, replace


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

    def transposed(self) -> "RegridSpec":
        """Swap source and target — same layout, opposite direction."""
        return replace(
            self,
            src_dims=self.dst_dims,
            dst_dims=self.src_dims,
            src_shape=self.dst_shape,
            dst_shape=self.src_shape,
        )
