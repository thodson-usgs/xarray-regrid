"""Conservative regridding for grids that aren't 1D-separable.

The existing ``conservative`` method uses axis-factored 1D overlap — fast and
elegant but strictly rectilinear. This module computes the full 2D cell
intersection via shapely, so it handles:

- curvilinear grids (2D ``lat[i, j]`` / ``lon[i, j]`` coordinate variables)
- unstructured meshes (arbitrary polygon cells, via
  :meth:`ConservativeRegridder.from_polygons`)
- grid-to-polygon aggregation (e.g. gridded data → country shapes)

For rectilinear grids a cheap analytic fast-path is used, but this module is
still slower and more memory-intensive than ``conservative``; prefer the
axis-factored path when your grid is 1D-separable.

Requires ``shapely >= 2.0``. If ``sparse`` is available, the weight matrix is
stored as ``sparse.COO``; otherwise a dense numpy matrix is used.
"""

import warnings
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from xarray_regrid.methods.conservative_2d import geometry as _geom
from xarray_regrid.methods.conservative_2d import weights as _weights
from xarray_regrid.methods.conservative_2d._deps import AreaMatrix, require_shapely
from xarray_regrid.methods.conservative_2d.apply import (
    apply_stored_weights as _apply_stored_weights,
)
from xarray_regrid.methods.conservative_2d.serialization import (
    load_regridder_netcdf,
    save_regridder_netcdf,
)
from xarray_regrid.methods.conservative_2d.spec import RegridSpec

NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"] | None

# We fill NaNs with 0 ourselves before matmul (see `_apply_core`), so sparse's
# "NaN will not be propagated" warning is spurious. A module-level filter
# avoids `warnings.catch_warnings` inside the hot matmul path, which is not
# thread-safe — dask threads would race on the global `warnings.filters` list.
warnings.filterwarnings(
    "ignore",
    message="Nan will not be propagated in matrix multiplication",
    category=RuntimeWarning,
)
class _Direction:
    """Lazy weights, apply matrix, and coverage mask for one regrid direction.

    Holds the raw ``(n_dst, n_src)`` area matrix in this direction's orientation
    and derives, on first access:

    - ``weights``: row-normalized weight matrix
    - ``apply_matrix``: pre-transposed and index-sorted weights, so
      ``_apply_core``'s matmul is ``(..., n_src) @ (n_src, n_dst)`` with no
      per-call sort
    - ``coverage`` / ``coverage_all``: which output cells have any source overlap

    A regridder holds two of these (forward, backward); transposing the
    regridder swaps them with no recomputation.
    """

    def __init__(self, areas: AreaMatrix) -> None:
        self.matrix = _weights.WeightMatrix(areas)

    @cached_property
    def weights(self) -> AreaMatrix:
        return self.matrix.row_normalized()

    @cached_property
    def apply_matrix(self) -> AreaMatrix:
        return _weights.transpose_weights(self.weights, sort=True)

    @cached_property
    def coverage(self) -> np.ndarray:
        return self.matrix.coverage()

    @cached_property
    def coverage_all(self) -> bool:
        return bool(self.coverage.all())


class ConservativeRegridder:
    """Reusable conservative regridder for grids that aren't 1D-separable:
    curvilinear (2D ``lat``/``lon``), unstructured (via :meth:`from_polygons`),
    or arbitrary polygon-to-polygon aggregation. For purely 1D-separable
    rectilinear grids, the ``.regrid.conservative`` accessor is faster.

    Build once, apply to many fields via :meth:`regrid` (or by calling the
    regridder); ``.T`` gives the backward regridder. Forward and backward
    weight matrices are cached lazily. Requires ``shapely >= 2.0``.

    The unnormalized cell-intersection area matrix
    ``A[i, j] = area(target_i ∩ source_j)`` is exposed as ``self.areas``
    (sparse ``(n_dst, n_src)`` if the ``sparse`` package is available, dense
    otherwise) — useful for conservation diagnostics and per-cell coverage
    analysis.
    """

    def __init__(
        self,
        source: xr.DataArray | xr.Dataset,
        target: xr.Dataset,
        x_coord: str = "longitude",
        y_coord: str = "latitude",
        spherical: bool = False,
        n_threads: int | None = None,
    ) -> None:
        require_shapely()
        source_grid, target_grid, src_x_sort_idx = _geom.normalize_longitude_coords(
            source, target, x_coord
        )
        src_dims = _geom.spatial_dims(source_grid, x_coord, y_coord)
        dst_dims = _geom.spatial_dims(target_grid, x_coord, y_coord)
        if not src_dims:
            msg = f"source has no dims for coords {x_coord!r}, {y_coord!r}"
            raise ValueError(msg)
        if not dst_dims:
            msg = f"target has no dims for coords {x_coord!r}, {y_coord!r}"
            raise ValueError(msg)
        src_grid = _geom.grid_from_coords(
            source_grid, x_coord, y_coord, src_dims, spherical=spherical
        )
        dst_grid = _geom.grid_from_coords(
            target_grid, x_coord, y_coord, dst_dims, spherical=spherical
        )
        self.spherical = spherical

        self.x_coord = x_coord
        self.y_coord = y_coord
        self._src_dims = src_dims
        self._dst_dims = dst_dims
        self._src_shape = tuple(int(source.sizes[d]) for d in src_dims)
        self._dst_shape = tuple(int(target.sizes[d]) for d in dst_dims)
        areas = _weights.build_intersection_areas(
            src_grid,
            dst_grid,
            n_threads=n_threads,
        )
        if src_x_sort_idx is not None:
            # Source x was sorted by _normalize_longitude_coords so polygon
            # construction stayed monotone. Relabel matrix columns so column
            # order matches the user's original (unsorted) data layout.
            x_dim_index = src_dims.index(source[x_coord].dims[0])
            areas = _geom.remap_columns_for_axis_sort(
                areas, src_x_sort_idx, self._src_shape, x_dim_index
            )
        self.areas = areas
        self._source_coords = source.coords.to_dataset()
        self._target_coords = target.coords.to_dataset()

    @property
    def spec(self) -> RegridSpec:
        """Canonical metadata describing this regridder's layout."""
        return RegridSpec(
            src_dims=self._src_dims,
            dst_dims=self._dst_dims,
            src_shape=self._src_shape,
            dst_shape=self._dst_shape,
            x_coord=self.x_coord,
            y_coord=self.y_coord,
            spherical=self.spherical,
        )

    @cached_property
    def _forward(self) -> _Direction:
        return _Direction(self.areas)

    @cached_property
    def _backward(self) -> _Direction:
        return _Direction(_weights.WeightMatrix(self.areas).transposed())

    @property
    def forward_weights(self) -> AreaMatrix:
        """The row-normalized forward weight matrix (source → target)."""
        return self._forward.weights

    @property
    def backward_weights(self) -> AreaMatrix:
        """The row-normalized backward weight matrix (target → source)."""
        return self._backward.weights

    @property
    def target_areas(self) -> np.ndarray:
        """Area of each target cell overlapped by the source domain."""
        return _weights.WeightMatrix(self.areas).sum_axis(axis=1)

    @property
    def source_coverage_areas(self) -> np.ndarray:
        """Area of each source cell covered by target cells."""
        return _weights.WeightMatrix(self.areas).sum_axis(axis=0)

    def regrid(
        self,
        data: xr.DataArray | xr.Dataset,
        skipna: bool = True,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid ``data`` forward (source → target)."""
        return _apply_stored_weights(
            data,
            direction=self._forward,
            spec=self.spec,
            target_coords=self._target_coords,
            skipna=skipna,
            nan_threshold=nan_threshold,
        )

    def __call__(
        self,
        data: xr.DataArray | xr.Dataset,
        skipna: bool = True,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray | xr.Dataset:
        return self.regrid(data, skipna=skipna, nan_threshold=nan_threshold)

    def transpose(self) -> "ConservativeRegridder":
        """Return the backward regridder (target → source). The original's
        forward Direction becomes the new's backward (and vice versa), so any
        already-computed weight matrices are reused, not recomputed."""
        new = type(self)._from_state(
            areas=_weights.transpose_weights(self.areas),
            source_coords=self._target_coords,
            target_coords=self._source_coords,
            spec=RegridSpec(
                src_dims=self._dst_dims,
                dst_dims=self._src_dims,
                src_shape=self._dst_shape,
                dst_shape=self._src_shape,
                x_coord=self.x_coord,
                y_coord=self.y_coord,
                spherical=self.spherical,
            ),
        )
        if "_forward" in self.__dict__:
            new.__dict__["_backward"] = self.__dict__["_forward"]
        if "_backward" in self.__dict__:
            new.__dict__["_forward"] = self.__dict__["_backward"]
        return new

    @property
    def T(self) -> "ConservativeRegridder":  # noqa: N802
        """Alias for :meth:`transpose` (numpy-style transpose)."""
        return self.transpose()

    def __repr__(self) -> str:
        nnz = getattr(self.areas, "nnz", None)
        shape = getattr(self.areas, "shape", (None, None))
        nnz_str = f"nnz={nnz}" if nnz is not None else "dense"
        return (
            f"ConservativeRegridder(src_dims={self._src_dims}, "
            f"dst_dims={self._dst_dims}, {shape[0]}x{shape[1]}, {nnz_str})"
        )

    def to_netcdf(self, path: str | Path, engine: NetcdfEngine = None) -> None:
        """Save the weight matrix and reproducibility metadata to a netCDF file.
        Requires a group-aware engine (``netcdf4`` or ``h5netcdf``); ``engine``
        is forwarded to :func:`xarray.Dataset.to_netcdf`."""
        save_regridder_netcdf(
            path=path,
            areas=self.areas,
            spec=self.spec,
            source_coords=self._source_coords,
            target_coords=self._target_coords,
            engine=engine,
        )

    @classmethod
    def from_netcdf(
        cls, path: str | Path, engine: NetcdfEngine = None
    ) -> "ConservativeRegridder":
        """Reload a regridder previously written with :meth:`to_netcdf`.

        Validates ``schema_version``; raises :class:`ValueError` if the file
        was written by an incompatible version.
        """
        areas, source_coords, target_coords, meta = load_regridder_netcdf(
            path=path, engine=engine
        )

        return cls._from_state(
            areas=areas,
            source_coords=source_coords,
            target_coords=target_coords,
            spec=meta,
        )

    @classmethod
    def _from_state(
        cls,
        *,
        areas: AreaMatrix,
        source_coords: xr.Dataset,
        target_coords: xr.Dataset,
        spec: RegridSpec,
    ) -> "ConservativeRegridder":
        """Construct a regridder directly from its canonical state. Shared
        bypass of ``__init__`` used by :meth:`from_netcdf` and
        :meth:`from_polygons`; keeps the list of private attrs in one place."""
        instance = object.__new__(cls)
        instance.x_coord = spec.x_coord
        instance.y_coord = spec.y_coord
        instance.spherical = spec.spherical
        instance._src_dims = spec.src_dims
        instance._dst_dims = spec.dst_dims
        instance._src_shape = spec.src_shape
        instance._dst_shape = spec.dst_shape
        instance.areas = areas
        instance._source_coords = source_coords
        instance._target_coords = target_coords
        return instance

    @classmethod
    def from_polygons(
        cls,
        source_polygons: np.ndarray,
        target_polygons: np.ndarray,
        source_dim: str = "cell",
        target_dim: str = "cell",
        target_coords: xr.Dataset | None = None,
        periodic: bool = False,
        n_threads: int | None = None,
        predicate_filter: bool = True,
    ) -> "ConservativeRegridder":
        """Build a regridder from explicit shapely polygon arrays — for
        unstructured meshes (MPAS, ICON), arbitrary polygon targets (countries,
        watersheds), or any non-rectilinear combination.

        Args:
            source_polygons, target_polygons: 1D arrays of shapely Polygons.
            source_dim, target_dim: Dim names for source/target cells on the
                input and output arrays.
            target_coords: Optional Dataset of coord variables along
                ``target_dim`` to reattach on the output (else: integer index).
            periodic: Unwrap polygons that cross the antimeridian (treats x
                as longitude on a 360-degree periodic axis).
            n_threads: Thread count for parallel GEOS intersection.
            predicate_filter: If True, filter STRtree candidates with GEOS
                ``intersects``. Set False for tight-bbox grid cells to skip
                the predicate (faster on that case, pathological otherwise).

        Geometry is planar in the polygons' own coordinate space. For lat/lon
        cells, project into an equal-area CRS first or use the structured
        path with ``spherical=True``.
        """
        require_shapely()
        src_polys = np.asarray(source_polygons)
        dst_polys = np.asarray(target_polygons)
        if src_polys.ndim != 1 or dst_polys.ndim != 1:
            msg = "source_polygons and target_polygons must be 1D arrays"
            raise ValueError(msg)
        if periodic:
            src_polys = _geom.normalize_periodic_polygons(src_polys)
            dst_polys = _geom.normalize_periodic_polygons(
                dst_polys, reference=_geom.polygon_reference_x(src_polys)
            )

        src_grid = _geom.Grid(
            polys=src_polys,
            bounds=_geom.shapely.bounds(src_polys),
            rectilinear=False,
        )
        dst_grid = _geom.Grid(
            polys=dst_polys,
            bounds=_geom.shapely.bounds(dst_polys),
            rectilinear=False,
        )
        n_src = int(src_polys.size)
        n_dst = int(dst_polys.size)
        tgt_ds = (
            target_coords
            if target_coords is not None
            else xr.Dataset(coords={target_dim: np.arange(n_dst)})
        )
        return cls._from_state(
            areas=_weights.build_intersection_areas(
                # preserve current default behavior for user-supplied polygons
                # where predicate filtering is typically beneficial
                src_grid,
                dst_grid,
                n_threads=n_threads,
                predicate_filter=predicate_filter,
            ),
            source_coords=xr.Dataset(coords={source_dim: np.arange(n_src)}),
            target_coords=tgt_ds,
            spec=RegridSpec(
                src_dims=(source_dim,),
                dst_dims=(target_dim,),
                src_shape=(n_src,),
                dst_shape=(n_dst,),
                x_coord="",
                y_coord="",
                spherical=False,
            ),
        )


def polygons_from_coords(
    x: np.ndarray,
    y: np.ndarray,
    spherical: bool = False,
    periodic: bool = False,
) -> np.ndarray:
    """Build a 1D row-major (y, x) array of shapely cell polygons from 1D or
    2D center coords. Convenience for mixing structured and unstructured paths
    via :meth:`ConservativeRegridder.from_polygons`. ``spherical=True``
    projects 1D lat/lon (degrees) into Lambert cylindrical equal-area space;
    ``periodic=True`` unwraps antimeridian-crossing cells."""
    require_shapely()
    x = np.asarray(x)
    y = np.asarray(y)
    if periodic:
        x = _geom.unwrap_longitude(x)
    if spherical:
        if x.ndim != 1 or y.ndim != 1:
            msg = "spherical=True requires 1D lat/lon arrays"
            raise ValueError(msg)
        return _geom.build_cea_grid(x, y).polys
    return _geom.build_grid(x, y).polys
