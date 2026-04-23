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

import os
import warnings
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import xarray as xr

from xarray_regrid import utils

NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"] | None


def _package_version() -> str:
    try:
        return version("xarray-regrid")
    except PackageNotFoundError:
        return "unknown"


# Bump on breaking change to the on-disk format in ConservativeRegridder.to_netcdf.
_SCHEMA_VERSION = 1


try:
    import shapely
    from shapely import affinity
    from shapely.strtree import STRtree

    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    shapely = None
    affinity = None
    STRtree = None
    _HAS_SHAPELY = False

try:
    import sparse

    _HAS_SPARSE = True
except ImportError:  # pragma: no cover
    sparse = None
    _HAS_SPARSE = False


# We fill NaNs with 0 ourselves before matmul (see `_apply_core`), so sparse's
# "NaN will not be propagated" warning is spurious. A module-level filter
# avoids `warnings.catch_warnings` inside the hot matmul path, which is not
# thread-safe — dask threads would race on the global `warnings.filters` list.
warnings.filterwarnings(
    "ignore",
    message="Nan will not be propagated in matrix multiplication",
    category=RuntimeWarning,
)


SHAPELY_IMPORT_ERROR = (
    "polygon conservative regridding requires shapely >= 2.0; "
    "install with `pip install shapely`."
)


def _check_shapely() -> None:
    if not _HAS_SHAPELY:
        raise ImportError(SHAPELY_IMPORT_ERROR)


class ConservativeRegridder:
    """Reusable conservative regridder for grids that aren't 1D-separable.

    Use this when your source or target isn't a pure rectilinear lat/lon
    grid: curvilinear coordinates (2D ``lat[i, j]`` / ``lon[i, j]``),
    unstructured meshes (via :meth:`from_polygons`), or arbitrary
    polygon-to-polygon aggregation. For plain 1D-separable rectilinear
    grids, the existing ``.conservative`` accessor is much faster.

    Build once from source and target grids; apply to many compatible fields
    via :meth:`regrid` (or by calling the regridder). The raw intersection
    area matrix is stored internally; the forward and backward row-normalized
    weight matrices are lazily cached on first use.

    Planar geometry only. Requires ``shapely >= 2.0``.

    Example::

        regridder = ConservativeRegridder(
            src_ds, tgt_ds, x_coord="lon", y_coord="lat"
        )
        out = regridder.regrid(da)              # forward
        back = regridder.T.regrid(out)          # backward
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
        _check_shapely()
        source_grid, target_grid = _normalize_longitude_coords(source, target, x_coord)
        src_dims = _spatial_dims(source_grid, x_coord, y_coord)
        dst_dims = _spatial_dims(target_grid, x_coord, y_coord)
        if not src_dims:
            msg = f"source has no dims for coords {x_coord!r}, {y_coord!r}"
            raise ValueError(msg)
        if not dst_dims:
            msg = f"target has no dims for coords {x_coord!r}, {y_coord!r}"
            raise ValueError(msg)
        src_grid = _grid_from_coords(
            source_grid, x_coord, y_coord, src_dims, spherical=spherical
        )
        dst_grid = _grid_from_coords(
            target_grid, x_coord, y_coord, dst_dims, spherical=spherical
        )
        self.spherical = spherical

        self.x_coord = x_coord
        self.y_coord = y_coord
        self._src_dims = src_dims
        self._dst_dims = dst_dims
        self._src_shape = tuple(int(source.sizes[d]) for d in src_dims)
        self._dst_shape = tuple(int(target.sizes[d]) for d in dst_dims)
        self._areas = _build_intersection_areas(src_grid, dst_grid, n_threads=n_threads)
        self._source_coords = source.coords.to_dataset()
        self._target_coords = target.coords.to_dataset()

    @cached_property
    def forward_weights(self) -> "sparse.COO | np.ndarray":
        """The row-normalized forward weight matrix (source → target)."""
        return _row_normalize(self._areas)

    @cached_property
    def backward_weights(self) -> "sparse.COO | np.ndarray":
        """The row-normalized backward weight matrix (target → source)."""
        return _row_normalize(_transpose_weights(self._areas))

    @cached_property
    def _forward_apply(self) -> "sparse.COO | np.ndarray":
        # Transposed + index-sorted once so the matmul in _apply_core is
        # (..., n_src) @ (n_src, n_dst) with no per-call sort.
        return _transpose_weights(self.forward_weights, sort=True)

    @cached_property
    def _backward_apply(self) -> "sparse.COO | np.ndarray":
        return _transpose_weights(self.backward_weights, sort=True)

    @cached_property
    def _forward_coverage(self) -> np.ndarray:
        return _coverage_mask(self._areas)

    @cached_property
    def _backward_coverage(self) -> np.ndarray:
        return _coverage_mask(_transpose_weights(self._areas))

    def regrid(
        self,
        data: xr.DataArray | xr.Dataset,
        skipna: bool = True,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid ``data`` forward (source → target)."""
        return _apply_stored_weights(
            data,
            apply_weights=self._forward_apply,
            coverage=self._forward_coverage,
            src_dims=self._src_dims,
            dst_dims=self._dst_dims,
            src_shape=self._src_shape,
            dst_shape=self._dst_shape,
            target_coords=self._target_coords,
            x_coord=self.x_coord,
            y_coord=self.y_coord,
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

    def regrid_blockwise(
        self,
        data: xr.DataArray | xr.Dataset,
        target_chunks: dict[Hashable, int] | None = None,
        skipna: bool = True,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid forward using per-target-block matmul.

        Unlike :meth:`regrid`, each target block pulls only the source cells
        that contribute to it (identified from the sparse weight matrix), so
        per-task source memory scales with the block's source footprint
        rather than the full source plane. Useful when the source is chunked
        spatially and the full plane doesn't fit in worker memory.

        ``target_chunks`` maps target dim names to chunk sizes, e.g.
        ``{"latitude": 90, "longitude": 180}``. Dims not listed default to
        one chunk.
        """
        return _apply_blockwise(
            data,
            apply_weights=self._forward_apply,
            coverage=self._forward_coverage,
            src_dims=self._src_dims,
            dst_dims=self._dst_dims,
            src_shape=self._src_shape,
            dst_shape=self._dst_shape,
            target_coords=self._target_coords,
            x_coord=self.x_coord,
            y_coord=self.y_coord,
            skipna=skipna,
            nan_threshold=nan_threshold,
            target_chunks=target_chunks or {},
        )

    def transpose(self) -> "ConservativeRegridder":
        """Return the backward regridder (target → source), sharing the
        underlying area matrix and any already-computed cached weight
        matrices (forward on the transposed regridder is backward on the
        original, and vice versa)."""
        new = object.__new__(ConservativeRegridder)
        new.x_coord = self.x_coord
        new.y_coord = self.y_coord
        new.spherical = self.spherical
        new._src_dims = self._dst_dims
        new._dst_dims = self._src_dims
        new._src_shape = self._dst_shape
        new._dst_shape = self._src_shape
        new._areas = _transpose_weights(self._areas)
        new._source_coords = self._target_coords
        new._target_coords = self._source_coords
        swap = {
            "forward_weights": "backward_weights",
            "backward_weights": "forward_weights",
            "_forward_apply": "_backward_apply",
            "_backward_apply": "_forward_apply",
            "_forward_coverage": "_backward_coverage",
            "_backward_coverage": "_forward_coverage",
        }
        for src, dst in swap.items():
            if src in self.__dict__:
                new.__dict__[dst] = self.__dict__[src]
        return new

    @property
    def T(self) -> "ConservativeRegridder":  # noqa: N802
        """Alias for :meth:`transpose` (numpy-style transpose)."""
        return self.transpose()

    def __repr__(self) -> str:
        nnz = getattr(self._areas, "nnz", None)
        shape = getattr(self._areas, "shape", (None, None))
        nnz_str = f"nnz={nnz}" if nnz is not None else "dense"
        return (
            f"ConservativeRegridder(src_dims={self._src_dims}, "
            f"dst_dims={self._dst_dims}, {shape[0]}x{shape[1]}, {nnz_str})"
        )

    def to_netcdf(self, path: str | Path, engine: NetcdfEngine = None) -> None:
        """Save the weight matrix and reproducibility metadata to a netCDF file.

        File layout:

        - root dataset: the sparse area matrix stored as three 1D variables
          ``_coo_row``, ``_coo_col``, ``_coo_data`` (``shape=(n_dst, n_src)``
          carried on root-dataset attributes), plus the regridder metadata as
          root attributes.
        - ``/source_coords`` group: coord-only Dataset capturing the source grid.
        - ``/target_coords`` group: coord-only Dataset capturing the target grid.

        Groups require an engine that supports them (``netcdf4`` or
        ``h5netcdf``); ``engine`` is forwarded to :func:`xarray.Dataset.to_netcdf`.
        """
        path = Path(path)
        row, col, data, shape = _coo_components(self._areas)
        ds_weights = xr.Dataset(
            {
                "_coo_row": (("nnz",), row),
                "_coo_col": (("nnz",), col),
                "_coo_data": (("nnz",), data),
            },
            attrs={
                **_metadata_attrs(self),
                "n_dst": int(shape[0]),
                "n_src": int(shape[1]),
            },
        )
        ds_weights.to_netcdf(path, mode="w", engine=engine)
        self._source_coords.to_netcdf(
            path, mode="a", group="source_coords", engine=engine
        )
        self._target_coords.to_netcdf(
            path, mode="a", group="target_coords", engine=engine
        )

    @classmethod
    def from_netcdf(
        cls, path: str | Path, engine: NetcdfEngine = None
    ) -> "ConservativeRegridder":
        """Reload a regridder previously written with :meth:`to_netcdf`.

        Validates ``schema_version``; raises :class:`ValueError` if the file
        was written by an incompatible version.
        """
        path = Path(path)
        with xr.open_dataset(path, engine=engine) as ds_weights:
            attrs = dict(ds_weights.attrs)
            n_dst = int(attrs.pop("n_dst"))
            n_src = int(attrs.pop("n_src"))
            row = np.asarray(ds_weights["_coo_row"].values)
            col = np.asarray(ds_weights["_coo_col"].values)
            data = np.asarray(ds_weights["_coo_data"].values)
        meta = _metadata_from_attrs(attrs, path)

        with xr.open_dataset(path, group="source_coords", engine=engine) as g:
            source_coords = g.load()
        with xr.open_dataset(path, group="target_coords", engine=engine) as g:
            target_coords = g.load()

        return cls._from_state(
            areas=_coo_from_components(row, col, data, (n_dst, n_src)),
            source_coords=source_coords,
            target_coords=target_coords,
            **meta,
        )

    @classmethod
    def _from_state(
        cls,
        *,
        areas: "sparse.COO | np.ndarray",
        source_coords: xr.Dataset,
        target_coords: xr.Dataset,
        src_dims: tuple[Hashable, ...],
        dst_dims: tuple[Hashable, ...],
        src_shape: tuple[int, ...],
        dst_shape: tuple[int, ...],
        x_coord: str,
        y_coord: str,
        spherical: bool,
    ) -> "ConservativeRegridder":
        """Construct a regridder directly from its canonical state. Shared
        bypass of ``__init__`` used by :meth:`from_netcdf` and
        :meth:`from_polygons`; keeps the list of private attrs in one place."""
        instance = object.__new__(cls)
        instance.x_coord = x_coord
        instance.y_coord = y_coord
        instance.spherical = spherical
        instance._src_dims = src_dims
        instance._dst_dims = dst_dims
        instance._src_shape = src_shape
        instance._dst_shape = dst_shape
        instance._areas = areas
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
        """Build a regridder from explicit shapely polygon arrays.

        Use this for unstructured meshes (MPAS, ICON, finite-element), arbitrary
        polygon targets (countries, watersheds), or any combination the
        structured-grid path cannot express.

        Args:
            source_polygons: 1D array of shapely Polygons for source cells.
            target_polygons: 1D array of shapely Polygons for target cells.
            source_dim: Name of the single dim carrying source cells on the
                data passed to :meth:`regrid`. Default ``"cell"``.
            target_dim: Name of the single dim carrying target cells on the
                output. Default ``"cell"``.
            target_coords: Optional xr.Dataset providing coordinate variables
                along ``target_dim`` (and any auxiliary coords) to reattach on
                the output. If None, the output is given a bare integer index.
            periodic: Treat polygon x coordinates as longitudes on a 360-degree
                periodic axis, so polygons that cross the antimeridian are
                unwrapped before intersection.
            n_threads: Thread count for GEOS intersection.
            predicate_filter: If True (default), the STRtree candidate query
                filters by GEOS ``intersects``. Safe for arbitrary polygons
                including thin/diagonal shapes with loose bboxes. Set False
                when your polygons have tight bboxes (low aspect ratio,
                roughly axis-aligned) to skip the predicate and let the
                ``area > 0`` filter drop false positives — usually faster
                in that case, pathological otherwise.

        Returns:
            A ``ConservativeRegridder`` that accepts data with ``source_dim``
            in place of the structured grid's spatial dims.

        Intersection geometry is planar in the input polygons' coordinate
        space. If the polygons represent lat/lon cells, project them into an
        equal-area CRS first (or use the structured path with ``spherical=True``).
        """
        _check_shapely()
        src_polys = _as_1d_polygon_array(source_polygons, name="source_polygons")
        dst_polys = _as_1d_polygon_array(target_polygons, name="target_polygons")
        if periodic:
            src_polys = _normalize_periodic_polygons(src_polys)
            src_reference = _polygon_reference_x(src_polys)
            dst_polys = _normalize_periodic_polygons(
                dst_polys,
                reference=src_reference if np.isfinite(src_reference) else None,
            )

        src_grid = _Grid(
            polys=src_polys,
            bounds=shapely.bounds(src_polys),
            rectilinear=False,
        )
        dst_grid = _Grid(
            polys=dst_polys,
            bounds=shapely.bounds(dst_polys),
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
            areas=_build_intersection_areas(
                src_grid,
                dst_grid,
                n_threads=n_threads,
                predicate_filter=predicate_filter,
            ),
            source_coords=xr.Dataset(coords={source_dim: np.arange(n_src)}),
            target_coords=tgt_ds,
            src_dims=(source_dim,),
            dst_dims=(target_dim,),
            src_shape=(n_src,),
            dst_shape=(n_dst,),
            x_coord="",
            y_coord="",
            spherical=False,
        )


def polygons_from_coords(
    x: np.ndarray,
    y: np.ndarray,
    spherical: bool = False,
    periodic: bool = False,
) -> np.ndarray:
    """Build a 1D array of shapely cell polygons from 1D or 2D center coords.

    Convenience for mixing structured and unstructured regridding. E.g. build
    target polygons for a regular lat/lon grid, then pass them together with
    an unstructured source mesh to :meth:`ConservativeRegridder.from_polygons`.

    Args:
        x: 1D or 2D array of cell-center x coordinates.
        y: 1D or 2D array of cell-center y coordinates.
        spherical: If True, apply a cylindrical equal-area projection to 1D
            lat/lon (degrees) before building rectangles — matches the
            structured-grid ``spherical=True`` path.
        periodic: Treat x coordinates as longitudes on a 360-degree periodic
            axis, so cells that cross the antimeridian are unwrapped before
            polygon construction.

    Returns:
        A 1D numpy array of shapely Polygons in row-major (y, x) order.
    """
    _check_shapely()
    x = np.asarray(x)
    y = np.asarray(y)
    if periodic:
        x = _unwrap_longitude(x)
    if spherical:
        if x.ndim != 1 or y.ndim != 1:
            msg = "spherical=True requires 1D lat/lon arrays"
            raise ValueError(msg)
        return _build_cea_grid(x, y).polys
    return _build_grid(x, y).polys


def conservative_2d_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    x_coord: str = "longitude",
    y_coord: str = "latitude",
    spherical: bool = False,
    skipna: bool = True,
    nan_threshold: float = 1.0,
    n_threads: int | None = None,
) -> xr.DataArray | xr.Dataset:
    """Conservative regridding via explicit 2D polygon intersection.

    One-shot convenience wrapper: constructs a :class:`ConservativeRegridder`
    and applies it once. For repeated regridding to the same target, construct
    a ``ConservativeRegridder`` directly and reuse it.

    Args:
        data: Input data on the source grid.
        target_ds: Dataset defining the target grid; must expose ``x_coord`` and
            ``y_coord`` as (1D or 2D) coordinate variables.
        x_coord: Name of the x (longitude-like) coordinate variable, shared
            between source and target.
        y_coord: Name of the y (latitude-like) coordinate variable, shared
            between source and target.
        spherical: If True, assume ``x_coord``/``y_coord`` are longitude/latitude
            in degrees and project cells into Lambert cylindrical equal-area
            space before intersecting — gives correct spherical area weights
            at the same cost as the planar fast path. Rectilinear (1D coord)
            grids only.
        skipna: If True, propagate NaNs into the weighted mean via a two-pass
            sum (values and valid mask), matching the ``conservative`` method.
        nan_threshold: Keep output cells whose valid source fraction is at
            least ``nan_threshold``.
        n_threads: Thread count for parallel GEOS intersection.

    Returns:
        Regridded data on the target grid, preserving non-spatial dims.
    """
    regridder = ConservativeRegridder(
        data,
        target_ds,
        x_coord=x_coord,
        y_coord=y_coord,
        spherical=spherical,
        n_threads=n_threads,
    )
    return regridder.regrid(data, skipna=skipna, nan_threshold=nan_threshold)


def _apply_stored_weights(
    data: xr.DataArray | xr.Dataset,
    apply_weights: "sparse.COO | np.ndarray",
    coverage: np.ndarray,
    src_dims: tuple[Hashable, ...],
    dst_dims: tuple[Hashable, ...],
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
    target_coords: xr.Dataset,
    x_coord: str,
    y_coord: str,
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray | xr.Dataset:
    """Apply a cached, pre-transposed weight matrix to ``data`` via
    ``xr.apply_ufunc``.

    ``apply_weights`` has shape ``(n_src, n_dst)`` so the matmul is
    ``(..., n_src) @ (n_src, n_dst) → (..., n_dst)`` with no per-call transpose.
    """
    # apply_ufunc(dask="parallelized") needs each core dim to be a single
    # chunk. Only rechunk if data is already dask-backed — don't inadvertently
    # dask-ify a numpy-backed input.
    if getattr(data, "chunks", None) is not None:
        split = {
            d: -1
            for d in src_dims
            if d in data.dims and len(data.chunksizes.get(d, ())) > 1
        }
        if split:
            data = data.chunk(split)

    actual_src_shape = tuple(int(data.sizes[d]) for d in src_dims if d in data.sizes)
    if actual_src_shape != src_shape:
        msg = (
            f"source spatial shape {actual_src_shape} on dims {src_dims} does "
            f"not match the regridder's expected shape {src_shape}"
        )
        raise ValueError(msg)

    src_tokens = tuple(f"__src_{d}" for d in src_dims)
    data_renamed = data.rename(dict(zip(src_dims, src_tokens, strict=True)))

    output_dtype = _result_dtype(data)
    result = xr.apply_ufunc(
        _apply_core,
        data_renamed,
        kwargs={
            "apply_weights": apply_weights,
            "coverage": coverage,
            "coverage_all": bool(coverage.all()),
            "src_shape": src_shape,
            "dst_shape": dst_shape,
            "skipna": skipna,
            "nan_threshold": nan_threshold,
            "output_dtype": output_dtype,
        },
        input_core_dims=[list(src_tokens)],
        output_core_dims=[list(dst_dims)],
        exclude_dims=set(src_tokens),
        dask="parallelized",
        output_dtypes=[output_dtype],
        dask_gufunc_kwargs={
            "output_sizes": {d: int(target_coords.sizes[d]) for d in dst_dims},
            "allow_rechunk": False,
        },
        keep_attrs=True,
    )

    result = _assign_target_coords(result, target_coords, dst_dims, x_coord, y_coord)
    return result


def _resolve_chunks(size: int, chunk: int) -> tuple[int, ...]:
    """Split ``size`` into chunks of at most ``chunk`` elements (last may be
    smaller). ``chunk <= 0`` or ``chunk >= size`` returns a single chunk."""
    if chunk <= 0 or chunk >= size:
        return (size,)
    full, rem = divmod(size, chunk)
    return (chunk,) * full + ((rem,) if rem else ())


def _block_weights(
    apply_weights: Any, dst_flat: np.ndarray
) -> tuple[np.ndarray, Any]:
    """Given a (n_src, n_dst) weight matrix and a list of target flat
    indices, return (src_rows, sub_weights) where ``sub_weights`` has shape
    ``(len(src_rows), len(dst_flat))`` and ``src_rows`` are the unique source
    indices with any nonzero in the block."""
    sub_cols = apply_weights[:, dst_flat]
    if _HAS_SPARSE and isinstance(sub_cols, sparse.COO):
        src_rows = np.unique(sub_cols.coords[0]) if sub_cols.nnz else np.empty(0, dtype=np.int64)
    else:
        src_rows = np.where(np.any(sub_cols != 0, axis=1))[0]
    return src_rows, sub_cols[src_rows, :]


def _apply_blockwise(
    data: xr.DataArray | xr.Dataset,
    apply_weights: Any,
    coverage: np.ndarray,
    src_dims: tuple[Hashable, ...],
    dst_dims: tuple[Hashable, ...],
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
    target_coords: xr.Dataset,
    x_coord: str,
    y_coord: str,
    skipna: bool,
    nan_threshold: float,
    target_chunks: dict[Hashable, int],
) -> xr.DataArray | xr.Dataset:
    """Apply weights block-by-block across the target spatial dims.

    For each target block, identify the subset of source cells that
    contribute (via the sparse weight matrix), slice the input along those
    indices, and matmul with the block-local weight submatrix. Output is a
    dask-backed array concatenated along the target spatial dims; leading
    dims keep the input's chunking.
    """
    if len(dst_dims) != 2:
        msg = "regrid_blockwise currently supports only 2D target grids."
        raise NotImplementedError(msg)

    ny, nx = (int(target_coords.sizes[d]) for d in dst_dims)
    y_chunks = _resolve_chunks(ny, int(target_chunks.get(dst_dims[0], ny)))
    x_chunks = _resolve_chunks(nx, int(target_chunks.get(dst_dims[1], nx)))

    # Stack source spatial dims → one flat axis so per-block .isel on
    # arbitrary src indices is a single operation (and stays lazy on dask).
    src_tokens = tuple(f"__src_{d}" for d in src_dims)
    data_renamed = data.rename(dict(zip(src_dims, src_tokens, strict=True)))
    stacked = "__src_flat"
    data_flat = data_renamed.stack({stacked: src_tokens})

    output_dtype = _result_dtype(data)

    def _block_fn(
        arr: np.ndarray,
        sub_weights: Any,
        coverage_block: np.ndarray,
        block_shape: tuple[int, int],
    ) -> np.ndarray:
        return _apply_core(
            arr,
            apply_weights=sub_weights,
            coverage=coverage_block,
            coverage_all=bool(coverage_block.all()),
            src_shape=(arr.shape[-1],),
            dst_shape=block_shape,
            skipna=skipna,
            nan_threshold=nan_threshold,
            output_dtype=output_dtype,
        )

    row_arrays = []
    for by, y_chunk in enumerate(y_chunks):
        y_start = sum(y_chunks[:by])
        col_arrays = []
        for bx, x_chunk in enumerate(x_chunks):
            x_start = sum(x_chunks[:bx])
            yy, xx = np.mgrid[y_start:y_start + y_chunk, x_start:x_start + x_chunk]
            dst_flat = (yy * nx + xx).ravel()
            src_rows, sub_weights = _block_weights(apply_weights, dst_flat)
            coverage_block = coverage[dst_flat]
            block_shape = (y_chunk, x_chunk)

            if src_rows.size == 0:
                # No source cells contribute; whole block is NaN.
                sliced = data_flat.isel({stacked: slice(0, 1)})
            else:
                sliced = data_flat.isel({stacked: src_rows})

            block_da = xr.apply_ufunc(
                _block_fn,
                sliced,
                kwargs={
                    "sub_weights": sub_weights,
                    "coverage_block": coverage_block,
                    "block_shape": block_shape,
                },
                input_core_dims=[[stacked]],
                output_core_dims=[list(dst_dims)],
                exclude_dims={stacked},
                dask="parallelized",
                output_dtypes=[output_dtype],
                dask_gufunc_kwargs={
                    "output_sizes": dict(zip(dst_dims, block_shape, strict=True)),
                    "allow_rechunk": True,
                },
                keep_attrs=True,
            )
            col_arrays.append(block_da)
        row_arrays.append(xr.concat(col_arrays, dim=dst_dims[1]))

    result = xr.concat(row_arrays, dim=dst_dims[0])
    result = _assign_target_coords(result, target_coords, dst_dims, x_coord, y_coord)
    return result


def _coverage_mask(areas: "sparse.COO | np.ndarray") -> np.ndarray:
    """Return a boolean ``(n_dst,)`` mask of target cells with any source
    overlap, derived from the raw area matrix."""
    if _HAS_SPARSE and isinstance(areas, sparse.COO):
        # COO sparse: row_sum is 0 iff row has no nonzero entries.
        n_dst = int(areas.shape[0])
        mask = np.zeros(n_dst, dtype=bool)
        mask[areas.coords[0]] = True
        return mask
    arr = np.asarray(areas)
    return np.asarray((arr > 0).any(axis=1))


def _coo_components(
    w: "sparse.COO | np.ndarray",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    if _HAS_SPARSE and isinstance(w, sparse.COO):
        coords = np.asarray(w.coords)
        return (
            coords[0].astype(np.int64, copy=False),
            coords[1].astype(np.int64, copy=False),
            np.asarray(w.data),
            w.shape,
        )
    arr = np.asarray(w)
    rows, cols = np.nonzero(arr)
    return (
        rows.astype(np.int64, copy=False),
        cols.astype(np.int64, copy=False),
        arr[rows, cols],
        arr.shape,
    )


def _coo_from_components(
    row: np.ndarray,
    col: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
) -> "sparse.COO | np.ndarray":
    if _HAS_SPARSE:
        return sparse.COO(
            coords=np.stack([row, col]),
            data=data,
            shape=shape,
            has_duplicates=False,
            sorted=False,
        )
    dense = np.zeros(shape, dtype=data.dtype if data.size else np.float64)
    dense[row, col] = data
    return dense


def _metadata_attrs(regridder: ConservativeRegridder) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "x_coord": regridder.x_coord,
        "y_coord": regridder.y_coord,
        "spherical": int(regridder.spherical),
        "src_dims": [str(d) for d in regridder._src_dims],
        "dst_dims": [str(d) for d in regridder._dst_dims],
        "src_shape": list(regridder._src_shape),
        "dst_shape": list(regridder._dst_shape),
        "xarray_regrid_version": _package_version(),
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "schema_version": _SCHEMA_VERSION,
    }
    for name, value in {
        "source_x_range": _coord_range(regridder._source_coords, regridder.x_coord),
        "source_y_range": _coord_range(regridder._source_coords, regridder.y_coord),
        "target_x_range": _coord_range(regridder._target_coords, regridder.x_coord),
        "target_y_range": _coord_range(regridder._target_coords, regridder.y_coord),
    }.items():
        if value is not None:
            attrs[name] = list(value)
    return attrs


def _metadata_from_attrs(attrs: dict[str, Any], path: Path) -> dict[str, Any]:
    """Parse the kwargs needed by :meth:`ConservativeRegridder._from_state` out
    of netCDF root attributes, validating ``schema_version``."""
    schema_version = int(attrs.get("schema_version", 0))
    if schema_version != _SCHEMA_VERSION:
        msg = (
            f"regridder file at {path} uses schema version {schema_version}; "
            f"this xarray-regrid understands {_SCHEMA_VERSION}. "
            "Upgrade xarray-regrid or re-save."
        )
        raise ValueError(msg)

    return {
        "x_coord": str(attrs["x_coord"]),
        "y_coord": str(attrs["y_coord"]),
        "spherical": bool(int(attrs["spherical"])),
        "src_dims": tuple(str(d) for d in np.atleast_1d(attrs["src_dims"])),
        "dst_dims": tuple(str(d) for d in np.atleast_1d(attrs["dst_dims"])),
        "src_shape": tuple(int(s) for s in np.atleast_1d(attrs["src_shape"])),
        "dst_shape": tuple(int(s) for s in np.atleast_1d(attrs["dst_shape"])),
    }


def _normalize_longitude_coords(
    source: xr.DataArray | xr.Dataset,
    target: xr.Dataset,
    x_coord: str,
) -> tuple[xr.DataArray | xr.Dataset, xr.Dataset]:
    """Unwrap x coordinates across the antimeridian so source and target share
    a contiguous longitude frame. No-op when the coord isn't present on both
    objects or doesn't look like a longitude."""
    if x_coord not in source.coords or x_coord not in target.coords:
        return source, target

    source_x = np.asarray(source[x_coord].values)
    target_x = np.asarray(target[x_coord].values)
    if not _looks_like_longitude(source_x) and not _looks_like_longitude(target_x):
        return source, target

    source_x = _unwrap_longitude(source_x)
    target_x = _align_longitude(_unwrap_longitude(target_x), source_x)
    return (
        utils.update_coord(source, x_coord, source_x),
        cast(xr.Dataset, utils.update_coord(target, x_coord, target_x)),
    )


def _looks_like_longitude(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    return bool(finite.min() >= -360.0 and finite.max() <= 360.0)


def _unwrap_longitude(values: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(np.asarray(values, dtype=float))
    for axis in range(radians.ndim):
        radians = np.unwrap(radians, axis=axis)
    return np.rad2deg(radians)


def _align_longitude(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if values.size == 0 or reference.size == 0:
        return values
    reference_center = _finite_mean(reference)
    values_center = _finite_mean(values)
    if reference_center is None or values_center is None:
        return values
    offset = _periodic_offset(reference_center, values_center)
    return values + offset


def _normalize_periodic_polygons(
    polygons: np.ndarray, reference: float | None = None
) -> np.ndarray:
    normalized = []
    current_reference = reference
    for polygon in polygons:
        new_polygon = _unwrap_polygon(polygon)
        center = _polygon_center_x(new_polygon)
        if current_reference is None and np.isfinite(center):
            current_reference = center
        if current_reference is None:
            normalized.append(new_polygon)
            continue
        offset = _periodic_offset(current_reference, center)
        if offset != 0.0:
            new_polygon = affinity.translate(new_polygon, xoff=offset)
        normalized.append(new_polygon)
    return np.array(normalized, dtype=object)


def _polygon_reference_x(polygons: np.ndarray) -> float:
    bounds = shapely.bounds(polygons)
    centers = 0.5 * (bounds[:, 0] + bounds[:, 2])
    center = _finite_mean(centers)
    return float("nan") if center is None else center


def _polygon_center_x(polygon: Any) -> float:
    minx, _, maxx, _ = polygon.bounds
    return 0.5 * (float(minx) + float(maxx))


def _periodic_offset(reference: float, value: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(value):
        return 0.0
    return 360.0 * round((reference - value) / 360.0)


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _as_1d_polygon_array(polygons: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(polygons)
    if arr.ndim != 1:
        msg = f"{name} must be a 1D array of shapely Polygons"
        raise ValueError(msg)
    return arr


def _unwrap_polygon(polygon: Any) -> Any:
    if polygon.is_empty:
        return polygon
    if polygon.geom_type == "Polygon":
        exterior = _unwrap_ring(np.asarray(polygon.exterior.coords))
        holes = [_unwrap_ring(np.asarray(ring.coords)) for ring in polygon.interiors]
        return shapely.Polygon(exterior, holes)
    if polygon.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([_unwrap_polygon(part) for part in polygon.geoms])
    return polygon


def _unwrap_ring(ring: np.ndarray) -> np.ndarray:
    new_ring = np.asarray(ring, dtype=float).copy()
    offset = 0.0
    for i in range(1, new_ring.shape[0]):
        x = new_ring[i, 0] + offset
        step = x - new_ring[i - 1, 0]
        if step > 180.0:
            offset -= 360.0
        elif step < -180.0:
            offset += 360.0
        new_ring[i, 0] += offset
    return new_ring


def _coord_range(ds: xr.Dataset, coord_name: str) -> tuple[float, float] | None:
    """Return ``(min, max)`` of a coord, or ``None`` when it isn't on the
    Dataset (e.g., the integer-index stub emitted by ``from_polygons``)."""
    if not coord_name or coord_name not in ds.coords:
        return None
    arr = np.asarray(ds[coord_name].values)
    if arr.size == 0:
        return None
    return float(arr.min()), float(arr.max())


def _transpose_weights(
    w: "sparse.COO | np.ndarray", *, sort: bool = False
) -> "sparse.COO | np.ndarray":
    """Materialize a transposed weight matrix.

    `sparse.COO.T` is a lazy view that re-sorts indices on each downstream
    matmul, so callers relying on ``.coords[0]`` being row indices or wanting
    a hot matmul path should materialize once here. Pass ``sort=True`` to
    additionally trigger the sort ahead of time (used for the apply matrix).
    """
    if _HAS_SPARSE and isinstance(w, sparse.COO):
        t = w.T
        out = sparse.COO(
            coords=np.asarray(t.coords),
            data=np.asarray(t.data),
            shape=t.shape,
            has_duplicates=False,
            sorted=False,
        )
        if sort:
            out._sort_indices()
        return out
    return np.asarray(w).T.copy()


def _spatial_dims(
    obj: xr.DataArray | xr.Dataset, x_coord: str, y_coord: str
) -> tuple[Hashable, ...]:
    """Return the spatial dim order to feed to apply_ufunc.

    For 1D rectilinear coords (x and y ride on separate dims), canonicalize to
    ``(y_dim, x_dim)`` so the flattened cell order matches the polygon order
    emitted by the fast-path in ``_build_grid``. For curvilinear (2D) coords,
    preserve the dim order already present on ``obj``.
    """
    if x_coord not in obj.coords or y_coord not in obj.coords:
        return ()
    xd = obj[x_coord].dims
    yd = obj[y_coord].dims
    if len(xd) == 1 and len(yd) == 1 and xd[0] != yd[0]:
        return (yd[0], xd[0])
    dims = set(xd) | set(yd)
    return tuple(d for d in obj.dims if d in dims)


def _grid_from_coords(
    obj: xr.DataArray | xr.Dataset,
    x_coord: str,
    y_coord: str,
    dims: tuple[Hashable, ...],
    spherical: bool = False,
) -> "_Grid":
    """Build a :class:`_Grid` from the object's x/y coordinates.

    If both coords are 1D and ride on separate dims, pass them as separate 1D
    vectors to trigger the rectilinear fast path. Otherwise broadcast to a
    common N-D array in ``dims`` order for the curvilinear path.

    If ``spherical`` is True, coordinates are assumed to be longitude (x) and
    latitude (y) in degrees, and cells are projected into a Lambert cylindrical
    equal-area space (x' = lon_rad, y' = sin(lat_rad)) before constructing the
    cell polygons. This gives mass-conservative weights on the sphere at the
    same cost as the planar fast path. Only supported for rectilinear (1D
    coord) grids in this version.
    """
    xd = obj[x_coord]
    yd = obj[y_coord]
    is_rectilinear = xd.ndim == 1 and yd.ndim == 1 and xd.dims[0] != yd.dims[0]
    if spherical:
        if not is_rectilinear:
            msg = (
                "spherical=True is only supported for rectilinear (1D lat/lon) "
                "coordinate arrays in this version."
            )
            raise NotImplementedError(msg)
        return _build_cea_grid(np.asarray(xd.values), np.asarray(yd.values))
    if is_rectilinear:
        return _build_grid(np.asarray(xd.values), np.asarray(yd.values))
    xc, yc = xr.broadcast(obj[x_coord], obj[y_coord])
    xc = xc.transpose(*dims)
    yc = yc.transpose(*dims)
    return _build_grid(np.asarray(xc.values), np.asarray(yc.values))


def _build_cea_grid(lon_centers: np.ndarray, lat_centers: np.ndarray) -> "_Grid":
    """Build a rectilinear :class:`_Grid` whose cell polygons are in Lambert
    cylindrical equal-area coordinates (x' = lon_rad, y' = sin(lat_rad)).

    Projecting *edges* analytically — rather than projecting centers and then
    re-midpointing — is required because ``sin()`` is nonlinear: the projected
    midpoint of two lat centers is not the same as the midpoint of two
    projected lat edges.
    """
    _check_shapely()
    if lon_centers.size < 2 or lat_centers.size < 2:
        msg = "spherical mode requires at least two cells per dimension"
        raise ValueError(msg)
    lat_edges_deg = np.clip(utils.infer_1d_edges(lat_centers), -90.0, 90.0)
    lon_edges_deg = utils.infer_1d_edges(lon_centers)
    return _rect_grid_from_edges(
        np.deg2rad(lon_edges_deg),
        np.sin(np.deg2rad(lat_edges_deg)),
    )


def _infer_2d_corners(xc: np.ndarray, yc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ny+1, nx+1) cell-corner arrays from 2D cell-center arrays.

    Interior corners are the mean of the four surrounding centers; boundary
    corners are reflected from the adjacent interior row/column.
    """
    if xc.shape != yc.shape or xc.ndim != 2:
        msg = "xc and yc must be 2D arrays of the same shape"
        raise ValueError(msg)

    def corners(a: np.ndarray) -> np.ndarray:
        ny, nx = a.shape
        pad = np.empty((ny + 2, nx + 2), dtype=a.dtype)
        pad[1:-1, 1:-1] = a
        pad[0, 1:-1] = 2 * a[0, :] - a[1, :]
        pad[-1, 1:-1] = 2 * a[-1, :] - a[-2, :]
        pad[1:-1, 0] = 2 * a[:, 0] - a[:, 1]
        pad[1:-1, -1] = 2 * a[:, -1] - a[:, -2]
        pad[0, 0] = 2 * pad[0, 1] - pad[0, 2]
        pad[0, -1] = 2 * pad[0, -2] - pad[0, -3]
        pad[-1, 0] = 2 * pad[-1, 1] - pad[-1, 2]
        pad[-1, -1] = 2 * pad[-1, -2] - pad[-1, -3]
        return 0.25 * (pad[:-1, :-1] + pad[1:, :-1] + pad[:-1, 1:] + pad[1:, 1:])

    return corners(xc.astype(float)), corners(yc.astype(float))


@dataclass
class _Grid:
    """Cached cell geometry for a structured grid.

    ``polys`` is a flat (n_cells,) object array of shapely Polygons.
    ``bounds`` is a (n_cells, 4) ``(minx, miny, maxx, maxy)`` array cached for
    the STRtree / candidate-search path. ``rectilinear`` is True when both the
    source x and y were 1D coordinate arrays (axis-aligned rectangles) — the
    weight builder uses this to skip GEOS polygon clipping and compute
    intersection areas analytically from the bounds.
    """

    polys: np.ndarray
    bounds: np.ndarray
    rectilinear: bool


def _rect_grid_from_edges(xe: np.ndarray, ye: np.ndarray) -> _Grid:
    """Build a rectilinear :class:`_Grid` from already-prepared edge arrays.

    Shared by the raw-planar (:func:`_build_grid` 1D branch) and the analytic
    equal-area (:func:`_build_cea_grid`) paths.
    """
    x0, y0 = np.meshgrid(xe[:-1], ye[:-1], indexing="xy")
    x1, y1 = np.meshgrid(xe[1:], ye[1:], indexing="xy")
    x0f, y0f, x1f, y1f = x0.ravel(), y0.ravel(), x1.ravel(), y1.ravel()
    polys = shapely.box(x0f, y0f, x1f, y1f)
    bounds = np.stack([x0f, y0f, x1f, y1f], axis=1)
    return _Grid(polys=polys, bounds=bounds, rectilinear=True)


def _build_grid(xc: np.ndarray, yc: np.ndarray) -> _Grid:
    """Return a _Grid of cell geometry for a structured grid.

    Accepts 1D (rectilinear, separate x and y vectors) or 2D (curvilinear,
    co-shaped center arrays) inputs. Output order is row-major in the input dim
    order: for 2D inputs of shape (ny, nx) the polygons correspond to cells
    reshaped as ``(ny, nx)``.
    """
    _check_shapely()
    if xc.ndim == 1 and yc.ndim == 1:
        xe = utils.infer_1d_edges(xc.astype(float))
        ye = utils.infer_1d_edges(yc.astype(float))
        return _rect_grid_from_edges(xe, ye)
    if xc.ndim == 2 and yc.ndim == 2:
        xcorn, ycorn = _infer_2d_corners(xc, yc)
        ny, nx = xc.shape
        c00 = np.stack([xcorn[:-1, :-1], ycorn[:-1, :-1]], axis=-1)
        c10 = np.stack([xcorn[:-1, 1:], ycorn[:-1, 1:]], axis=-1)
        c11 = np.stack([xcorn[1:, 1:], ycorn[1:, 1:]], axis=-1)
        c01 = np.stack([xcorn[1:, :-1], ycorn[1:, :-1]], axis=-1)
        rings = np.stack([c00, c10, c11, c01, c00], axis=2).reshape(ny * nx, 5, 2)
        polys = shapely.polygons(rings)
        return _Grid(polys=polys, bounds=shapely.bounds(polys), rectilinear=False)
    msg = "x and y coordinate arrays must both be 1D or both 2D"
    raise ValueError(msg)


def _build_intersection_areas(
    src: _Grid,
    dst: _Grid,
    n_threads: int | None = None,
    *,
    predicate_filter: bool = False,
) -> "sparse.COO | np.ndarray":
    """Build the (n_dst, n_src) raw area-intersection matrix ``A[i, j] =
    area(dst_i ∩ src_j)``.

    This is the unnormalized matrix. Row-normalize via :func:`_row_normalize`
    to get forward weights; transpose first for backward (target → source).

    When both grids are rectilinear (axis-aligned rectangles) intersection
    areas are computed analytically from the bounds, skipping GEOS clipping.

    ``predicate_filter=False`` (default) uses a bbox-only STRtree query and
    relies on the ``area > 0`` filter below to drop bbox-false-positives.
    For structured cells whose bboxes are tight (quadrilaterals) this is a
    large win — the GEOS ``intersects`` predicate inside STRtree is much
    more expensive than the extra no-op intersections it avoids. Set
    ``predicate_filter=True`` for user-supplied polygons with loose bboxes
    (long, thin, diagonal shapes) where the predicate pays for itself.
    """
    _check_shapely()
    n_dst = len(dst.polys)
    n_src = len(src.polys)

    tree = STRtree(src.polys)
    if predicate_filter:
        pairs = tree.query(dst.polys, predicate="intersects")
    else:
        pairs = tree.query(dst.polys)
    dst_idx = np.asarray(pairs[0])
    src_idx = np.asarray(pairs[1])

    if dst_idx.size == 0:
        return _empty_weights(n_dst, n_src)

    if src.rectilinear and dst.rectilinear:
        sb = src.bounds[src_idx]
        db = dst.bounds[dst_idx]
        dx = np.minimum(sb[:, 2], db[:, 2]) - np.maximum(sb[:, 0], db[:, 0])
        dy = np.minimum(sb[:, 3], db[:, 3]) - np.maximum(sb[:, 1], db[:, 1])
        areas = np.maximum(dx, 0.0) * np.maximum(dy, 0.0)
    else:
        areas = _intersection_areas_threaded(
            dst.polys[dst_idx], src.polys[src_idx], n_threads=n_threads
        )

    keep = areas > 0
    dst_idx = dst_idx[keep]
    src_idx = src_idx[keep]
    areas = areas[keep]

    if dst_idx.size == 0:
        return _empty_weights(n_dst, n_src)

    if _HAS_SPARSE:
        return sparse.COO(
            coords=np.stack([dst_idx, src_idx]),
            data=areas.astype(np.float64),
            shape=(n_dst, n_src),
            has_duplicates=False,
            sorted=False,
        )
    a_dense = np.zeros((n_dst, n_src), dtype=np.float64)
    a_dense[dst_idx, src_idx] = areas
    return a_dense


def _row_normalize(
    areas: "sparse.COO | np.ndarray",
) -> "sparse.COO | np.ndarray":
    """Normalize rows of an area matrix so each row sums to 1 (rows with no
    overlap stay all-zero, which produces NaN output under the apply path)."""
    if _HAS_SPARSE and isinstance(areas, sparse.COO):
        n_dst = areas.shape[0]
        dst_idx = areas.coords[0]
        src_idx = areas.coords[1]
        data = areas.data
        if data.size == 0:
            return areas
        row_sum = np.bincount(dst_idx, weights=data, minlength=n_dst)
        new_data = data / row_sum[dst_idx]
        return sparse.COO(
            coords=np.stack([dst_idx, src_idx]),
            data=new_data,
            shape=areas.shape,
            has_duplicates=False,
            sorted=False,
        )
    row_sum = areas.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return areas / row_sum


def _intersection_areas_threaded(
    a: np.ndarray, b: np.ndarray, n_threads: int | None
) -> np.ndarray:
    """Compute per-pair intersection areas ``area(a[i] & b[i])`` over numpy
    arrays of shapely geometries, optionally parallelized via threads.

    Shapely 2.x releases the GIL for GEOS ops, so a ``ThreadPoolExecutor``
    gives near-linear speedup on multi-core machines without pickling data.
    """
    _check_shapely()
    n = len(a)
    if n_threads is None:
        # Below ~1k pairs the pool spin-up (~0.3 ms) dominates sub-ms work.
        # Above that, scaling is near-linear with logical cores — shapely
        # releases the GIL inside its GEOS ufuncs. Cap at 16 to avoid
        # oversubscription on unusually wide machines.
        n_threads = 1 if n < 1_000 else min(os.cpu_count() or 1, 16)
    if n_threads <= 1 or n == 0:
        return shapely.area(shapely.intersection(a, b))

    splits = np.array_split(np.arange(n), n_threads)

    def _work(idx: np.ndarray) -> np.ndarray:
        return shapely.area(shapely.intersection(a[idx], b[idx]))

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        parts = list(pool.map(_work, splits))
    return np.concatenate(parts)


def _empty_weights(n_dst: int, n_src: int) -> "sparse.COO | np.ndarray":
    if _HAS_SPARSE:
        return sparse.COO(
            coords=np.zeros((2, 0), dtype=np.int64),
            data=np.zeros(0, dtype=np.float64),
            shape=(n_dst, n_src),
        )
    return np.zeros((n_dst, n_src), dtype=np.float64)


def _apply_core(
    arr: np.ndarray,
    apply_weights: Any,
    coverage: np.ndarray,
    coverage_all: bool,
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
    skipna: bool,
    nan_threshold: float,
    output_dtype: np.dtype,
) -> np.ndarray:
    """Apply a pre-transposed weight matrix along the trailing spatial dims.

    ``arr`` has shape ``(..., *src_shape)``; ``apply_weights`` has shape
    ``(n_src, n_dst)`` — returns ``(..., *dst_shape)``.

    ``coverage`` is a boolean ``(n_dst,)`` mask: target cells with any source
    overlap. ``coverage_all`` is precomputed so every block skips the
    ``coverage.all()`` scan. Uncovered cells are always masked to NaN in the
    output, regardless of ``skipna`` — domain boundaries and polygon holes
    produce NaN (matches the axis-factored ``conservative`` method).
    """
    n_spatial = len(src_shape)
    leading_shape = arr.shape[:-n_spatial] if n_spatial > 0 else arr.shape
    n_src = int(np.prod(src_shape))
    flat = arr.reshape(-1, n_src) if leading_shape else arr.reshape(1, n_src)

    if skipna and np.issubdtype(flat.dtype, np.floating):
        nan_mask = np.isnan(flat)
        has_nan = nan_mask.any()
    else:
        has_nan = False

    if has_nan:
        mask = (~nan_mask).astype(flat.dtype)
        filled = np.where(nan_mask, flat.dtype.type(0.0), flat)
        numerator = _matmul_last(filled, apply_weights)
        fraction = _matmul_last(mask, apply_weights)
        threshold = 1.0 - min(max(nan_threshold, 1e-6), 1.0 - 1e-6)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = numerator / fraction
        result = np.where(fraction >= threshold, result, np.nan)
    else:
        result = _matmul_last(flat, apply_weights)
        if not coverage_all:
            result = np.where(coverage[np.newaxis, :], result, np.nan)

    # sparse.matmul promotes to float64 regardless of the input dtype — cast
    # back to the requested output dtype so float32-in really produces
    # float32-out (halves memory for float32 pipelines).
    if result.dtype != output_dtype:
        result = result.astype(output_dtype, copy=False)

    out_shape = (*leading_shape, *dst_shape) if leading_shape else dst_shape
    return result.reshape(out_shape)


def _matmul_last(flat: np.ndarray, apply_weights: Any) -> np.ndarray:
    """Compute ``flat @ apply_weights`` where ``flat`` is dense ``(N, n_src)``
    and ``apply_weights`` is ``(n_src, n_dst)`` (dense or pre-sorted sparse)."""
    if _HAS_SPARSE and isinstance(apply_weights, sparse.COO):
        return np.asarray(sparse.matmul(flat, apply_weights))
    return flat @ apply_weights


def _result_dtype(obj: xr.DataArray | xr.Dataset) -> np.dtype:
    if isinstance(obj, xr.DataArray):
        return np.result_type(np.float32, obj.dtype)
    dtypes = [v.dtype for v in obj.data_vars.values()]
    if not dtypes:
        return np.dtype(np.float64)
    return np.result_type(np.float32, *dtypes)


def _assign_target_coords(
    obj: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    dst_dims: tuple[Hashable, ...],
    x_coord: str,
    y_coord: str,
) -> xr.DataArray | xr.Dataset:
    """Attach target coordinates that name a spatial axis or live on the
    output spatial dims. Scalar coords (``dims == ()``) on the target also
    ride along, since they represent per-regrid metadata (e.g. a pinned
    time stamp on the target template)."""
    dst_dim_set = set(dst_dims)
    new_coords: dict[Hashable, Any] = {}
    for name, coord in target_ds.coords.items():
        if name in (x_coord, y_coord) or set(coord.dims).issubset(dst_dim_set):
            new_coords[name] = coord
    if new_coords:
        obj = obj.assign_coords(new_coords)
    return obj
