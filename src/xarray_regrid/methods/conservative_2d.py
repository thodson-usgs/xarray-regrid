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
from xarray_regrid.methods.conservative import get_valid_threshold

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

    def __init__(self, areas: "sparse.COO | np.ndarray") -> None:
        self.areas = areas

    @cached_property
    def weights(self) -> "sparse.COO | np.ndarray":
        return _row_normalize(self.areas)

    @cached_property
    def apply_matrix(self) -> "sparse.COO | np.ndarray":
        return _transpose_weights(self.weights, sort=True)

    @cached_property
    def coverage(self) -> np.ndarray:
        return _coverage_mask(self.areas)

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
        source_grid, target_grid, src_x_sort_idx = _normalize_longitude_coords(
            source, target, x_coord
        )
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
        areas = _build_intersection_areas(src_grid, dst_grid, n_threads=n_threads)
        if src_x_sort_idx is not None:
            # Source x was sorted by _normalize_longitude_coords so polygon
            # construction stayed monotone. Relabel matrix columns so column
            # order matches the user's original (unsorted) data layout.
            x_dim_index = src_dims.index(source[x_coord].dims[0])
            areas = _remap_columns_for_axis_sort(
                areas, src_x_sort_idx, self._src_shape, x_dim_index
            )
        self._areas = areas
        self._source_coords = source.coords.to_dataset()
        self._target_coords = target.coords.to_dataset()

    @cached_property
    def _forward(self) -> _Direction:
        return _Direction(self._areas)

    @cached_property
    def _backward(self) -> _Direction:
        return _Direction(_transpose_weights(self._areas))

    @property
    def forward_weights(self) -> "sparse.COO | np.ndarray":
        """The row-normalized forward weight matrix (source → target)."""
        return self._forward.weights

    @property
    def backward_weights(self) -> "sparse.COO | np.ndarray":
        """The row-normalized backward weight matrix (target → source)."""
        return self._backward.weights

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

    def transpose(self) -> "ConservativeRegridder":
        """Return the backward regridder (target → source). The original's
        forward Direction becomes the new's backward (and vice versa), so any
        already-computed weight matrices are reused, not recomputed."""
        new = type(self)._from_state(
            areas=_transpose_weights(self._areas),
            source_coords=self._target_coords,
            target_coords=self._source_coords,
            src_dims=self._dst_dims,
            dst_dims=self._src_dims,
            src_shape=self._dst_shape,
            dst_shape=self._src_shape,
            x_coord=self.x_coord,
            y_coord=self.y_coord,
            spherical=self.spherical,
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
        nnz = getattr(self._areas, "nnz", None)
        shape = getattr(self._areas, "shape", (None, None))
        nnz_str = f"nnz={nnz}" if nnz is not None else "dense"
        return (
            f"ConservativeRegridder(src_dims={self._src_dims}, "
            f"dst_dims={self._dst_dims}, {shape[0]}x{shape[1]}, {nnz_str})"
        )

    def to_netcdf(self, path: str | Path, engine: NetcdfEngine = None) -> None:
        """Save the weight matrix and reproducibility metadata to a netCDF file.
        Requires a group-aware engine (``netcdf4`` or ``h5netcdf``); ``engine``
        is forwarded to :func:`xarray.Dataset.to_netcdf`."""
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
        _check_shapely()
        src_polys = np.asarray(source_polygons)
        dst_polys = np.asarray(target_polygons)
        if src_polys.ndim != 1 or dst_polys.ndim != 1:
            msg = "source_polygons and target_polygons must be 1D arrays"
            raise ValueError(msg)
        if periodic:
            src_polys = _normalize_periodic_polygons(src_polys)
            dst_polys = _normalize_periodic_polygons(
                dst_polys, reference=_polygon_reference_x(src_polys)
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
    """Build a 1D row-major (y, x) array of shapely cell polygons from 1D or
    2D center coords. Convenience for mixing structured and unstructured paths
    via :meth:`ConservativeRegridder.from_polygons`. ``spherical=True``
    projects 1D lat/lon (degrees) into Lambert cylindrical equal-area space;
    ``periodic=True`` unwraps antimeridian-crossing cells."""
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


def _apply_stored_weights(
    data: xr.DataArray | xr.Dataset,
    direction: _Direction,
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
    """Apply ``direction``'s cached, pre-transposed weight matrix to ``data``
    via ``xr.apply_ufunc``.

    The apply matrix has shape ``(n_src, n_dst)`` so the matmul is
    ``(..., n_src) @ (n_src, n_dst) → (..., n_dst)`` with no per-call transpose.
    """
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
            "apply_weights": direction.apply_matrix,
            "coverage": direction.coverage,
            "coverage_all": direction.coverage_all,
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
            "allow_rechunk": True,
        },
        keep_attrs=True,
    )

    return _assign_target_coords(result, target_coords, dst_dims, x_coord, y_coord)


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
    for prefix, ds, coord in [
        ("source_x", regridder._source_coords, regridder.x_coord),
        ("source_y", regridder._source_coords, regridder.y_coord),
        ("target_x", regridder._target_coords, regridder.x_coord),
        ("target_y", regridder._target_coords, regridder.y_coord),
    ]:
        if coord and coord in ds.coords and ds[coord].size:
            attrs[f"{prefix}_range"] = [float(ds[coord].min()), float(ds[coord].max())]
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
) -> tuple[xr.DataArray | xr.Dataset, xr.Dataset, np.ndarray | None]:
    """Unwrap x coordinates across the antimeridian so source and target share
    a contiguous longitude frame. No-op when the coord isn't present on both
    objects or doesn't look like a longitude.

    For 1D rectilinear longitudes on both sides this mirrors the per-value
    wrap done by :func:`xarray_regrid.utils.format_lon` for the axis-factored
    path, which is what makes a source on ``[0, 360]`` align with a target on
    ``[-180, 180]`` (and vice versa). If wrapping breaks source monotonicity
    the source coord is sorted in place; the caller is expected to remap the
    area-matrix columns by the returned ``src_x_sort_idx`` so the final matrix
    columns line up with the user's original data layout. For 2D / curvilinear
    coords the existing uniform-shift fallback is kept.
    """
    if x_coord not in source.coords or x_coord not in target.coords:
        return source, target, None

    source_x = np.asarray(source[x_coord].values)
    target_x = np.asarray(target[x_coord].values)
    if not _looks_like_longitude(source_x) and not _looks_like_longitude(target_x):
        return source, target, None

    source_x = _unwrap_longitude(source_x)
    target_x = _unwrap_longitude(target_x)
    src_finite = source_x[np.isfinite(source_x)]
    tgt_finite = target_x[np.isfinite(target_x)]

    src_x_sort_idx: np.ndarray | None = None
    if (
        source_x.ndim == 1
        and target_x.ndim == 1
        and src_finite.size
        and tgt_finite.size
    ):
        # Per-value wrap source into target's 360° window — mirrors format_lon
        # so source [0, 360] vs target [-180, 180] (and either reversed)
        # aligns. A uniform offset can't reconcile cross-convention grids:
        # mean diff is exactly 180° and round() is banker's-rounded to 0.
        wrap_point = float((tgt_finite[0] + tgt_finite[-1] + 360.0) / 2.0)
        source_x = np.where(
            source_x < wrap_point - 360.0, source_x + 360.0, source_x
        )
        source_x = np.where(source_x > wrap_point, source_x - 360.0, source_x)
        diffs = np.diff(source_x)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            src_x_sort_idx = np.argsort(source_x, kind="stable")
            source_x = source_x[src_x_sort_idx]
    elif src_finite.size and tgt_finite.size:
        # 2D / curvilinear: fall back to uniform shift of target into source's
        # window. Doesn't handle cross-convention but preserves the existing
        # antimeridian-crossing behavior for 2D coords.
        target_x = target_x + _periodic_offset(
            float(src_finite.mean()), float(tgt_finite.mean())
        )
    return (
        utils.update_coord(source, x_coord, source_x),
        cast(xr.Dataset, utils.update_coord(target, x_coord, target_x)),
        src_x_sort_idx,
    )


def _looks_like_longitude(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size and finite.min() >= -360.0 and finite.max() <= 360.0)


def _unwrap_longitude(values: np.ndarray) -> np.ndarray:
    """Unwrap longitudes along the trailing axis (CF convention). For 2D
    coords, latitude doesn't wrap mod 360°, so unwrapping it is incorrect."""
    radians = np.deg2rad(np.asarray(values, dtype=float))
    return np.rad2deg(np.unwrap(radians, axis=-1))


def _normalize_periodic_polygons(
    polygons: np.ndarray, reference: float | None = None
) -> np.ndarray:
    """Unwrap each polygon across the antimeridian, then shift each into the
    same 360-degree window as ``reference`` (or as the first finite center if
    ``reference`` is None)."""
    unwrapped = [_unwrap_polygon(p) for p in polygons]
    if reference is None:
        for poly in unwrapped:
            center = _polygon_center_x(poly)
            if np.isfinite(center):
                reference = center
                break
    if reference is None:
        return np.array(unwrapped, dtype=object)

    out = []
    for poly in unwrapped:
        offset = _periodic_offset(reference, _polygon_center_x(poly))
        out.append(affinity.translate(poly, xoff=offset) if offset != 0.0 else poly)
    return np.array(out, dtype=object)


def _polygon_reference_x(polygons: np.ndarray) -> float | None:
    """Mean polygon-center x across an array of polygons, or None if all
    polygons have non-finite centers."""
    bounds = shapely.bounds(polygons)
    centers = 0.5 * (bounds[:, 0] + bounds[:, 2])
    finite = centers[np.isfinite(centers)]
    return float(finite.mean()) if finite.size else None


def _polygon_center_x(polygon: Any) -> float:
    minx, _, maxx, _ = polygon.bounds
    return 0.5 * (float(minx) + float(maxx))


def _periodic_offset(reference: float, value: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(value):
        return 0.0
    return 360.0 * round((reference - value) / 360.0)


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


def _remap_columns_for_axis_sort(
    areas: "sparse.COO | np.ndarray",
    sort_idx: np.ndarray,
    src_shape: tuple[int, ...],
    axis_index: int,
) -> "sparse.COO | np.ndarray":
    """Relabel the column indices of an ``(n_dst, prod(src_shape))`` area
    matrix so that columns appear in the user's original source-data order
    after ``sort_idx`` was applied along ``axis_index`` of ``src_shape``.

    A row-major source flat index ``c = unravel(c, src_shape)`` has its
    ``axis_index`` component ``i_sorted`` permuted via
    ``i_orig = sort_idx[i_sorted]``; all other components are untouched. So
    the new flat index is ``c_new = c // stride * stride + (i_orig - i_sorted) * inner + ...``.
    For separable 1D-rect grids (the only case that triggers this today) the
    sorted axis sits between an outer block of size ``outer`` and an inner
    block of size ``inner`` with ``stride = nx * inner``.
    """
    nx = int(src_shape[axis_index])
    inner = int(np.prod(src_shape[axis_index + 1 :]))
    stride = nx * inner
    sort_idx = np.asarray(sort_idx, dtype=np.int64)

    if _HAS_SPARSE and isinstance(areas, sparse.COO):
        old_col = np.asarray(areas.coords[1], dtype=np.int64)
        outer_block = (old_col // stride) * stride
        within = old_col % stride
        i_sorted = within // inner
        rest = within % inner
        new_col = outer_block + sort_idx[i_sorted] * inner + rest
        coords = np.stack([np.asarray(areas.coords[0], dtype=np.int64), new_col])
        return sparse.COO(
            coords=coords,
            data=np.asarray(areas.data),
            shape=areas.shape,
            has_duplicates=False,
            sorted=False,
        )

    arr = np.asarray(areas)
    n_cells = arr.shape[1]
    inv_sort_idx = np.empty_like(sort_idx)
    inv_sort_idx[sort_idx] = np.arange(sort_idx.size, dtype=sort_idx.dtype)
    cells = np.arange(n_cells, dtype=np.int64)
    outer_block = (cells // stride) * stride
    within = cells % stride
    i_orig = within // inner
    rest = within % inner
    inv_perm = outer_block + inv_sort_idx[i_orig] * inner + rest
    return arr[:, inv_perm]


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

    Rectilinear (both coords 1D on separate dims) takes the fast path.
    Curvilinear coords are broadcast to a common N-D array in ``dims`` order.

    If ``spherical`` is True, coordinates are assumed to be longitude (x) and
    latitude (y) in degrees, and cells are projected into a Lambert cylindrical
    equal-area space (x' = lon_rad, y' = sin(lat_rad)) before constructing the
    cell polygons. This gives mass-conservative weights on the sphere at the
    same cost as the planar fast path. Rectilinear-only.
    """
    xd = obj[x_coord]
    yd = obj[y_coord]
    is_rectilinear = xd.ndim == 1 and yd.ndim == 1 and xd.dims[0] != yd.dims[0]

    if spherical and not is_rectilinear:
        msg = "spherical=True is only supported for rectilinear (1D lat/lon) coords"
        raise NotImplementedError(msg)

    if is_rectilinear:
        x = np.asarray(xd.values)
        y = np.asarray(yd.values)
        return _build_cea_grid(x, y) if spherical else _build_grid(x, y)

    xc, yc = xr.broadcast(xd, yd)
    return _build_grid(
        np.asarray(xc.transpose(*dims).values),
        np.asarray(yc.transpose(*dims).values),
    )


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


def _infer_2d_corners(a: np.ndarray) -> np.ndarray:
    """Infer (ny+1, nx+1) cell corners from a 2D cell-center array. Interior
    corners are the mean of the 4 surrounding centers; boundary corners are
    reflected from the adjacent interior row/column."""
    a = np.asarray(a, dtype=float)
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
    if xc.ndim == 2 and yc.ndim == 2 and xc.shape == yc.shape:
        xcorn = _infer_2d_corners(xc)
        ycorn = _infer_2d_corners(yc)
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
        numerator = np.asarray(filled @ apply_weights)
        fraction = np.asarray(mask @ apply_weights)
        threshold = get_valid_threshold(nan_threshold)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = numerator / fraction
        result = np.where(fraction >= threshold, result, np.nan)
    else:
        result = np.asarray(flat @ apply_weights)
        if not coverage_all:
            result = np.where(coverage[np.newaxis, :], result, np.nan)

    # sparse.matmul promotes to float64 regardless of the input dtype — cast
    # back to the requested output dtype so float32-in really produces
    # float32-out (halves memory for float32 pipelines).
    if result.dtype != output_dtype:
        result = result.astype(output_dtype, copy=False)

    out_shape = (*leading_shape, *dst_shape) if leading_shape else dst_shape
    return result.reshape(out_shape)


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
    output spatial dims. Scalar coords (``dims == ()``) ride along too, so
    pinned target metadata (e.g. a fixed timestamp) is preserved."""
    dst_dim_set = set(dst_dims)
    new_coords = {
        name: coord
        for name, coord in target_ds.coords.items()
        if name in (x_coord, y_coord) or set(coord.dims).issubset(dst_dim_set)
    }
    return obj.assign_coords(new_coords) if new_coords else obj
