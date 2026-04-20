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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


@cache
def _package_version() -> str:
    """Cached ``importlib.metadata`` lookup. Cheap on warm call."""
    try:
        from importlib.metadata import version
        return version("xarray-regrid")
    except Exception:
        return "unknown"


# Bump on breaking change to the on-disk format in ConservativeRegridder.to_netcdf.
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RegridderMetadata:
    """Serialized build parameters of a :class:`ConservativeRegridder`.

    Fields become netCDF attributes via :meth:`to_attrs`; the loader
    reconstructs via :meth:`from_attrs`. Keep new fields optional so older
    files remain loadable.
    """

    x_coord: str
    y_coord: str
    spherical: bool
    src_dims: tuple[str, ...]
    dst_dims: tuple[str, ...]
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    source_x_range: tuple[float, float] | None = None
    source_y_range: tuple[float, float] | None = None
    target_x_range: tuple[float, float] | None = None
    target_y_range: tuple[float, float] | None = None
    xarray_regrid_version: str = field(default_factory=_package_version)
    created: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    schema_version: int = _SCHEMA_VERSION

    def to_attrs(self) -> dict[str, Any]:
        d = asdict(self)
        d["spherical"] = int(d["spherical"])
        d["src_dims"] = list(d["src_dims"])
        d["dst_dims"] = list(d["dst_dims"])
        d["src_shape"] = list(d["src_shape"])
        d["dst_shape"] = list(d["dst_shape"])
        for key in (
            "source_x_range", "source_y_range",
            "target_x_range", "target_y_range",
        ):
            if d[key] is None:
                del d[key]
            else:
                d[key] = list(d[key])
        return d

    @classmethod
    def from_attrs(cls, attrs: dict[str, Any]) -> "RegridderMetadata":
        """Parse back from netCDF attributes. Missing optional fields default."""
        def _range(key: str) -> tuple[float, float] | None:
            v = attrs.get(key)
            if v is None:
                return None
            arr = np.atleast_1d(v)
            return (float(arr[0]), float(arr[1]))

        return cls(
            x_coord=str(attrs["x_coord"]),
            y_coord=str(attrs["y_coord"]),
            spherical=bool(int(attrs["spherical"])),
            src_dims=tuple(str(d) for d in np.atleast_1d(attrs["src_dims"])),
            dst_dims=tuple(str(d) for d in np.atleast_1d(attrs["dst_dims"])),
            src_shape=tuple(int(s) for s in np.atleast_1d(attrs["src_shape"])),
            dst_shape=tuple(int(s) for s in np.atleast_1d(attrs["dst_shape"])),
            source_x_range=_range("source_x_range"),
            source_y_range=_range("source_y_range"),
            target_x_range=_range("target_x_range"),
            target_y_range=_range("target_y_range"),
            xarray_regrid_version=str(attrs.get("xarray_regrid_version", "unknown")),
            created=str(attrs.get("created", "")),
            # Default to 0 so files written before this attr existed fail the
            # schema check rather than silently appearing current.
            schema_version=int(attrs.get("schema_version", 0)),
        )

try:
    import shapely
    from shapely.strtree import STRtree

    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    shapely = None  # type: ignore[assignment]
    STRtree = None  # type: ignore[assignment]
    _HAS_SHAPELY = False

try:
    import sparse  # type: ignore

    _HAS_SPARSE = True
except ImportError:  # pragma: no cover
    sparse = None  # type: ignore[assignment]
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

    Example:
        >>> regridder = ConservativeRegridder(src_ds, tgt_ds, x_coord="lon", y_coord="lat")
        >>> out = regridder.regrid(da)              # forward
        >>> back = regridder.T.regrid(out)          # backward
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
        src_dims = _spatial_dims(source, x_coord, y_coord)
        dst_dims = _spatial_dims(target, x_coord, y_coord)
        if not src_dims:
            msg = f"source has no dims for coords {x_coord!r}, {y_coord!r}"
            raise ValueError(msg)
        if not dst_dims:
            msg = f"target has no dims for coords {x_coord!r}, {y_coord!r}"
            raise ValueError(msg)
        src_grid = _grid_from_coords(
            source, x_coord, y_coord, src_dims, spherical=spherical
        )
        dst_grid = _grid_from_coords(
            target, x_coord, y_coord, dst_dims, spherical=spherical
        )
        self.spherical = spherical

        self.x_coord = x_coord
        self.y_coord = y_coord
        self._src_dims = src_dims
        self._dst_dims = dst_dims
        self._src_shape = tuple(int(source.sizes[d]) for d in src_dims)
        self._dst_shape = tuple(int(target.sizes[d]) for d in dst_dims)
        self._areas = _build_intersection_areas(
            src_grid, dst_grid, n_threads=n_threads
        )
        self._source_coords = source.coords.to_dataset()
        self._target_coords = target.coords.to_dataset()
        # `_*_apply` caches the transposed, index-sorted weight matrix used for
        # `data @ W` matmul — avoiding sparse's per-call `.T` + `_sort_indices`.
        self._fwd_weights: "sparse.COO | np.ndarray | None" = None
        self._bwd_weights: "sparse.COO | np.ndarray | None" = None
        self._fwd_apply: "sparse.COO | np.ndarray | None" = None
        self._bwd_apply: "sparse.COO | np.ndarray | None" = None
        self._fwd_coverage: np.ndarray | None = None
        self._bwd_coverage: np.ndarray | None = None

    @property
    def forward_weights(self) -> "sparse.COO | np.ndarray":
        """The row-normalized forward weight matrix (source → target)."""
        if self._fwd_weights is None:
            self._fwd_weights = _row_normalize(self._areas)
        return self._fwd_weights

    @property
    def backward_weights(self) -> "sparse.COO | np.ndarray":
        """The row-normalized backward weight matrix (target → source)."""
        if self._bwd_weights is None:
            self._bwd_weights = _row_normalize(_transpose_weights(self._areas))
        return self._bwd_weights

    def _forward_apply_matrix(self) -> "sparse.COO | np.ndarray":
        if self._fwd_apply is None:
            self._fwd_apply = _transpose_weights(self.forward_weights, sort=True)
        return self._fwd_apply

    def _backward_apply_matrix(self) -> "sparse.COO | np.ndarray":
        if self._bwd_apply is None:
            self._bwd_apply = _transpose_weights(self.backward_weights, sort=True)
        return self._bwd_apply

    def _forward_coverage(self) -> np.ndarray:
        """Boolean (n_dst,) mask: which destination cells have any source
        overlap. Lazily computed and cached alongside the forward weights."""
        if self._fwd_coverage is None:
            self._fwd_coverage = _coverage_mask(self._areas)
        return self._fwd_coverage

    def _backward_coverage(self) -> np.ndarray:
        if self._bwd_coverage is None:
            self._bwd_coverage = _coverage_mask(_transpose_weights(self._areas))
        return self._bwd_coverage

    def regrid(
        self,
        data: xr.DataArray | xr.Dataset,
        skipna: bool = True,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid ``data`` forward (source → target)."""
        return _apply_stored_weights(
            data,
            apply_weights=self._forward_apply_matrix(),
            coverage=self._forward_coverage(),
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
        """Return the backward regridder (target → source), sharing the
        underlying area matrix and both cached weight matrices."""
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
        new._fwd_weights = self._bwd_weights
        new._bwd_weights = self._fwd_weights
        new._fwd_apply = self._bwd_apply
        new._bwd_apply = self._fwd_apply
        new._fwd_coverage = self._bwd_coverage
        new._bwd_coverage = self._fwd_coverage
        return new

    @property
    def T(self) -> "ConservativeRegridder":
        """Alias for :meth:`transpose`."""
        return self.transpose()

    def __repr__(self) -> str:
        nnz = getattr(self._areas, "nnz", None)
        shape = getattr(self._areas, "shape", (None, None))
        nnz_str = f"nnz={nnz}" if nnz is not None else "dense"
        return (
            f"ConservativeRegridder(src_dims={self._src_dims}, "
            f"dst_dims={self._dst_dims}, {shape[0]}x{shape[1]}, {nnz_str})"
        )

    def metadata(self) -> RegridderMetadata:
        """Return the :class:`RegridderMetadata` that :meth:`to_netcdf` would
        write — useful for inspection without touching disk."""
        return RegridderMetadata(
            x_coord=self.x_coord,
            y_coord=self.y_coord,
            spherical=self.spherical,
            src_dims=tuple(str(d) for d in self._src_dims),
            dst_dims=tuple(str(d) for d in self._dst_dims),
            src_shape=self._src_shape,
            dst_shape=self._dst_shape,
            source_x_range=_coord_range(self._source_coords, self.x_coord),
            source_y_range=_coord_range(self._source_coords, self.y_coord),
            target_x_range=_coord_range(self._target_coords, self.x_coord),
            target_y_range=_coord_range(self._target_coords, self.y_coord),
        )

    def to_netcdf(
        self, path: str | Path, engine: str | None = None
    ) -> None:
        """Save the weight matrix and reproducibility metadata to a netCDF file.

        File layout:

        - root dataset: the sparse area matrix stored as three 1D variables
          ``_coo_row``, ``_coo_col``, ``_coo_data`` (``shape=(n_dst, n_src)``
          carried on root-dataset attributes), plus :class:`RegridderMetadata`
          fields as attributes.
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
                **self.metadata().to_attrs(),
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
        cls, path: str | Path, engine: str | None = None
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
        meta = RegridderMetadata.from_attrs(attrs)
        if meta.schema_version != _SCHEMA_VERSION:
            msg = (
                f"regridder file at {path} uses schema version "
                f"{meta.schema_version}; this xarray-regrid understands "
                f"{_SCHEMA_VERSION}. Upgrade xarray-regrid or re-save."
            )
            raise ValueError(msg)

        with xr.open_dataset(path, group="source_coords", engine=engine) as g:
            source_coords = g.load()
        with xr.open_dataset(path, group="target_coords", engine=engine) as g:
            target_coords = g.load()

        return cls._from_state(
            areas=_coo_from_components(row, col, data, (n_dst, n_src)),
            source_coords=source_coords,
            target_coords=target_coords,
            src_dims=meta.src_dims,
            dst_dims=meta.dst_dims,
            src_shape=meta.src_shape,
            dst_shape=meta.dst_shape,
            x_coord=meta.x_coord,
            y_coord=meta.y_coord,
            spherical=meta.spherical,
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
        instance._fwd_weights = None
        instance._bwd_weights = None
        instance._fwd_apply = None
        instance._bwd_apply = None
        instance._fwd_coverage = None
        instance._bwd_coverage = None
        return instance

    @classmethod
    def from_polygons(
        cls,
        source_polygons: np.ndarray,
        target_polygons: np.ndarray,
        source_dim: str = "cell",
        target_dim: str = "cell",
        target_coords: xr.Dataset | None = None,
        n_threads: int | None = None,
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
            n_threads: Thread count for GEOS intersection.

        Returns:
            A ``ConservativeRegridder`` that accepts data with ``source_dim``
            in place of the structured grid's spatial dims.

        Intersection geometry is planar in the input polygons' coordinate
        space. If the polygons represent lat/lon cells, project them into an
        equal-area CRS first (or use the structured path with ``spherical=True``).
        Polygons that cross the antimeridian must be split beforehand.
        """
        _check_shapely()
        src_polys = np.asarray(source_polygons)
        dst_polys = np.asarray(target_polygons)
        if src_polys.ndim != 1:
            msg = "source_polygons must be a 1D array of shapely Polygons"
            raise ValueError(msg)
        if dst_polys.ndim != 1:
            msg = "target_polygons must be a 1D array of shapely Polygons"
            raise ValueError(msg)

        src_grid = _Grid(
            polys=src_polys, bounds=shapely.bounds(src_polys), rectilinear=False,
        )
        dst_grid = _Grid(
            polys=dst_polys, bounds=shapely.bounds(dst_polys), rectilinear=False,
        )
        n_src = int(src_polys.size)
        n_dst = int(dst_polys.size)
        tgt_ds = (
            target_coords
            if target_coords is not None
            else xr.Dataset(coords={target_dim: np.arange(n_dst)})
        )
        return cls._from_state(
            areas=_build_intersection_areas(src_grid, dst_grid, n_threads=n_threads),
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

    Returns:
        A 1D numpy array of shapely Polygons in row-major (y, x) order.
    """
    _check_shapely()
    x = np.asarray(x)
    y = np.asarray(y)
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
        data, target_ds,
        x_coord=x_coord, y_coord=y_coord,
        spherical=spherical, n_threads=n_threads,
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
            d: -1 for d in src_dims
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
    data_renamed = data.rename({s: t for s, t in zip(src_dims, src_tokens)})

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
    return (arr > 0).any(axis=1)


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
    return rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False), arr[rows, cols], arr.shape


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


def _coord_range(
    ds: xr.Dataset, coord_name: str
) -> tuple[float, float] | None:
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
    is_rectilinear = (
        xd.ndim == 1 and yd.ndim == 1 and xd.dims[0] != yd.dims[0]
    )
    if spherical:
        if not is_rectilinear:
            msg = (
                "spherical=True is only supported for rectilinear (1D lat/lon) "
                "coordinate arrays in this version."
            )
            raise NotImplementedError(msg)
        return _build_cea_grid(
            np.asarray(xd.values), np.asarray(yd.values)
        )
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
    lat_edges_deg = np.clip(_infer_1d_edges(lat_centers), -90.0, 90.0)
    lon_edges_deg = _infer_1d_edges(lon_centers)
    return _rect_grid_from_edges(
        np.deg2rad(lon_edges_deg),
        np.sin(np.deg2rad(lat_edges_deg)),
    )


def _infer_1d_edges(centers: np.ndarray) -> np.ndarray:
    """Return cell edges from 1D centers: midpoints between consecutive
    centers, with symmetric reflection for the two outer bounds."""
    c = np.asarray(centers, dtype=float)
    if c.size < 2:
        msg = "need at least two centers to infer cell edges"
        raise ValueError(msg)
    mids = 0.5 * (c[:-1] + c[1:])
    left = 2 * c[0] - mids[0]
    right = 2 * c[-1] - mids[-1]
    return np.concatenate([[left], mids, [right]])


def _infer_2d_corners(
    xc: np.ndarray, yc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ny+1, nx+1) cell-corner arrays from 2D cell-center arrays.

    Interior corners are the mean of the four surrounding centers; boundary
    corners are reflected from the adjacent interior row/column.
    """
    assert xc.shape == yc.shape and xc.ndim == 2

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
        xe = _infer_1d_edges(xc.astype(float))
        ye = _infer_1d_edges(yc.astype(float))
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
    src: _Grid, dst: _Grid, n_threads: int | None = None
) -> "sparse.COO | np.ndarray":
    """Build the (n_dst, n_src) raw area-intersection matrix ``A[i, j] =
    area(dst_i ∩ src_j)``.

    This is the unnormalized matrix. Row-normalize via :func:`_row_normalize`
    to get forward weights; transpose first for backward (target → source).

    When both grids are rectilinear (axis-aligned rectangles) intersection
    areas are computed analytically from the bounds, skipping GEOS clipping.
    """
    _check_shapely()
    n_dst = int(len(dst.polys))
    n_src = int(len(src.polys))

    tree = STRtree(src.polys)
    pairs = tree.query(dst.polys, predicate="intersects")
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
        n_threads = min(os.cpu_count() or 1, 8)
        # Amortize thread-pool overhead only when there's meaningful work.
        if n < 50_000:
            n_threads = 1
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
    """Attach the target dataset's dim coords and auxiliary lat/lon coords."""
    new_coords: dict[Hashable, Any] = {}
    for d in dst_dims:
        if d in target_ds.coords:
            new_coords[d] = target_ds[d]
    for name in (x_coord, y_coord):
        if name in target_ds.coords and name not in new_coords:
            new_coords[name] = target_ds[name]
    if new_coords:
        obj = obj.assign_coords(new_coords)
    return obj
