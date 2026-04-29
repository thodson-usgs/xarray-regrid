from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from xarray_regrid import utils
from xarray_regrid.methods.conservative_2d._deps import (
    HAS_SPARSE,
    affinity,
    require_shapely,
    shapely,
    sparse,
)


def check_shapely() -> None:
    require_shapely()


@dataclass
class Grid:
    polys: np.ndarray
    bounds: np.ndarray
    rectilinear: bool


def looks_like_longitude(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size and finite.min() >= -360.0 and finite.max() <= 360.0)


def unwrap_longitude(values: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(np.asarray(values, dtype=float))
    return np.asarray(np.rad2deg(np.unwrap(radians, axis=-1)))


def periodic_offset(reference: float, value: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(value):
        return 0.0
    return 360.0 * round((reference - value) / 360.0)


def normalize_periodic_polygons(
    polygons: np.ndarray, reference: float | None = None
) -> np.ndarray:
    unwrapped = [unwrap_polygon(p) for p in polygons]
    if reference is None:
        for poly in unwrapped:
            center = polygon_center_x(poly)
            if np.isfinite(center):
                reference = center
                break
    if reference is None:
        return np.array(unwrapped, dtype=object)

    out = []
    for poly in unwrapped:
        offset = periodic_offset(reference, polygon_center_x(poly))
        out.append(affinity.translate(poly, xoff=offset) if offset != 0.0 else poly)
    return np.array(out, dtype=object)


def polygon_reference_x(polygons: np.ndarray) -> float | None:
    bounds = shapely.bounds(polygons)
    centers = 0.5 * (bounds[:, 0] + bounds[:, 2])
    finite = centers[np.isfinite(centers)]
    return float(finite.mean()) if finite.size else None


def polygon_center_x(polygon: Any) -> float:
    minx, _, maxx, _ = polygon.bounds
    return 0.5 * (float(minx) + float(maxx))


def unwrap_polygon(polygon: Any) -> Any:
    if polygon.is_empty:
        return polygon
    if polygon.geom_type == "Polygon":
        exterior = unwrap_ring(np.asarray(polygon.exterior.coords))
        holes = [unwrap_ring(np.asarray(ring.coords)) for ring in polygon.interiors]
        return shapely.Polygon(exterior, holes)
    if polygon.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([unwrap_polygon(part) for part in polygon.geoms])
    return polygon


def unwrap_ring(ring: np.ndarray) -> np.ndarray:
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
    return np.asarray(new_ring)


def spatial_dims(
    obj: xr.DataArray | xr.Dataset, x_coord: str, y_coord: str
) -> tuple[Hashable, ...]:
    if x_coord not in obj.coords or y_coord not in obj.coords:
        return ()
    xd = obj[x_coord].dims
    yd = obj[y_coord].dims
    if len(xd) == 1 and len(yd) == 1 and xd[0] != yd[0]:
        return (yd[0], xd[0])
    dims = set(xd) | set(yd)
    return tuple(d for d in obj.dims if d in dims)


def grid_from_coords(
    obj: xr.DataArray | xr.Dataset,
    x_coord: str,
    y_coord: str,
    dims: tuple[Hashable, ...],
    spherical: bool = False,
) -> Grid:
    xd = obj[x_coord]
    yd = obj[y_coord]
    is_rectilinear = xd.ndim == 1 and yd.ndim == 1 and xd.dims[0] != yd.dims[0]

    if spherical and not is_rectilinear:
        msg = "spherical=True is only supported for rectilinear (1D lat/lon) coords"
        raise NotImplementedError(msg)

    if is_rectilinear:
        x = np.asarray(xd.values)
        y = np.asarray(yd.values)
        return build_cea_grid(x, y) if spherical else build_grid(x, y)

    xc, yc = xr.broadcast(xd, yd)
    return build_grid(
        np.asarray(xc.transpose(*dims).values),
        np.asarray(yc.transpose(*dims).values),
    )


def build_cea_grid(lon_centers: np.ndarray, lat_centers: np.ndarray) -> Grid:
    check_shapely()
    if lon_centers.size < 2 or lat_centers.size < 2:
        msg = "spherical mode requires at least two cells per dimension"
        raise ValueError(msg)
    lat_edges_deg = np.clip(utils.infer_1d_edges(lat_centers), -90.0, 90.0)
    lon_edges_deg = utils.infer_1d_edges(lon_centers)
    return rect_grid_from_edges(
        np.deg2rad(lon_edges_deg),
        np.sin(np.deg2rad(lat_edges_deg)),
    )


def infer_2d_corners(a: np.ndarray) -> np.ndarray:
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


def rect_grid_from_edges(xe: np.ndarray, ye: np.ndarray) -> Grid:
    x0, y0 = np.meshgrid(xe[:-1], ye[:-1], indexing="xy")
    x1, y1 = np.meshgrid(xe[1:], ye[1:], indexing="xy")
    x0f, y0f, x1f, y1f = x0.ravel(), y0.ravel(), x1.ravel(), y1.ravel()
    polys = shapely.box(x0f, y0f, x1f, y1f)
    bounds = np.stack([x0f, y0f, x1f, y1f], axis=1)
    return Grid(polys=polys, bounds=bounds, rectilinear=True)


def build_grid(xc: np.ndarray, yc: np.ndarray) -> Grid:
    check_shapely()
    if xc.ndim == 1 and yc.ndim == 1:
        xe = utils.infer_1d_edges(xc.astype(float))
        ye = utils.infer_1d_edges(yc.astype(float))
        return rect_grid_from_edges(xe, ye)
    if xc.ndim == 2 and yc.ndim == 2 and xc.shape == yc.shape:
        xcorn = infer_2d_corners(xc)
        ycorn = infer_2d_corners(yc)
        ny, nx = xc.shape
        c00 = np.stack([xcorn[:-1, :-1], ycorn[:-1, :-1]], axis=-1)
        c10 = np.stack([xcorn[:-1, 1:], ycorn[:-1, 1:]], axis=-1)
        c11 = np.stack([xcorn[1:, 1:], ycorn[1:, 1:]], axis=-1)
        c01 = np.stack([xcorn[1:, :-1], ycorn[1:, :-1]], axis=-1)
        rings = np.stack([c00, c10, c11, c01, c00], axis=2).reshape(ny * nx, 5, 2)
        polys = shapely.polygons(rings)
        return Grid(polys=polys, bounds=shapely.bounds(polys), rectilinear=False)
    msg = "x and y coordinate arrays must both be 1D or both 2D"
    raise ValueError(msg)


def remap_columns_for_axis_sort(
    areas: Any,
    sort_idx: np.ndarray,
    src_shape: tuple[int, ...],
    axis_index: int,
) -> Any:
    nx = int(src_shape[axis_index])
    inner = int(np.prod(src_shape[axis_index + 1 :]))
    stride = nx * inner
    sort_idx = np.asarray(sort_idx, dtype=np.int64)

    if HAS_SPARSE and isinstance(areas, sparse.COO):
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


def normalize_longitude_coords(
    source: xr.DataArray | xr.Dataset,
    target: xr.Dataset,
    x_coord: str,
) -> tuple[xr.DataArray | xr.Dataset, xr.Dataset, np.ndarray | None]:
    if x_coord not in source.coords or x_coord not in target.coords:
        return source, target, None

    source_x = np.asarray(source[x_coord].values)
    target_x = np.asarray(target[x_coord].values)
    if not looks_like_longitude(source_x) and not looks_like_longitude(target_x):
        return source, target, None

    source_x = unwrap_longitude(source_x)
    target_x = unwrap_longitude(target_x)
    src_finite = source_x[np.isfinite(source_x)]
    tgt_finite = target_x[np.isfinite(target_x)]

    src_x_sort_idx: np.ndarray | None = None
    if (
        source_x.ndim == 1
        and target_x.ndim == 1
        and src_finite.size
        and tgt_finite.size
    ):
        wrap_point = float((tgt_finite[0] + tgt_finite[-1] + 360.0) / 2.0)
        source_x = np.where(source_x < wrap_point - 360.0, source_x + 360.0, source_x)
        source_x = np.where(source_x > wrap_point, source_x - 360.0, source_x)
        diffs = np.diff(source_x)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            src_x_sort_idx = np.argsort(source_x, kind="stable")
            source_x = source_x[src_x_sort_idx]
    elif src_finite.size and tgt_finite.size:
        target_x = target_x + periodic_offset(
            float(src_finite.mean()),
            float(tgt_finite.mean()),
        )

    return (
        utils.update_coord(source, x_coord, source_x),
        utils.update_coord(target, x_coord, target_x),
        src_x_sort_idx,
    )
