import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from xarray_regrid.methods.conservative_2d._deps import (
    HAS_SPARSE,
    AreaMatrix,
    STRtree,
    require_shapely,
    shapely,
    sparse,
)


def coverage_mask(areas: AreaMatrix) -> np.ndarray:
    if HAS_SPARSE and isinstance(areas, sparse.COO):
        n_dst = int(areas.shape[0])
        mask = np.zeros(n_dst, dtype=bool)
        mask[areas.coords[0]] = True
        return mask
    arr = np.asarray(areas)
    return np.asarray((arr > 0).any(axis=1))


def sum_matrix_axis_1d(areas: AreaMatrix, axis: int) -> np.ndarray:
    summed = areas.sum(axis=axis)
    if hasattr(summed, "todense"):
        summed = summed.todense()
    return np.asarray(summed, dtype=np.float64).reshape(-1)


def transpose_weights(w: AreaMatrix, *, sort: bool = False) -> AreaMatrix:
    if HAS_SPARSE and isinstance(w, sparse.COO):
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


def row_normalize(areas: AreaMatrix) -> AreaMatrix:
    if HAS_SPARSE and isinstance(areas, sparse.COO):
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


def empty_weights(n_dst: int, n_src: int) -> AreaMatrix:
    if HAS_SPARSE:
        return sparse.COO(
            coords=np.zeros((2, 0), dtype=np.int64),
            data=np.zeros(0, dtype=np.float64),
            shape=(n_dst, n_src),
        )
    return np.zeros((n_dst, n_src), dtype=np.float64)


def intersection_areas_threaded(
    a: np.ndarray, b: np.ndarray, n_threads: int | None
) -> np.ndarray:
    n = len(a)
    if n_threads is None:
        n_threads = 1 if n < 1_000 else min(os.cpu_count() or 1, 16)
    if n_threads <= 1 or n == 0:
        return np.asarray(shapely.area(shapely.intersection(a, b)))

    splits = np.array_split(np.arange(n), n_threads)

    def _work(idx: np.ndarray) -> np.ndarray:
        return np.asarray(shapely.area(shapely.intersection(a[idx], b[idx])))

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        parts = list(pool.map(_work, splits))
    return np.concatenate(parts)


def build_intersection_areas(
    src: Any,
    dst: Any,
    n_threads: int | None = None,
    *,
    predicate_filter: bool = False,
) -> AreaMatrix:
    require_shapely()
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
        return empty_weights(n_dst, n_src)

    if src.rectilinear and dst.rectilinear:
        sb = src.bounds[src_idx]
        db = dst.bounds[dst_idx]
        dx = np.minimum(sb[:, 2], db[:, 2]) - np.maximum(sb[:, 0], db[:, 0])
        dy = np.minimum(sb[:, 3], db[:, 3]) - np.maximum(sb[:, 1], db[:, 1])
        areas = np.maximum(dx, 0.0) * np.maximum(dy, 0.0)
    else:
        areas = intersection_areas_threaded(
            dst.polys[dst_idx],
            src.polys[src_idx],
            n_threads,
        )

    keep = areas > 0
    dst_idx = dst_idx[keep]
    src_idx = src_idx[keep]
    areas = areas[keep]

    if dst_idx.size == 0:
        return empty_weights(n_dst, n_src)

    if HAS_SPARSE:
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
