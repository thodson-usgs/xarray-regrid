from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

from xarray_regrid.methods.conservative import get_valid_threshold
from xarray_regrid.methods.conservative_2d.spec import RegridSpec


def apply_stored_weights(
    data: xr.DataArray | xr.Dataset,
    direction: Any,
    spec: RegridSpec,
    target_coords: xr.Dataset,
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray | xr.Dataset:
    """Apply ``direction``'s cached, pre-transposed weight matrix to ``data``
    via ``xr.apply_ufunc``."""
    actual_src_shape = tuple(
        int(data.sizes[d]) for d in spec.src_dims if d in data.sizes
    )
    if actual_src_shape != spec.src_shape:
        msg = (
            f"source spatial shape {actual_src_shape} on dims {spec.src_dims} does "
            f"not match the regridder's expected shape {spec.src_shape}"
        )
        raise ValueError(msg)

    src_tokens = tuple(f"__src_{d}" for d in spec.src_dims)
    data_renamed = data.rename(dict(zip(spec.src_dims, src_tokens, strict=True)))

    output_dtype = result_dtype(data)
    result = xr.apply_ufunc(
        apply_core,
        data_renamed,
        kwargs={
            "apply_weights": direction.apply_matrix,
            "coverage": direction.coverage,
            "coverage_all": direction.coverage_all,
            "src_shape": spec.src_shape,
            "dst_shape": spec.dst_shape,
            "skipna": skipna,
            "nan_threshold": nan_threshold,
            "output_dtype": output_dtype,
        },
        input_core_dims=[list(src_tokens)],
        output_core_dims=[list(spec.dst_dims)],
        exclude_dims=set(src_tokens),
        dask="parallelized",
        output_dtypes=[output_dtype],
        dask_gufunc_kwargs={
            "output_sizes": {d: int(target_coords.sizes[d]) for d in spec.dst_dims},
            "allow_rechunk": True,
        },
        keep_attrs=True,
    )

    return assign_target_coords(
        result,
        target_coords,
        spec.dst_dims,
        spec.x_coord,
        spec.y_coord,
    )


def apply_core(
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
    """Apply a pre-transposed weight matrix along the trailing spatial dims."""
    n_spatial = len(src_shape)
    leading_shape = arr.shape[:-n_spatial] if n_spatial > 0 else arr.shape
    n_src = int(np.prod(src_shape))
    flat = arr.reshape(-1, n_src) if leading_shape else arr.reshape(1, n_src)

    nan_mask = (
        np.isnan(flat)
        if skipna and np.issubdtype(flat.dtype, np.floating)
        else None
    )

    if nan_mask is not None and nan_mask.any():
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

    if result.dtype != output_dtype:
        result = result.astype(output_dtype, copy=False)

    out_shape = (*leading_shape, *dst_shape) if leading_shape else dst_shape
    return result.reshape(out_shape)


def result_dtype(obj: xr.DataArray | xr.Dataset) -> np.dtype:
    if isinstance(obj, xr.DataArray):
        return np.result_type(np.float32, obj.dtype)
    dtypes = [v.dtype for v in obj.data_vars.values()]
    if not dtypes:
        return np.dtype(np.float64)
    return np.result_type(np.float32, *dtypes)


def assign_target_coords(
    obj: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    dst_dims: tuple[Hashable, ...],
    x_coord: str,
    y_coord: str,
) -> xr.DataArray | xr.Dataset:
    """Attach target coordinates that name a spatial axis or live on output dims."""
    dst_dim_set = set(dst_dims)
    new_coords = {
        name: coord
        for name, coord in target_ds.coords.items()
        if name in (x_coord, y_coord) or set(coord.dims).issubset(dst_dim_set)
    }
    return obj.assign_coords(new_coords) if new_coords else obj
