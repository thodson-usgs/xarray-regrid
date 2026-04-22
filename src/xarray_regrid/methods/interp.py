"""Methods based on xr.interp."""

from typing import Literal, overload

import xarray as xr


@overload
def interp_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.DataArray: ...


@overload
def interp_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.Dataset: ...


def interp_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.DataArray | xr.Dataset:
    """Refine a dataset using xarray's interp method.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        method: Which interpolation method to use (e.g. 'linear', 'nearest').

    Returns:
        Regridded input dataset
    """
    coord_names = set(target_ds.coords).intersection(set(data.coords))
    coord_attrs = {coord: data[coord].attrs for coord in coord_names}

    # For cubic on rectilinear (1D-coord) targets, apply interp one axis at
    # a time rather than through xarray's multi-coord dispatch. The single
    # multi-coord call routes through scipy.interpolate.interpn's N-D
    # tensor-product B-spline path (``make_ndbspl`` plus an iterative
    # solver — ~250 ms on a 360x720 -> 120x240 global workload). Applying
    # a 1D cubic spline along each axis in turn gives essentially the same
    # output (relative RMS ~1e-5) at ~1/15 the cost.
    if method == "cubic" and _all_1d_coords(data, target_ds, coord_names):
        interped = data
        for name in sorted(coord_names):
            interped = interped.interp({name: target_ds[name]}, method="cubic")
    else:
        coords = {name: target_ds[name] for name in coord_names}
        interped = data.interp(coords=coords, method=method)

    # xarray's interp drops some of the coordinate's attributes (e.g. long_name)
    for coord in coord_names:
        interped[coord].attrs = coord_attrs[coord]

    return interped


def _all_1d_coords(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coord_names: set,
) -> bool:
    """True when every regrid coord is 1D on both source and target.

    Gates the axis-factored cubic path: factoring requires separable axes,
    which is only guaranteed for rectilinear coordinates.
    """
    for name in coord_names:
        if data[name].ndim != 1 or target_ds[name].ndim != 1:
            return False
    return True
