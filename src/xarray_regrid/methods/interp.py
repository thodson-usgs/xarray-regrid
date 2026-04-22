"""Methods based on xr.interp."""

import warnings
from typing import Literal, overload

import numpy as np
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

    # For cubic on rectilinear (1D-coord) targets whose bounds are fully
    # inside the source bounds, apply interp one axis at a time rather than
    # through xarray's multi-coord dispatch. The single multi-coord call
    # routes through scipy.interpolate.interpn's N-D tensor-product B-spline
    # path (``make_ndbspl`` plus an iterative solver — ~250 ms on a 360x720
    # -> 120x240 global workload). Applying a 1D cubic spline along each
    # axis in turn gives essentially the same output (relative RMS ~1e-5)
    # at ~1/15 the cost.
    #
    # We gate on "target within source" because any out-of-bounds target
    # point produces NaN after the first axis step, and scipy's 1D cubic
    # spline cannot be constructed from NaN-containing inputs — the second
    # step would propagate NaN across the entire output, not just the
    # boundary cells. The N-D path handles this case cell-by-cell, so we
    # fall through to it when any target coord extends beyond the source.
    # In practice, the ``.regrid`` accessor pre-pads the source via
    # ``format_for_regrid`` (pole padding for global lat, longitude
    # wraparound), so this fallback path almost never fires end-to-end.
    if method == "cubic" and _all_1d_coords(data, target_ds, coord_names):
        if _target_within_source(data, target_ds, coord_names):
            interped = data
            for name in sorted(coord_names):
                interped = interped.interp({name: target_ds[name]}, method="cubic")
        else:
            warnings.warn(
                "cubic regrid is falling back to the scipy N-D B-spline path "
                "because the target coordinates extend beyond the source on "
                "at least one axis. This is ~15-90x slower than the factored "
                "path. If you're calling .regrid.cubic() on a near-global "
                "lat/lon grid and hit this, it's usually from mixing "
                "coordinate conventions (e.g. cell-centered source with "
                "edge-anchored target); clipping the target to the source "
                "bounds, or matching conventions, will keep you on the fast "
                "path.",
                RuntimeWarning,
                stacklevel=2,
            )
            coords = {name: target_ds[name] for name in coord_names}
            interped = data.interp(coords=coords, method=method)
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


def _target_within_source(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coord_names: set,
) -> bool:
    """True when every target coord's range is fully inside the source's.

    The factored path interpolates one axis at a time. If a target point
    lies outside the source's range on any axis, the first step produces
    NaN there, and scipy's 1D cubic spline on the next axis cannot be
    constructed from NaN inputs — it returns NaN across that axis' batch,
    not just at the out-of-bounds cell. Falling back to the N-D path
    preserves the per-cell NaN behaviour users expect at domain edges.

    A small relative tolerance (1e-9 times the source span) absorbs
    floating-point rounding between otherwise-identical bounds so we
    don't spuriously reject machine-equal endpoints.
    """
    for name in coord_names:
        src = np.asarray(data[name].values)
        tgt = np.asarray(target_ds[name].values)
        if src.size == 0 or tgt.size == 0:
            return False
        src_lo, src_hi = float(src.min()), float(src.max())
        tol = 1e-9 * max(abs(src_hi - src_lo), 1.0)
        if float(tgt.min()) < src_lo - tol or float(tgt.max()) > src_hi + tol:
            return False
    return True
