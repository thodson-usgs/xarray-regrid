from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from xarray_regrid.methods.conservative_2d.spec import RegridSpec

try:
    import sparse

    _HAS_SPARSE = True
except ImportError:  # pragma: no cover
    sparse = None
    _HAS_SPARSE = False

# Bump on breaking change to the on-disk format in ConservativeRegridder.to_netcdf.
_SCHEMA_VERSION = 1
NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"] | None


def _package_version() -> str:
    try:
        return version("xarray-regrid")
    except PackageNotFoundError:
        return "unknown"


def _coo_components(
    weights: "sparse.COO | np.ndarray",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    if _HAS_SPARSE and isinstance(weights, sparse.COO):
        coords = np.asarray(weights.coords)
        return (
            coords[0].astype(np.int64, copy=False),
            coords[1].astype(np.int64, copy=False),
            np.asarray(weights.data),
            weights.shape,
        )
    arr = np.asarray(weights)
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


def _metadata_attrs(
    spec: RegridSpec, source_coords: xr.Dataset, target_coords: xr.Dataset
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "x_coord": spec.x_coord,
        "y_coord": spec.y_coord,
        "spherical": int(spec.spherical),
        "src_dims": [str(d) for d in spec.src_dims],
        "dst_dims": [str(d) for d in spec.dst_dims],
        "src_shape": list(spec.src_shape),
        "dst_shape": list(spec.dst_shape),
        "xarray_regrid_version": _package_version(),
        "created": datetime.now(tz=timezone.utc).isoformat(),
        "schema_version": _SCHEMA_VERSION,
    }
    for prefix, ds, coord in [
        ("source_x", source_coords, spec.x_coord),
        ("source_y", source_coords, spec.y_coord),
        ("target_x", target_coords, spec.x_coord),
        ("target_y", target_coords, spec.y_coord),
    ]:
        if coord and coord in ds.coords and ds[coord].size:
            attrs[f"{prefix}_range"] = [float(ds[coord].min()), float(ds[coord].max())]
    return attrs


def _metadata_from_attrs(attrs: dict[str, Any], path: Path) -> RegridSpec:
    """Parse and validate regridding spec metadata from netCDF attrs."""
    schema_version = int(attrs.get("schema_version", 0))
    if schema_version != _SCHEMA_VERSION:
        msg = (
            f"regridder file at {path} uses schema version {schema_version}; "
            f"this xarray-regrid understands {_SCHEMA_VERSION}. "
            "Upgrade xarray-regrid or re-save."
        )
        raise ValueError(msg)

    return RegridSpec(
        x_coord=str(attrs["x_coord"]),
        y_coord=str(attrs["y_coord"]),
        spherical=bool(int(attrs["spherical"])),
        src_dims=tuple(str(d) for d in np.atleast_1d(attrs["src_dims"])),
        dst_dims=tuple(str(d) for d in np.atleast_1d(attrs["dst_dims"])),
        src_shape=tuple(int(s) for s in np.atleast_1d(attrs["src_shape"])),
        dst_shape=tuple(int(s) for s in np.atleast_1d(attrs["dst_shape"])),
    )


def save_regridder_netcdf(
    path: str | Path,
    areas: "sparse.COO | np.ndarray",
    spec: RegridSpec,
    source_coords: xr.Dataset,
    target_coords: xr.Dataset,
    engine: NetcdfEngine = None,
) -> None:
    path = Path(path)
    row, col, data, shape = _coo_components(areas)
    ds_weights = xr.Dataset(
        {
            "_coo_row": (("nnz",), row),
            "_coo_col": (("nnz",), col),
            "_coo_data": (("nnz",), data),
        },
        attrs={
            **_metadata_attrs(spec, source_coords, target_coords),
            "n_dst": int(shape[0]),
            "n_src": int(shape[1]),
        },
    )
    ds_weights.to_netcdf(path, mode="w", engine=engine)
    source_coords.to_netcdf(path, mode="a", group="source_coords", engine=engine)
    target_coords.to_netcdf(path, mode="a", group="target_coords", engine=engine)


def load_regridder_netcdf(
    path: str | Path,
    engine: NetcdfEngine = None,
) -> tuple["sparse.COO | np.ndarray", xr.Dataset, xr.Dataset, RegridSpec]:
    path = Path(path)
    with xr.open_dataset(path, engine=engine) as ds_weights:
        attrs = dict(ds_weights.attrs)
        n_dst = int(attrs.pop("n_dst"))
        n_src = int(attrs.pop("n_src"))
        row = np.asarray(ds_weights["_coo_row"].values)
        col = np.asarray(ds_weights["_coo_col"].values)
        data = np.asarray(ds_weights["_coo_data"].values)
    spec = _metadata_from_attrs(attrs, path)

    with xr.open_dataset(path, group="source_coords", engine=engine) as g:
        source_coords = g.load()
    with xr.open_dataset(path, group="target_coords", engine=engine) as g:
        target_coords = g.load()

    areas = _coo_from_components(row, col, data, (n_dst, n_src))
    return areas, source_coords, target_coords, spec
