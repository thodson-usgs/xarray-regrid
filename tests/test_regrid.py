import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

try:
    import xesmf
except ImportError:
    xesmf = None

import xarray_regrid
from xarray_regrid.methods import interp as _cubic_interp

DATA_PATH = Path(__file__).parent.parent / "docs" / "notebooks" / "benchmarks" / "data"

CDO_FILES = {
    "linear": DATA_PATH / "cdo_bilinear_64b.nc",
    "nearest": DATA_PATH / "cdo_nearest_64b.nc",
    "conservative": DATA_PATH / "cdo_conservative_64b.nc",
}

CHUNK_SCHEMES = [{}, {"time": 1}, {"longitude": 100, "latitude": 100}]


@pytest.fixture(scope="session")
def sample_input_data() -> xr.Dataset:
    ds = xr.open_dataset(DATA_PATH / "era5_2m_dewpoint_temperature_2000_monthly.nc")
    # Convert to dask so regridding is lazy for attr-only tests
    return ds.compute().chunk()


@pytest.fixture(scope="session")
def conservative_input_data() -> xr.Dataset:
    ds = xr.open_dataset(DATA_PATH / "era5_total_precipitation_2020_monthly.nc")
    return ds.compute().chunk()


@pytest.fixture(scope="session")
def cdo_comparison_data() -> dict[str, xr.Dataset]:
    data = {}
    for method, path in CDO_FILES.items():
        data[method] = xr.open_dataset(path).compute()
    return data


@pytest.fixture(scope="session")
def sample_grid_ds():
    grid = xarray_regrid.Grid(
        north=90,
        east=180,
        south=45,
        west=90,
        resolution_lat=0.17,
        resolution_lon=0.17,
    )

    return xarray_regrid.create_regridding_dataset(grid)


@pytest.fixture(scope="session")
def conservative_sample_grid():
    grid = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=2.2,
        resolution_lon=2.2,
    )

    return xarray_regrid.create_regridding_dataset(grid)


@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("chunks", CHUNK_SCHEMES)
def test_basic_regridders_ds(
    sample_input_data, sample_grid_ds, cdo_comparison_data, method, chunks
):
    """Test the dataset regridders (except conservative)."""
    regridder = getattr(sample_input_data.chunk(chunks).regrid, method)
    ds_regrid = regridder(sample_grid_ds)
    ds_cdo = cdo_comparison_data[method]
    xr.testing.assert_allclose(ds_regrid, ds_cdo, rtol=0.002, atol=2e-5)


@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("chunks", CHUNK_SCHEMES)
def test_basic_regridders_da(
    sample_input_data, sample_grid_ds, cdo_comparison_data, method, chunks
):
    """Test the dataarray regridders (except conservative)."""
    regridder = getattr(sample_input_data["d2m"].chunk(chunks).regrid, method)
    da_regrid = regridder(sample_grid_ds)
    da_cdo = cdo_comparison_data[method]["d2m"]
    xr.testing.assert_allclose(da_regrid, da_cdo, rtol=0.002, atol=2e-5)


@pytest.mark.parametrize("chunks", CHUNK_SCHEMES)
def test_conservative_regridder(
    conservative_input_data,
    conservative_sample_grid,
    cdo_comparison_data,
    chunks,
):
    input_data = conservative_input_data.chunk(chunks)
    ds_regrid = input_data.regrid.conservative(
        conservative_sample_grid, latitude_coord="latitude"
    )
    ds_cdo = cdo_comparison_data["conservative"]

    xr.testing.assert_allclose(
        ds_regrid["tp"],
        ds_cdo["tp"],
        rtol=0.002,
        atol=4e-5,
    )


def test_conservative_nans(
    conservative_input_data, conservative_sample_grid, cdo_comparison_data
):
    ds = conservative_input_data
    ds["tp"] = ds["tp"].where(ds.latitude >= 0).where(ds.longitude < 180)
    ds_regrid = ds.regrid.conservative(
        conservative_sample_grid, latitude_coord="latitude"
    )
    ds_cdo = cdo_comparison_data["conservative"]

    no_nans = {"latitude": slice(1, 90), "longitude": slice(1, 179)}
    xr.testing.assert_allclose(
        ds_regrid["tp"].sel(no_nans),
        ds_cdo["tp"].sel(no_nans),
        rtol=0.002,
        atol=4e-5,
    )


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
def test_attrs_dataarray(sample_input_data, sample_grid_ds, method):
    regridder = getattr(sample_input_data["d2m"].regrid, method)
    da_regrid = regridder(sample_grid_ds)
    assert da_regrid.attrs == sample_input_data["d2m"].attrs
    assert da_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


def test_attrs_dataarray_conservative(sample_input_data, sample_grid_ds):
    da_regrid = sample_input_data["d2m"].regrid.conservative(
        sample_grid_ds, latitude_coord="latitude"
    )
    assert da_regrid.attrs == sample_input_data["d2m"].attrs
    assert da_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
def test_attrs_dataset(sample_input_data, sample_grid_ds, method):
    regridder = getattr(sample_input_data.regrid, method)
    ds_regrid = regridder(sample_grid_ds)
    assert ds_regrid.attrs == sample_input_data.attrs
    assert ds_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


def test_attrs_dataset_conservative(sample_input_data, sample_grid_ds):
    ds_regrid = sample_input_data.regrid.conservative(
        sample_grid_ds, latitude_coord="latitude"
    )
    assert ds_regrid.attrs == sample_input_data.attrs
    assert ds_regrid["d2m"].attrs == sample_input_data["d2m"].attrs
    assert ds_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


def test_conservative_nan_aggregation_over_dims():
    """Check the behavior of valid cell aggregation across multiple dimensions.
    If we correctly accumulate the NaN count across dims, this should output 1.666,
    vs 1.5 or 1.75 if we naively aggregate dimensions separately without accumulating
    the NaN count. Also checks the ability to handle a singleton target grid."""
    data = xr.DataArray([[1, np.nan], [2, 2]], coords={"x": [-1, 1], "y": [-1, 1]})
    # Add a time dim and mask off a single slice to make sure we keep all possible
    # valid points
    data = data.expand_dims(time=[0, 1, 2], axis=0)
    data = data.where(data.time < 2)
    target = xr.Dataset(coords={"x": [0], "y": [0]})

    result = data.regrid.conservative(target, skipna=True, nan_threshold=1)
    np.testing.assert_allclose(result[0].mean().item(), data[0].mean().item())


@pytest.mark.parametrize("nan_threshold", [0, 1])
def test_conservative_nan_thresholds_against_coarsen(nan_threshold):
    """Compare nan_threshold regridding to behavior of xarray coarsen, where coarsen
    with skipna=True should map to nan_threshold=1.0 and coarsen with skipna=False
    should map to nan_threshold=0.0 when the source grid evenly divides the target
    grid."""
    da = xr.DataArray(
        [
            [1, np.nan, np.nan, np.nan],
            [2, 2, np.nan, np.nan],
            [3, 3, 3, np.nan],
            [4, 4, 4, 4],
        ],
        coords={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        name="foo",
    )
    da_coarsen = da.coarsen(x=2, y=2).mean(skipna=bool(nan_threshold))
    target = da_coarsen.to_dataset()[["x", "y"]]
    da_regrid = da.regrid.conservative(target, skipna=True, nan_threshold=nan_threshold)

    xr.testing.assert_allclose(da_coarsen, da_regrid)


def test_conservative_output_chunks():
    data = xr.DataArray(
        np.ones((8, 8)),
        coords={"x": np.linspace(0, 1, 8), "y": np.linspace(0, 1, 8)},
        dims=("x", "y"),
    )
    target = xr.Dataset(coords={"x": np.linspace(0, 1, 4), "y": np.linspace(0, 1, 4)})

    # Non-dask input should return non-dask output
    result = data.regrid.conservative(target)
    assert result.chunks is None

    # Dask input with unspecified output chunks should match the input chunks
    result = data.chunk({"x": 2, "y": 2}).regrid.conservative(target)
    assert result.chunks == ((2, 2), (2, 2))

    # Specified output chunks should be respected
    result = data.chunk({"x": 4, "y": 4}).regrid.conservative(
        target, output_chunks={"x": 2, "y": 2}
    )
    assert result.chunks == ((2, 2), (2, 2))


@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_conservative_nan_thresholds_against_xesmf():
    ds = xr.tutorial.open_dataset("ersstv5").sst.isel(time=[0]).compute()
    ds = ds.rename(lon="longitude", lat="latitude")
    new_grid = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=2,
        resolution_lon=2,
    )
    target_dataset = xarray_regrid.create_regridding_dataset(new_grid)
    regridder = xesmf.Regridder(ds, target_dataset, "conservative")

    for nan_threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
        data_regrid = ds.regrid.conservative(
            target_dataset, skipna=True, nan_threshold=nan_threshold
        )
        data_esmf = regridder(ds, keep_attrs=True, na_thres=nan_threshold, skipna=True)
        xr.testing.assert_equal(data_regrid.isnull(), data_esmf.isnull())


class TestCoordOrder:
    @pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_original(self, sample_input_data, sample_grid_ds, method, dataarray):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        regridder = getattr(input_data.regrid, method)
        ds_regrid = regridder(sample_grid_ds)
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("coord", ["latitude", "longitude"])
    @pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_reversed(
        self, sample_input_data, sample_grid_ds, method, coord, dataarray
    ):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        regridder = getattr(input_data.regrid, method)
        sample_grid_ds[coord] = list(reversed(sample_grid_ds[coord]))
        ds_regrid = regridder(sample_grid_ds)
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("dataarray", [True, False])
    def test_conservative_original(self, sample_input_data, sample_grid_ds, dataarray):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        ds_regrid = input_data.regrid.conservative(
            sample_grid_ds, latitude_coord="latitude"
        )
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("coord", ["latitude", "longitude"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_conservative_reversed(
        self, sample_input_data, sample_grid_ds, coord, dataarray
    ):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        sample_grid_ds[coord] = list(reversed(sample_grid_ds[coord]))
        ds_regrid = input_data.regrid.conservative(
            sample_grid_ds, latitude_coord="latitude"
        )
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])


# --- factored cubic path (interp.py) --------------------------------------


def _cubic_grid(ny, nx):
    """Cell-centered lat/lon grid, used as source and target in the tests."""
    lat = np.linspace(-90, 90, ny, endpoint=False) + 90 / ny
    lon = np.linspace(-180, 180, nx, endpoint=False) + 180 / nx
    return lat, lon


def test_cubic_factored_matches_nd_path():
    """On rectilinear grids where target is inside source, the factored
    cubic path and the N-D cubic path agree to noise level (relRMS ~1e-5)."""
    src_lat, src_lon = _cubic_grid(60, 120)
    tgt_lat, tgt_lon = _cubic_grid(30, 60)
    rng = np.random.default_rng(0)
    src = xr.DataArray(
        rng.standard_normal((60, 120)),
        dims=("latitude", "longitude"),
        coords={"latitude": src_lat, "longitude": src_lon},
    )
    tgt_ds = xr.Dataset(coords={"latitude": tgt_lat, "longitude": tgt_lon})

    # Factored (gate passes because 30->60 cell-centered at coarser step is
    # strictly inside the finer-step source)
    out_fac = _cubic_interp.interp_regrid(src, tgt_ds, "cubic")

    # N-D (force fallback); silence the expected fallback warning since
    # we're deliberately triggering it to compare outputs.
    saved = _cubic_interp._target_within_source
    _cubic_interp._target_within_source = lambda *_args, **_kw: False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out_nd = _cubic_interp.interp_regrid(src, tgt_ds, "cubic")
    finally:
        _cubic_interp._target_within_source = saved

    diff = np.asarray(out_fac.values - out_nd.values)
    rms = float(np.sqrt(np.nanmean(diff**2)))
    rel = rms / float(np.sqrt(np.nanmean(np.asarray(out_nd.values) ** 2)))
    assert rel < 1e-4, f"factored and N-D diverged: relRMS={rel:.3e}"


def test_cubic_fallback_warns_when_target_exceeds_source():
    """Factored path cannot handle out-of-bounds target points (scipy's
    1D cubic would propagate NaN across the batch). We fall back to the
    N-D path and emit a RuntimeWarning so callers notice the slowdown."""
    # Source centers [-87, 87]; target centers [-89.25, 89.25] extend beyond.
    src_lat, src_lon = _cubic_grid(30, 60)
    tgt_lat, tgt_lon = _cubic_grid(120, 240)
    rng = np.random.default_rng(0)
    src = xr.DataArray(
        rng.standard_normal((30, 60)),
        dims=("latitude", "longitude"),
        coords={"latitude": src_lat, "longitude": src_lon},
    )
    tgt_ds = xr.Dataset(coords={"latitude": tgt_lat, "longitude": tgt_lon})

    with pytest.warns(RuntimeWarning, match="scipy N-D"):
        out = _cubic_interp.interp_regrid(src, tgt_ds, "cubic")
    # Output should be finite except at the out-of-bounds boundary cells —
    # most of the interior interpolates successfully via the N-D fallback.
    assert np.isfinite(out.values).sum() > 0.9 * out.size


def test_cubic_factored_tolerates_tiny_float_drift():
    """When target bounds equal source bounds up to floating-point rounding,
    we should still take the factored path instead of falling back."""
    src_lat, src_lon = _cubic_grid(60, 120)
    # Target bounds are the same centers but after a float round-trip.
    tgt_lat = src_lat.astype(np.float32).astype(np.float64)
    tgt_lon = src_lon.astype(np.float32).astype(np.float64)
    rng = np.random.default_rng(0)
    src = xr.DataArray(
        rng.standard_normal((60, 120)),
        dims=("latitude", "longitude"),
        coords={"latitude": src_lat, "longitude": src_lon},
    )
    tgt_ds = xr.Dataset(coords={"latitude": tgt_lat, "longitude": tgt_lon})

    # Turn the fallback warning into an error so we notice if the tolerance
    # stops absorbing float32 round-trip drift.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cubic_interp.interp_regrid(src, tgt_ds, "cubic")
