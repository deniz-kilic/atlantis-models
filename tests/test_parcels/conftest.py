"""Pytest fixtures for parcel tests."""

import numpy as np
import pytest
import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon, box
from pyproj import CRS

from atmod.base import Raster, VoxelModel


# Create CRS object for Dutch RD New
RD_NEW_CRS = CRS.from_proj4(
    "+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 "
    "+k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel "
    "+towgs84=565.417,50.3319,465.552,-0.398957,0.343988,-1.8774,4.0725 "
    "+units=m +no_defs"
)


@pytest.fixture
def sample_parcels_gdf():
    """Create sample parcel GeoDataFrame with 5 parcels."""
    # Create 5 square parcels in a row
    parcels = []
    for i in range(5):
        x_start = 100 + i * 100
        parcels.append(
            box(x_start, 100, x_start + 80, 180)
        )

    gdf = gpd.GeoDataFrame(
        {
            "Perceel_ID": [f"P{i+1}" for i in range(5)],
            "area_m2": [80 * 80] * 5,
            "soil_type": ["clay", "peat", "sand", "clay", "peat"],
        },
        geometry=parcels,
        crs=RD_NEW_CRS,
    )
    return gdf


@pytest.fixture
def sample_parcels_numeric_id(sample_parcels_gdf):
    """Sample parcels with numeric IDs."""
    gdf = sample_parcels_gdf.copy()
    gdf["parcel_num"] = [101, 102, 103, 104, 105]
    return gdf


@pytest.fixture
def sample_raster():
    """Create sample 2D raster covering the parcel area."""
    # Create 10x10 grid covering 100-600 x 50-250
    x = np.arange(105, 600, 50)  # cell centers
    y = np.arange(225, 50, -50)  # cell centers (descending)

    # Create data - simple gradient
    data = np.zeros((len(y), len(x)))
    for i, yi in enumerate(y):
        for j, xi in enumerate(x):
            data[i, j] = -1.0 + (xi - 100) / 1000  # slight gradient

    da = xr.DataArray(
        data,
        coords={"y": y, "x": x},
        dims=["y", "x"],
    )

    return Raster(da, cellsize=50, crs=RD_NEW_CRS)


@pytest.fixture
def sample_voxelmodel():
    """Create sample 3D voxel model covering the parcel area."""
    # Create grid
    x = np.arange(105, 600, 50)
    y = np.arange(225, 50, -50)
    z = np.arange(-5.25, 0.5, 0.5)  # 12 layers from -5 to 0

    nx, ny, nz = len(x), len(y), len(z)

    # Create data arrays
    lithology = np.random.choice([1, 2, 3, 4, 5], size=(ny, nx, nz))
    geology = np.ones((ny, nx, nz))  # All Holocene
    geology[:, :, nz//2:] = 2  # Bottom half is older
    thickness = np.full((ny, nx, nz), 0.5)

    # Create probability distributions (kans_1 to kans_9)
    kans = {}
    for i in range(1, 10):
        kans[f"kans_{i}"] = np.random.rand(ny, nx, nz) * 0.2
    # Normalize so probabilities sum to ~1
    kans_sum = sum(kans.values())
    for i in range(1, 10):
        kans[f"kans_{i}"] = kans[f"kans_{i}"] / kans_sum

    # Make lithology 2 (clay) dominant in some areas
    kans["kans_2"][:, :2, :] = 0.6

    data_vars = {
        "lithology": (["y", "x", "z"], lithology.astype(float)),
        "geology": (["y", "x", "z"], geology),
        "thickness": (["y", "x", "z"], thickness),
        "mass_fraction_organic": (["y", "x", "z"], np.full((ny, nx, nz), 0.3)),
        "mass_fraction_lutum": (["y", "x", "z"], np.full((ny, nx, nz), 0.4)),
    }
    data_vars.update({k: (["y", "x", "z"], v) for k, v in kans.items()})

    ds = xr.Dataset(
        data_vars,
        coords={"y": y, "x": x, "z": z},
    )

    return VoxelModel(ds, cellsize=50, dz=0.5, epsg=28992)


@pytest.fixture
def sample_ahn_raster(sample_raster):
    """Sample AHN (surface elevation) raster."""
    # Modify raster to have typical AHN values
    raster = sample_raster
    # Surface level around 0 m NAP
    raster.ds.values[:] = np.random.uniform(-0.5, 0.5, raster.ds.shape)
    return raster


@pytest.fixture
def sample_glg_raster(sample_raster):
    """Sample GLG (groundwater level) raster."""
    raster = sample_raster
    # GLG typically 0.5-1.5m below surface
    raster.ds.values[:] = raster.ds.values - np.random.uniform(0.5, 1.5, raster.ds.shape)
    return raster


@pytest.fixture
def single_cell_parcel():
    """Create a parcel centered exactly on a grid cell."""
    # Cell centered at x=155, y=175 (inside the grid)
    geom = box(130, 150, 180, 200)  # 50x50 parcel

    gdf = gpd.GeoDataFrame(
        {"parcel_id": ["SINGLE"]},
        geometry=[geom],
        crs=RD_NEW_CRS,
    )
    return gdf


@pytest.fixture
def large_parcel():
    """Create a large parcel spanning multiple grid cells."""
    geom = box(100, 100, 400, 200)  # 300x100 parcel

    gdf = gpd.GeoDataFrame(
        {"parcel_id": ["LARGE"]},
        geometry=[geom],
        crs=RD_NEW_CRS,
    )
    return gdf


@pytest.fixture
def tiny_parcel():
    """Create a tiny parcel smaller than a grid cell."""
    geom = box(150, 170, 160, 180)  # 10x10 parcel

    gdf = gpd.GeoDataFrame(
        {"parcel_id": ["TINY"]},
        geometry=[geom],
        crs=RD_NEW_CRS,
    )
    return gdf


@pytest.fixture
def outside_bounds_parcel():
    """Create a parcel outside the grid bounds."""
    geom = box(1000, 1000, 1100, 1100)

    gdf = gpd.GeoDataFrame(
        {"parcel_id": ["OUTSIDE"]},
        geometry=[geom],
        crs=RD_NEW_CRS,
    )
    return gdf


@pytest.fixture
def mock_atlans_parcel_output(sample_parcels_gdf):
    """
    Create mock Atlans.jl output in virtual grid format (x=N, y=1, time).
    Used to test result remapping.
    """
    n_parcels = len(sample_parcels_gdf)
    n_times = 10

    # Create synthetic subsidence results (increasing over time)
    subsidence = np.zeros((n_parcels, 1, n_times), dtype=np.float32)
    for t in range(n_times):
        # Each parcel subsides differently
        for p in range(n_parcels):
            subsidence[p, 0, t] = -0.01 * (t + 1) * (1 + p * 0.1)

    return xr.Dataset(
        {
            "subsidence": (["x", "y", "time"], subsidence),
            "consolidation": (["x", "y", "time"], subsidence * 0.6),
            "oxidation": (["x", "y", "time"], subsidence * 0.3),
            "shrinkage": (["x", "y", "time"], subsidence * 0.1),
        },
        coords={
            "x": np.arange(1, n_parcels + 1, dtype=float),
            "y": np.array([0.0]),
            "time": np.arange(n_times),
        },
    )


@pytest.fixture
def sample_parcels_wgs84():
    """Create sample parcels in WGS84 (EPSG:4326) for CRS transformation testing."""
    # Approximate WGS84 coordinates for the Netherlands
    parcels = []
    for i in range(3):
        # Small parcels in WGS84
        lon_start = 4.5 + i * 0.001  # ~100m spacing
        lat_start = 52.0
        parcels.append(
            box(lon_start, lat_start, lon_start + 0.0008, lat_start + 0.0008)
        )

    gdf = gpd.GeoDataFrame(
        {"parcel_id": [f"WGS_{i+1}" for i in range(3)]},
        geometry=parcels,
        crs="EPSG:4326",
    )
    return gdf


@pytest.fixture
def extraction_result(sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster):
    """Pre-computed extraction result for virtual grid tests."""
    from atmod.parcels.data import ParcelData
    from atmod.parcels.extractor import ParcelExtractor

    parcel_data = ParcelData.from_geodataframe(sample_parcels_gdf, "Perceel_ID")
    extractor = ParcelExtractor(parcel_data, method="centroid")

    return extractor.extract(
        voxelmodel=sample_voxelmodel,
        rasters={"surface_level": sample_ahn_raster},
        variables_3d=["lithology", "geology", "thickness"],
    )
