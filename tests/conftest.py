import numpy as np
import pytest
import xarray as xr

from atmod.bro_models.geology import StratGeoTop
from atmod.bro_models.voxelmodels import GeoTop

# --- Existing fixtures ---

@pytest.fixture
def test_subsurface_model():
    nlayers = 4
    layers = np.arange(nlayers) + 1
    x = [150, 250, 350]
    y = [350, 250]

    nx, ny = len(x), len(y)

    data = dict(
        lithology=(["layer", "y", "x"], np.full((nlayers, ny, nx), 5)),
        thickness=(["layer", "y", "x"], np.full((nlayers, ny, nx), 0.5)),
        mass_fraction_organic=(["layer", "y", "x"], np.full((nlayers, ny, nx), 0.5)),
        geology=(["layer", "y", "x"], np.full((nlayers, ny, nx), 1)),
        rho_bulk=(["layer", "y", "x"], np.full((nlayers, ny, nx), 833.0)),
        mass_fraction_lutum=(["layer", "y", "x"], np.full((nlayers, ny, nx), 0.5)),
        shrinkage_degree=(["layer", "y", "x"], np.full((nlayers, ny, nx), 0.7)),
        surface_level=(["y", "x"], np.full((ny, nx), 0)),
        phreatic_level=(["y", "x"], np.full((ny, nx), -0.75)),
        zbase=(["y", "x"], np.full((ny, nx), -5)),
        domainbase=(["y", "x"], np.full((ny, nx), -2)),
        no_oxidation_thickness=(["y", "x"], np.full((ny, nx), 0.0)),
        no_shrinkage_thickness=(["y", "x"], np.full((ny, nx), 0.0)),
        max_oxidation_depth=(["y", "x"], np.full((ny, nx), -1.2)),
    )

    coords = dict(
        layer=(["layer"], layers),
        y=(["y"], y),
        x=(["x"], x),
    )

    return xr.Dataset(data, coords)


# --- Uncertainty / kans testing fixtures ---

@pytest.fixture
def mock_geotop_with_kans():
    """
    Create synthetic GeoTop with controlled kans probability distributions.

    Structure:
    - 10x10x5 voxels (nx, ny, nz)
    - Top 2 layers: Holocene (strat from HOLOCENE dict)
    - Bottom 3 layers: Older (strat from OLDER dict)
    - kans_1-9 with known probabilities for testing

    The lithok values are set to match the most likely kans class.
    """
    nx, ny, nz = 10, 10, 5
    x = np.arange(100, 100 + nx * 100, 100) + 50  # cell centers
    y = np.arange(100, 100 + ny * 100, 100) + 50
    z = np.arange(-2.25, -2.25 - nz * 0.5, -0.5)  # 0.5m layers

    # Create stratigraphy: top 2 layers Holocene, bottom 3 Older
    strat = np.zeros((nz, ny, nx), dtype=np.int32)
    holocene_code = list(StratGeoTop.holocene.values)[0]  # e.g., 1010 (NIGR)
    older_code = list(StratGeoTop.older.values)[0]  # e.g., 3000 (BXKO)

    strat[:2, :, :] = holocene_code  # top 2 layers = Holocene
    strat[2:, :, :] = older_code     # bottom 3 layers = Older

    # Create deterministic lithology (most likely class)
    lithok = np.full((nz, ny, nx), 5, dtype=np.int8)  # default: fine sand (5)
    lithok[:2, :, :] = 2  # Holocene: clay (2)

    # Create kans distributions
    # Holocene: 80% clay (kans_2), 15% loam (kans_3), 5% organic (kans_1)
    # Older: 90% fine sand (kans_5), 10% medium sand (kans_6)
    kans = {}
    for i in range(1, 10):
        kans[f'kans_{i}'] = np.zeros((nz, ny, nx), dtype=np.float32)

    # Holocene layers (top 2): clay-dominated
    kans['kans_1'][:2, :, :] = 5.0   # organic
    kans['kans_2'][:2, :, :] = 80.0  # clay
    kans['kans_3'][:2, :, :] = 15.0  # loam

    # Older layers (bottom 3): sand-dominated
    kans['kans_5'][2:, :, :] = 90.0  # fine sand
    kans['kans_6'][2:, :, :] = 10.0  # medium sand

    # Build dataset
    ds = xr.Dataset(
        {
            'strat': (['z', 'y', 'x'], strat),
            'lithok': (['z', 'y', 'x'], lithok),
            **{k: (['z', 'y', 'x'], v) for k, v in kans.items()},
        },
        coords={
            'x': x,
            'y': y,
            'z': z,
        }
    )

    return GeoTop(ds, cellsize=100, dz=0.5, epsg=28992)


@pytest.fixture
def mock_geotop_without_kans():
    """
    Create synthetic GeoTop with only strat and lithok (no kans variables).
    Used to test backward compatibility and error handling.
    """
    nx, ny, nz = 5, 5, 3
    x = np.arange(100, 100 + nx * 100, 100) + 50
    y = np.arange(100, 100 + ny * 100, 100) + 50
    z = np.arange(-0.25, -0.25 - nz * 0.5, -0.5)

    strat = np.full((nz, ny, nx), 1010, dtype=np.int32)  # All Holocene
    lithok = np.full((nz, ny, nx), 2, dtype=np.int8)     # All clay

    ds = xr.Dataset(
        {
            'strat': (['z', 'y', 'x'], strat),
            'lithok': (['z', 'y', 'x'], lithok),
        },
        coords={'x': x, 'y': y, 'z': z}
    )

    return GeoTop(ds, cellsize=100, dz=0.5, epsg=28992)


@pytest.fixture
def mock_geotop_uniform_kans():
    """
    GeoTop with uniform probability distribution (maximum uncertainty).
    Each lithology class has ~11.1% probability.
    Used to test entropy calculations and edge cases.
    """
    nx, ny, nz = 5, 5, 3
    x = np.arange(100, 100 + nx * 100, 100) + 50
    y = np.arange(100, 100 + ny * 100, 100) + 50
    z = np.arange(-0.25, -0.25 - nz * 0.5, -0.5)

    strat = np.full((nz, ny, nx), 1010, dtype=np.int32)
    lithok = np.full((nz, ny, nx), 1, dtype=np.int8)

    # Uniform distribution: each class gets 100/9 ≈ 11.1%
    uniform_prob = 100.0 / 9.0
    kans = {f'kans_{i}': np.full((nz, ny, nx), uniform_prob, dtype=np.float32)
            for i in range(1, 10)}

    ds = xr.Dataset(
        {
            'strat': (['z', 'y', 'x'], strat),
            'lithok': (['z', 'y', 'x'], lithok),
            **{k: (['z', 'y', 'x'], v) for k, v in kans.items()},
        },
        coords={'x': x, 'y': y, 'z': z}
    )

    return GeoTop(ds, cellsize=100, dz=0.5, epsg=28992)


@pytest.fixture
def mock_geotop_deterministic_kans():
    """
    GeoTop with 100% probability for one class (no uncertainty).
    kans_2 (clay) = 100%, all others = 0%.
    Used to test deterministic edge case.
    """
    nx, ny, nz = 5, 5, 3
    x = np.arange(100, 100 + nx * 100, 100) + 50
    y = np.arange(100, 100 + ny * 100, 100) + 50
    z = np.arange(-0.25, -0.25 - nz * 0.5, -0.5)

    strat = np.full((nz, ny, nx), 1010, dtype=np.int32)
    lithok = np.full((nz, ny, nx), 2, dtype=np.int8)  # clay

    kans = {f'kans_{i}': np.zeros((nz, ny, nx), dtype=np.float32)
            for i in range(1, 10)}
    kans['kans_2'][:] = 100.0  # 100% clay

    ds = xr.Dataset(
        {
            'strat': (['z', 'y', 'x'], strat),
            'lithok': (['z', 'y', 'x'], lithok),
            **{k: (['z', 'y', 'x'], v) for k, v in kans.items()},
        },
        coords={'x': x, 'y': y, 'z': z}
    )

    return GeoTop(ds, cellsize=100, dz=0.5, epsg=28992)


@pytest.fixture
def mock_geotop_with_nan():
    """
    GeoTop with NaN values simulating areas outside model domain.
    Some voxels have NaN in kans and lithok (edge of Netherlands).
    """
    nx, ny, nz = 5, 5, 3
    x = np.arange(100, 100 + nx * 100, 100) + 50
    y = np.arange(100, 100 + ny * 100, 100) + 50
    z = np.arange(-0.25, -0.25 - nz * 0.5, -0.5)

    strat = np.full((nz, ny, nx), 1010, dtype=np.int32)
    lithok = np.full((nz, ny, nx), 2.0, dtype=np.float32)
    lithok[:, :, -1] = np.nan  # last column is NaN (outside domain)

    kans = {}
    for i in range(1, 10):
        k = np.zeros((nz, ny, nx), dtype=np.float32)
        if i == 2:
            k[:] = 100.0  # clay
        k[:, :, -1] = np.nan  # NaN at edge
        kans[f'kans_{i}'] = k

    ds = xr.Dataset(
        {
            'strat': (['z', 'y', 'x'], strat),
            'lithok': (['z', 'y', 'x'], lithok),
            **{k: (['z', 'y', 'x'], v) for k, v in kans.items()},
        },
        coords={'x': x, 'y': y, 'z': z}
    )

    return GeoTop(ds, cellsize=100, dz=0.5, epsg=28992)


# --- Mock AHN and data fixtures for build pipeline testing ---

@pytest.fixture
def mock_ahn(mock_geotop_with_kans):
    """
    Create synthetic AHN raster matching mock_geotop spatial extent.
    Surface elevation at ~1m NAP (typical Dutch polder).
    """
    from atmod.base import Raster

    x = mock_geotop_with_kans.ds.x.values
    y = mock_geotop_with_kans.ds.y.values

    # Create surface elevation data (1-2m NAP, typical low-lying area)
    xx, yy = np.meshgrid(x, y)
    elevation = 1.0 + 0.5 * np.sin(xx / 500) + 0.3 * np.cos(yy / 500)

    da = xr.DataArray(
        elevation.astype(np.float32),
        dims=['y', 'x'],
        coords={'x': x, 'y': y},
        name='elevation'
    )
    da = da.rio.write_crs(28992)

    return Raster(da, cellsize=100, crs=28992)


@pytest.fixture
def mock_atlans_output():
    """
    Create mock Atlans.jl output with subsidence results.
    Format: (x, y, time) with standard output variables.
    Used to test result remapping.
    """
    nx, ny = 10, 1
    nt = 5
    x = np.arange(100, 100 + nx * 100, 100) + 50
    y = np.array([150.0])
    time = np.arange(nt)

    # Create synthetic subsidence results (increasing over time)
    subsidence = np.zeros((nx, ny, nt), dtype=np.float32)
    for t in range(nt):
        subsidence[:, :, t] = -0.01 * (t + 1)  # -1cm per timestep

    ds = xr.Dataset(
        {
            'subsidence': (['x', 'y', 'time'], subsidence),
            'consolidation': (['x', 'y', 'time'], subsidence * 0.6),
            'oxidation': (['x', 'y', 'time'], subsidence * 0.3),
            'shrinkage': (['x', 'y', 'time'], subsidence * 0.1),
        },
        coords={'x': x, 'y': y, 'time': time}
    )

    return ds


@pytest.fixture
def mock_complete_atlantis_model(mock_geotop_with_kans):
    """
    Create a complete Atlantis model dataset for testing build pipelines.
    Contains all required variables for analysis.
    """
    from atmod.analysis_tools import SOURCE_BODEMKAART, SOURCE_GEOTOP

    x = mock_geotop_with_kans.ds.x.values
    y = mock_geotop_with_kans.ds.y.values
    z = mock_geotop_with_kans.ds.z.values

    nx, ny, nz = len(x), len(y), len(z)
    shape = (ny, nx, nz)

    # Create standard model variables
    thickness = np.full(shape, 0.5, dtype=np.float32)
    lithology = np.full(shape, 2.0, dtype=np.float32)  # clay
    geology = np.full(shape, 1, dtype=np.int32)  # Holocene
    mass_organic = np.full(shape, 0.1, dtype=np.float32)
    rho_bulk = np.full(shape, 833.0, dtype=np.float32)
    mass_lutum = np.full(shape, 0.3, dtype=np.float32)
    shrinkage_degree = np.full(shape, 0.7, dtype=np.float32)

    # Data source tracking
    data_source = np.full(shape, SOURCE_GEOTOP, dtype=np.int8)
    data_source[:, :, -1] = SOURCE_BODEMKAART  # top layer from bodemkaart

    # 2D variables
    surface_level = np.full((ny, nx), 1.0, dtype=np.float32)
    phreatic_level = np.full((ny, nx), -0.5, dtype=np.float32)

    return xr.Dataset(
        {
            'thickness': (['y', 'x', 'z'], thickness),
            'lithology': (['y', 'x', 'z'], lithology),
            'geology': (['y', 'x', 'z'], geology),
            'mass_fraction_organic': (['y', 'x', 'z'], mass_organic),
            'rho_bulk': (['y', 'x', 'z'], rho_bulk),
            'mass_fraction_lutum': (['y', 'x', 'z'], mass_lutum),
            'shrinkage_degree': (['y', 'x', 'z'], shrinkage_degree),
            'data_source': (['y', 'x', 'z'], data_source),
            'surface_level': (['y', 'x'], surface_level),
            'phreatic_level': (['y', 'x'], phreatic_level),
        },
        coords={'x': x, 'y': y, 'z': z}
    )
