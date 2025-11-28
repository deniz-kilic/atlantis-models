import numpy as np
import pytest
import xarray as xr

from atmod.bro_models.voxelmodels import GeoTop
from atmod.bro_models.geology import StratGeoTop


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
