"""
Lithology uncertainty quantification for GeoTOP.

This module provides functions for sampling lithology from GeoTOP probability
distributions (kans_1-9) to enable Monte Carlo uncertainty quantification of
subsidence predictions.

The main workflow is:
1. Load GeoTOP with kans data: GeoTop.from_opendap(..., include_kans=True)
2. Sample lithology: sampled = sample_lithology_from_kans(geotop, seed=42)
3. Create realization: realization = create_geotop_realization(geotop, sampled)
4. Build model with realization: build_atlantis_model(geotop=realization, ...)

For ensemble generation:
    ensemble = generate_lithology_ensemble(geotop, n_realizations=100)
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr

from atmod.bro_models.geology import StratGeoTop

if TYPE_CHECKING:
    from atmod.bro_models.voxelmodels import GeoTop


def get_holocene_units() -> np.ndarray:
    """
    Get array of all GeoTOP stratigraphic unit codes considered Holocene.

    This includes:
    - Holocene units (NASC, NAWA, etc.)
    - Channel belts (AEC, BEC, etc.) - compressible Holocene sediments
    - Anthropogenic units (AAOP, AAES) - recent deposits

    Returns
    -------
    np.ndarray
        Array of integer stratigraphic codes for Holocene units.
    """
    return np.concatenate([
        StratGeoTop.holocene.values,
        StratGeoTop.channel_belts.values,
        StratGeoTop.anthropogenic.values,
    ])


HOLOCENE_UNITS = get_holocene_units()


def sample_lithology_from_kans(
    geotop: "GeoTop",
    seed: Optional[int] = None,
    holocene_only: bool = True
) -> xr.DataArray:
    """
    Sample lithology class from GeoTOP kans probability distributions.

    Uses the kans_1 through kans_9 probability variables to draw a random
    lithology class for each voxel. The kans values represent the probability
    (as percentages summing to 100) of each lithology class.

    Parameters
    ----------
    geotop : GeoTop
        GeoTOP object with kans_1-9 loaded (via include_kans=True).
    seed : int, optional
        Random seed for reproducibility. If None, results are random.
    holocene_only : bool, default True
        If True, only sample Holocene voxels (including channel belts and
        anthropogenic units). Non-Holocene voxels keep their deterministic
        lithok values. This is recommended for subsidence uncertainty since
        older materials have less variability impact.

    Returns
    -------
    xr.DataArray
        Sampled lithology array with same shape and coordinates as lithok.
        Values are integers 1-9 representing lithology classes.

    Raises
    ------
    ValueError
        If geotop does not have kans data loaded.

    Examples
    --------
    >>> geotop = GeoTop.from_opendap(url, bbox, include_kans=True)
    >>> sampled = sample_lithology_from_kans(geotop, seed=42)
    >>> sampled.values  # array of integers 1-9
    """
    if not geotop.has_kans:
        raise ValueError(
            "GeoTop object does not have kans probability data. "
            "Load with include_kans=True."
        )

    if seed is not None:
        np.random.seed(seed)

    # Stack kans into (9, z, y, x) array - one layer per lithology class
    kans = np.stack(
        [geotop.ds[f'kans_{i}'].values for i in range(1, 10)],
        axis=0
    )

    # Handle NaN values (outside model domain) - set to 0 probability
    kans = np.nan_to_num(kans, nan=0.0)

    # Normalize to proper probabilities (kans are percentages, may not sum to 100)
    kans_sum = kans.sum(axis=0, keepdims=True)
    # Avoid division by zero where all kans are 0/NaN
    kans_norm = np.divide(
        kans, kans_sum,
        where=kans_sum > 0,
        out=np.zeros_like(kans)
    )

    # Vectorized sampling using cumulative probability approach
    # This is more efficient than looping over voxels
    shape = kans.shape[1:]  # (z, y, x)
    n_voxels = np.prod(shape)

    # Reshape to (9, n_voxels) for vectorized operations
    kans_flat = kans_norm.reshape(9, n_voxels)

    # Compute cumulative probabilities along class axis
    cum_probs = np.cumsum(kans_flat, axis=0)

    # Draw uniform random values for each voxel
    draws = np.random.rand(n_voxels)

    # Find first class where cumulative probability exceeds random draw
    # argmax returns first True, +1 to convert 0-indexed to 1-indexed lithology
    sampled_flat = (cum_probs > draws).argmax(axis=0) + 1
    sampled = sampled_flat.reshape(shape)

    # Apply Holocene mask if requested
    if holocene_only:
        strat = geotop.ds['strat'].values
        holocene_mask = np.isin(strat, HOLOCENE_UNITS)
        lithok = geotop.ds['lithok'].values

        # Use sampled values for Holocene, original lithok for older materials
        result = np.where(holocene_mask, sampled, lithok)
    else:
        result = sampled

    # Create DataArray with same metadata as original lithok
    return xr.DataArray(
        result,
        dims=geotop.ds['lithok'].dims,
        coords=geotop.ds['lithok'].coords,
        name='sampled_lithology',
        attrs={'description': 'Lithology sampled from kans probability distribution'}
    )


def generate_lithology_ensemble(
    geotop: "GeoTop",
    n_realizations: int = 100,
    holocene_only: bool = True,
    base_seed: int = 42
) -> xr.DataArray:
    """
    Generate ensemble of lithology realizations from GeoTOP kans distributions.

    Creates N independent samples from the probability distributions, suitable
    for Monte Carlo uncertainty analysis.

    Parameters
    ----------
    geotop : GeoTop
        GeoTOP object with kans_1-9 loaded.
    n_realizations : int, default 100
        Number of lithology realizations to generate.
    holocene_only : bool, default True
        If True, only sample Holocene materials; keep older as deterministic.
    base_seed : int, default 42
        Base random seed. Each realization uses seed = base_seed + i.

    Returns
    -------
    xr.DataArray
        Array with shape (n_realizations, z, y, x) containing sampled lithology.
        First dimension is 'realization' with values 0 to n_realizations-1.

    Examples
    --------
    >>> ensemble = generate_lithology_ensemble(geotop, n_realizations=50)
    >>> ensemble.shape  # (50, nz, ny, nx)
    >>> realization_5 = ensemble.isel(realization=5)
    """
    realizations = []

    for i in range(n_realizations):
        sampled = sample_lithology_from_kans(
            geotop,
            seed=base_seed + i,
            holocene_only=holocene_only
        )
        realizations.append(sampled.values)

    # Stack into single array with realization as first dimension
    ensemble = np.stack(realizations, axis=0)

    return xr.DataArray(
        ensemble,
        dims=['realization', *geotop.ds['lithok'].dims],
        coords={
            'realization': np.arange(n_realizations),
            **{k: v for k, v in geotop.ds['lithok'].coords.items()}
        },
        name='lithology_ensemble',
        attrs={
            'description': 'Ensemble of lithology realizations from kans sampling',
            'n_realizations': n_realizations,
            'base_seed': base_seed,
            'holocene_only': holocene_only
        }
    )


def create_geotop_realization(
    geotop: "GeoTop",
    sampled_lithology: xr.DataArray
) -> "GeoTop":
    """
    Create a GeoTop object with sampled lithology replacing the deterministic lithok.

    This allows using the existing build pipeline (build_atlantis_model) with
    sampled lithologies for uncertainty quantification.

    Parameters
    ----------
    geotop : GeoTop
        Original GeoTop object (provides structure and other variables).
    sampled_lithology : xr.DataArray
        Sampled lithology array from sample_lithology_from_kans().

    Returns
    -------
    GeoTop
        New GeoTop instance with lithok replaced by sampled values.
        Stratigraphy and other variables are preserved.

    Examples
    --------
    >>> sampled = sample_lithology_from_kans(geotop, seed=42)
    >>> realization = create_geotop_realization(geotop, sampled)
    >>> model = build_atlantis_model(geotop=realization, ...)
    """
    # Import here to avoid circular imports
    from atmod.bro_models.voxelmodels import GeoTop as GeoTopClass

    # Create a copy of the dataset
    new_ds = geotop.ds.copy(deep=True)

    # Replace lithok with sampled values
    new_ds['lithok'] = sampled_lithology.rename('lithok')

    # Return new GeoTop instance with same properties
    return GeoTopClass(new_ds, geotop.cellsize, geotop.dz, geotop.crs)


def compute_lithology_entropy(geotop: "GeoTop") -> xr.DataArray:
    """
    Compute Shannon entropy of lithology probability distributions.

    High entropy indicates high uncertainty (probabilities spread across classes).
    Low entropy indicates low uncertainty (one class dominates).

    Parameters
    ----------
    geotop : GeoTop
        GeoTOP object with kans_1-9 loaded.

    Returns
    -------
    xr.DataArray
        Entropy values for each voxel. Higher values = more uncertainty.
        Maximum possible entropy is log(9) ≈ 2.2 (uniform distribution).

    Examples
    --------
    >>> entropy = compute_lithology_entropy(geotop)
    >>> high_uncertainty = entropy > 1.5  # mask of uncertain voxels
    """
    if not geotop.has_kans:
        raise ValueError("GeoTop object does not have kans probability data.")

    # Stack and normalize probabilities
    kans = np.stack(
        [geotop.ds[f'kans_{i}'].values for i in range(1, 10)],
        axis=0
    )
    kans = np.nan_to_num(kans, nan=0.0)
    kans_sum = kans.sum(axis=0, keepdims=True)
    probs = np.divide(
        kans, kans_sum,
        where=kans_sum > 0,
        out=np.zeros_like(kans)
    )

    # Compute entropy: -sum(p * log(p)), handling p=0
    with np.errstate(divide='ignore', invalid='ignore'):
        log_probs = np.log(probs)
        log_probs = np.where(probs > 0, log_probs, 0)  # 0 * log(0) = 0
        entropy = -np.sum(probs * log_probs, axis=0)

    return xr.DataArray(
        entropy,
        dims=geotop.ds['lithok'].dims,
        coords=geotop.ds['lithok'].coords,
        name='lithology_entropy',
        attrs={
            'description': 'Shannon entropy of lithology probability distribution',
            'units': 'nats'
        }
    )


def compute_mode_probability(geotop: "GeoTop") -> xr.DataArray:
    """
    Compute the probability of the most likely lithology class at each voxel.

    Low mode probability indicates that no single class is dominant,
    suggesting high lithology uncertainty.

    Parameters
    ----------
    geotop : GeoTop
        GeoTOP object with kans_1-9 loaded.

    Returns
    -------
    xr.DataArray
        Maximum probability (0-100) for each voxel.
        100 = completely certain, ~11 = maximum uncertainty (uniform).

    Examples
    --------
    >>> mode_prob = compute_mode_probability(geotop)
    >>> uncertain = mode_prob < 50  # less than 50% confident
    """
    if not geotop.has_kans:
        raise ValueError("GeoTop object does not have kans probability data.")

    # Stack kans values
    kans = np.stack(
        [geotop.ds[f'kans_{i}'].values for i in range(1, 10)],
        axis=0
    )

    # Find maximum probability for each voxel
    mode_prob = np.nanmax(kans, axis=0)

    return xr.DataArray(
        mode_prob,
        dims=geotop.ds['lithok'].dims,
        coords=geotop.ds['lithok'].coords,
        name='mode_probability',
        attrs={
            'description': 'Probability of most likely lithology class',
            'units': 'percent'
        }
    )
