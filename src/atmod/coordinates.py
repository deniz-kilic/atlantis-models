"""
Coordinate system utilities for Atlantis models.

This module provides functions to add auxiliary coordinate systems
for ArcGIS compatibility while preserving Atlantis layer structure.
"""

import numpy as np
import xarray as xr


def calculate_level_coordinate(ds):
    """
    Calculate 3D level auxiliary coordinate for ArcGIS visualization.

    Level numbering: 1 (top/surface) to K (bottom-most valid layer)
    Layer numbering: 1 (bottom) to N (top) [Atlantis convention - unchanged]

    The level coordinate provides a discrete vertical index suitable for
    ArcGIS voxel visualization. Unlike depth (which varies with thickness),
    level is a simple ordinal index counting valid layers from the surface.

    For each (y, x) location:
    - Count valid layers (non-NaN thickness)
    - Assign level = 1 to top-most valid layer (surface/topsoil)
    - Assign level = K to bottom-most valid layer
    - Invalid layers (NaN thickness) get level = NaN

    Parameters
    ----------
    ds : xr.Dataset
        Atlantis model dataset with 'thickness' variable and 'layer' dimension

    Returns
    -------
    level : np.ndarray
        3D array with shape (layer, y, x) containing level numbers.
        Values range from 1 (top) to K (bottom) for valid layers at each location.
        Invalid layers have NaN values.

    Examples
    --------
    >>> ds = xr.open_dataset('atlantis_model.nc')
    >>> level = calculate_level_coordinate(ds)
    >>> level.shape
    (282, 3240, 2700)
    >>> # At location (y=100, x=100) with 50 valid layers:
    >>> # level[:, 100, 100] will have values 1-50 for valid layers, NaN elsewhere

    Notes
    -----
    - Level 1 always corresponds to the topmost valid layer (closest to surface)
    - Different (y,x) locations can have different numbers of valid levels
    - The layer dimension maintains Atlantis convention (1=bottom, N=top)
    - Level coordinate reverses this for visualization (1=top, K=bottom)
    """
    if 'thickness' not in ds.data_vars:
        raise ValueError("Dataset must contain 'thickness' variable")

    if 'layer' not in ds.dims:
        raise ValueError("Dataset must contain 'layer' dimension")

    thickness = ds['thickness'].values  # (layer, y, x)
    n_layers, ny, nx = thickness.shape

    # Initialize level array with NaN
    level = np.full((n_layers, ny, nx), np.nan, dtype='float32')

    print(f"Calculating level coordinate for {ny}×{nx} grid with {n_layers} layers...")

    # Process in chunks to reduce memory usage
    chunk_size = 100  # Process 100 rows at a time
    n_processed = 0

    for chunk_start in range(0, ny, chunk_size):
        chunk_end = min(chunk_start + chunk_size, ny)
        progress = (chunk_start / ny) * 100
        print(f"  Progress: {progress:.0f}% (rows {chunk_start}-{chunk_end}/{ny})")

        # Process chunk
        for i in range(chunk_start, chunk_end):
            for j in range(nx):
                # Get thickness column at this location
                thickness_column = thickness[:, i, j]

                # Find valid layers (non-NaN thickness)
                valid_mask = ~np.isnan(thickness_column)
                n_valid = np.sum(valid_mask)

                if n_valid > 0:
                    # Get indices of valid layers
                    valid_indices = np.where(valid_mask)[0]

                    # valid_indices[0] = lowest layer in Atlantis (layer 1)
                    # valid_indices[-1] = highest layer (topsoil)
                    #
                    # Assign levels in reverse order:
                    # - Highest layer (topsoil) gets level = 1
                    # - Lowest layer gets level = n_valid

                    for k, layer_idx in enumerate(valid_indices):
                        # k=0 (bottom layer) gets level=n_valid
                        # k=n_valid-1 (top layer) gets level=1
                        level[layer_idx, i, j] = n_valid - k

                    n_processed += 1

    print(f"  Complete! Processed {n_processed:,} valid locations")

    # Calculate statistics
    valid_levels = level[~np.isnan(level)]
    if len(valid_levels) > 0:
        print(f"  Level range: {int(valid_levels.min())} to {int(valid_levels.max())}")
        print(f"  Total valid level assignments: {len(valid_levels):,}")

    return level


def add_level_coordinate(ds):
    """
    Add level auxiliary coordinate to dataset for ArcGIS compatibility.

    This function adds a 3D 'level' coordinate variable that provides
    discrete vertical indexing suitable for ArcGIS voxel visualization.
    The level coordinate complements the existing layer dimension:

    - layer: Atlantis indexing (1=bottom, N=top) - preserved unchanged
    - level: Visualization indexing (1=top/surface, K=bottom) - added

    The function also:
    - Updates layer dimension metadata to clarify its purpose
    - Adds 'coordinates' attribute to 3D variables linking them to level
    - Ensures CF-1.8 compliance for auxiliary coordinates

    Parameters
    ----------
    ds : xr.Dataset
        Atlantis model dataset with layer dimension and thickness variable

    Returns
    -------
    ds : xr.Dataset
        Dataset with added level coordinate and updated metadata

    Examples
    --------
    >>> ds = xr.open_dataset('atlantis_model.nc')
    >>> ds = add_level_coordinate(ds)
    >>> print(ds.coords)
    Coordinates:
      * layer    (layer) int32: 1, 2, 3, ..., 282
      * x        (x) float64: ...
      * y        (y) float64: ...
        level    (layer, y, x) float32: varies by location

    Notes
    -----
    - The level coordinate is stored as float32 to handle NaN values
    - 3D variables get 'coordinates' attribute pointing to level
    - Layer dimension metadata is updated to clarify difference from level
    - This is a CF-1.8 compliant auxiliary coordinate

    See Also
    --------
    calculate_level_coordinate : Function that computes the level values
    """
    print("\nAdding level coordinate to dataset...")

    # Calculate level coordinate
    level = calculate_level_coordinate(ds)

    # Add as coordinate variable
    ds['level'] = (('layer', 'y', 'x'), level)

    # Add comprehensive metadata
    ds['level'].attrs.update({
        'long_name': 'vertical level number from surface',
        'description': (
            'Discrete vertical index for ArcGIS visualization. '
            'Level 1 is topsoil/surface layer, increases with depth. '
            'Number of levels varies by (y,x) location based on valid thickness.'
        ),
        'units': '1',
        'axis': 'Z',
        'positive': 'down',  # Level numbers increase downward
        'comment': (
            'Auxiliary coordinate for visualization. '
            'Use layer dimension for Atlantis model processing. '
            'Level coordinate reverses layer ordering: layer 1 (bottom) → level K, '
            'layer N (top) → level 1.'
        ),
        '_FillValue': np.float32(np.nan)
    })

    print("  Added level coordinate variable")

    # Identify 3D variables (those with layer dimension)
    vars_3d = [
        'lithology', 'thickness', 'geology',
        'mass_fraction_organic', 'rho_bulk',
        'mass_fraction_lutum', 'shrinkage_degree'
    ]

    # Link 3D variables to level coordinate
    n_linked = 0
    for var in vars_3d:
        if var in ds.data_vars:
            if 'layer' in ds[var].dims:  # Only 3D variables
                ds[var].attrs['coordinates'] = 'level y x'
                n_linked += 1

    print(f"  Linked {n_linked} 3D variables to level coordinate")

    # Update layer dimension metadata for clarity
    ds['layer'].attrs.update({
        'long_name': 'layer number from bottom to top',
        'description': (
            'Atlantis layer indexing convention: layer 1 is bottom-most layer, '
            'layer N is top-most layer. This is the primary dimension for Atlantis '
            'model processing.'
        ),
        'units': '1',
        'comment': (
            'Use this dimension for Atlantis Julia calculations. '
            'For visualization in ArcGIS, use the level auxiliary coordinate instead.'
        )
    })

    print("  Updated layer dimension metadata")
    print("✓ Level coordinate added successfully\n")

    return ds


def validate_level_thickness_consistency(ds):
    """
    Validate that level coordinate matches thickness structure.

    This function checks that:
    - Level values exist exactly where thickness is valid
    - Level numbering is continuous from 1 to K at each location
    - Top layer (highest in Atlantis) has level = 1
    - Bottom layer (lowest in Atlantis) has level = K

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with both level and thickness variables

    Returns
    -------
    is_valid : bool
        True if level coordinate is consistent with thickness
    issues : list of str
        List of validation issues found (empty if valid)

    Examples
    --------
    >>> ds = add_level_coordinate(ds)
    >>> is_valid, issues = validate_level_thickness_consistency(ds)
    >>> if not is_valid:
    ...     for issue in issues:
    ...         print(issue)
    """
    if 'level' not in ds:
        return False, ["Level coordinate not present in dataset"]

    if 'thickness' not in ds.data_vars:
        return False, ["Thickness variable not present in dataset"]

    thickness = ds['thickness'].values
    level = ds['level'].values

    issues = []
    n_layers, ny, nx = thickness.shape

    # Sample some locations for validation (checking all would be too slow)
    sample_size = min(100, ny * nx)
    sample_indices = np.random.choice(ny * nx, size=sample_size, replace=False)

    for idx in sample_indices:
        i = idx // nx  # y index
        j = idx % nx   # x index

        thickness_col = thickness[:, i, j]
        level_col = level[:, i, j]

        valid_thickness = ~np.isnan(thickness_col)
        valid_level = ~np.isnan(level_col)

        # Check that valid locations match
        if not np.array_equal(valid_thickness, valid_level):
            issues.append(
                f"Level/thickness mismatch at (y={i}, x={j}): "
                f"{np.sum(valid_thickness)} thickness vs {np.sum(valid_level)} level"
            )
            continue

        # Check level numbering
        if np.any(valid_level):
            valid_levels = level_col[valid_level]
            expected_levels = np.arange(1, len(valid_levels) + 1)

            # Levels should be continuous from bottom to top
            # Get sorted indices by layer (bottom to top in Atlantis)
            layer_indices = np.where(valid_level)[0]
            sorted_levels = level_col[layer_indices]

            # Top layer (last in layer_indices) should have level=1
            if sorted_levels[-1] != 1:
                issues.append(
                    f"Top layer at (y={i}, x={j}) has level={sorted_levels[-1]}, expected 1"
                )

            # Check continuity
            unique_levels = np.unique(sorted_levels)
            if not np.array_equal(unique_levels, expected_levels):
                issues.append(
                    f"Non-continuous levels at (y={i}, x={j}): {unique_levels}"
                )

    is_valid = len(issues) == 0

    return is_valid, issues


def remove_level_coordinate(ds):
    """
    Remove level coordinate from dataset.

    Useful for reverting to Atlantis-only format or troubleshooting.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with level coordinate

    Returns
    -------
    ds : xr.Dataset
        Dataset with level coordinate removed
    """
    if 'level' in ds:
        # Remove coordinates attribute from variables
        for var in ds.data_vars:
            if 'coordinates' in ds[var].attrs:
                if 'level' in ds[var].attrs['coordinates']:
                    del ds[var].attrs['coordinates']

        # Drop level coordinate
        ds = ds.drop_vars('level')
        print("Level coordinate removed")
    else:
        print("No level coordinate to remove")

    return ds
