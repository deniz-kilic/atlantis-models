"""
Cell ID assignment for Atlantis subsurface models.

This module provides functionality to add unique cell identifiers to NetCDF models,
enabling spatial subsetting and area-specific analysis.
"""

from pathlib import Path
from typing import Optional

import netCDF4 as nc
import numpy as np


def calculate_cell_ids(surface_level: np.ndarray) -> np.ndarray:
    """
    Calculate cell IDs based on valid surface_level values.

    Cell IDs are assigned in row-major order (top-left to bottom-right),
    with only valid cells (non-NaN surface_level) receiving IDs.

    Parameters
    ----------
    surface_level : np.ndarray
        2D array (ny, nx) of surface level values

    Returns
    -------
    cell_id : np.ndarray
        2D array (ny, nx) with sequential cell IDs for valid cells

    Notes
    -----
    - Ordering: Northwest (top-left) to Southeast (bottom-right)
    - Row-major: Processes left-to-right within each row, top-to-bottom
    - First valid cell gets ID=1, last valid cell gets ID=N
    - Invalid cells (NaN surface_level) get NaN cell_id
    """
    ny, nx = surface_level.shape

    # Create valid mask
    if np.ma.isMaskedArray(surface_level):
        valid_mask = ~np.ma.getmaskarray(surface_level)
    else:
        valid_mask = ~np.isnan(surface_level)

    # Initialize cell_id array
    cell_id = np.full((ny, nx), np.nan, dtype='float32')

    # Assign IDs in row-major order
    current_id = 1
    for i in range(ny):  # Rows (y dimension)
        for j in range(nx):  # Columns (x dimension)
            if valid_mask[i, j]:
                cell_id[i, j] = current_id
                current_id += 1

    return cell_id


def add_cell_id_to_dataset(ds, overwrite: bool = False):
    """
    Add cell_id variable to an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with surface_level variable
    overwrite : bool, optional
        If True, overwrite existing cell_id variable (default: False)

    Returns
    -------
    ds : xarray.Dataset
        Dataset with cell_id variable added

    Raises
    ------
    ValueError
        If surface_level variable is missing
        If cell_id already exists and overwrite=False
    """
    import xarray as xr

    if 'surface_level' not in ds.data_vars:
        raise ValueError("Dataset must contain 'surface_level' variable")

    if 'cell_id' in ds.data_vars and not overwrite:
        raise ValueError(
            "cell_id already exists. Set overwrite=True to replace it."
        )

    # Calculate cell IDs
    surface_level = ds['surface_level'].values
    cell_id = calculate_cell_ids(surface_level)

    # Add to dataset
    ds['cell_id'] = (('y', 'x'), cell_id)

    # Add metadata
    ds['cell_id'].attrs.update({
        'long_name': 'cell identifier for spatial subsetting',
        'description': (
            'Unique cell identifier for valid cells. Numbered sequentially from '
            'top-left (northwest) to bottom-right (southeast) in row-major order. '
            'Only cells with valid surface_level are assigned IDs. First valid cell = 1.'
        ),
        'units': '1',
        'comment': 'Use for spatial subsetting and indexing valid cells',
        'valid_range': np.array([1, np.nanmax(cell_id)], dtype='float32'),
    })

    return ds


def add_cell_id_to_netcdf(
    netcdf_path: str,
    output_path: Optional[str] = None,
    in_place: bool = False,
    chunk_size: int = 500
) -> str:
    """
    Add cell_id variable to NetCDF file using memory-efficient processing.

    This function processes the file in chunks to minimize memory usage,
    making it suitable for large models.

    Parameters
    ----------
    netcdf_path : str
        Path to input NetCDF file with surface_level variable
    output_path : str, optional
        Path for output file. If None and in_place=False,
        appends '_with_cellid' to input name
    in_place : bool, optional
        If True, modifies input file directly (default: False)
        WARNING: Makes permanent changes, ensure you have a backup!
    chunk_size : int, optional
        Number of rows to process at once (default: 500)

    Returns
    -------
    output_path : str
        Path to the output file with cell_id variable

    Examples
    --------
    >>> # Create new file with cell_id
    >>> output = add_cell_id_to_netcdf(
    ...     'model.nc',
    ...     'model_with_cellid.nc'
    ... )

    >>> # Modify existing file (backup first!)
    >>> add_cell_id_to_netcdf('model.nc', in_place=True)

    Notes
    -----
    Memory usage: ~chunk_size × nx × 8 bytes for float64 surface_level
    For 500 rows × 2700 cols: ~11 MB peak memory
    """
    netcdf_path = Path(netcdf_path)

    if in_place:
        output_path = netcdf_path
        print(f"WARNING: Modifying file in-place: {netcdf_path}")
        print("Ensure you have a backup!")

        # Use temp file for in-place modification
        temp_path = netcdf_path.parent / f".tmp_{netcdf_path.name}"
        _add_cellid_chunked(str(netcdf_path), str(temp_path), chunk_size)

        # Replace original
        import shutil
        shutil.move(str(temp_path), str(netcdf_path))
        return str(netcdf_path)

    else:
        if output_path is None:
            output_path = netcdf_path.parent / f"{netcdf_path.stem}_with_cellid{netcdf_path.suffix}"

        output_path = Path(output_path)
        _add_cellid_chunked(str(netcdf_path), str(output_path), chunk_size)
        return str(output_path)


def _add_cellid_chunked(input_path: str, output_path: str, chunk_size: int):
    """
    Internal function to add cell_id in chunks.

    Memory-efficient approach:
    1. Load surface_level to calculate all cell IDs (required for sequential numbering)
    2. Copy file structure and all existing variables
    3. Write cell_id in chunks
    """
    print(f"\n{'='*70}")
    print("Adding Cell ID to NetCDF File")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Chunk size: {chunk_size} rows")
    print()

    # Step 1: Open input and calculate cell IDs
    print("Step 1: Opening input and calculating cell IDs...")
    ds_in = nc.Dataset(input_path, 'r')

    # Get dimensions
    ny = len(ds_in.dimensions['y'])
    nx = len(ds_in.dimensions['x'])

    print(f"  Dimensions: y={ny}, x={nx}")
    print(f"  Total cells: {ny * nx:,}")

    # Load surface_level (needed for cell ID calculation)
    surface_level = ds_in.variables['surface_level'][:]

    # Calculate cell IDs
    print("  Calculating cell IDs...")
    cell_id = calculate_cell_ids(surface_level)

    n_valid = np.sum(~np.isnan(cell_id))
    print(f"  Valid cells: {n_valid:,} ({100*n_valid/(ny*nx):.1f}%)")
    print()

    # Step 2: Create output file structure
    print("Step 2: Creating output file...")
    ds_out = nc.Dataset(output_path, 'w')

    # Copy dimensions
    for name, dimension in ds_in.dimensions.items():
        ds_out.createDimension(
            name,
            len(dimension) if not dimension.isunlimited() else None
        )

    # Copy global attributes
    ds_out.setncatts(ds_in.__dict__)

    # Copy all existing variables
    print("  Copying existing variables...")
    for name, variable in ds_in.variables.items():
        out_var = ds_out.createVariable(
            name,
            variable.datatype,
            variable.dimensions,
            zlib=True,
            complevel=4,
            fill_value=variable._FillValue if hasattr(variable, '_FillValue') else None
        )
        # Copy attributes
        out_var.setncatts({
            k: variable.getncattr(k)
            for k in variable.ncattrs()
            if k != '_FillValue'
        })

    # Create cell_id variable
    cellid_var = ds_out.createVariable(
        'cell_id',
        'f4',
        ('y', 'x'),
        zlib=True,
        complevel=4,
        fill_value=np.float32(np.nan)
    )

    # Add metadata
    cellid_var.long_name = 'cell identifier for spatial subsetting'
    cellid_var.description = (
        'Unique cell identifier for valid cells. Numbered sequentially from '
        'top-left (northwest) to bottom-right (southeast) in row-major order. '
        'Only cells with valid surface_level are assigned IDs. First valid cell = 1.'
    )
    cellid_var.units = '1'
    cellid_var.comment = 'Use for spatial subsetting and indexing valid cells'
    cellid_var.valid_range = np.array([1, n_valid], dtype='float32')

    print()

    # Step 3: Copy data in chunks
    print("Step 3: Writing data in chunks...")
    n_chunks = (ny + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, ny)

        progress = (chunk_idx / n_chunks) * 100
        print(f"  Chunk {chunk_idx+1}/{n_chunks} (rows {chunk_start}-{chunk_end}) - {progress:.1f}%")

        # Copy variable data in chunks
        for var_name in ds_in.variables.keys():
            var_in = ds_in.variables[var_name]
            var_out = ds_out.variables[var_name]

            # Check dimensions
            if 'y' in var_in.dimensions and 'x' in var_in.dimensions:
                if 'layer' in var_in.dimensions:
                    # 3D variable (layer, y, x)
                    chunk_data = var_in[chunk_start:chunk_end, :, :]
                    var_out[chunk_start:chunk_end, :, :] = chunk_data
                else:
                    # 2D variable (y, x)
                    chunk_data = var_in[chunk_start:chunk_end, :]
                    var_out[chunk_start:chunk_end, :] = chunk_data
            elif chunk_idx == 0:
                # 1D or coordinate variable - copy once
                var_out[:] = var_in[:]

        # Write cell_id chunk
        ds_out.variables['cell_id'][chunk_start:chunk_end, :] = cell_id[chunk_start:chunk_end, :]

        # Sync to disk
        ds_out.sync()

    print()

    # Step 4: Finalize
    print("Step 4: Finalizing...")
    ds_in.close()
    ds_out.close()

    # Get file size
    import os
    file_size_mb = os.path.getsize(output_path) / (1024**2)

    print()
    print(f"{'='*70}")
    print("✓ Complete!")
    print(f"{'='*70}")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Cell IDs assigned: {n_valid:,} (range: 1 to {n_valid})")
    print()


def get_cells_by_id(ds, cell_ids):
    """
    Extract data for specific cell IDs.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with cell_id variable
    cell_ids : int, list of int, or numpy.ndarray
        Cell ID(s) to extract

    Returns
    -------
    subset : xarray.Dataset
        Subset of data for requested cell IDs

    Examples
    --------
    >>> # Get single cell
    >>> cell_data = get_cells_by_id(ds, 12345)

    >>> # Get multiple cells
    >>> cells_data = get_cells_by_id(ds, [100, 200, 300])
    """
    import xarray as xr

    if 'cell_id' not in ds.data_vars:
        raise ValueError("Dataset must contain 'cell_id' variable")

    # Ensure cell_ids is array
    if isinstance(cell_ids, (int, float)):
        cell_ids = [cell_ids]
    cell_ids = np.array(cell_ids)

    # Find (y, x) positions for requested cell IDs
    cellid_array = ds['cell_id'].values
    mask = np.isin(cellid_array, cell_ids)

    # Extract subset
    subset = ds.where(mask, drop=True)

    return subset


def get_cell_coordinates(ds, cell_id: int):
    """
    Get (y, x) coordinates for a specific cell ID.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with cell_id variable
    cell_id : int
        Cell ID to find

    Returns
    -------
    coords : tuple or None
        (y_idx, x_idx) if found, None if not found

    Examples
    --------
    >>> y_idx, x_idx = get_cell_coordinates(ds, 12345)
    >>> surface = ds['surface_level'].values[y_idx, x_idx]
    """
    if 'cell_id' not in ds.data_vars:
        raise ValueError("Dataset must contain 'cell_id' variable")

    cellid_array = ds['cell_id'].values
    positions = np.where(cellid_array == cell_id)

    if len(positions[0]) == 0:
        return None

    return (positions[0][0], positions[1][0])
