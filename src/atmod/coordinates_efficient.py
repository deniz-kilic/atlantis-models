"""
Memory-efficient coordinate calculations for large Atlantis models.

This module provides chunked processing approaches that work with
very large NetCDF files without loading the entire dataset into memory.
Uses netCDF4 library for direct file modification.
"""

from typing import Optional

import netCDF4 as nc
import numpy as np


def calculate_level_coordinate_chunked(
    netcdf_path: str,
    output_path: Optional[str] = None,
    chunk_size: int = 100,
    in_place: bool = False
):
    """
    Add level coordinate to NetCDF file using memory-efficient chunked processing.

    This function processes the file in chunks, reading only what's needed
    and writing results directly without loading the entire dataset.

    **How it works (Step-by-step in memory and computation):**

    1. Open NetCDF file in read mode
       - Memory: ~1MB (metadata only, no data loaded)

    2. Create output file (or open input in append mode if in_place=True)
       - Memory: ~1MB (structure only)
       - Copies all variables and attributes from input
       - Creates empty 'level' variable (no data yet)

    3. For each chunk (e.g., 100 rows):
       a. Read thickness[chunk_start:chunk_end, :, :]
          - Memory: 100 × 2700 × 282 × 8 bytes = ~610 MB

       b. Calculate level for this chunk
          - Process column by column within chunk
          - Memory: Same 610 MB (working in-place on chunk)

       c. Write level[chunk_start:chunk_end, :, :] to file
          - I/O: Flush to disk immediately

       d. Release chunk from memory
          - Memory: Back to ~1MB

    4. Close files
       - Final file has level coordinate added

    **Total peak memory: ~610 MB (one chunk + overhead)**
    vs loading full model: ~20 GB

    Parameters
    ----------
    netcdf_path : str
        Path to input NetCDF file
    output_path : str, optional
        Path for output file. If None and in_place=False, appends '_with_level' to input name
    chunk_size : int, optional
        Number of rows (y dimension) to process at once (default: 100)
        Smaller = less memory, more I/O operations
        Larger = more memory, fewer I/O operations
        Recommended: 50-200 depending on available RAM
    in_place : bool, optional
        If True, modifies input file directly (default: False)
        WARNING: Makes permanent changes, ensure you have a backup!

    Returns
    -------
    output_path : str
        Path to the output file with level coordinate

    Examples
    --------
    >>> # Safe: Create new file
    >>> output = calculate_level_coordinate_chunked(
    ...     'model.nc',
    ...     'model_with_level.nc',
    ...     chunk_size=100
    ... )

    >>> # In-place: Modify original (backup first!)
    >>> calculate_level_coordinate_chunked(
    ...     'model.nc',
    ...     in_place=True
    ... )

    Notes
    -----
    Memory efficiency comparison for 3240×2700×282 model:

    Method                     Peak Memory    Time
    ─────────────────────────  ───────────    ─────
    xarray (load all)          ~20 GB         Fast
    xarray (dask chunks)       ~5-10 GB       Medium
    netCDF4 (this function)    ~610 MB        Medium
    Row-by-row processing      ~2 MB          Slow

    This function balances memory and speed with chunked processing.
    """
    from pathlib import Path

    netcdf_path = Path(netcdf_path)

    if in_place:
        output_path = netcdf_path
        print(f"WARNING: Modifying file in-place: {netcdf_path}")
        print("Ensure you have a backup!")

        # For in-place, we need to use a temp file and then replace
        # because netCDF4 doesn't allow adding variables to open file
        temp_path = netcdf_path.parent / f".tmp_{netcdf_path.name}"
        _process_file_chunked(str(netcdf_path), str(temp_path), chunk_size)

        # Replace original with temp
        import shutil
        shutil.move(str(temp_path), str(netcdf_path))
        return str(netcdf_path)

    else:
        if output_path is None:
            output_path = netcdf_path.parent / f"{netcdf_path.stem}_with_level{netcdf_path.suffix}"

        output_path = Path(output_path)
        _process_file_chunked(str(netcdf_path), str(output_path), chunk_size)
        return str(output_path)


def _process_file_chunked(input_path: str, output_path: str, chunk_size: int):
    """
    Internal function to process file in chunks.

    Detailed memory and computation flow:

    Step 1: Open input file (read mode)
    ────────────────────────────────────
    Memory: ~1 MB (metadata only)
    - Reads dimensions, variable names, attributes
    - Does NOT load any data arrays
    """
    print(f"\n{'='*70}")
    print("Memory-Efficient Level Coordinate Calculation")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Chunk size: {chunk_size} rows")
    print()

    # Step 1: Open input (read-only, no data loaded)
    print("Step 1: Opening input file (metadata only)...")
    ds_in = nc.Dataset(input_path, 'r')

    # Get dimensions
    n_layers = len(ds_in.dimensions['layer'])
    ny = len(ds_in.dimensions['y'])
    nx = len(ds_in.dimensions['x'])

    print(f"  Dimensions: layer={n_layers}, y={ny}, x={nx}")
    print(f"  Estimated chunk memory: {chunk_size * nx * n_layers * 4 / (1024**2):.1f} MB")
    print()

    """
    Step 2: Create output file structure
    ─────────────────────────────────────
    Memory: ~2 MB (two file handles + metadata)
    - Copies all dimensions, variables (structure only), attributes
    - Creates 'level' variable structure (no data yet)
    - Still no large arrays in memory
    """
    print("Step 2: Creating output file structure...")
    ds_out = nc.Dataset(output_path, 'w')

    # Copy dimensions
    for name, dimension in ds_in.dimensions.items():
        ds_out.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    # Copy global attributes
    ds_out.setncatts(ds_in.__dict__)

    # Copy all existing variables (structure only, no data yet)
    for name, variable in ds_in.variables.items():
        out_var = ds_out.createVariable(
            name, variable.datatype, variable.dimensions,
            zlib=True, complevel=4,
            fill_value=variable._FillValue if hasattr(variable, '_FillValue') else None
        )
        # Copy attributes
        out_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs() if k != '_FillValue'})

    # Create level variable (structure only)
    level_var = ds_out.createVariable(
        'level', 'f4', ('layer', 'y', 'x'),
        zlib=True, complevel=4,
        fill_value=np.float32(np.nan)
    )

    # Add level metadata
    level_var.long_name = 'vertical level number from surface'
    level_var.description = (
        'Discrete vertical index for ArcGIS visualization. '
        'Level 1 is topsoil/surface layer, increases with depth.'
    )
    level_var.units = '1'
    level_var.axis = 'Z'
    level_var.positive = 'down'
    level_var.comment = 'Auxiliary coordinate for visualization. Use layer dimension for Atlantis.'

    print("  Created output structure (no data yet)")
    print(f"  Variables: {list(ds_out.variables.keys())}")
    print()

    """
    Step 3: Process data in chunks
    ───────────────────────────────
    This is where the work happens, but memory stays controlled.
    """
    print("Step 3: Processing and copying data in chunks...")
    print()

    n_chunks = (ny + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, ny)
        n_rows = chunk_end - chunk_start

        progress = (chunk_idx / n_chunks) * 100
        print(f"  Chunk {chunk_idx+1}/{n_chunks} (rows {chunk_start}-{chunk_end}) - {progress:.1f}%")

        """
        Step 3a: Read thickness chunk
        ──────────────────────────────
        Memory increases: +chunk_memory
        - Reads thickness[chunk_start:chunk_end, :, :]
        - This is the only large array in memory at this point
        """
        thickness_chunk = ds_in.variables['thickness'][chunk_start:chunk_end, :, :]

        # Memory: ~610 MB for 100×2700×282 float64 array
        print(f"    Loaded thickness chunk: {thickness_chunk.shape} " +
              f"({thickness_chunk.nbytes / (1024**2):.1f} MB)")

        """
        Step 3b: Calculate level for this chunk
        ────────────────────────────────────────
        Memory: Same (working in-place on chunk)
        - Allocates level_chunk array (same size as thickness_chunk)
        - Processes column by column
        - Total memory: ~1.2 GB (thickness + level chunks)
        """
        level_chunk = np.full((n_rows, nx, n_layers), np.nan, dtype='float32')

        # Process each column in this chunk
        n_processed = 0
        for i in range(n_rows):
            for j in range(nx):
                # Get thickness column: n_layers values
                thickness_col = thickness_chunk[i, j, :]

                # Find valid layers
                valid_mask = ~np.isnan(thickness_col)
                n_valid = np.sum(valid_mask)

                if n_valid > 0:
                    valid_indices = np.where(valid_mask)[0]

                    # Assign levels (top=1, bottom=n_valid)
                    for k, layer_idx in enumerate(valid_indices):
                        level_chunk[i, j, layer_idx] = n_valid - k

                    n_processed += 1

        print(f"    Calculated levels for {n_processed:,} valid columns")

        """
        Step 3c: Write both thickness and level chunks to output
        ──────────────────────────────────────────────────────────
        Memory: Same (still have both chunks)
        I/O: Writes to disk
        """
        # Write thickness chunk to output
        ds_out.variables['thickness'][chunk_start:chunk_end, :, :] = thickness_chunk

        # Write level chunk to output (transpose to match dimensions)
        ds_out.variables['level'][:, chunk_start:chunk_end, :] = level_chunk.transpose(2, 0, 1)

        # Sync to disk to free buffer memory
        ds_out.sync()

        print("    Written to output file")

        """
        Step 3d: Release chunk from memory
        ───────────────────────────────────
        Memory decreases: back to ~1 MB
        - thickness_chunk and level_chunk go out of scope
        - Python garbage collector frees memory
        - Ready for next chunk
        """
        del thickness_chunk, level_chunk
        print("    Released chunk from memory")
        print()

    """
    Step 4: Copy remaining variables
    ─────────────────────────────────
    Process other variables in chunks too
    """
    print("Step 4: Copying remaining variables...")

    for var_name in ds_in.variables.keys():
        if var_name in ['thickness', 'level']:
            continue  # Already done

        var_in = ds_in.variables[var_name]
        var_out = ds_out.variables[var_name]

        # Check dimensions
        if 'y' in var_in.dimensions and 'x' in var_in.dimensions:
            # 2D or 3D variable - copy in chunks
            if 'layer' in var_in.dimensions:
                # 3D variable
                print(f"  Copying 3D variable: {var_name}")
                for chunk_idx in range(n_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, ny)
                    chunk_data = var_in[chunk_start:chunk_end, :, :]
                    var_out[chunk_start:chunk_end, :, :] = chunk_data
                    del chunk_data
            else:
                # 2D variable
                print(f"  Copying 2D variable: {var_name}")
                for chunk_idx in range(n_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, ny)
                    chunk_data = var_in[chunk_start:chunk_end, :]
                    var_out[chunk_start:chunk_end, :] = chunk_data
                    del chunk_data
        else:
            # 1D variable or coordinate - copy all at once (small)
            print(f"  Copying coordinate: {var_name}")
            var_out[:] = var_in[:]

    """
    Step 5: Finalize
    ────────────────
    Memory: ~1 MB
    - Closes file handles
    - Flushes any remaining buffers
    - Releases all resources
    """
    print()
    print("Step 5: Finalizing...")
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
    print(f"Peak memory used: ~{chunk_size * nx * n_layers * 8 / (1024**2):.0f} MB (estimate)")
    print()


def calculate_optimal_chunk_size(ny: int, nx: int, n_layers: int, available_ram_gb: float = 4.0) -> int:
    """
    Calculate optimal chunk size based on available RAM.

    Parameters
    ----------
    ny : int
        Number of rows in model
    nx : int
        Number of columns in model
    n_layers : int
        Number of layers in model
    available_ram_gb : float
        Available RAM in GB (default: 4.0)
        Conservative default leaves room for OS and other processes

    Returns
    -------
    chunk_size : int
        Recommended number of rows to process at once

    Examples
    --------
    >>> # For 3240×2700×282 model with 4GB RAM
    >>> chunk_size = calculate_optimal_chunk_size(3240, 2700, 282, 4.0)
    >>> print(chunk_size)
    50
    """
    # Memory per row: nx * n_layers * 8 bytes (float64) * 2 (thickness + level)
    bytes_per_row = nx * n_layers * 8 * 2

    # Available bytes (use 80% to be safe)
    available_bytes = available_ram_gb * 1024**3 * 0.8

    # Calculate chunk size
    chunk_size = int(available_bytes / bytes_per_row)

    # Clamp to reasonable range
    chunk_size = max(10, min(chunk_size, ny))  # At least 10, at most all rows

    print(f"Optimal chunk size for {ny}×{nx}×{n_layers} model:")
    print(f"  Available RAM: {available_ram_gb:.1f} GB")
    print(f"  Memory per row: {bytes_per_row / (1024**2):.1f} MB")
    print(f"  Recommended chunk size: {chunk_size} rows")
    print(f"  Peak memory: {chunk_size * bytes_per_row / (1024**3):.2f} GB")

    return chunk_size
