"""
ArcGIS compatibility module for Atlantis subsurface models.

This module provides functions to add CF-1.8 compliant metadata and CRS information
to make NetCDF files compatible with ArcGIS Pro.
"""

from pathlib import Path
from typing import Dict, Optional

import netCDF4 as nc
import numpy as np

# Standard metadata for all Atlantis subsurface model variables
VARIABLE_METADATA = {
    # 3D variables (layer, y, x)
    'lithology': {
        'standard_name': 'lithology_class',
        'long_name': 'Lithological class identifier',
        'units': '1',
        'comment': 'Categorical variable indicating lithology type'
    },
    'thickness': {
        'standard_name': 'layer_thickness',
        'long_name': 'Layer thickness',
        'units': 'm',
        'positive': 'down',
        'comment': 'Thickness of each subsurface layer'
    },
    'geology': {
        'standard_name': 'geological_unit',
        'long_name': 'Geological unit identifier',
        'units': '1',
        'comment': 'Geological unit classification'
    },
    'top': {
        'standard_name': 'layer_top_depth',
        'long_name': 'Top depth of layer',
        'units': 'm',
        'positive': 'down',
        'comment': 'Depth to top of layer below surface'
    },
    'bottom': {
        'standard_name': 'layer_bottom_depth',
        'long_name': 'Bottom depth of layer',
        'units': 'm',
        'positive': 'down',
        'comment': 'Depth to bottom of layer below surface'
    },
    'level': {
        'standard_name': 'level_number',
        'long_name': 'Vertical level number from surface',
        'units': '1',
        'axis': 'Z',
        'positive': 'down',
        'comment': 'Auxiliary coordinate for visualization. Use layer dimension for Atlantis.'
    },

    # 2D variables (y, x)
    'surface_level': {
        'standard_name': 'surface_altitude',
        'long_name': 'Surface elevation',
        'units': 'm',
        'positive': 'up',
        'comment': 'Surface elevation relative to NAP (Dutch Ordnance Datum)'
    },
    'phreatic_level': {
        'standard_name': 'water_table_depth',
        'long_name': 'Phreatic level (depth to water table)',
        'units': 'm',
        'positive': 'down',
        'comment': 'Depth to groundwater table below surface'
    },
    'cell_id': {
        'standard_name': 'cell_identifier',
        'long_name': 'Cell identifier for spatial subsetting',
        'units': '1',
        'comment': 'Unique identifier for valid cells, numbered sequentially from northwest to southeast'
    },

    # 1D variables (layer)
    'layer': {
        'standard_name': 'layer_index',
        'long_name': 'Layer index',
        'units': '1',
        'axis': 'Z',
        'positive': 'up',
        'comment': 'Layer numbering: 1=bottom layer, N=top layer (Atlantis convention)'
    }
}


# EPSG:28992 CRS metadata (Amersfoort / RD New)
CRS_METADATA = {
    'grid_mapping_name': 'oblique_stereographic',
    'epsg_code': 'EPSG:28992',
    'proj4_string': '+proj=sterea +lat_0=52.1561605555556 +lon_0=5.38763888888889 +k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel +units=m +no_defs',
    'latitude_of_projection_origin': 52.1561605555556,
    'longitude_of_projection_origin': 5.38763888888889,
    'scale_factor_at_projection_origin': 0.9999079,
    'false_easting': 155000.0,
    'false_northing': 463000.0,
    'semi_major_axis': 6377397.155,
    'semi_minor_axis': 6356078.96282,
    'inverse_flattening': 299.1528128,
    'crs_wkt': '''PROJCS["Amersfoort / RD New",
    GEOGCS["Amersfoort",
        DATUM["Amersfoort",
            SPHEROID["Bessel 1841",6377397.155,299.1528128,
                AUTHORITY["EPSG","7004"]],
            AUTHORITY["EPSG","6289"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4289"]],
    PROJECTION["Oblique_Stereographic"],
    PARAMETER["latitude_of_origin",52.1561605555556],
    PARAMETER["central_meridian",5.38763888888889],
    PARAMETER["scale_factor",0.9999079],
    PARAMETER["false_easting",155000],
    PARAMETER["false_northing",463000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","28992"]]''',
    'spatial_ref': '''PROJCS["Amersfoort / RD New",
    GEOGCS["Amersfoort",
        DATUM["Amersfoort",
            SPHEROID["Bessel 1841",6377397.155,299.1528128,
                AUTHORITY["EPSG","7004"]],
            AUTHORITY["EPSG","6289"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4289"]],
    PROJECTION["Oblique_Stereographic"],
    PARAMETER["latitude_of_origin",52.1561605555556],
    PARAMETER["central_meridian",5.38763888888889],
    PARAMETER["scale_factor",0.9999079],
    PARAMETER["false_easting",155000],
    PARAMETER["false_northing",463000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","28992"]]'''
}


def enhance_coordinate_attributes(var, coord_name: str, data: np.ndarray):
    """
    Add enhanced attributes to coordinate variables.

    Parameters
    ----------
    var : netCDF4.Variable
        The coordinate variable to enhance
    coord_name : str
        Name of coordinate ('x', 'y', or 'layer')
    data : np.ndarray
        Coordinate data for computing actual_range
    """
    if coord_name == 'x':
        var.standard_name = 'projection_x_coordinate'
        var.long_name = 'x-coordinate in Cartesian system'
        var.units = 'm'
        var.axis = 'X'
        var.epsg = '28992'
        var.actual_range = np.array([float(np.min(data)), float(np.max(data))], dtype='float64')

    elif coord_name == 'y':
        var.standard_name = 'projection_y_coordinate'
        var.long_name = 'y-coordinate in Cartesian system'
        var.units = 'm'
        var.axis = 'Y'
        var.epsg = '28992'
        var.actual_range = np.array([float(np.min(data)), float(np.max(data))], dtype='float64')

    elif coord_name == 'layer':
        var.standard_name = 'layer_index'
        var.long_name = 'Layer index'
        var.units = '1'
        var.axis = 'Z'
        var.positive = 'up'
        var.comment = 'Layer numbering: 1=bottom layer, N=top layer (Atlantis convention)'


def add_variable_metadata(var, var_name: str, add_grid_mapping: bool = True, has_level: bool = False):
    """
    Add standard metadata to a data variable.

    Parameters
    ----------
    var : netCDF4.Variable
        The variable to enhance
    var_name : str
        Name of the variable
    add_grid_mapping : bool, optional
        Whether to add grid_mapping attribute (default: True)
    has_level : bool, optional
        Whether the file has a 'level' auxiliary coordinate (default: False)
    """
    if var_name in VARIABLE_METADATA:
        metadata = VARIABLE_METADATA[var_name]

        for attr_name, attr_value in metadata.items():
            setattr(var, attr_name, attr_value)

        # Add grid_mapping for variables with spatial dimensions
        if add_grid_mapping and hasattr(var, 'dimensions'):
            if 'y' in var.dimensions and 'x' in var.dimensions:
                var.grid_mapping = 'spatial_ref'

                # Add coordinates attribute for 3D variables to link to level auxiliary coordinate
                if 'layer' in var.dimensions and has_level:
                    var.coordinates = 'level'


def prepare_for_arcgis(
    input_path: str,
    output_path: Optional[str] = None,
    chunk_size: int = 300,
    add_cell_id: bool = False,
    overwrite: bool = False
) -> str:
    """
    Prepare Atlantis subsurface model for ArcGIS Pro.

    Adds:
    - Enhanced coordinate metadata (x, y, layer)
    - EPSG:28992 CRS information
    - CF-1.8 compliant variable metadata
    - Optional cell_id variable

    Uses memory-efficient chunked processing for large files.

    Parameters
    ----------
    input_path : str
        Path to input NetCDF file
    output_path : str, optional
        Path for output file. If None, appends '_arcgis' to input name
    chunk_size : int, optional
        Number of rows to process at once (default: 300)
    add_cell_id : bool, optional
        Whether to add cell_id variable (default: False)
    overwrite : bool, optional
        Whether to overwrite existing output file (default: False)

    Returns
    -------
    output_path : str
        Path to the output file

    Examples
    --------
    >>> # Basic usage
    >>> output = prepare_for_arcgis('model.nc', 'model_arcgis.nc')

    >>> # With cell_id
    >>> output = prepare_for_arcgis('model.nc', add_cell_id=True)

    >>> # Auto output name
    >>> output = prepare_for_arcgis('model.nc')  # Creates model_arcgis.nc
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_arcgis{input_path.suffix}"
    else:
        output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Set overwrite=True to replace it."
        )

    print("="*70)
    print("Preparing Atlantis Model for ArcGIS Pro")
    print("="*70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Chunk size: {chunk_size} rows")
    print()

    # Process the file
    _process_file_chunked(
        str(input_path),
        str(output_path),
        chunk_size,
        add_cell_id
    )

    return str(output_path)


def _process_file_chunked(
    input_path: str,
    output_path: str,
    chunk_size: int,
    add_cell_id: bool
):
    """
    Internal function to process file in chunks.

    Memory-efficient approach:
    1. Create output file structure with all metadata
    2. Copy data in chunks
    3. Optionally calculate and add cell_id
    """
    print("Step 1: Opening input and reading structure...")
    ds_in = nc.Dataset(input_path, 'r')

    # Get dimensions
    dims = {name: len(dim) for name, dim in ds_in.dimensions.items()}
    print(f"  Dimensions: {dims}")

    # Check if file has 'level' auxiliary coordinate
    has_level_coord = 'level' in ds_in.variables
    if has_level_coord:
        print("  ✓ Found 'level' auxiliary coordinate - will link 3D variables")

    # Check if we need to calculate cell_id
    cell_id_array = None
    if add_cell_id and 'surface_level' in ds_in.variables:
        print("\n  Calculating cell_id from surface_level...")
        from .cellid import calculate_cell_ids
        surface_level = ds_in.variables['surface_level'][:]
        cell_id_array = calculate_cell_ids(surface_level)
        n_valid = np.sum(~np.isnan(cell_id_array))
        print(f"    Valid cells: {n_valid:,}")

    print("\nStep 2: Creating output file with enhanced metadata...")
    ds_out = nc.Dataset(output_path, 'w')

    # Copy dimensions
    for name, size in dims.items():
        ds_out.createDimension(name, size)

    # Copy global attributes and add CF conventions
    ds_out.setncatts(ds_in.__dict__)
    if not hasattr(ds_out, 'Conventions'):
        ds_out.Conventions = 'CF-1.8'

    # Create spatial_ref (CRS) variable first
    print("  Creating spatial_ref (CRS) variable...")
    crs_var = ds_out.createVariable('spatial_ref', 'i4')
    for attr_name, attr_value in CRS_METADATA.items():
        setattr(crs_var, attr_name, attr_value)

    # Create all other variables with enhanced metadata
    print("  Creating variables with metadata...")
    for var_name in ds_in.variables.keys():
        var_in = ds_in.variables[var_name]

        # Coordinate variables (x, y, layer) should NEVER have _FillValue (CF-1.8 requirement)
        is_coordinate = var_name in ['x', 'y', 'layer']

        # Create variable
        var_out = ds_out.createVariable(
            var_name,
            var_in.datatype,
            var_in.dimensions,
            zlib=True,
            complevel=4,
            fill_value=None if is_coordinate else (var_in._FillValue if hasattr(var_in, '_FillValue') else None)
        )

        # Copy existing attributes (skip _FillValue)
        for attr in var_in.ncattrs():
            if attr != '_FillValue':
                setattr(var_out, attr, var_in.getncattr(attr))

        # Add/enhance metadata
        if var_name in ['x', 'y', 'layer']:
            # Will add coordinate metadata after copying data (need data for actual_range)
            pass
        else:
            # Add variable metadata
            add_variable_metadata(var_out, var_name, add_grid_mapping=True, has_level=has_level_coord)

        print(f"    {var_name}")

    # Create cell_id variable if requested
    if add_cell_id and cell_id_array is not None:
        print("  Creating cell_id variable...")
        cellid_var = ds_out.createVariable(
            'cell_id',
            'f4',
            ('y', 'x'),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan)
        )
        add_variable_metadata(cellid_var, 'cell_id', add_grid_mapping=True, has_level=False)

    print("\nStep 3: Copying data in chunks...")
    ny = dims.get('y', 0)
    n_chunks = (ny + chunk_size - 1) // chunk_size if ny > 0 else 1

    # Track coordinates for actual_range
    coord_data = {}

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, ny) if ny > 0 else 0

        if ny > 0:
            progress = ((chunk_idx + 1) / n_chunks) * 100
            print(f"  Chunk {chunk_idx+1}/{n_chunks} (rows {chunk_start}-{chunk_end}) - {progress:.1f}%")

        for var_name in ds_in.variables.keys():
            var_in = ds_in.variables[var_name]
            var_out = ds_out.variables[var_name]

            dims_var = var_in.dimensions

            if not dims_var:
                # Scalar variable
                if chunk_idx == 0:
                    var_out[:] = var_in[:]
            elif dims_var == ('x',):
                # 1D x coordinate
                if chunk_idx == 0:
                    data = var_in[:]
                    var_out[:] = data
                    coord_data['x'] = data
            elif dims_var == ('y',):
                # 1D y coordinate
                if chunk_idx == 0:
                    data = var_in[:]
                    var_out[:] = data
                    coord_data['y'] = data
            elif dims_var == ('layer',):
                # 1D layer coordinate
                if chunk_idx == 0:
                    data = var_in[:]
                    var_out[:] = data
                    coord_data['layer'] = data
            elif dims_var == ('y', 'x'):
                # 2D variable
                if ny > 0:
                    chunk_data = var_in[chunk_start:chunk_end, :]
                    var_out[chunk_start:chunk_end, :] = chunk_data
                else:
                    var_out[:] = var_in[:]
            elif dims_var == ('layer', 'y', 'x'):
                # 3D variable
                if ny > 0:
                    chunk_data = var_in[:, chunk_start:chunk_end, :]
                    var_out[:, chunk_start:chunk_end, :] = chunk_data
                else:
                    var_out[:] = var_in[:]

        # Write cell_id chunk if applicable
        if add_cell_id and cell_id_array is not None and ny > 0:
            ds_out.variables['cell_id'][chunk_start:chunk_end, :] = cell_id_array[chunk_start:chunk_end, :]

        # Sync to disk
        ds_out.sync()

    print("\nStep 4: Adding coordinate metadata with actual_range...")
    # Now add coordinate metadata with actual_range
    for coord_name in ['x', 'y', 'layer']:
        if coord_name in ds_out.variables and coord_name in coord_data:
            var_out = ds_out.variables[coord_name]
            data = coord_data[coord_name]
            enhance_coordinate_attributes(var_out, coord_name, data)
            print(f"  {coord_name}: {np.min(data):.2f} to {np.max(data):.2f}")

    print("\nStep 5: Finalizing...")
    ds_in.close()
    ds_out.close()

    # Get file sizes
    import os
    input_size_mb = os.path.getsize(input_path) / (1024**2)
    output_size_mb = os.path.getsize(output_path) / (1024**2)

    print()
    print("="*70)
    print("✓ Complete!")
    print("="*70)
    print(f"Input file:  {input_size_mb:.1f} MB")
    print(f"Output file: {output_size_mb:.1f} MB")
    if add_cell_id:
        print(f"Added:       {output_size_mb - input_size_mb:.1f} MB (metadata + cell_id)")
    else:
        print(f"Added:       {output_size_mb - input_size_mb:.1f} MB (metadata only)")
    print()
