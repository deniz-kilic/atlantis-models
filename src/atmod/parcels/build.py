"""Main entry points for building parcel-based subsurface models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import xarray as xr

from atmod.parcels.aggregation import AggregationMethod
from atmod.parcels.data import ParcelData, ParcelModelConfig
from atmod.parcels.extractor import ExtractionResult, ParcelExtractor
from atmod.parcels.remapper import ParcelResultMapper, remap_parcel_results
from atmod.parcels.virtual_grid import VirtualGridBuilder, create_virtual_grid

if TYPE_CHECKING:
    from atmod.base import AtlansParameters, Mapping, Raster, VoxelModel

logger = logging.getLogger(__name__)


def build_parcel_model(
    parcels: Union[gpd.GeoDataFrame, ParcelData, str, Path],
    parcel_id_column: str,
    ahn: Raster,
    geotop: VoxelModel,
    nl3d: Optional[VoxelModel] = None,
    bodemkaart: Optional[Mapping] = None,
    glg: Optional[Raster] = None,
    parameters: Optional[AtlansParameters] = None,
    aggregation_method: Union[str, AggregationMethod] = "centroid",
    use_probability: bool = True,
    attribute_columns: Optional[list[str]] = None,
    config: Optional[ParcelModelConfig] = None,
) -> Tuple[xr.Dataset, dict]:
    """
    Build a parcel-based subsurface model for Atlans.jl.

    Creates a "virtual grid" NetCDF where parcels are mapped to a 1×N pseudo-grid
    that Atlans.jl can process unchanged. Each parcel becomes a single column in
    the virtual grid, and results can be remapped back to parcel IDs after simulation.

    Parameters
    ----------
    parcels : GeoDataFrame, ParcelData, str, or Path
        Parcel polygons. Can be:
        - GeoDataFrame with parcel geometries
        - ParcelData object (already processed)
        - Path to shapefile or GeoPackage
    parcel_id_column : str
        Name of column containing unique parcel identifiers.
    ahn : Raster
        AHN (surface elevation) raster.
    geotop : VoxelModel
        GeoTOP 3D voxel model with lithology, geology, etc.
        If combined with NL3D, pass the combined model here.
    nl3d : VoxelModel, optional
        NL3D 3D voxel model for gap-filling. If provided, will be combined
        with GeoTOP using combine_geotop_nl3d().
        **Deprecated**: Pre-combine with geotop instead.
    bodemkaart : Mapping, optional
        BRO Bodemkaart for top 1.2m soil profile.
        **Note**: Currently not implemented for parcel extraction.
    glg : Raster, optional
        GLG (lowest groundwater level) raster for phreatic level.
    parameters : AtlansParameters, optional
        Atlans model parameters. If None, defaults are used.
    aggregation_method : str or AggregationMethod, default "centroid"
        Method for aggregating grid cells to parcels:
        - "centroid": Sample at parcel centroid (fastest)
        - "mode": Most common value (for categorical data)
        - "area_weighted": Area-weighted average (for continuous data)
        - "probability": Use GeoTOP kans_* for most likely lithology
    use_probability : bool, default True
        If True and method is "probability", use GeoTOP probability
        distributions to derive most likely lithology.
    attribute_columns : list[str], optional
        Additional columns from parcels to preserve in output.
    config : ParcelModelConfig, optional
        Advanced configuration options.

    Returns
    -------
    tuple[xr.Dataset, dict]
        - virtual_grid: NetCDF dataset for Atlans.jl (dims: y=1, x=N, layer=K)
        - mapping_info: Dict for remapping results back to parcel IDs

    Examples
    --------
    >>> import geopandas as gpd
    >>> from atmod.parcels import build_parcel_model, remap_parcel_results
    >>> from atmod.bro_models import GeoTop
    >>> from atmod.base import Raster
    >>>
    >>> parcels = gpd.read_file("parcels.shp")
    >>> geotop = GeoTop.from_netcdf("geotop.nc")
    >>> ahn = Raster.from_tif("ahn.tif")
    >>>
    >>> virtual_grid, mapping = build_parcel_model(
    ...     parcels=parcels,
    ...     parcel_id_column="Perceel_ID",
    ...     ahn=ahn,
    ...     geotop=geotop,
    ...     aggregation_method="centroid",
    ... )
    >>>
    >>> # Save for Atlans.jl
    >>> virtual_grid.to_netcdf("parcel_model.nc")
    >>>
    >>> # After running Atlans.jl, remap results
    >>> results = remap_parcel_results("atlans_output.nc", mapping)

    Notes
    -----
    The virtual grid format:
    - Dimensions: y=1, x=N, layer=K where N=number of parcels
    - x coordinates are pseudo-coordinates (1, 2, 3, ..., N)
    - Auxiliary variables store parcel_id, centroid_x, centroid_y

    Atlans.jl processes each column independently (no lateral flow),
    so this representation is physically valid.
    """
    import warnings

    # Handle config
    if config is None:
        config = ParcelModelConfig(
            aggregation_method=aggregation_method
            if isinstance(aggregation_method, str)
            else aggregation_method.value,
            use_probability=use_probability,
        )

    # Handle nl3d deprecation
    if nl3d is not None:
        warnings.warn(
            "The 'nl3d' parameter is deprecated. Use combine_geotop_nl3d() "
            "to combine GeoTOP and NL3D before calling build_parcel_model().",
            DeprecationWarning,
            stacklevel=2,
        )
        from atmod.merge import combine_geotop_nl3d

        geotop, _ = combine_geotop_nl3d(geotop, nl3d)

    # Handle bodemkaart warning
    if bodemkaart is not None:
        logger.warning(
            "bodemkaart support for parcels is not yet implemented. "
            "The top 1.2m will use GeoTOP data only."
        )

    # Parse parcels input
    parcel_data = _parse_parcels(parcels, parcel_id_column, attribute_columns)

    logger.info(
        f"Building parcel model: {len(parcel_data)} parcels, "
        f"method={config.aggregation_method}"
    )

    # Create extractor
    extractor = ParcelExtractor(
        parcel_data,
        method=config.aggregation_method,
        use_probability=config.use_probability,
        strict=config.strict,
        max_warnings=config.max_warnings,
    )

    # Prepare rasters dict
    rasters = {"surface_level": ahn}
    if glg is not None:
        rasters["phreatic_level"] = glg

    # Determine 3D variables to extract
    variables_3d = ["lithology", "geology", "thickness"]
    # Add mass fractions if available
    for var in ["mass_fraction_organic", "mass_fraction_lutum"]:
        if var in geotop.data_vars:
            variables_3d.append(var)

    # Extract data
    extraction_result = extractor.extract(
        voxelmodel=geotop,
        rasters=rasters,
        variables_3d=variables_3d,
    )

    # Log warnings
    if extraction_result.warnings:
        logger.info(f"Extraction completed with {len(extraction_result.warnings)} warnings")

    # Build virtual grid
    builder = VirtualGridBuilder(extraction_result)
    virtual_grid = builder.build(
        compression_level=config.compression_level,
        include_provenance=True,
    )

    # Add Atlans-specific variables
    virtual_grid = _add_atlans_variables(virtual_grid, parameters)

    mapping_info = builder.get_mapping_info()

    return virtual_grid, mapping_info


def _parse_parcels(
    parcels: Union[gpd.GeoDataFrame, ParcelData, str, Path],
    parcel_id_column: str,
    attribute_columns: Optional[list[str]] = None,
) -> ParcelData:
    """Parse various parcel input formats to ParcelData."""
    if isinstance(parcels, ParcelData):
        return parcels

    if isinstance(parcels, (str, Path)):
        path = Path(parcels)
        if path.suffix.lower() in (".shp", ".geojson", ".json"):
            return ParcelData.from_shapefile(
                path, parcel_id_column, attribute_columns
            )
        elif path.suffix.lower() in (".gpkg",):
            return ParcelData.from_geopackage(
                path, parcel_id_column, attribute_columns=attribute_columns
            )
        else:
            # Try reading with geopandas
            gdf = gpd.read_file(path)
            return ParcelData.from_geodataframe(
                gdf, parcel_id_column, attribute_columns
            )

    if isinstance(parcels, gpd.GeoDataFrame):
        return ParcelData.from_geodataframe(
            parcels, parcel_id_column, attribute_columns
        )

    raise TypeError(
        f"parcels must be GeoDataFrame, ParcelData, or path, got {type(parcels)}"
    )


def _add_atlans_variables(
    ds: xr.Dataset,
    parameters: Optional[AtlansParameters] = None,
) -> xr.Dataset:
    """Add Atlans.jl-specific derived variables to the virtual grid."""
    from atmod.base import AtlansParameters, AtlansStrat

    if parameters is None:
        parameters = AtlansParameters()

    # Get shapes
    n_y = ds.dims["y"]
    n_x = ds.dims["x"]

    # Add zbase (model base elevation)
    if "zbase" not in ds:
        ds["zbase"] = (["y", "x"], np.full((n_y, n_x), parameters.modelbase))

    # Add max_oxidation_depth
    if "max_oxidation_depth" not in ds:
        ds["max_oxidation_depth"] = (
            ["y", "x"],
            np.full((n_y, n_x), parameters.max_oxidation_depth),
        )

    # Add no_oxidation_thickness
    if "no_oxidation_thickness" not in ds:
        ds["no_oxidation_thickness"] = (
            ["y", "x"],
            np.full((n_y, n_x), parameters.no_oxidation_thickness),
        )

    # Add no_shrinkage_thickness
    if "no_shrinkage_thickness" not in ds:
        ds["no_shrinkage_thickness"] = (
            ["y", "x"],
            np.full((n_y, n_x), parameters.no_shrinkage_thickness),
        )

    # Calculate domainbase (base of Holocene)
    if "domainbase" not in ds and "geology" in ds and "thickness" in ds:
        geology = ds["geology"].values  # (y, x, z)
        thickness = ds["thickness"].values  # (y, x, z)

        # Sum thickness of Holocene layers
        holocene_mask = geology == AtlansStrat.holocene
        thickness_holocene = np.nansum(
            np.where(holocene_mask, thickness, 0), axis=2
        )

        # Surface level minus Holocene thickness
        if "surface_level" in ds:
            surface_level = ds["surface_level"].values  # (y, x)
            domainbase = surface_level - thickness_holocene
            domainbase = np.where(thickness_holocene > 0, domainbase, np.nan)
            ds["domainbase"] = (["y", "x"], domainbase)

    # Calculate rho_bulk if organic mass fraction available
    if "rho_bulk" not in ds and "mass_fraction_organic" in ds:
        organic = ds["mass_fraction_organic"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_bulk = (100 / organic) * (1 - np.exp(-organic / 0.12))
        rho_bulk = np.where(np.isfinite(rho_bulk), rho_bulk, parameters.rho_bulk)
        ds["rho_bulk"] = (["y", "x", "z"], rho_bulk)

    # Rename z to layer for Atlans.jl compatibility
    ds = ds.rename({"z": "layer"})
    ds["layer"] = np.arange(len(ds["layer"])) + 1

    return ds


def build_parcel_forcing(
    gridded_forcing: xr.Dataset,
    parcels: Union[gpd.GeoDataFrame, ParcelData],
    parcel_id_column: Optional[str] = None,
    aggregation_method: Union[str, AggregationMethod] = "centroid",
    forcing_variables: Optional[list[str]] = None,
) -> xr.Dataset:
    """
    Convert gridded time-varying forcing to parcel format for Atlans.jl.

    Parameters
    ----------
    gridded_forcing : xr.Dataset
        Time-varying forcing data with dimensions (time, y, x).
    parcels : GeoDataFrame or ParcelData
        Parcel geometries.
    parcel_id_column : str, optional
        Column name for parcel IDs. Required if parcels is GeoDataFrame.
    aggregation_method : str or AggregationMethod, default "centroid"
        Aggregation method for forcing values.
    forcing_variables : list[str], optional
        Variables to aggregate. If None, all time-varying variables.

    Returns
    -------
    xr.Dataset
        Forcing data with dimensions (time, y=1, x=n_parcels).

    Notes
    -----
    This function aggregates each timestep independently to preserve
    temporal dynamics (e.g., seasonal groundwater fluctuations).
    """
    from atmod.base import Raster
    from atmod.parcels.aggregation import aggregate_2d

    # Parse parcels
    if isinstance(parcels, gpd.GeoDataFrame):
        if parcel_id_column is None:
            raise ValueError("parcel_id_column required when parcels is GeoDataFrame")
        parcel_data = ParcelData.from_geodataframe(parcels, parcel_id_column)
    else:
        parcel_data = parcels

    # Parse method
    if isinstance(aggregation_method, str):
        method = AggregationMethod(aggregation_method)
    else:
        method = aggregation_method

    # Determine variables
    if forcing_variables is None:
        forcing_variables = [
            v for v in gridded_forcing.data_vars if "time" in gridded_forcing[v].dims
        ]

    n_parcels = len(parcel_data)
    n_times = len(gridded_forcing.time)

    # Get cellsize from forcing
    if "x" in gridded_forcing.coords:
        x_vals = gridded_forcing.x.values
        cellsize = abs(x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 100
    else:
        cellsize = 100  # Default

    # Pre-allocate result arrays
    result_data = {var: np.full((n_times, 1, n_parcels), np.nan) for var in forcing_variables}

    logger.info(f"Building parcel forcing: {n_times} timesteps, {n_parcels} parcels")

    # Process each timestep
    for t in range(n_times):
        time_slice = gridded_forcing.isel(time=t)

        for var in forcing_variables:
            if var not in time_slice:
                continue

            # Create temporary Raster
            var_data = time_slice[var]
            raster = Raster(var_data, cellsize)

            # Aggregate to parcels
            result_data[var][t, 0, :] = aggregate_2d(
                parcel_data.geometries,
                raster,
                method,
                centroids=parcel_data.centroids,
            )

    # Create output dataset
    result = xr.Dataset(
        {var: (["time", "y", "x"], data) for var, data in result_data.items()},
        coords={
            "time": gridded_forcing.time,
            "y": np.array([0.0]),
            "x": np.arange(1, n_parcels + 1, dtype=np.float64),
        },
    )

    # Add parcel mapping info
    result["parcel_id"] = (["x"], parcel_data.parcel_id)
    result["centroid_x"] = (["x"], parcel_data.centroids[:, 0])
    result["centroid_y"] = (["x"], parcel_data.centroids[:, 1])

    result.attrs["grid_type"] = "virtual_parcel_grid"
    result.attrs["n_parcels"] = n_parcels
    result.attrs["aggregation_method"] = method.value

    return result


# Re-export remap_parcel_results for convenience
__all__ = [
    "build_parcel_model",
    "build_parcel_forcing",
    "remap_parcel_results",
]
