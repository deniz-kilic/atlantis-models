"""
Parcel-based subsurface model support for Atlantis.

This module enables land subsidence simulation at parcel resolution
(1 parcel = 1 subsurface column) using a "virtual grid" approach
that works with the existing Atlans.jl implementation unchanged.

Main entry points:
    - build_parcel_model(): Build parcel-based subsurface model
    - remap_parcel_results(): Convert Atlans.jl output to parcel format

Example usage:
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
    >>> # Save and run Atlans.jl, then remap results
    >>> results = remap_parcel_results("atlans_output.nc", mapping)
"""

__version__ = "0.1.0"

from atmod.parcels.aggregation import (
    AggregationMethod,
    aggregate_2d,
    aggregate_3d,
    aggregate_3d_probability,
)
from atmod.parcels.build import build_parcel_forcing, build_parcel_model
from atmod.parcels.data import ParcelData, ParcelModelConfig
from atmod.parcels.extractor import ExtractionResult, ParcelExtractor
from atmod.parcels.remapper import ParcelResultMapper, remap_parcel_results
from atmod.parcels.virtual_grid import VirtualGridBuilder, create_virtual_grid

__all__ = [
    # Data structures
    "ParcelData",
    "ParcelModelConfig",
    "AggregationMethod",
    "ExtractionResult",
    # Core classes
    "ParcelExtractor",
    "VirtualGridBuilder",
    "ParcelResultMapper",
    # Functions
    "aggregate_2d",
    "aggregate_3d",
    "aggregate_3d_probability",
    "create_virtual_grid",
    "build_parcel_model",
    "build_parcel_forcing",
    "remap_parcel_results",
]
