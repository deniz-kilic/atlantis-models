from .base import AtlansParameters
from .build import build_atlantis_model, build_model_in_chunks
from .read import read_ahn, read_glg

__version__ = "0.1.0"

# ArcGIS compatibility modules (optional - may not all be needed for UQ)
# from .coordinates_efficient import calculate_level_coordinate_chunked, calculate_optimal_chunk_size
# from .validation import validate_atlantis_model, AtlantisModelValidator, ValidationReport
# from .cellid import add_cell_id_to_netcdf, add_cell_id_to_dataset, calculate_cell_ids
# from .arcgis import prepare_for_arcgis, VARIABLE_METADATA, CRS_METADATA
