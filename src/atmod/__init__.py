from .analysis_tools import (
    SOURCE_BODEMKAART,
    SOURCE_GEOTOP,
    SOURCE_NAMES,
    SOURCE_NL3D,
    SOURCE_NODATA,
    compare_models,
    compute_data_quality_flags,
    compute_data_source_fractions,
    compute_ensemble_statistics,
    compute_holocene_statistics,
    compute_model_summary,
    get_data_source_3d,
    plot_cross_section,
    plot_data_source_map,
    plot_ensemble_spread,
    plot_holocene_thickness_map,
    plot_lithology_distribution,
)
from .base import AtlansParameters
from .build import build_atlantis_model, build_ensemble_models, build_model_in_chunks
from .read import read_ahn, read_glg
from .uncertainty import (
    compute_lithology_entropy,
    compute_mode_probability,
    create_geotop_realization,
    generate_lithology_ensemble,
    sample_lithology_from_kans,
)

__version__ = "0.1.0"

# ArcGIS compatibility modules (optional - may not all be needed for UQ)
# from .coordinates_efficient import calculate_level_coordinate_chunked, calculate_optimal_chunk_size
# from .validation import validate_atlantis_model, AtlantisModelValidator, ValidationReport
# from .cellid import add_cell_id_to_netcdf, add_cell_id_to_dataset, calculate_cell_ids
# from .arcgis import prepare_for_arcgis, VARIABLE_METADATA, CRS_METADATA
