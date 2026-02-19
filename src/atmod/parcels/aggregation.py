"""Aggregation methods for extracting raster/voxel data to parcels."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from rasterio import features
from scipy import stats
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    import geopandas as gpd

    from atmod.base import Raster, VoxelModel

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating grid cells to parcel values.

    Attributes
    ----------
    CENTROID : str
        Sample at parcel centroid using nearest neighbor.
        Fast, suitable when parcel size is similar to grid cell size.
    MODE : str
        Most common value within parcel (for categorical data).
        Appropriate for lithology, geology codes.
    AREA_WEIGHTED : str
        Area-weighted average of overlapping cells.
        Best for continuous variables like thickness, elevation.
    PROBABILITY : str
        Use GeoTOP probability distributions (kans_*) to find
        most likely lithology via area-weighted joint probability.
    """

    CENTROID = "centroid"
    MODE = "mode"
    AREA_WEIGHTED = "area_weighted"
    PROBABILITY = "probability"


def aggregate_2d(
    geometries: gpd.GeoSeries,
    raster: Raster,
    method: Union[AggregationMethod, str],
    centroids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aggregate 2D raster values to parcels.

    Parameters
    ----------
    geometries : gpd.GeoSeries
        Parcel polygon geometries.
    raster : Raster
        2D raster data to aggregate.
    method : AggregationMethod or str
        Aggregation method to use.
    centroids : np.ndarray, optional
        Pre-computed centroid coordinates (n_parcels, 2).
        Required for CENTROID method. If None, computed from geometries.

    Returns
    -------
    np.ndarray
        Aggregated values for each parcel (n_parcels,).
    """
    if isinstance(method, str):
        method = AggregationMethod(method)

    if method == AggregationMethod.CENTROID:
        return _aggregate_2d_centroid(geometries, raster, centroids)
    elif method == AggregationMethod.MODE:
        return _aggregate_2d_mode(geometries, raster)
    elif method == AggregationMethod.AREA_WEIGHTED:
        return _aggregate_2d_area_weighted(geometries, raster)
    else:
        raise ValueError(
            f"Method {method} not supported for 2D aggregation. "
            f"Use CENTROID, MODE, or AREA_WEIGHTED."
        )


def aggregate_3d(
    geometries: gpd.GeoSeries,
    voxelmodel: VoxelModel,
    variable: str,
    method: Union[AggregationMethod, str],
    centroids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aggregate 3D voxelmodel values to parcels.

    Parameters
    ----------
    geometries : gpd.GeoSeries
        Parcel polygon geometries.
    voxelmodel : VoxelModel
        3D voxel model data to aggregate.
    variable : str
        Name of variable in voxelmodel to aggregate.
    method : AggregationMethod or str
        Aggregation method to use.
    centroids : np.ndarray, optional
        Pre-computed centroid coordinates (n_parcels, 2).
        Required for CENTROID method. If None, computed from geometries.

    Returns
    -------
    np.ndarray
        Aggregated values for each parcel (n_parcels, n_layers).
    """
    if isinstance(method, str):
        method = AggregationMethod(method)

    if method == AggregationMethod.CENTROID:
        return _aggregate_3d_centroid(geometries, voxelmodel, variable, centroids)
    elif method == AggregationMethod.MODE:
        return _aggregate_3d_mode(geometries, voxelmodel, variable)
    elif method == AggregationMethod.AREA_WEIGHTED:
        return _aggregate_3d_area_weighted(geometries, voxelmodel, variable)
    elif method == AggregationMethod.PROBABILITY:
        raise ValueError(
            "PROBABILITY method requires kans_* variables. "
            "Use aggregate_3d_probability() instead."
        )
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_3d_probability(
    geometries: gpd.GeoSeries,
    voxelmodel: VoxelModel,
    centroids: Optional[np.ndarray] = None,
    n_classes: int = 9,
) -> np.ndarray:
    """
    Aggregate lithology using GeoTOP probability distributions.

    For each layer, computes area-weighted joint probability distribution
    across all cells in the parcel, then selects argmax as most likely
    lithology class.

    Parameters
    ----------
    geometries : gpd.GeoSeries
        Parcel polygon geometries.
    voxelmodel : VoxelModel
        VoxelModel containing kans_1 through kans_9 probability variables.
    centroids : np.ndarray, optional
        Pre-computed centroid coordinates (n_parcels, 2).
        Used as fallback for small parcels.
    n_classes : int, default 9
        Number of lithology classes (kans_1 through kans_n).

    Returns
    -------
    np.ndarray
        Most likely lithology for each parcel and layer (n_parcels, n_layers).
        Values are 1-indexed lithology codes (1=organic, 2=clay, etc.).
        NaN where no probability data available.

    Notes
    -----
    Lithology code mapping (GeoTOP):
        1: organic/peat
        2: clay
        3: loam
        4: fine sand
        5: medium sand
        6: coarse sand
        7: gravel
        8: shells
        9: anthropogenic
    """
    # Verify required variables exist
    kans_vars = [f"kans_{i}" for i in range(1, n_classes + 1)]
    missing = [v for v in kans_vars if v not in voxelmodel.data_vars]
    if missing:
        raise ValueError(
            f"VoxelModel missing probability variables: {missing}. "
            f"Required: kans_1 through kans_{n_classes}"
        )

    n_parcels = len(geometries)
    n_layers = voxelmodel.nz
    result = np.full((n_parcels, n_layers), np.nan)

    # Get raster metadata for geometry operations
    affine = voxelmodel.get_affine()
    out_shape = (voxelmodel.nrows, voxelmodel.ncols)

    for parcel_idx, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue

        # Get cell coverage for this parcel
        coverage_mask, weights = _get_cell_coverage(
            geom, affine, out_shape, voxelmodel.cellsize
        )

        if coverage_mask is None or weights.sum() == 0:
            # Fallback to centroid for tiny/outside parcels
            if centroids is not None:
                cx, cy = centroids[parcel_idx]
                result[parcel_idx, :] = _extract_centroid_probability(
                    cx, cy, voxelmodel, n_classes
                )
            continue

        # For each layer, compute area-weighted probability
        for layer_idx in range(n_layers):
            aggregated_probs = np.zeros(n_classes)

            for class_idx in range(n_classes):
                kans_var = f"kans_{class_idx + 1}"
                kans_data = voxelmodel[kans_var].values

                # Handle dimension ordering (y, x, z) or (z, y, x)
                if voxelmodel.ds[kans_var].dims[0] == "z":
                    layer_data = kans_data[layer_idx, :, :]
                else:
                    layer_data = kans_data[:, :, layer_idx]

                # Apply coverage mask and weights
                masked_values = layer_data[coverage_mask]
                valid_mask = ~np.isnan(masked_values)

                if valid_mask.any():
                    valid_values = masked_values[valid_mask]
                    valid_weights = weights[valid_mask]
                    aggregated_probs[class_idx] = np.average(
                        valid_values, weights=valid_weights
                    )

            # Find most likely lithology (1-indexed)
            if aggregated_probs.sum() > 0:
                result[parcel_idx, layer_idx] = np.argmax(aggregated_probs) + 1

    return result


# -----------------------------------------------------------------------------
# Private helper functions
# -----------------------------------------------------------------------------


def _aggregate_2d_centroid(
    geometries: gpd.GeoSeries,
    raster: Raster,
    centroids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Aggregate 2D raster by sampling at parcel centroids."""
    n_parcels = len(geometries)
    result = np.full(n_parcels, np.nan)

    # Compute centroids if not provided
    if centroids is None:
        centroids_series = geometries.centroid
        centroids = np.column_stack([centroids_series.x, centroids_series.y])

    # Get raster coordinates
    x_coords = raster.xcoords
    y_coords = raster.ycoords

    for i in range(n_parcels):
        cx, cy = centroids[i]

        # Check bounds
        if not (raster.xmin <= cx <= raster.xmax and raster.ymin <= cy <= raster.ymax):
            continue

        # Find nearest cell
        x_idx = np.argmin(np.abs(x_coords - cx))
        y_idx = np.argmin(np.abs(y_coords - cy))

        result[i] = raster.values[y_idx, x_idx]

    return result


def _aggregate_2d_mode(
    geometries: gpd.GeoSeries,
    raster: Raster,
) -> np.ndarray:
    """Aggregate 2D raster by computing mode within parcel."""
    n_parcels = len(geometries)
    result = np.full(n_parcels, np.nan)

    affine = raster.get_affine()
    out_shape = (raster.nrows, raster.ncols)

    for i, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue

        # Rasterize parcel geometry
        mask = features.rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=affine,
            fill=0,
            dtype=np.uint8,
        )

        # Extract values within parcel
        parcel_values = raster.values[mask == 1]
        valid_values = parcel_values[~np.isnan(parcel_values)]

        if len(valid_values) > 0:
            # Compute mode
            mode_result = stats.mode(valid_values, keepdims=False)
            result[i] = mode_result.mode

    return result


def _aggregate_2d_area_weighted(
    geometries: gpd.GeoSeries,
    raster: Raster,
) -> np.ndarray:
    """Aggregate 2D raster using area-weighted average."""
    n_parcels = len(geometries)
    result = np.full(n_parcels, np.nan)

    affine = raster.get_affine()
    out_shape = (raster.nrows, raster.ncols)

    for i, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue

        coverage_mask, weights = _get_cell_coverage(
            geom, affine, out_shape, raster.cellsize
        )

        if coverage_mask is None:
            continue

        # Extract values and compute weighted average
        values = raster.values[coverage_mask]
        valid_mask = ~np.isnan(values)

        if valid_mask.any():
            result[i] = np.average(values[valid_mask], weights=weights[valid_mask])

    return result


def _aggregate_3d_centroid(
    geometries: gpd.GeoSeries,
    voxelmodel: VoxelModel,
    variable: str,
    centroids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Aggregate 3D voxelmodel by sampling at parcel centroids."""
    n_parcels = len(geometries)
    n_layers = voxelmodel.nz
    result = np.full((n_parcels, n_layers), np.nan)

    # Compute centroids if not provided
    if centroids is None:
        centroids_series = geometries.centroid
        centroids = np.column_stack([centroids_series.x, centroids_series.y])

    # Get voxelmodel coordinates
    x_coords = voxelmodel.xcoords
    y_coords = voxelmodel.ycoords
    data = voxelmodel[variable].values

    # Determine dimension ordering
    dims = voxelmodel.ds[variable].dims
    z_first = dims[0] == "z"

    for i in range(n_parcels):
        cx, cy = centroids[i]

        # Check bounds
        if not (
            voxelmodel.xmin <= cx <= voxelmodel.xmax
            and voxelmodel.ymin <= cy <= voxelmodel.ymax
        ):
            continue

        # Find nearest cell
        x_idx = np.argmin(np.abs(x_coords - cx))
        y_idx = np.argmin(np.abs(y_coords - cy))

        # Extract column
        if z_first:
            result[i, :] = data[:, y_idx, x_idx]
        else:
            result[i, :] = data[y_idx, x_idx, :]

    return result


def _aggregate_3d_mode(
    geometries: gpd.GeoSeries,
    voxelmodel: VoxelModel,
    variable: str,
) -> np.ndarray:
    """Aggregate 3D voxelmodel by computing mode within parcel for each layer."""
    n_parcels = len(geometries)
    n_layers = voxelmodel.nz
    result = np.full((n_parcels, n_layers), np.nan)

    affine = voxelmodel.get_affine()
    out_shape = (voxelmodel.nrows, voxelmodel.ncols)
    data = voxelmodel[variable].values

    # Determine dimension ordering
    dims = voxelmodel.ds[variable].dims
    z_first = dims[0] == "z"

    for i, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue

        # Rasterize parcel geometry
        mask = features.rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=affine,
            fill=0,
            dtype=np.uint8,
        )

        for layer_idx in range(n_layers):
            if z_first:
                layer_data = data[layer_idx, :, :]
            else:
                layer_data = data[:, :, layer_idx]

            # Extract values within parcel
            parcel_values = layer_data[mask == 1]
            valid_values = parcel_values[~np.isnan(parcel_values)]

            if len(valid_values) > 0:
                mode_result = stats.mode(valid_values, keepdims=False)
                result[i, layer_idx] = mode_result.mode

    return result


def _aggregate_3d_area_weighted(
    geometries: gpd.GeoSeries,
    voxelmodel: VoxelModel,
    variable: str,
) -> np.ndarray:
    """Aggregate 3D voxelmodel using area-weighted average for each layer."""
    n_parcels = len(geometries)
    n_layers = voxelmodel.nz
    result = np.full((n_parcels, n_layers), np.nan)

    affine = voxelmodel.get_affine()
    out_shape = (voxelmodel.nrows, voxelmodel.ncols)
    data = voxelmodel[variable].values

    # Determine dimension ordering
    dims = voxelmodel.ds[variable].dims
    z_first = dims[0] == "z"

    for i, geom in enumerate(geometries):
        if geom is None or geom.is_empty:
            continue

        coverage_mask, weights = _get_cell_coverage(
            geom, affine, out_shape, voxelmodel.cellsize
        )

        if coverage_mask is None:
            continue

        for layer_idx in range(n_layers):
            if z_first:
                layer_data = data[layer_idx, :, :]
            else:
                layer_data = data[:, :, layer_idx]

            values = layer_data[coverage_mask]
            valid_mask = ~np.isnan(values)

            if valid_mask.any():
                result[i, layer_idx] = np.average(
                    values[valid_mask], weights=weights[valid_mask]
                )

    return result


def _get_cell_coverage(
    geometry: Union[Polygon, MultiPolygon],
    affine: tuple,
    out_shape: tuple,
    cellsize: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get cell coverage mask and area weights for a parcel geometry.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon
        Parcel geometry.
    affine : tuple
        Rasterio affine transform tuple.
    out_shape : tuple
        Output raster shape (nrows, ncols).
    cellsize : float
        Grid cell size in meters.

    Returns
    -------
    tuple
        (coverage_mask, weights) where:
        - coverage_mask: Boolean array of shape out_shape indicating covered cells
        - weights: 1D array of coverage fractions for covered cells
        Returns (None, None) if no cells are covered.
    """
    from rasterio.transform import Affine

    # Convert tuple to Affine object if needed
    if isinstance(affine, tuple):
        affine = Affine(*affine)

    # Rasterize to get binary coverage
    mask = features.rasterize(
        [(geometry, 1)],
        out_shape=out_shape,
        transform=affine,
        fill=0,
        dtype=np.uint8,
        all_touched=True,  # Include partially covered cells
    )

    coverage_mask = mask == 1

    if not coverage_mask.any():
        return None, None

    # For now, use uniform weights (simplified)
    # A more accurate implementation would compute fractional coverage
    # by intersecting each cell with the geometry
    n_covered = coverage_mask.sum()
    weights = np.ones(n_covered)

    return coverage_mask, weights


def _extract_centroid_probability(
    cx: float,
    cy: float,
    voxelmodel: VoxelModel,
    n_classes: int,
) -> np.ndarray:
    """Extract most likely lithology at centroid location."""
    n_layers = voxelmodel.nz
    result = np.full(n_layers, np.nan)

    x_coords = voxelmodel.xcoords
    y_coords = voxelmodel.ycoords

    # Check bounds
    if not (
        voxelmodel.xmin <= cx <= voxelmodel.xmax
        and voxelmodel.ymin <= cy <= voxelmodel.ymax
    ):
        return result

    # Find nearest cell
    x_idx = np.argmin(np.abs(x_coords - cx))
    y_idx = np.argmin(np.abs(y_coords - cy))

    for layer_idx in range(n_layers):
        probs = np.zeros(n_classes)

        for class_idx in range(n_classes):
            kans_var = f"kans_{class_idx + 1}"
            kans_data = voxelmodel[kans_var].values

            # Handle dimension ordering
            if voxelmodel.ds[kans_var].dims[0] == "z":
                probs[class_idx] = kans_data[layer_idx, y_idx, x_idx]
            else:
                probs[class_idx] = kans_data[y_idx, x_idx, layer_idx]

        # Find most likely (1-indexed)
        if not np.all(np.isnan(probs)) and np.nansum(probs) > 0:
            result[layer_idx] = np.nanargmax(probs) + 1

    return result
