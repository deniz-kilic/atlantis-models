"""Data structures for parcel-based modeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
from numpy.typing import ArrayLike

from atmod.parcels.exceptions import (
    DuplicateParcelIdError,
    InvalidGeometryError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Target CRS for all operations (Dutch RD New)
TARGET_EPSG = 28992

# Proj4 string for Dutch RD New (used when EPSG database is unavailable)
RD_NEW_PROJ4 = (
    "+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 "
    "+k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel "
    "+towgs84=565.417,50.3319,465.552,-0.398957,0.343988,-1.8774,4.0725 "
    "+units=m +no_defs"
)


@dataclass
class ParcelData:
    """
    Container for parcel data used in subsurface model extraction.

    Attributes
    ----------
    parcel_id : np.ndarray
        Unique identifiers for each parcel. Can be integers or strings.
    geometries : gpd.GeoSeries
        Polygon geometries for each parcel in EPSG:28992.
    centroids : np.ndarray
        Centroid coordinates (n_parcels, 2) in EPSG:28992. Column 0 is x, column 1 is y.
    areas : np.ndarray
        Parcel areas in square meters.
    crs : str
        Coordinate reference system (always "EPSG:28992" after initialization).
    attributes : dict
        Optional additional attributes from the source GeoDataFrame.
    """

    parcel_id: np.ndarray
    geometries: gpd.GeoSeries
    centroids: np.ndarray
    areas: np.ndarray
    crs: str = "EPSG:28992"
    attributes: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of parcels."""
        return len(self.parcel_id)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ParcelData(n_parcels={len(self)}, "
            f"crs={self.crs}, "
            f"bounds={self.bounds})"
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return total bounds (xmin, ymin, xmax, ymax)."""
        return tuple(self.geometries.total_bounds)

    @property
    def xmin(self) -> float:
        """Minimum x coordinate."""
        return self.bounds[0]

    @property
    def ymin(self) -> float:
        """Minimum y coordinate."""
        return self.bounds[1]

    @property
    def xmax(self) -> float:
        """Maximum x coordinate."""
        return self.bounds[2]

    @property
    def ymax(self) -> float:
        """Maximum y coordinate."""
        return self.bounds[3]

    @classmethod
    def from_geodataframe(
        cls,
        gdf: gpd.GeoDataFrame,
        parcel_id_column: str,
        attribute_columns: Optional[list[str]] = None,
        repair_geometries: bool = True,
    ) -> "ParcelData":
        """
        Create ParcelData from a GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with parcel polygons.
        parcel_id_column : str
            Name of column containing unique parcel identifiers.
        attribute_columns : list[str], optional
            Additional columns to preserve as attributes.
        repair_geometries : bool, default True
            If True, attempt to repair invalid geometries.

        Returns
        -------
        ParcelData
            Validated parcel data container.

        Raises
        ------
        ValidationError
            If required column is missing or CRS is undefined.
        DuplicateParcelIdError
            If duplicate parcel IDs are found.
        """
        # Validate parcel_id column exists
        if parcel_id_column not in gdf.columns:
            raise ValidationError(
                f"Column '{parcel_id_column}' not found in GeoDataFrame. "
                f"Available columns: {list(gdf.columns)}"
            )

        # Check for duplicate parcel IDs
        if gdf[parcel_id_column].duplicated().any():
            n_dups = gdf[parcel_id_column].duplicated().sum()
            dup_ids = gdf[parcel_id_column][gdf[parcel_id_column].duplicated()].head(5)
            raise DuplicateParcelIdError(
                f"Found {n_dups} duplicate parcel IDs. Examples: {list(dup_ids)}"
            )

        # Validate CRS
        if gdf.crs is None:
            raise ValidationError(
                "GeoDataFrame must have a CRS defined. "
                "Set it with: gdf.set_crs(epsg=28992)"
            )

        # Transform to target CRS if needed
        # Check if already in target CRS (handle both EPSG code and proj4 definitions)
        current_epsg = gdf.crs.to_epsg()
        if current_epsg != TARGET_EPSG:
            logger.info(f"Transforming parcels from {gdf.crs} to EPSG:{TARGET_EPSG}")
            try:
                gdf = gdf.to_crs(epsg=TARGET_EPSG)
            except Exception:
                # Fallback to proj4 string if EPSG lookup fails
                gdf = gdf.to_crs(RD_NEW_PROJ4)

        # Repair invalid geometries if requested
        if repair_geometries:
            gdf = _repair_geometries(gdf)

        # Extract data
        parcel_id = gdf[parcel_id_column].values
        geometries = gdf.geometry.copy()

        # Calculate centroids
        centroids_series = geometries.centroid
        centroids = np.column_stack([centroids_series.x, centroids_series.y])

        # Calculate areas
        areas = geometries.area.values

        # Extract optional attributes
        attributes = {}
        if attribute_columns:
            for col in attribute_columns:
                if col in gdf.columns:
                    attributes[col] = gdf[col].values
                else:
                    logger.warning(f"Attribute column '{col}' not found, skipping")

        return cls(
            parcel_id=parcel_id,
            geometries=geometries,
            centroids=centroids,
            areas=areas,
            crs=f"EPSG:{TARGET_EPSG}",
            attributes=attributes,
        )

    @classmethod
    def from_shapefile(
        cls,
        path: Union[str, Path],
        parcel_id_column: str,
        attribute_columns: Optional[list[str]] = None,
        repair_geometries: bool = True,
    ) -> "ParcelData":
        """
        Create ParcelData from a shapefile.

        Parameters
        ----------
        path : str or Path
            Path to shapefile (.shp).
        parcel_id_column : str
            Name of column containing unique parcel identifiers.
        attribute_columns : list[str], optional
            Additional columns to preserve as attributes.
        repair_geometries : bool, default True
            If True, attempt to repair invalid geometries.

        Returns
        -------
        ParcelData
            Validated parcel data container.
        """
        gdf = gpd.read_file(path)
        return cls.from_geodataframe(
            gdf,
            parcel_id_column=parcel_id_column,
            attribute_columns=attribute_columns,
            repair_geometries=repair_geometries,
        )

    @classmethod
    def from_geopackage(
        cls,
        path: Union[str, Path],
        parcel_id_column: str,
        layer: Optional[str] = None,
        attribute_columns: Optional[list[str]] = None,
        repair_geometries: bool = True,
    ) -> "ParcelData":
        """
        Create ParcelData from a GeoPackage.

        Parameters
        ----------
        path : str or Path
            Path to GeoPackage (.gpkg).
        parcel_id_column : str
            Name of column containing unique parcel identifiers.
        layer : str, optional
            Layer name if GeoPackage contains multiple layers.
        attribute_columns : list[str], optional
            Additional columns to preserve as attributes.
        repair_geometries : bool, default True
            If True, attempt to repair invalid geometries.

        Returns
        -------
        ParcelData
            Validated parcel data container.
        """
        gdf = gpd.read_file(path, layer=layer)
        return cls.from_geodataframe(
            gdf,
            parcel_id_column=parcel_id_column,
            attribute_columns=attribute_columns,
            repair_geometries=repair_geometries,
        )

    def subset(self, indices: ArrayLike) -> "ParcelData":
        """
        Create a subset of parcels by indices.

        Parameters
        ----------
        indices : array-like
            Indices of parcels to include.

        Returns
        -------
        ParcelData
            Subset of parcels.
        """
        indices = np.asarray(indices)
        return ParcelData(
            parcel_id=self.parcel_id[indices],
            geometries=self.geometries.iloc[indices].reset_index(drop=True),
            centroids=self.centroids[indices],
            areas=self.areas[indices],
            crs=self.crs,
            attributes={k: v[indices] for k, v in self.attributes.items()},
        )

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert ParcelData back to a GeoDataFrame.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with parcel data.
        """
        data = {
            "parcel_id": self.parcel_id,
            "centroid_x": self.centroids[:, 0],
            "centroid_y": self.centroids[:, 1],
            "area_m2": self.areas,
        }
        data.update(self.attributes)

        # Use the CRS from geometries to avoid mismatch
        gdf = gpd.GeoDataFrame(
            data,
            geometry=self.geometries.reset_index(drop=True),
        )
        return gdf


def _repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Repair invalid geometries in a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with potentially invalid geometries.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with repaired geometries.
    """
    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = invalid_mask.sum()

    if n_invalid == 0:
        return gdf

    logger.info(f"Repairing {n_invalid} invalid geometries using buffer(0)")

    # Create a copy to avoid modifying original
    gdf = gdf.copy()

    # Apply buffer(0) fix to invalid geometries
    gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)

    # Check if repair was successful
    still_invalid = ~gdf.geometry.is_valid
    n_still_invalid = still_invalid.sum()

    if n_still_invalid > 0:
        logger.warning(
            f"{n_still_invalid} geometries could not be repaired. "
            "These parcels may cause issues during extraction."
        )

    return gdf


@dataclass
class ParcelModelConfig:
    """
    Configuration for parcel model building.

    Attributes
    ----------
    aggregation_method : str
        Method for aggregating grid cells to parcels.
        One of: "centroid", "mode", "area_weighted", "probability".
    use_probability : bool
        If True and aggregation_method is "probability",
        use GeoTOP kans_* variables for most likely lithology.
    chunk_size : int
        Number of parcels to process at once (for memory efficiency).
    n_workers : int
        Number of parallel workers (0 = sequential processing).
    verbose : bool
        If True, show progress bar and info messages.
    strict : bool
        If True, raise on any error. If False, skip problematic parcels.
    max_warnings : int
        Maximum number of warnings before suppressing further warnings.
    compression_level : int
        Compression level for output NetCDF files (0-9).
    validate_output : bool
        If True, verify output file after writing.
    """

    # Aggregation settings
    aggregation_method: str = "centroid"
    use_probability: bool = True

    # Processing settings
    chunk_size: int = 1000
    n_workers: int = 0
    verbose: bool = True

    # Error handling
    strict: bool = False
    max_warnings: int = 100

    # Output settings
    compression_level: int = 4
    validate_output: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        valid_methods = {"centroid", "mode", "area_weighted", "probability"}
        if self.aggregation_method not in valid_methods:
            raise ValueError(
                f"Invalid aggregation_method: {self.aggregation_method}. "
                f"Must be one of: {valid_methods}"
            )

        if not 0 <= self.compression_level <= 9:
            raise ValueError(
                f"compression_level must be 0-9, got {self.compression_level}"
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ParcelModelConfig":
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML configuration file.

        Returns
        -------
        ParcelModelConfig
            Configuration loaded from file.
        """
        import yaml

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to save YAML configuration file.
        """
        import yaml
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
