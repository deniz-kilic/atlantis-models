"""Extract subsurface data at parcel locations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from atmod.parcels.aggregation import (
    AggregationMethod,
    aggregate_2d,
    aggregate_3d,
    aggregate_3d_probability,
)
from atmod.parcels.data import ParcelData
from atmod.parcels.exceptions import (
    NoSubsurfaceDataError,
    ParcelExtractionError,
    ParcelOutsideBoundsError,
)

if TYPE_CHECKING:
    from atmod.base import Raster, VoxelModel

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """
    Result of extracting subsurface data for parcels.

    Attributes
    ----------
    parcel_data : ParcelData
        The parcel data that was used for extraction.
    data_3d : dict[str, np.ndarray]
        3D variable data, each array is (n_parcels, n_layers).
    data_2d : dict[str, np.ndarray]
        2D variable data, each array is (n_parcels,).
    layer_coords : np.ndarray
        Z coordinates for layers.
    n_layers : int
        Number of vertical layers.
    aggregation_method : str
        Method used for aggregation.
    skipped_parcels : list[int]
        Indices of parcels that were skipped due to errors.
    warnings : list[str]
        Warning messages generated during extraction.
    """

    parcel_data: ParcelData
    data_3d: dict
    data_2d: dict
    layer_coords: np.ndarray
    n_layers: int
    aggregation_method: str
    skipped_parcels: list
    warnings: list

    @property
    def n_parcels(self) -> int:
        """Number of parcels processed."""
        return len(self.parcel_data)

    @property
    def n_valid_parcels(self) -> int:
        """Number of parcels with valid data."""
        return self.n_parcels - len(self.skipped_parcels)


class ParcelExtractor:
    """
    Extract subsurface data at parcel locations.

    This class handles extraction of 2D and 3D data from rasters and
    voxel models at parcel locations using various aggregation methods.

    Parameters
    ----------
    parcels : ParcelData
        Parcel data containing geometries and IDs.
    method : AggregationMethod or str, default "centroid"
        Aggregation method to use for extraction.
    use_probability : bool, default True
        If True and method is "probability", use GeoTOP probability
        distributions (kans_*) to derive most likely lithology.
    strict : bool, default False
        If True, raise exceptions on errors. If False, skip
        problematic parcels and log warnings.
    max_warnings : int, default 100
        Maximum number of warnings before suppressing further warnings.

    Examples
    --------
    >>> from atmod.parcels import ParcelData, ParcelExtractor
    >>> from atmod.bro_models import GeoTop
    >>> from atmod.base import Raster
    >>>
    >>> parcels = ParcelData.from_shapefile("parcels.shp", "Perceel_ID")
    >>> geotop = GeoTop.from_netcdf("geotop.nc")
    >>> ahn = Raster.from_tif("ahn.tif")
    >>>
    >>> extractor = ParcelExtractor(parcels, method="centroid")
    >>> result = extractor.extract(
    ...     voxelmodel=geotop,
    ...     rasters={"surface_level": ahn},
    ...     variables_3d=["lithology", "geology"],
    ... )
    """

    def __init__(
        self,
        parcels: ParcelData,
        method: Union[AggregationMethod, str] = "centroid",
        use_probability: bool = True,
        strict: bool = False,
        max_warnings: int = 100,
    ):
        self.parcels = parcels
        self.method = (
            AggregationMethod(method) if isinstance(method, str) else method
        )
        self.use_probability = use_probability
        self.strict = strict
        self.max_warnings = max_warnings
        self._warning_count = 0
        self._warnings: list[str] = []

    def extract(
        self,
        voxelmodel: VoxelModel,
        rasters: Optional[dict[str, Raster]] = None,
        variables_3d: Optional[list[str]] = None,
        variables_2d_from_voxel: Optional[list[str]] = None,
    ) -> ExtractionResult:
        """
        Extract subsurface data for all parcels.

        Parameters
        ----------
        voxelmodel : VoxelModel
            3D voxel model (e.g., GeoTOP) to extract from.
        rasters : dict[str, Raster], optional
            Named 2D rasters to extract (e.g., {"surface_level": ahn}).
        variables_3d : list[str], optional
            Variable names to extract from voxelmodel.
            If None, extracts all available variables.
        variables_2d_from_voxel : list[str], optional
            2D variables to extract from voxelmodel (if present).

        Returns
        -------
        ExtractionResult
            Container with extracted data and metadata.

        Raises
        ------
        ParcelExtractionError
            If strict=True and an error occurs during extraction.
        """
        self._warning_count = 0
        self._warnings = []
        skipped_parcels = []

        # Validate spatial overlap
        self._validate_spatial_overlap(voxelmodel)

        # Determine variables to extract
        if variables_3d is None:
            variables_3d = voxelmodel.data_vars
            # Filter out kans_* if not using probability method
            if self.method != AggregationMethod.PROBABILITY:
                variables_3d = [v for v in variables_3d if not v.startswith("kans_")]

        # Extract 3D data
        data_3d = {}
        for var_name in variables_3d:
            if var_name not in voxelmodel.data_vars:
                self._log_warning(f"Variable '{var_name}' not found in voxelmodel")
                continue

            logger.info(f"Extracting 3D variable: {var_name}")
            data_3d[var_name] = self._extract_3d_variable(voxelmodel, var_name)

        # Handle probability-based lithology extraction
        if (
            self.method == AggregationMethod.PROBABILITY
            and self.use_probability
            and "lithology" not in data_3d
        ):
            logger.info("Extracting lithology using probability distributions")
            data_3d["lithology"] = aggregate_3d_probability(
                self.parcels.geometries,
                voxelmodel,
                centroids=self.parcels.centroids,
            )

        # Extract 2D data from rasters
        data_2d = {}
        if rasters:
            for name, raster in rasters.items():
                logger.info(f"Extracting 2D raster: {name}")
                data_2d[name] = self._extract_2d_raster(raster)

        # Extract 2D variables from voxelmodel if specified
        if variables_2d_from_voxel:
            for var_name in variables_2d_from_voxel:
                if var_name in voxelmodel.data_vars:
                    # Take first layer or surface value
                    data_3d_temp = self._extract_3d_variable(voxelmodel, var_name)
                    data_2d[var_name] = data_3d_temp[:, 0]

        return ExtractionResult(
            parcel_data=self.parcels,
            data_3d=data_3d,
            data_2d=data_2d,
            layer_coords=voxelmodel.zcoords,
            n_layers=voxelmodel.nz,
            aggregation_method=self.method.value,
            skipped_parcels=skipped_parcels,
            warnings=self._warnings,
        )

    def extract_from_raster(self, raster: Raster) -> np.ndarray:
        """
        Extract 2D raster values at parcel locations.

        Parameters
        ----------
        raster : Raster
            2D raster to extract from.

        Returns
        -------
        np.ndarray
            Extracted values (n_parcels,).
        """
        return self._extract_2d_raster(raster)

    def extract_from_voxelmodel(
        self,
        voxelmodel: VoxelModel,
        variables: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Extract 3D voxelmodel variables at parcel locations.

        Parameters
        ----------
        voxelmodel : VoxelModel
            3D voxel model to extract from.
        variables : list[str], optional
            Variables to extract. If None, extracts all.

        Returns
        -------
        dict[str, np.ndarray]
            Dict mapping variable names to (n_parcels, n_layers) arrays.
        """
        if variables is None:
            variables = voxelmodel.data_vars

        result = {}
        for var_name in variables:
            if var_name in voxelmodel.data_vars:
                result[var_name] = self._extract_3d_variable(voxelmodel, var_name)

        return result

    def _extract_3d_variable(
        self,
        voxelmodel: VoxelModel,
        variable: str,
    ) -> np.ndarray:
        """Extract a single 3D variable."""
        # For lithology with probability method, use probability aggregation
        if (
            variable == "lithology"
            and self.method == AggregationMethod.PROBABILITY
            and self.use_probability
        ):
            # Check if kans_* variables exist
            has_kans = any(v.startswith("kans_") for v in voxelmodel.data_vars)
            if has_kans:
                return aggregate_3d_probability(
                    self.parcels.geometries,
                    voxelmodel,
                    centroids=self.parcels.centroids,
                )

        # Use standard aggregation
        # For categorical variables (lithology, geology), prefer MODE
        method = self.method

        # PROBABILITY method only applies to lithology via kans_*
        # For other variables, fall back to CENTROID
        if method == AggregationMethod.PROBABILITY:
            method = AggregationMethod.CENTROID

        if variable in ("lithology", "geology", "lithoklasse", "lithostrat"):
            if method == AggregationMethod.AREA_WEIGHTED:
                method = AggregationMethod.MODE
                self._log_warning(
                    f"Using MODE instead of AREA_WEIGHTED for categorical variable '{variable}'"
                )

        return aggregate_3d(
            self.parcels.geometries,
            voxelmodel,
            variable,
            method,
            centroids=self.parcels.centroids,
        )

    def _extract_2d_raster(self, raster: Raster) -> np.ndarray:
        """Extract a 2D raster."""
        # For 2D, use appropriate method
        method = self.method
        if method == AggregationMethod.PROBABILITY:
            # Probability only applies to 3D, fall back to centroid for 2D
            method = AggregationMethod.CENTROID

        return aggregate_2d(
            self.parcels.geometries,
            raster,
            method,
            centroids=self.parcels.centroids,
        )

    def _validate_spatial_overlap(self, voxelmodel: VoxelModel) -> None:
        """Validate that parcels overlap with voxelmodel extent."""
        parcel_bounds = self.parcels.bounds
        voxel_bounds = (
            voxelmodel.xmin,
            voxelmodel.ymin,
            voxelmodel.xmax,
            voxelmodel.ymax,
        )

        # Check for any overlap
        overlap = (
            parcel_bounds[0] < voxel_bounds[2]  # parcel xmin < voxel xmax
            and parcel_bounds[2] > voxel_bounds[0]  # parcel xmax > voxel xmin
            and parcel_bounds[1] < voxel_bounds[3]  # parcel ymin < voxel ymax
            and parcel_bounds[3] > voxel_bounds[1]  # parcel ymax > voxel ymin
        )

        if not overlap:
            msg = (
                f"No spatial overlap between parcels and voxelmodel. "
                f"Parcel bounds: {parcel_bounds}, "
                f"Voxelmodel bounds: {voxel_bounds}"
            )
            if self.strict:
                raise ParcelOutsideBoundsError(msg)
            else:
                logger.warning(msg)

        # Check what fraction of parcels are within bounds
        centroids_x = self.parcels.centroids[:, 0]
        centroids_y = self.parcels.centroids[:, 1]
        within_bounds = (
            (centroids_x >= voxel_bounds[0])
            & (centroids_x <= voxel_bounds[2])
            & (centroids_y >= voxel_bounds[1])
            & (centroids_y <= voxel_bounds[3])
        )
        n_within = within_bounds.sum()
        n_outside = len(self.parcels) - n_within

        if n_outside > 0:
            pct_outside = 100 * n_outside / len(self.parcels)
            self._log_warning(
                f"{n_outside} parcels ({pct_outside:.1f}%) are outside voxelmodel bounds"
            )

    def _log_warning(self, message: str) -> None:
        """Log a warning, respecting max_warnings limit."""
        self._warning_count += 1
        self._warnings.append(message)

        if self._warning_count <= self.max_warnings:
            logger.warning(message)
        elif self._warning_count == self.max_warnings + 1:
            logger.warning(
                f"Suppressing further warnings (max_warnings={self.max_warnings})"
            )
