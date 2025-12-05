"""Build virtual grid NetCDF for Atlans.jl from parcel data."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import xarray as xr

from atmod.parcels.extractor import ExtractionResult

if TYPE_CHECKING:
    from atmod.parcels.data import ParcelData

logger = logging.getLogger(__name__)


class VirtualGridBuilder:
    """
    Build a virtual grid NetCDF compatible with Atlans.jl from parcel data.

    The virtual grid represents parcels as a 1×N pseudo-grid where:
    - y dimension = 1 (single row)
    - x dimension = N (number of parcels)
    - layer dimension = K (number of vertical layers)

    Atlans.jl can process this as a regular grid because each column
    (grid cell) is simulated independently with no lateral flow.

    Parameters
    ----------
    extraction_result : ExtractionResult
        Result from ParcelExtractor.extract() containing extracted data.

    Attributes
    ----------
    ds : xr.Dataset
        The virtual grid dataset after build() is called.

    Examples
    --------
    >>> from atmod.parcels import ParcelExtractor, VirtualGridBuilder
    >>> extractor = ParcelExtractor(parcels, method="centroid")
    >>> result = extractor.extract(geotop, rasters={"surface_level": ahn})
    >>> builder = VirtualGridBuilder(result)
    >>> virtual_grid = builder.build()
    >>> virtual_grid.to_netcdf("parcel_model.nc")
    """

    def __init__(self, extraction_result: ExtractionResult):
        self.result = extraction_result
        self.ds: Optional[xr.Dataset] = None
        self._mapping_info: Optional[dict] = None

    def build(
        self,
        compression_level: int = 4,
        include_provenance: bool = True,
    ) -> xr.Dataset:
        """
        Build the virtual grid dataset.

        Parameters
        ----------
        compression_level : int, default 4
            Compression level for output (0-9, 0=no compression).
        include_provenance : bool, default True
            If True, add provenance metadata to the dataset.

        Returns
        -------
        xr.Dataset
            Virtual grid dataset with dimensions (y=1, x=N, layer=K).
        """
        n_parcels = self.result.n_parcels
        n_layers = self.result.n_layers

        logger.info(f"Building virtual grid: {n_parcels} parcels × {n_layers} layers")

        # Create coordinates
        # x: pseudo-coordinates 1, 2, 3, ..., N
        # y: single value [0.0]
        # layer/z: from voxelmodel
        coords = {
            "y": np.array([0.0]),
            "x": np.arange(1, n_parcels + 1, dtype=np.float64),
            "z": self.result.layer_coords,
        }

        # Create data variables
        data_vars = {}

        # Add 3D variables (y, x, z) or (z, y, x) depending on convention
        # Atlans.jl expects (y, x, z) ordering
        for var_name, var_data in self.result.data_3d.items():
            # var_data is (n_parcels, n_layers)
            # Reshape to (1, n_parcels, n_layers) = (y, x, z)
            data_3d = var_data.reshape(1, n_parcels, n_layers)
            data_vars[var_name] = (["y", "x", "z"], data_3d)

        # Add 2D variables (y, x)
        for var_name, var_data in self.result.data_2d.items():
            # var_data is (n_parcels,)
            # Reshape to (1, n_parcels) = (y, x)
            data_2d = var_data.reshape(1, n_parcels)
            data_vars[var_name] = (["y", "x"], data_2d)

        # Add auxiliary coordinates for parcel mapping
        parcel_ids = self.result.parcel_data.parcel_id
        centroids = self.result.parcel_data.centroids

        data_vars["parcel_id"] = (["x"], parcel_ids)
        data_vars["centroid_x"] = (["x"], centroids[:, 0])
        data_vars["centroid_y"] = (["x"], centroids[:, 1])
        data_vars["parcel_area"] = (["x"], self.result.parcel_data.areas)

        # Create dataset
        self.ds = xr.Dataset(data_vars, coords=coords)

        # Add attributes
        self.ds.attrs["crs"] = self.result.parcel_data.crs
        self.ds.attrs["grid_type"] = "virtual_parcel_grid"
        self.ds.attrs["n_parcels"] = n_parcels
        self.ds.attrs["n_layers"] = n_layers
        self.ds.attrs["aggregation_method"] = self.result.aggregation_method
        self.ds.attrs["Conventions"] = "CF-1.8"

        if include_provenance:
            self._add_provenance()

        # Store mapping info for result remapping
        self._mapping_info = {
            "parcel_id": parcel_ids.tolist(),
            "centroid_x": centroids[:, 0].tolist(),
            "centroid_y": centroids[:, 1].tolist(),
            "n_parcels": n_parcels,
            "n_layers": n_layers,
            "crs": self.result.parcel_data.crs,
            "aggregation_method": self.result.aggregation_method,
        }

        logger.info(f"Virtual grid built successfully: {self.ds.dims}")
        return self.ds

    def get_mapping_info(self) -> dict:
        """
        Get mapping information for result remapping.

        This dict contains all information needed to convert
        Atlans.jl output back to parcel-indexed format.

        Returns
        -------
        dict
            Mapping information including parcel_id, centroids, etc.

        Raises
        ------
        ValueError
            If build() has not been called yet.
        """
        if self._mapping_info is None:
            raise ValueError("Must call build() before get_mapping_info()")
        return self._mapping_info

    def to_netcdf(
        self,
        path: Union[str, Path],
        compression_level: int = 4,
    ) -> None:
        """
        Save the virtual grid to a NetCDF file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        compression_level : int, default 4
            Compression level (0-9).
        """
        if self.ds is None:
            raise ValueError("Must call build() before to_netcdf()")

        path = Path(path)

        # Set up encoding with compression
        encoding = {}
        for var in self.ds.data_vars:
            dtype = self.ds[var].dtype
            # Use appropriate dtype
            if np.issubdtype(dtype, np.floating):
                encoding[var] = {
                    "zlib": compression_level > 0,
                    "complevel": compression_level,
                    "dtype": "float32",
                }
            elif np.issubdtype(dtype, np.integer):
                encoding[var] = {
                    "zlib": compression_level > 0,
                    "complevel": compression_level,
                }
            else:
                # For object dtype (strings), use vlen string
                encoding[var] = {
                    "zlib": compression_level > 0,
                    "complevel": compression_level,
                }

        self.ds.to_netcdf(
            path,
            encoding=encoding,
            format="NETCDF4",
            engine="netcdf4",
        )

        logger.info(f"Saved virtual grid to {path}")

    def save_mapping(self, path: Union[str, Path]) -> None:
        """
        Save mapping information to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path (.json).
        """
        import json

        if self._mapping_info is None:
            raise ValueError("Must call build() before save_mapping()")

        path = Path(path)
        with open(path, "w") as f:
            json.dump(self._mapping_info, f, indent=2)

        logger.info(f"Saved mapping info to {path}")

    def _add_provenance(self) -> None:
        """Add provenance metadata to the dataset."""
        try:
            import atmod

            version = getattr(atmod, "__version__", "unknown")
        except ImportError:
            version = "unknown"

        self.ds.attrs["created_by"] = "atmod.parcels.VirtualGridBuilder"
        self.ds.attrs["atmod_version"] = version
        self.ds.attrs["creation_date"] = datetime.now().isoformat()
        self.ds.attrs["source"] = "parcel-based subsurface extraction"

        # Add variable-level metadata
        if "parcel_id" in self.ds:
            self.ds["parcel_id"].attrs["long_name"] = "unique parcel identifier"

        if "centroid_x" in self.ds:
            self.ds["centroid_x"].attrs["long_name"] = "parcel centroid x coordinate"
            self.ds["centroid_x"].attrs["units"] = "m"

        if "centroid_y" in self.ds:
            self.ds["centroid_y"].attrs["long_name"] = "parcel centroid y coordinate"
            self.ds["centroid_y"].attrs["units"] = "m"

        if "parcel_area" in self.ds:
            self.ds["parcel_area"].attrs["long_name"] = "parcel area"
            self.ds["parcel_area"].attrs["units"] = "m2"


def create_virtual_grid(
    extraction_result: ExtractionResult,
    output_path: Optional[Union[str, Path]] = None,
    compression_level: int = 4,
) -> tuple[xr.Dataset, dict]:
    """
    Convenience function to create virtual grid from extraction result.

    Parameters
    ----------
    extraction_result : ExtractionResult
        Result from ParcelExtractor.extract().
    output_path : str or Path, optional
        If provided, save the virtual grid to this path.
    compression_level : int, default 4
        Compression level for NetCDF output.

    Returns
    -------
    tuple[xr.Dataset, dict]
        Virtual grid dataset and mapping information for result remapping.

    Examples
    --------
    >>> virtual_grid, mapping = create_virtual_grid(result, "model.nc")
    """
    builder = VirtualGridBuilder(extraction_result)
    ds = builder.build(compression_level=compression_level)

    if output_path is not None:
        builder.to_netcdf(output_path, compression_level=compression_level)

    return ds, builder.get_mapping_info()
