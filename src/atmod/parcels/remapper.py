"""Remap Atlans.jl output from virtual grid to parcel-indexed format."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class ParcelResultMapper:
    """
    Convert Atlans.jl output from virtual grid format to parcel-indexed format.

    The virtual grid format uses dimensions (y=1, x=N, time) where x is a
    pseudo-coordinate representing parcel indices. This class converts the
    output to a proper parcel-indexed format with (parcel_id, time) dimensions.

    Parameters
    ----------
    mapping_info : dict
        Mapping information from VirtualGridBuilder.get_mapping_info().
        Contains parcel_id, centroid_x, centroid_y, etc.

    Examples
    --------
    >>> from atmod.parcels import ParcelResultMapper
    >>> import json
    >>> with open("mapping.json") as f:
    ...     mapping_info = json.load(f)
    >>> mapper = ParcelResultMapper(mapping_info)
    >>> results = mapper.remap("atlans_output.nc")
    >>> results.to_netcdf("parcel_results.nc")
    """

    def __init__(self, mapping_info: dict):
        self.mapping_info = mapping_info
        self._validate_mapping_info()

    def _validate_mapping_info(self) -> None:
        """Validate mapping information has required fields."""
        required = ["parcel_id", "centroid_x", "centroid_y", "n_parcels"]
        missing = [k for k in required if k not in self.mapping_info]
        if missing:
            raise ValueError(f"Mapping info missing required fields: {missing}")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ParcelResultMapper":
        """
        Create mapper from a JSON mapping file.

        Parameters
        ----------
        path : str or Path
            Path to JSON file with mapping information.

        Returns
        -------
        ParcelResultMapper
            Initialized mapper.
        """
        path = Path(path)
        with open(path) as f:
            mapping_info = json.load(f)
        return cls(mapping_info)

    @classmethod
    def from_virtual_grid(cls, virtual_grid_path: Union[str, Path]) -> "ParcelResultMapper":
        """
        Create mapper by extracting mapping info from virtual grid NetCDF.

        Parameters
        ----------
        virtual_grid_path : str or Path
            Path to virtual grid NetCDF file.

        Returns
        -------
        ParcelResultMapper
            Initialized mapper.
        """
        with xr.open_dataset(virtual_grid_path) as ds:
            mapping_info = {
                "parcel_id": ds["parcel_id"].values.tolist(),
                "centroid_x": ds["centroid_x"].values.tolist(),
                "centroid_y": ds["centroid_y"].values.tolist(),
                "n_parcels": ds.dims["x"],
                "crs": ds.attrs.get("crs", "EPSG:28992"),
                "aggregation_method": ds.attrs.get("aggregation_method", "unknown"),
            }
            if "parcel_area" in ds:
                mapping_info["parcel_area"] = ds["parcel_area"].values.tolist()

        return cls(mapping_info)

    def remap(
        self,
        atlans_output_path: Union[str, Path],
        output_variables: Optional[list[str]] = None,
    ) -> xr.Dataset:
        """
        Remap Atlans.jl output to parcel-indexed format.

        Parameters
        ----------
        atlans_output_path : str or Path
            Path to Atlans.jl output NetCDF file.
        output_variables : list[str], optional
            Variables to include in output. If None, includes all
            time-varying variables from the input.

        Returns
        -------
        xr.Dataset
            Dataset with dimensions (parcel_id, time) instead of (y, x, time).

        Notes
        -----
        Input format:
            - Dimensions: y=1, x=N, time=T
            - Variables: subsidence(y, x, time), consolidation(y, x, time), etc.

        Output format:
            - Dimensions: parcel_id=N, time=T
            - Variables: subsidence(parcel_id, time), etc.
            - Coordinates: parcel_id, time, centroid_x, centroid_y
        """
        atlans_output_path = Path(atlans_output_path)
        logger.info(f"Remapping Atlans output: {atlans_output_path}")

        # Open Atlans output
        with xr.open_dataset(atlans_output_path) as ds:
            # Validate dimensions
            self._validate_atlans_output(ds)

            # Determine output variables
            if output_variables is None:
                output_variables = self._get_time_varying_variables(ds)

            # Create output dataset
            result = self._create_output_dataset(ds, output_variables)

        logger.info(f"Remapping complete: {result.dims}")
        return result

    def _validate_atlans_output(self, ds: xr.Dataset) -> None:
        """Validate Atlans output has expected structure."""
        # Check dimensions
        if "x" not in ds.dims:
            raise ValueError("Atlans output missing 'x' dimension")
        if "time" not in ds.dims:
            raise ValueError("Atlans output missing 'time' dimension")

        # Check x dimension matches expected number of parcels
        n_x = ds.dims["x"]
        expected = self.mapping_info["n_parcels"]
        if n_x != expected:
            raise ValueError(
                f"Atlans output has {n_x} x values, expected {expected} (n_parcels)"
            )

    def _get_time_varying_variables(self, ds: xr.Dataset) -> list[str]:
        """Get list of time-varying variables from dataset."""
        time_vars = []
        for var in ds.data_vars:
            if "time" in ds[var].dims and var not in (
                "parcel_id",
                "centroid_x",
                "centroid_y",
                "parcel_area",
            ):
                time_vars.append(var)
        return time_vars

    def _create_output_dataset(
        self,
        input_ds: xr.Dataset,
        variables: list[str],
    ) -> xr.Dataset:
        """Create output dataset with parcel_id dimension."""
        parcel_ids = np.array(self.mapping_info["parcel_id"])
        centroids_x = np.array(self.mapping_info["centroid_x"])
        centroids_y = np.array(self.mapping_info["centroid_y"])

        # Get time coordinate
        time_coord = input_ds["time"].values

        # Create coordinates
        coords = {
            "parcel_id": parcel_ids,
            "time": time_coord,
        }

        # Create data variables
        data_vars = {}

        for var_name in variables:
            if var_name not in input_ds:
                logger.warning(f"Variable '{var_name}' not found in Atlans output")
                continue

            var_data = input_ds[var_name]
            dims = var_data.dims  # Use tuple for ordered comparison

            # Handle different dimension orderings based on actual dimension order
            # Atlans.jl outputs in ("x", "y", "time") order
            if dims == ("x", "y", "time"):
                # Atlans.jl native format: (x, y, time) -> (parcel_id, time)
                # For virtual grid: y=0, x=0..N-1
                data = var_data.values[:, 0, :]  # (n_parcels, n_times)
                data_vars[var_name] = (["parcel_id", "time"], data)

            elif dims == ("y", "x", "time"):
                # Alternative: (y, x, time) -> (parcel_id, time)
                # y=0, x=0..N-1
                data = var_data.values[0, :, :]  # (n_parcels, n_times)
                data_vars[var_name] = (["parcel_id", "time"], data)

            elif dims == ("time", "y", "x"):
                # (time, y, x) -> (parcel_id, time)
                data = var_data.values[:, 0, :]  # (n_times, n_parcels)
                data_vars[var_name] = (["parcel_id", "time"], data.T)

            elif dims == ("time", "x", "y"):
                # (time, x, y) -> (parcel_id, time)
                data = var_data.values[:, :, 0]  # (n_times, n_parcels)
                data_vars[var_name] = (["parcel_id", "time"], data.T)

            elif dims == ("x", "time"):
                # (x, time) -> (parcel_id, time)
                data = var_data.values  # (n_parcels, n_times)
                data_vars[var_name] = (["parcel_id", "time"], data)

            elif dims == ("time", "x"):
                # (time, x) -> (parcel_id, time)
                data = var_data.values  # (n_times, n_parcels)
                data_vars[var_name] = (["parcel_id", "time"], data.T)

            else:
                logger.warning(
                    f"Variable '{var_name}' has unexpected dimensions {dims}, skipping"
                )
                continue

        # Add centroid coordinates as auxiliary variables
        data_vars["centroid_x"] = (["parcel_id"], centroids_x)
        data_vars["centroid_y"] = (["parcel_id"], centroids_y)

        if "parcel_area" in self.mapping_info:
            data_vars["parcel_area"] = (
                ["parcel_id"],
                np.array(self.mapping_info["parcel_area"]),
            )

        # Create output dataset
        output = xr.Dataset(data_vars, coords=coords)

        # Add attributes
        output.attrs["crs"] = self.mapping_info.get("crs", "EPSG:28992")
        output.attrs["aggregation_method"] = self.mapping_info.get(
            "aggregation_method", "unknown"
        )
        output.attrs["source_file"] = str(input_ds.encoding.get("source", "unknown"))
        output.attrs["Conventions"] = "CF-1.8"

        # Variable attributes
        output["centroid_x"].attrs = {
            "long_name": "parcel centroid x coordinate",
            "units": "m",
        }
        output["centroid_y"].attrs = {
            "long_name": "parcel centroid y coordinate",
            "units": "m",
        }
        if "parcel_area" in output:
            output["parcel_area"].attrs = {
                "long_name": "parcel area",
                "units": "m2",
            }

        # Copy attributes from input variables
        for var_name in variables:
            if var_name in output and var_name in input_ds:
                for attr_name, attr_value in input_ds[var_name].attrs.items():
                    output[var_name].attrs[attr_name] = attr_value

        return output

    def to_geodataframe(
        self,
        results: xr.Dataset,
        time_index: Optional[int] = None,
        time_value: Optional[str] = None,
    ):
        """
        Convert results to a GeoDataFrame for spatial analysis.

        Parameters
        ----------
        results : xr.Dataset
            Remapped results from remap().
        time_index : int, optional
            Index of time step to include. If None, includes all.
        time_value : str, optional
            Time value to select (e.g., "2023-01-01").

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with point geometries at parcel centroids.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        # Select time if specified
        if time_value is not None:
            results = results.sel(time=time_value)
        elif time_index is not None:
            results = results.isel(time=time_index)

        # Convert to DataFrame
        df = results.to_dataframe().reset_index()

        # Create point geometries from centroids
        geometry = [
            Point(x, y)
            for x, y in zip(df["centroid_x"], df["centroid_y"])
        ]

        gdf = gpd.GeoDataFrame(
            df,
            geometry=geometry,
            crs=self.mapping_info.get("crs", "EPSG:28992"),
        )

        return gdf


def remap_parcel_results(
    atlans_output_path: Union[str, Path],
    mapping_info: Union[dict, str, Path],
    output_path: Optional[Union[str, Path]] = None,
    output_variables: Optional[list[str]] = None,
) -> xr.Dataset:
    """
    Convenience function to remap Atlans.jl output to parcel format.

    Parameters
    ----------
    atlans_output_path : str or Path
        Path to Atlans.jl output NetCDF file.
    mapping_info : dict or str or Path
        Mapping information dict, or path to JSON file with mapping info,
        or path to virtual grid NetCDF file.
    output_path : str or Path, optional
        If provided, save remapped results to this path.
    output_variables : list[str], optional
        Variables to include. If None, includes all time-varying variables.

    Returns
    -------
    xr.Dataset
        Remapped results with (parcel_id, time) dimensions.

    Examples
    --------
    >>> results = remap_parcel_results(
    ...     "atlans_output.nc",
    ...     "mapping.json",
    ...     "parcel_results.nc",
    ... )
    """
    # Create mapper
    if isinstance(mapping_info, dict):
        mapper = ParcelResultMapper(mapping_info)
    elif isinstance(mapping_info, (str, Path)):
        path = Path(mapping_info)
        if path.suffix == ".json":
            mapper = ParcelResultMapper.from_json(path)
        elif path.suffix in (".nc", ".nc4", ".netcdf"):
            mapper = ParcelResultMapper.from_virtual_grid(path)
        else:
            # Try JSON first
            try:
                mapper = ParcelResultMapper.from_json(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                mapper = ParcelResultMapper.from_virtual_grid(path)
    else:
        raise TypeError(f"mapping_info must be dict, str, or Path, got {type(mapping_info)}")

    # Remap
    results = mapper.remap(atlans_output_path, output_variables)

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        results.to_netcdf(
            output_path,
            encoding={
                var: {"zlib": True, "complevel": 4}
                for var in results.data_vars
            },
        )
        logger.info(f"Saved remapped results to {output_path}")

    return results
