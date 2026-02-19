from pathlib import WindowsPath
from typing import Optional, TypeVar

import numpy as np
import xarray as xr

from atmod.base import VoxelModel
from atmod.utils import _follow_gdal_conventions, get_crs_object

ArrayLike = TypeVar("ArrayLike")

# GeoTOP kans (probability) variables for lithology uncertainty
KANS_VARS = [f'kans_{i}' for i in range(1, 10)]


class GeoTop(VoxelModel):
    """
    GeoTOP voxel model for the Netherlands subsurface.

    GeoTOP provides both deterministic lithology (lithok) and probability
    distributions (kans_1 through kans_9) for each voxel. The kans variables
    represent the probability of each lithology class occurring at each location.
    """
    @classmethod
    def from_netcdf(
        cls,
        nc_path: str | WindowsPath,
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        lazy: bool = True,
        include_kans: bool = False,
        **xr_kwargs,
    ):
        """
        Read the BRO GeoTop subsurface model from a netcdf dataset into a GeoTop
        VoxelModel instance. GeoTop can be downloaded from: https://dinodata.nl/opendap.

        Parameters
        ----------
        nc_path : str | WindowsPath
            Path to the netcdf file of GeoTop.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple, optional
            Enter a tuple (xmin, ymin, xmax, ymax) to return a selected area of GeoTop.
            The default is None.
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.
        include_kans : bool, optional
            If True, include kans_1 through kans_9 probability variables for
            lithology uncertainty quantification. Default is False.

        Returns
        -------
        GeoTop
            GeoTop instance of the netcdf file.

        """
        cellsize = 100
        dz = 0.5
        crs = 28992

        if lazy and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = "auto"

        ds = xr.open_dataset(nc_path, **xr_kwargs)
        ds = cls.coordinates_to_cellcenters(ds, cellsize, dz)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        # Handle data_vars selection with optional kans inclusion
        if data_vars is not None:
            vars_to_load = list(data_vars)
            if include_kans:
                vars_to_load = vars_to_load + [k for k in KANS_VARS if k not in vars_to_load]
            ds = ds[vars_to_load]
        elif include_kans:
            # Load default vars plus kans
            default_vars = ['strat', 'lithok']
            vars_to_load = default_vars + KANS_VARS
            # Only select vars that exist in dataset
            vars_to_load = [v for v in vars_to_load if v in ds.data_vars]
            ds = ds[vars_to_load]

        if not lazy:
            print("Load data")
            ds = ds.load()

        ds = _follow_gdal_conventions(ds)
        return cls(ds, cellsize, dz, crs)

    @classmethod
    def from_opendap(
        cls,
        url: str = r"https://dinodata.nl/opendap/GeoTOP/geotop.nc",
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        lazy: bool = True,
        include_kans: bool = False,
        **xr_kwargs,
    ):
        """
        Download an area of GeoTop directly from the OPeNDAP data server into a GeoTop
        VoxelModel instance.

        Parameters
        ----------
        url : str
            Url to the netcdf file on the OPeNDAP server. See:
            https://www.dinoloket.nl/modelbestanden-aanvragen
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple, optional
            Enter a tuple (xmin, ymin, xmax, ymax) to return a selected area of GeoTop.
            The default is None but for practical reasons, specifying a bounding box is
            advised (TODO: find max downloadsize for server).
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.
        include_kans : bool, optional
            If True, include kans_1 through kans_9 probability variables for
            lithology uncertainty quantification. Default is False.

        Returns
        -------
        GeoTop
            GeoTop instance for the selected area.

        """
        return cls.from_netcdf(url, data_vars, bbox, lazy, include_kans, **xr_kwargs)

    @property
    def has_kans(self) -> bool:
        """Check if kans probability data is loaded."""
        return all(f'kans_{i}' in self.ds for i in range(1, 10))

    def validate_kans(self, tolerance: float = 5.0) -> bool:
        """
        Validate that kans probabilities sum to approximately 100% for valid voxels.

        Parameters
        ----------
        tolerance : float
            Allowed deviation from 100% (default 5.0 to account for rounding).

        Returns
        -------
        bool
            True if kans data is valid (sums to ~100% where data exists).
        """
        if not self.has_kans:
            return False

        # Sum all kans values
        total = sum(self.ds[f'kans_{i}'] for i in range(1, 10))

        # Check where data is valid (not NaN) and sums approximately to 100
        valid_mask = ~np.isnan(total.values)
        if not np.any(valid_mask):
            return False

        valid_totals = total.values[valid_mask]
        return np.all(np.abs(valid_totals - 100) <= tolerance)


class Nl3d(VoxelModel):
    @classmethod
    def from_netcdf(
        cls,
        nc_path: str | WindowsPath,
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        """
        Read the NL3D subsurface model from a netcdf dataset into a Nl3d VoxelModel
        instance. NL3D can be downloaded from: https://dinodata.nl/opendap.

        Parameters
        ----------
        nc_path : str | WindowsPath
            Path to the netcdf file of NL3D.
        data_vars : ArrayLike
            List or array-like object specifying which data variables to return.
        bbox : tuple, optional
            Enter a tuple (xmin, ymin, xmax, ymax) to return a selected area of NL3D.
            The default is None.
        lazy : bool, optional
            If True, netcdf loads lazily. Use False for speed improvements for larger
            areas but that still fit into memory. The default is False.

        Returns
        -------
        Nl3d
            Nl3d instance of the netcdf file.

        """
        cellsize = 250
        dz = 1.0
        crs = 28992

        if lazy and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = "auto"

        ds = xr.open_dataset(nc_path, **xr_kwargs)
        ds = cls.coordinates_to_cellcenters(ds, cellsize, dz)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print("Load data")
            ds = ds.load()

        ds = _follow_gdal_conventions(ds)
        return cls(ds, cellsize, dz, crs)

    @classmethod
    def from_opendap(
        cls,
        strat_url=r"https://dinodata.nl/opendap/NL3D/nl3d_lithostrat.nc",
        lithok_url=r"https://dinodata.nl/opendap/NL3D/nl3d_lithoklasse.nc",
        data_vars: ArrayLike = None,
        bbox: tuple = None,
        lazy: bool = True,
        **xr_kwargs,
    ):
        cellsize = 250
        dz = 1.0
        crs = 28992

        if lazy and "chunks" not in xr_kwargs:
            xr_kwargs["chunks"] = "auto"

        ds = xr.Dataset()

        if "lithostrat" in data_vars or data_vars is None:
            strat = xr.open_dataset(
                strat_url, drop_variables=["crs", "lat", "lon"], **xr_kwargs
            )

            for var in strat.data_vars:
                ds[var] = strat[var]

        if "lithoklasse" in data_vars or data_vars is None:
            lithok = xr.open_dataset(
                lithok_url, drop_variables=["crs", "lat", "lon"], **xr_kwargs
            )

            for var in lithok.data_vars:
                ds[var] = lithok[var]

        ds = cls.coordinates_to_cellcenters(ds, cellsize, dz)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        if data_vars is not None:
            ds = ds[data_vars]

        if not lazy:
            print("Load data")
            ds = ds.load()

        ds = _follow_gdal_conventions(ds)

        return cls(ds, cellsize, dz, crs)
