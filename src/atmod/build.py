from pathlib import Path, WindowsPath
from typing import List, Optional, TypeVar

import numpy as np
import xarray as xr

from atmod.base import AtlansParameters, AtlansStrat, Mapping, Raster, VoxelModel
from atmod.bro_models import BroBodemKaart
from atmod.merge import combine_data_sources
from atmod.preprocessing import (
    NumbaDicts,
    map_geotop_strat,
    map_nl3d_strat,
    soilmap_to_raster,
)
from atmod.templates import build_template, dask_output_model_like
from atmod.utils import COMPRESSION, find_overlapping_areas
from atmod.warnings import suppress_warnings

BodemKaartDicts = TypeVar("BodemKaartDicts")


def get_2d_template_like(model: Raster | VoxelModel) -> Raster:
    xmin_center = model.xmin + (0.5 * model.cellsize)
    ymin_center = model.ymin + (0.5 * model.cellsize)

    return build_template(
        model.ncols, model.nrows, xmin_center, ymin_center, model.cellsize
    )


@suppress_warnings(RuntimeWarning)
def _calc_rho_bulk(voxelmodel, parameters):
    organic = voxelmodel["mass_fraction_organic"]
    rho_bulk = (100 / organic) * (1 - np.exp(-organic / 0.12))
    rho_bulk = rho_bulk.fillna(parameters.rho_bulk).where(voxelmodel.isvalid)
    return rho_bulk


def calculate_domainbase(voxelmodel, parameters):
    thickness_holocene = np.nansum(
        voxelmodel["thickness"].where(voxelmodel["geology"] == AtlansStrat.holocene),
        axis=2,
    )
    surface_level_voxels = parameters.modelbase + np.nansum(
        voxelmodel["thickness"], axis=2
    )
    domainbase = surface_level_voxels - thickness_holocene
    domainbase[thickness_holocene == 0] = np.nan
    return domainbase


def create_atlantis_variables(voxelmodel, parameters, glg=None):
    if glg is not None:
        voxelmodel["phreatic_level"] = (glg.dims, glg.values)

    voxelmodel["rho_bulk"] = _calc_rho_bulk(voxelmodel, parameters)
    voxelmodel["zbase"] = xr.full_like(
        voxelmodel["surface_level"], parameters.modelbase
    )

    voxelmodel["max_oxidation_depth"] = xr.full_like(
        voxelmodel["surface_level"], parameters.max_oxidation_depth
    )
    voxelmodel["no_oxidation_thickness"] = xr.full_like(
        voxelmodel["surface_level"], parameters.no_oxidation_thickness
    )
    voxelmodel["no_shrinkage_thickness"] = xr.full_like(
        voxelmodel["surface_level"], parameters.no_shrinkage_thickness
    )
    domainbase = calculate_domainbase(voxelmodel, parameters)
    voxelmodel["domainbase"] = (("y", "x"), domainbase)

    return voxelmodel


def build_atlantis_model(
    ahn: Raster,
    geotop: VoxelModel,  # TODO: Also make input for geotop optional
    nl3d: VoxelModel = None,
    bodemkaart: Mapping = None,
    glg: Raster = None,
    parameters: AtlansParameters = None,
):
    """
    Create a 3D subsurface model that can be used as input for subsidence modelling in
    Atlantis Julia. A minimal subsurface model combines AHN (Algemeen Hoogtebestand
    Nederland) with the GeoTOP 3D voxelmodel. For a more detailed model or for areas
    where GeoTOP is unavailable, the BRO Bodemkaart and the NL3D 3D voxelmodel can be
    included. When the BRO Bodemkaart is included, the top 1.2 m below the surface level
    is replaced by the buildup as specified in the Bodemkaart.

    Parameters
    ----------
    ahn : Raster
        Raster instance with the AHN data to use for the surface level for the area
        where the model is created for.
    geotop : VoxelModel
        VoxelModel instance with the GeoTOP 3D voxelmodel data. See atmod.bro_models.GeoTop.
    bodemkaart : Mapping, optional
        Mapping instance containing the BRO Bodemkaart data. See atmod.bro_models.BroBodemKaart.
        The default is None.
    glg : Raster, optional
        Optional Raster instance with the GLG data to use for the phreatic level for the area
        where the model is created for. The default is None, then the GLG needs to be added
        manually because a phreatic level is mandatory input for an Atlantis subsurface model.
    parameters : AtlansParameters, optional
        Optional parameter class to specify default parameters for variables to use in the
        model. The default is None, then all Atlantis defaults will be used for each parameter.

    Returns
    -------
    xr.Dataset
        Xarray Dataset with the required input variables for an Atlantis subsurface model
        (N.B. mind the optional GLG input) that can be stored as a Netcdf.

    """  # noqa: E501
    if bodemkaart is not None:
        soilmap_dicts = NumbaDicts.from_soilmap(bodemkaart)
        soilmap = soilmap_to_raster(bodemkaart, ahn)
    else:
        soilmap_dicts = NumbaDicts.empty()
        soilmap = None

    geotop = map_geotop_strat(geotop)

    if nl3d is not None:
        nl3d = map_nl3d_strat(nl3d)

    if parameters is None:
        parameters = AtlansParameters()  # use all defaults

    voxelmodel = combine_data_sources(
        ahn, geotop, parameters, nl3d, soilmap, soilmap_dicts
    )
    voxelmodel = create_atlantis_variables(voxelmodel, parameters, glg)

    voxelmodel = voxelmodel.ds
    voxelmodel = voxelmodel.rename({"z": "layer"})  # No longer a vald atmod.VoxelModel
    voxelmodel["layer"] = np.arange(len(voxelmodel["layer"])) + 1

    return voxelmodel.astype("float64")


def build_model_in_chunks(
    ahn: Raster,
    geotop: VoxelModel,
    nl3d: VoxelModel,
    bodemkaart: Mapping,
    glg: Raster,
    parameters: AtlansParameters = None,
    chunksize: int = 250,
):
    if parameters is None:
        parameters = AtlansParameters()  # use all defaults

    geotop = geotop.select(z=slice(parameters.modelbase, parameters.modeltop))

    overlapping_area = find_overlapping_areas(ahn, geotop, nl3d, glg)
    geotop = geotop.select_in_bbox(overlapping_area)

    soilmap_dicts = NumbaDicts.from_soilmap(bodemkaart)
    soilmap = soilmap_to_raster(bodemkaart, ahn)

    model = dask_output_model_like(geotop, chunksize, True, True)

    model = model.map_blocks(
        _write_model_chunk,
        kwargs={
            "ahn": ahn,
            "geotop": geotop,
            "nl3d": nl3d,
            "soilmap": soilmap,
            "soilmap_dicts": soilmap_dicts,
            "glg": glg,
            "parameters": parameters,
        },
        template=model,
    )
    model = model.rename({"z": "layer"})
    model["layer"] = np.arange(len(model["layer"])) + 1

    return model


def _write_model_chunk(chunk, **kwargs):

    xmin, xmax = chunk["x"].min(), chunk["x"].max()
    ymin, ymax = chunk["y"].min(), chunk["y"].max()

    ahn = kwargs["ahn"].select(x=slice(xmin, xmax), y=slice(ymax, ymin))
    geotop = kwargs["geotop"].select(x=slice(xmin, xmax), y=slice(ymax, ymin))
    nl3d = kwargs["nl3d"].select(x=slice(xmin, xmax), y=slice(ymax, ymin))
    soilmap = kwargs["soilmap"].select(x=slice(xmin, xmax), y=slice(ymax, ymin))
    glg = kwargs["glg"].select(x=slice(xmin, xmax), y=slice(ymax, ymin))
    soilmap_dicts = kwargs["soilmap_dicts"]
    params = kwargs["parameters"]

    geotop = map_geotop_strat(geotop)
    nl3d = map_nl3d_strat(nl3d)

    model = combine_data_sources(ahn, geotop, params, nl3d, soilmap, soilmap_dicts)
    model = create_atlantis_variables(model, params, glg)

    for var in model.data_vars:
        chunk[var].data = model[var]

    return chunk


def build_ensemble_models(
    ahn: Raster,
    geotop: VoxelModel,
    bodemkaart: Mapping = None,
    glg: Raster = None,
    nl3d: VoxelModel = None,
    parameters: AtlansParameters = None,
    n_realizations: int = 100,
    output_dir: Optional[Path] = None,
    base_seed: int = 42,
    holocene_only: bool = True,
) -> List[xr.Dataset]:
    """
    Build ensemble of Atlantis subsurface models from lithology realizations.

    Uses GeoTOP kans probability distributions to generate multiple lithology
    realizations, enabling Monte Carlo uncertainty quantification of subsidence
    predictions.

    Parameters
    ----------
    ahn : Raster
        Raster instance with AHN surface elevation data.
    geotop : VoxelModel
        GeoTop instance with kans_1-9 loaded (via include_kans=True).
    bodemkaart : Mapping, optional
        BRO Bodemkaart data for top 1.2m soil profile. Default is None.
    glg : Raster, optional
        GLG phreatic level data. Default is None.
    nl3d : VoxelModel, optional
        NL3D voxelmodel for areas outside GeoTOP coverage. Default is None.
    parameters : AtlansParameters, optional
        Model parameters. Default uses Atlantis defaults.
    n_realizations : int, default 100
        Number of lithology realizations to generate.
    output_dir : Path, optional
        If provided, save each model to NetCDF as atlantis_realization_XXX.nc.
    base_seed : int, default 42
        Base random seed. Each realization uses seed = base_seed + i.
    holocene_only : bool, default True
        If True, only sample Holocene materials; keep older deterministic.

    Returns
    -------
    List[xr.Dataset]
        List of N Atlantis model datasets, one per realization.

    Examples
    --------
    >>> geotop = GeoTop.from_opendap(url, bbox, include_kans=True)
    >>> models = build_ensemble_models(
    ...     ahn=ahn, geotop=geotop, bodemkaart=bodemkaart, glg=glg,
    ...     n_realizations=50, output_dir=Path("./ensemble")
    ... )
    >>> # Analyze ensemble variance
    >>> rho_bulk_std = np.std([m['rho_bulk'] for m in models], axis=0)
    """
    from atmod.uncertainty import (
        create_geotop_realization,
        generate_lithology_ensemble,
    )

    # Validate that geotop has kans data
    if not geotop.has_kans:
        raise ValueError(
            "GeoTop must have kans probability data loaded. "
            "Use GeoTop.from_opendap(..., include_kans=True)."
        )

    if parameters is None:
        parameters = AtlansParameters()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all lithology realizations upfront
    ensemble = generate_lithology_ensemble(
        geotop,
        n_realizations=n_realizations,
        holocene_only=holocene_only,
        base_seed=base_seed,
    )

    models = []
    for i in range(n_realizations):
        # Extract single realization
        sampled_lith = ensemble.isel(realization=i)

        # Create GeoTop with sampled lithology
        geotop_i = create_geotop_realization(geotop, sampled_lith)

        # Build model using existing pipeline
        model_i = build_atlantis_model(
            ahn=ahn,
            geotop=geotop_i,
            nl3d=nl3d,
            bodemkaart=bodemkaart,
            glg=glg,
            parameters=parameters,
        )

        # Add realization metadata
        model_i.attrs['realization'] = i
        model_i.attrs['seed'] = base_seed + i
        model_i.attrs['holocene_only_sampling'] = holocene_only

        if output_dir is not None:
            model_i.to_netcdf(
                output_dir / f'atlantis_realization_{i:03d}.nc',
                encoding={var: COMPRESSION for var in model_i.data_vars}
            )

        models.append(model_i)

    return models
