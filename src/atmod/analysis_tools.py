"""
Analysis tools for Atlantis model development visualization and documentation.

This module provides functions to:
1. Analyze data source provenance from the model's data_source variable
2. Compute model summary statistics
3. Visualize model inputs and ensemble results
4. Generate documentation for model development

The main workflow is:
1. Build Atlantis model: model = build_atlantis_model(...)
   - This automatically creates a 'data_source' variable tracking provenance
2. Analyze provenance: fractions = compute_data_source_fractions(model)
3. Visualize: plot_data_source_map(fractions)

Data source codes (from atmod.merge):
- 0 = No Data (invalid cells)
- 1 = Bodemkaart (top soil layers from BRO soil map)
- 2 = GeoTOP (3D voxel model from TNO)
- 3 = NL3D (gap-fill voxel model for areas outside GeoTOP)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes
    from atmod.base import AtlansParameters, Raster, VoxelModel, Mapping


# =============================================================================
# Constants
# =============================================================================

# Data source identifiers for provenance tracking (must match merge.py)
SOURCE_NODATA = 0
SOURCE_BODEMKAART = 1
SOURCE_GEOTOP = 2
SOURCE_NL3D = 3

SOURCE_NAMES = {
    SOURCE_NODATA: "No Data",
    SOURCE_BODEMKAART: "Bodemkaart",
    SOURCE_GEOTOP: "GeoTOP",
    SOURCE_NL3D: "NL3D",
}


# =============================================================================
# Data Source Provenance Functions
# =============================================================================

def compute_data_source_fractions(
    model: xr.Dataset,
    holocene_only: bool = True,
) -> xr.Dataset:
    """
    Compute 2D maps showing fraction of model from each data source.

    This function uses the 'data_source' variable that is automatically created
    during model building (build_atlantis_model). The data_source variable
    tracks the exact origin of each voxel:
    - 1 = Bodemkaart (top soil layers from BRO soil map)
    - 2 = GeoTOP (3D voxel model from TNO)
    - 3 = NL3D (gap-fill voxel model for areas outside GeoTOP)

    Parameters
    ----------
    model : xr.Dataset
        Atlantis model with 'data_source' and 'thickness' variables.
        Created by build_atlantis_model().
    holocene_only : bool, default True
        If True, only compute fractions for Holocene layers (geology == 1).
        If False, compute for all valid layers.

    Returns
    -------
    xr.Dataset
        Dataset with 2D maps (y, x):
        - frac_bodemkaart: Fraction from Bodemkaart (0-1)
        - frac_geotop: Fraction from GeoTOP (0-1)
        - frac_nl3d: Fraction from NL3D (0-1)
        - total_thickness: Total thickness analyzed (m)
        - bodemkaart_thickness: Thickness from Bodemkaart (m)
        - geotop_thickness: Thickness from GeoTOP (m)
        - nl3d_thickness: Thickness from NL3D (m)

    Raises
    ------
    ValueError
        If model doesn't have 'data_source' variable.

    Examples
    --------
    >>> model = build_atlantis_model(ahn, geotop, bodemkaart=bodemkaart)
    >>> fractions = compute_data_source_fractions(model)
    >>> fractions['frac_bodemkaart'].plot()  # Map of bodemkaart contribution
    >>> print(f"Mean Bodemkaart fraction: {fractions['frac_bodemkaart'].mean().values:.1%}")
    """
    if 'data_source' not in model.data_vars:
        raise ValueError(
            "Model must have 'data_source' variable. "
            "Ensure model was built with build_atlantis_model() from atmod >= current version."
        )

    # Get arrays
    data_source = model['data_source'].values
    thickness = model['thickness'].values

    # Apply Holocene mask if requested
    if holocene_only and 'geology' in model.data_vars:
        holocene_mask = model['geology'].values == 1  # AtlansStrat.holocene
        thickness_masked = np.where(holocene_mask, thickness, np.nan)
    else:
        thickness_masked = thickness

    # Get coordinates
    y_coords = model.coords['y'].values
    x_coords = model.coords['x'].values
    shape = (len(y_coords), len(x_coords))

    # Calculate thickness per source for each column
    bodemkaart_thickness = np.zeros(shape, dtype=np.float32)
    geotop_thickness = np.zeros(shape, dtype=np.float32)
    nl3d_thickness = np.zeros(shape, dtype=np.float32)

    # Sum thickness by source (layer dimension is last)
    layer_dim = 2 if len(thickness.shape) == 3 else -1
    n_layers = thickness.shape[layer_dim] if layer_dim >= 0 else 1

    for k in range(n_layers):
        layer_thickness = thickness_masked[:, :, k]
        layer_source = data_source[:, :, k]
        valid = ~np.isnan(layer_thickness)

        bodemkaart_thickness += np.where(valid & (layer_source == SOURCE_BODEMKAART), layer_thickness, 0)
        geotop_thickness += np.where(valid & (layer_source == SOURCE_GEOTOP), layer_thickness, 0)
        nl3d_thickness += np.where(valid & (layer_source == SOURCE_NL3D), layer_thickness, 0)

    # Calculate total thickness
    total_thickness = bodemkaart_thickness + geotop_thickness + nl3d_thickness

    # Calculate fractions (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_bodemkaart = np.where(total_thickness > 0, bodemkaart_thickness / total_thickness, np.nan)
        frac_geotop = np.where(total_thickness > 0, geotop_thickness / total_thickness, np.nan)
        frac_nl3d = np.where(total_thickness > 0, nl3d_thickness / total_thickness, np.nan)

    # Mark invalid columns (no valid data) as NaN
    invalid = total_thickness == 0
    frac_bodemkaart[invalid] = np.nan
    frac_geotop[invalid] = np.nan
    frac_nl3d[invalid] = np.nan

    # Build output dataset
    coords = {'y': y_coords, 'x': x_coords}

    return xr.Dataset({
        'frac_bodemkaart': xr.DataArray(
            frac_bodemkaart.astype(np.float32), dims=['y', 'x'], coords=coords,
            attrs={'description': 'Fraction from Bodemkaart', 'units': '-'}
        ),
        'frac_geotop': xr.DataArray(
            frac_geotop.astype(np.float32), dims=['y', 'x'], coords=coords,
            attrs={'description': 'Fraction from GeoTOP', 'units': '-'}
        ),
        'frac_nl3d': xr.DataArray(
            frac_nl3d.astype(np.float32), dims=['y', 'x'], coords=coords,
            attrs={'description': 'Fraction from NL3D', 'units': '-'}
        ),
        'total_thickness': xr.DataArray(
            total_thickness.astype(np.float32), dims=['y', 'x'], coords=coords,
            attrs={'description': 'Total thickness analyzed', 'units': 'm'}
        ),
        'bodemkaart_thickness': xr.DataArray(
            bodemkaart_thickness.astype(np.float32), dims=['y', 'x'], coords=coords,
            attrs={'description': 'Thickness from Bodemkaart', 'units': 'm'}
        ),
        'geotop_thickness': xr.DataArray(
            geotop_thickness.astype(np.float32), dims=['y', 'x'], coords=coords,
            attrs={'description': 'Thickness from GeoTOP', 'units': 'm'}
        ),
        'nl3d_thickness': xr.DataArray(
            nl3d_thickness.astype(np.float32), dims=['y', 'x'], coords=coords,
            attrs={'description': 'Thickness from NL3D', 'units': 'm'}
        ),
    })


def get_data_source_3d(model: xr.Dataset) -> xr.DataArray:
    """
    Get the 3D data source array from a model.

    The data_source variable is automatically created during model building
    (build_atlantis_model) and tracks the exact origin of each voxel.

    Parameters
    ----------
    model : xr.Dataset
        Atlantis model with 'data_source' variable.

    Returns
    -------
    xr.DataArray
        3D array with source codes:
        - 0 = No data
        - 1 = Bodemkaart
        - 2 = GeoTOP
        - 3 = NL3D

    Raises
    ------
    ValueError
        If model doesn't have 'data_source' variable.
    """
    if 'data_source' not in model.data_vars:
        raise ValueError(
            "Model must have 'data_source' variable. "
            "Ensure model was built with build_atlantis_model() from atmod >= current version."
        )

    data_source = model['data_source']

    # Add metadata if not present
    if 'description' not in data_source.attrs:
        data_source.attrs['description'] = 'Data source for each voxel'
        data_source.attrs['flag_values'] = [SOURCE_NODATA, SOURCE_BODEMKAART, SOURCE_GEOTOP, SOURCE_NL3D]
        data_source.attrs['flag_meanings'] = 'no_data bodemkaart geotop nl3d'

    return data_source


# =============================================================================
# Model Summary Statistics
# =============================================================================

def compute_model_summary(model: xr.Dataset) -> Dict:
    """
    Compute summary statistics for model documentation.

    Parameters
    ----------
    model : xr.Dataset
        Atlantis model dataset.

    Returns
    -------
    dict
        Summary statistics including:
        - domain_bounds: (xmin, ymin, xmax, ymax)
        - n_columns: Number of (x,y) locations
        - n_layers: Number of vertical layers
        - depth_range: (min_z, max_z)
        - lithology_distribution: {class: percentage}
        - organic_content: {mean, max, std}
        - data_completeness: Fraction of non-NaN voxels

    Examples
    --------
    >>> summary = compute_model_summary(model)
    >>> print(f"Domain: {summary['domain_bounds']}")
    >>> print(f"Completeness: {summary['data_completeness']:.1%}")
    """
    summary = {}

    # Domain bounds
    x = model.coords['x'].values
    y = model.coords['y'].values
    summary['domain_bounds'] = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))

    # Grid dimensions
    summary['n_columns'] = len(x) * len(y)
    summary['grid_shape'] = (len(y), len(x))

    # Vertical extent
    if 'z' in model.dims:
        z = model.coords['z'].values
        summary['n_layers'] = len(z)
        summary['depth_range'] = (float(z.min()), float(z.max()))

    # Lithology distribution
    if 'lithology' in model.data_vars:
        lith = model['lithology'].values
        valid_lith = lith[~np.isnan(lith)].astype(int)
        if len(valid_lith) > 0:
            unique, counts = np.unique(valid_lith, return_counts=True)
            total = counts.sum()
            summary['lithology_distribution'] = {
                int(u): float(c) / total * 100 for u, c in zip(unique, counts)
            }

    # Organic content statistics
    if 'mass_fraction_organic' in model.data_vars:
        organic = model['mass_fraction_organic'].values
        valid_org = organic[~np.isnan(organic)]
        if len(valid_org) > 0:
            summary['organic_content'] = {
                'mean': float(np.mean(valid_org)),
                'max': float(np.max(valid_org)),
                'std': float(np.std(valid_org)),
            }

    # Data completeness
    if 'thickness' in model.data_vars:
        thickness = model['thickness'].values
        valid_frac = np.sum(~np.isnan(thickness)) / thickness.size
        summary['data_completeness'] = float(valid_frac)

    return summary


def compute_holocene_statistics(
    model: xr.Dataset,
    geology_var: str = 'geology',
    holocene_code: int = 1,
) -> xr.Dataset:
    """
    Compute per-column Holocene statistics.

    Parameters
    ----------
    model : xr.Dataset
        Atlantis model with thickness and geology variables.
    geology_var : str, default 'geology'
        Name of geology variable in model.
    holocene_code : int, default 1
        Code for Holocene geology in the model.

    Returns
    -------
    xr.Dataset
        2D maps (y, x):
        - holocene_thickness: Total thickness (m)
        - holocene_base: Elevation of Holocene base (m NAP)
        - n_holocene_layers: Number of Holocene layers
        - mean_organic: Mean organic content in Holocene

    Examples
    --------
    >>> holo_stats = compute_holocene_statistics(model)
    >>> holo_stats['holocene_thickness'].plot()
    """
    # Get dimensions
    y_coords = model.coords['y'].values
    x_coords = model.coords['x'].values
    shape = (len(y_coords), len(x_coords))

    # Initialize outputs
    holocene_thickness = np.zeros(shape, dtype=np.float32)
    holocene_base = np.full(shape, np.nan, dtype=np.float32)
    n_layers = np.zeros(shape, dtype=np.int16)
    mean_organic = np.full(shape, np.nan, dtype=np.float32)

    # Get required arrays
    if 'thickness' not in model.data_vars:
        raise ValueError("Model must have 'thickness' variable")

    thickness = model['thickness'].values

    if geology_var in model.data_vars:
        geology = model[geology_var].values
    else:
        # If no geology, assume all is Holocene for simplicity
        geology = np.ones_like(thickness)

    has_organic = 'mass_fraction_organic' in model.data_vars
    if has_organic:
        organic = model['mass_fraction_organic'].values

    # Get z coordinates for base calculation
    if 'z' in model.coords:
        z_coords = model.coords['z'].values
    else:
        z_coords = None

    # Calculate per-column statistics
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Get column data
            col_thickness = thickness[i, j, :]
            col_geology = geology[i, j, :]

            # Find Holocene voxels
            is_holocene = col_geology == holocene_code
            valid = ~np.isnan(col_thickness) & is_holocene

            if not np.any(valid):
                continue

            # Thickness
            holocene_thickness[i, j] = np.nansum(col_thickness[valid])

            # Number of layers
            n_layers[i, j] = np.sum(valid)

            # Base elevation (lowest Holocene voxel)
            if z_coords is not None:
                holocene_z = z_coords[valid]
                holocene_base[i, j] = np.min(holocene_z)

            # Mean organic content
            if has_organic:
                col_organic = organic[i, j, :]
                org_valid = valid & ~np.isnan(col_organic)
                if np.any(org_valid):
                    mean_organic[i, j] = np.nanmean(col_organic[org_valid])

    # Build output dataset
    coords = {'y': y_coords, 'x': x_coords}

    return xr.Dataset({
        'holocene_thickness': xr.DataArray(
            holocene_thickness, dims=['y', 'x'], coords=coords,
            attrs={'description': 'Total Holocene thickness', 'units': 'm'}
        ),
        'holocene_base': xr.DataArray(
            holocene_base, dims=['y', 'x'], coords=coords,
            attrs={'description': 'Elevation of Holocene base', 'units': 'm NAP'}
        ),
        'n_holocene_layers': xr.DataArray(
            n_layers, dims=['y', 'x'], coords=coords,
            attrs={'description': 'Number of Holocene layers'}
        ),
        'mean_organic': xr.DataArray(
            mean_organic, dims=['y', 'x'], coords=coords,
            attrs={'description': 'Mean organic fraction in Holocene', 'units': '-'}
        ),
    })


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_data_source_map(
    source_fractions: xr.Dataset,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: str = "Data Source Contributions",
    show_legend: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Create RGB map showing data source contributions.

    Colors:
    - Red channel: Bodemkaart fraction
    - Green channel: GeoTOP fraction
    - Blue channel: NL3D fraction

    Parameters
    ----------
    source_fractions : xr.Dataset
        Output from compute_data_source_fractions().
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, default "Data Source Contributions"
        Plot title.
    show_legend : bool, default True
        Whether to show color legend.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Examples
    --------
    >>> fractions = compute_data_source_fractions(...)
    >>> fig = plot_data_source_map(fractions)
    >>> fig.savefig('source_map.png')
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Get fractions
    r = source_fractions['frac_bodemkaart'].values
    g = source_fractions['frac_geotop'].values
    b = source_fractions['frac_nl3d'].values

    # Handle NaN values
    invalid = np.isnan(r) | np.isnan(g) | np.isnan(b)
    r = np.nan_to_num(r, nan=0.5)
    g = np.nan_to_num(g, nan=0.5)
    b = np.nan_to_num(b, nan=0.5)

    # Stack into RGB image
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 1)

    # Set invalid areas to gray
    rgb[invalid] = [0.5, 0.5, 0.5]

    # Get extent from coordinates
    x = source_fractions.coords['x'].values
    y = source_fractions.coords['y'].values
    dx = (x[1] - x[0]) / 2 if len(x) > 1 else 50
    dy = (y[1] - y[0]) / 2 if len(y) > 1 else 50
    extent = [x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy]

    # Plot
    ax.imshow(rgb, origin='upper', extent=extent, aspect='equal')
    ax.set_xlabel('X (m RD)')
    ax.set_ylabel('Y (m RD)')
    ax.set_title(title)

    # Add legend
    if show_legend:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Bodemkaart'),
            Patch(facecolor='green', label='GeoTOP'),
            Patch(facecolor='blue', label='NL3D'),
            Patch(facecolor='gray', label='No data'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig


def plot_cross_section(
    model: xr.Dataset,
    start: Tuple[float, float],
    end: Tuple[float, float],
    variable: str = 'lithology',
    source_mask: Optional[xr.DataArray] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    cmap: Optional[str] = None,
    n_points: int = 100,
) -> "matplotlib.figure.Figure":
    """
    Plot vertical cross-section through model.

    Parameters
    ----------
    model : xr.Dataset
        Atlantis model with 3D variables.
    start : tuple
        (x, y) start point of cross-section in RD coordinates.
    end : tuple
        (x, y) end point of cross-section.
    variable : str, default 'lithology'
        Variable to plot.
    source_mask : xr.DataArray, optional
        3D source mask from model['data_source'] or get_data_source_3d().
        If provided, overlays source boundaries on the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, optional
        Colormap name. If None, uses appropriate default.
    n_points : int, default 100
        Number of points along cross-section.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()

    # Create points along cross-section
    x_line = np.linspace(start[0], end[0], n_points)
    y_line = np.linspace(start[1], end[1], n_points)
    distance = np.sqrt((x_line - start[0])**2 + (y_line - start[1])**2)

    # Get z/layer coordinates (models can have either 'z' or 'layer' dimension)
    if 'z' in model.coords:
        z_coords = model.coords['z'].values
        z_dim = 'z'
    elif 'layer' in model.dims:
        # For layer-based models, use layer indices or domainbase if available
        z_coords = np.arange(model.dims['layer'])
        z_dim = 'layer'
    else:
        raise ValueError("Model must have either 'z' coordinate or 'layer' dimension")

    # Interpolate model along cross-section
    if variable not in model.data_vars:
        raise ValueError(f"Variable '{variable}' not in model")

    # Create cross-section array
    section = np.full((len(z_coords), n_points), np.nan)

    for i, (xi, yi) in enumerate(zip(x_line, y_line)):
        try:
            col = model[variable].sel(x=xi, y=yi, method='nearest')
            section[:, i] = col.values
        except Exception:
            pass

    # Choose colormap
    if cmap is None:
        if variable == 'lithology':
            cmap = 'tab10'
        elif 'organic' in variable.lower():
            cmap = 'YlOrBr'
        else:
            cmap = 'viridis'

    # Plot
    im = ax.pcolormesh(
        distance, z_coords, section,
        cmap=cmap, shading='auto'
    )

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Elevation (m NAP)')
    ax.set_title(f'Cross-section: {variable}')

    plt.colorbar(im, ax=ax, label=variable)

    # Overlay source boundaries if provided
    if source_mask is not None:
        # Extract source along same line and plot contours
        source_section = np.full((len(z_coords), n_points), np.nan)
        for i, (xi, yi) in enumerate(zip(x_line, y_line)):
            try:
                col = source_mask.sel(x=xi, y=yi, method='nearest')
                source_section[:, i] = col.values
            except Exception:
                pass

        # Add contour lines at source boundaries
        ax.contour(
            distance, z_coords, source_section,
            levels=[0.5, 1.5, 2.5], colors='white',
            linewidths=1, linestyles='--'
        )

    plt.tight_layout()
    return fig


def plot_holocene_thickness_map(
    model_or_stats: Union[xr.Dataset, xr.DataArray],
    ax: Optional["matplotlib.axes.Axes"] = None,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "matplotlib.figure.Figure":
    """
    2D map of Holocene package thickness.

    Parameters
    ----------
    model_or_stats : xr.Dataset or xr.DataArray
        Either output from compute_holocene_statistics() or a 2D DataArray.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default 'viridis'
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Get thickness data
    if isinstance(model_or_stats, xr.Dataset):
        if 'holocene_thickness' in model_or_stats:
            thickness = model_or_stats['holocene_thickness']
        else:
            raise ValueError("Dataset must contain 'holocene_thickness'")
    else:
        thickness = model_or_stats

    # Get extent
    x = thickness.coords['x'].values
    y = thickness.coords['y'].values
    dx = (x[1] - x[0]) / 2 if len(x) > 1 else 50
    dy = (y[1] - y[0]) / 2 if len(y) > 1 else 50
    extent = [x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy]

    # Plot
    im = ax.imshow(
        thickness.values, origin='upper', extent=extent,
        cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal'
    )

    ax.set_xlabel('X (m RD)')
    ax.set_ylabel('Y (m RD)')
    ax.set_title('Holocene Thickness')

    plt.colorbar(im, ax=ax, label='Thickness (m)')
    plt.tight_layout()

    return fig


def plot_lithology_distribution(
    model: xr.Dataset,
    lithology_names: Optional[Dict[int, str]] = None,
    ax: Optional["matplotlib.axes.Axes"] = None,
    by_depth: bool = False,
) -> "matplotlib.figure.Figure":
    """
    Bar chart of lithology class distribution.

    Parameters
    ----------
    model : xr.Dataset
        Atlantis model with lithology variable.
    lithology_names : dict, optional
        Mapping of lithology codes to names.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    by_depth : bool, default False
        If True, show distribution by depth layer.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if 'lithology' not in model.data_vars:
        raise ValueError("Model must have 'lithology' variable")

    # Default lithology names
    if lithology_names is None:
        lithology_names = {
            0: 'Anthropogenic',
            1: 'Organic',
            2: 'Clay',
            3: 'Loam',
            4: 'Fine sand',
            5: 'Medium sand',
            6: 'Coarse sand',
            7: 'Gravel',
            8: 'Shells',
            9: 'Peat (not NL)',
        }

    lith = model['lithology'].values
    valid = ~np.isnan(lith)
    lith_valid = lith[valid].astype(int)

    if len(lith_valid) == 0:
        ax.text(0.5, 0.5, 'No valid lithology data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    # Count occurrences
    unique, counts = np.unique(lith_valid, return_counts=True)
    percentages = counts / counts.sum() * 100

    # Create labels
    labels = [lithology_names.get(u, f'Class {u}') for u in unique]

    # Plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
    bars = ax.bar(labels, percentages, color=colors)

    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Lithology Class')
    ax.set_title('Lithology Distribution')
    ax.tick_params(axis='x', rotation=45)

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        if pct > 2:  # Only label bars > 2%
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


# =============================================================================
# Ensemble Analysis Functions
# =============================================================================

def compute_ensemble_statistics(
    models: List[xr.Dataset],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Compute statistics across ensemble realizations.

    Parameters
    ----------
    models : list of xr.Dataset
        List of Atlantis model realizations.
    variables : list of str, optional
        Variables to analyze. If None, analyzes common numeric variables.

    Returns
    -------
    xr.Dataset
        Statistics including for each variable:
        - {var}_mean: Ensemble mean
        - {var}_std: Ensemble standard deviation
        - {var}_p10, {var}_p50, {var}_p90: Percentiles

    Examples
    --------
    >>> models = [build_atlantis_model(...) for _ in range(10)]
    >>> stats = compute_ensemble_statistics(models)
    >>> stats['lithology_std'].plot()  # Uncertainty map
    """
    if len(models) == 0:
        raise ValueError("models list cannot be empty")

    # Determine variables to analyze
    if variables is None:
        # Find common numeric variables
        common_vars = set(models[0].data_vars)
        for m in models[1:]:
            common_vars &= set(m.data_vars)
        variables = [v for v in common_vars
                    if np.issubdtype(models[0][v].dtype, np.number)]

    stats = xr.Dataset()

    for var in variables:
        if var not in models[0].data_vars:
            continue

        # Stack values across realizations
        stacked = np.stack([m[var].values for m in models], axis=0)

        # Compute statistics
        with np.errstate(all='ignore'):
            mean_vals = np.nanmean(stacked, axis=0)
            std_vals = np.nanstd(stacked, axis=0)
            p10_vals = np.nanpercentile(stacked, 10, axis=0)
            p50_vals = np.nanpercentile(stacked, 50, axis=0)
            p90_vals = np.nanpercentile(stacked, 90, axis=0)

        # Create DataArrays
        template = models[0][var]
        stats[f'{var}_mean'] = xr.DataArray(
            mean_vals, dims=template.dims, coords=template.coords,
            attrs={'description': f'Ensemble mean of {var}'}
        )
        stats[f'{var}_std'] = xr.DataArray(
            std_vals, dims=template.dims, coords=template.coords,
            attrs={'description': f'Ensemble std of {var}'}
        )
        stats[f'{var}_p10'] = xr.DataArray(
            p10_vals, dims=template.dims, coords=template.coords,
            attrs={'description': f'10th percentile of {var}'}
        )
        stats[f'{var}_p50'] = xr.DataArray(
            p50_vals, dims=template.dims, coords=template.coords,
            attrs={'description': f'Median of {var}'}
        )
        stats[f'{var}_p90'] = xr.DataArray(
            p90_vals, dims=template.dims, coords=template.coords,
            attrs={'description': f'90th percentile of {var}'}
        )

    stats.attrs['n_realizations'] = len(models)
    return stats


def plot_ensemble_spread(
    ensemble_stats: xr.Dataset,
    variable: str,
    metric: str = 'std',
    ax: Optional["matplotlib.axes.Axes"] = None,
    cmap: str = 'Reds',
) -> "matplotlib.figure.Figure":
    """
    Map showing ensemble spread (std or IQR) for a variable.

    Parameters
    ----------
    ensemble_stats : xr.Dataset
        Output from compute_ensemble_statistics().
    variable : str
        Variable name (without _std suffix).
    metric : str, default 'std'
        Spread metric: 'std' or 'iqr' (p90-p10).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    cmap : str, default 'Reds'
        Colormap for spread values.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Get spread data
    if metric == 'std':
        var_name = f'{variable}_std'
        label = 'Standard Deviation'
    elif metric == 'iqr':
        if f'{variable}_p90' in ensemble_stats and f'{variable}_p10' in ensemble_stats:
            spread = ensemble_stats[f'{variable}_p90'] - ensemble_stats[f'{variable}_p10']
            var_name = None
            label = 'Interquartile Range (P90-P10)'
        else:
            raise ValueError("IQR requires p10 and p90 percentiles in stats")
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if var_name is not None:
        if var_name not in ensemble_stats:
            raise ValueError(f"'{var_name}' not in ensemble_stats")
        spread = ensemble_stats[var_name]

    # Reduce to 2D if needed (take column mean)
    if len(spread.dims) == 3:
        spread_2d = spread.mean(dim='z')
    else:
        spread_2d = spread

    # Get extent
    x = spread_2d.coords['x'].values
    y = spread_2d.coords['y'].values
    dx = (x[1] - x[0]) / 2 if len(x) > 1 else 50
    dy = (y[1] - y[0]) / 2 if len(y) > 1 else 50
    extent = [x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy]

    # Plot
    im = ax.imshow(
        spread_2d.values, origin='upper', extent=extent,
        cmap=cmap, aspect='equal'
    )

    ax.set_xlabel('X (m RD)')
    ax.set_ylabel('Y (m RD)')
    ax.set_title(f'Ensemble Spread: {variable} ({label})')

    plt.colorbar(im, ax=ax, label=label)
    plt.tight_layout()

    return fig


# =============================================================================
# Data Quality Functions
# =============================================================================

def compute_data_quality_flags(
    model: xr.Dataset,
    geotop_valid: Optional[xr.DataArray] = None,
    bodemkaart_mask: Optional[xr.DataArray] = None,
    entropy: Optional[xr.DataArray] = None,
) -> xr.Dataset:
    """
    Flag potential data quality issues for documentation.

    Parameters
    ----------
    model : xr.Dataset
        Atlantis model.
    geotop_valid : xr.DataArray, optional
        Boolean mask of GeoTOP coverage.
    bodemkaart_mask : xr.DataArray, optional
        Boolean mask of bodemkaart coverage.
    entropy : xr.DataArray, optional
        Lithology entropy from compute_lithology_entropy().

    Returns
    -------
    xr.Dataset
        Quality flags (2D maps):
        - uses_nl3d: Boolean, True where NL3D is used
        - no_bodemkaart: Boolean, True where bodemkaart is missing
        - high_entropy: Boolean, True where lithology uncertainty is high
        - has_gaps: Boolean, True where data gaps exist
    """
    y_coords = model.coords['y'].values
    x_coords = model.coords['x'].values
    shape = (len(y_coords), len(x_coords))
    coords = {'y': y_coords, 'x': x_coords}

    result = xr.Dataset()

    # NL3D usage flag
    if geotop_valid is not None:
        if isinstance(geotop_valid, xr.DataArray):
            gv = geotop_valid.values
        else:
            gv = geotop_valid
        result['uses_nl3d'] = xr.DataArray(
            ~gv, dims=['y', 'x'], coords=coords,
            attrs={'description': 'True where NL3D gap-fill is used'}
        )

    # Bodemkaart coverage flag
    if bodemkaart_mask is not None:
        if isinstance(bodemkaart_mask, xr.DataArray):
            bk = bodemkaart_mask.values
        else:
            bk = bodemkaart_mask
        result['no_bodemkaart'] = xr.DataArray(
            ~bk, dims=['y', 'x'], coords=coords,
            attrs={'description': 'True where bodemkaart is missing'}
        )

    # High entropy flag
    if entropy is not None:
        # High entropy threshold: > 1.0 nat (moderate uncertainty)
        high_entropy_mask = entropy.values > 1.0
        if len(high_entropy_mask.shape) == 3:
            # Reduce 3D to 2D: flag if any layer has high entropy
            high_entropy_2d = np.any(high_entropy_mask, axis=2)
        else:
            high_entropy_2d = high_entropy_mask
        result['high_entropy'] = xr.DataArray(
            high_entropy_2d, dims=['y', 'x'], coords=coords,
            attrs={'description': 'True where lithology entropy > 1.0 nat'}
        )

    # Data gaps flag
    if 'thickness' in model.data_vars:
        thickness = model['thickness'].values
        # Check for columns with all NaN
        if len(thickness.shape) == 3:
            has_gaps = np.all(np.isnan(thickness), axis=2)
        else:
            has_gaps = np.isnan(thickness)
        result['has_gaps'] = xr.DataArray(
            has_gaps, dims=['y', 'x'], coords=coords,
            attrs={'description': 'True where data gaps exist'}
        )

    return result


def compare_models(
    model_a: xr.Dataset,
    model_b: xr.Dataset,
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Compute differences between two models.

    Useful for comparing deterministic vs sampled models.

    Parameters
    ----------
    model_a, model_b : xr.Dataset
        Two Atlantis models to compare.
    variables : list of str, optional
        Variables to compare. If None, compares common numeric variables.

    Returns
    -------
    xr.Dataset
        Difference maps:
        - {var}_diff: model_b - model_a
        - {var}_abs_diff: |model_b - model_a|
        - {var}_pct_diff: Percentage difference where model_a != 0
    """
    if variables is None:
        common_vars = set(model_a.data_vars) & set(model_b.data_vars)
        variables = [v for v in common_vars
                    if np.issubdtype(model_a[v].dtype, np.number)]

    result = xr.Dataset()

    for var in variables:
        if var not in model_a.data_vars or var not in model_b.data_vars:
            continue

        a = model_a[var].values.astype(float)
        b = model_b[var].values.astype(float)

        diff = b - a
        abs_diff = np.abs(diff)

        with np.errstate(divide='ignore', invalid='ignore'):
            pct_diff = np.where(a != 0, diff / np.abs(a) * 100, np.nan)

        template = model_a[var]
        result[f'{var}_diff'] = xr.DataArray(
            diff, dims=template.dims, coords=template.coords,
            attrs={'description': f'Difference (B - A) for {var}'}
        )
        result[f'{var}_abs_diff'] = xr.DataArray(
            abs_diff, dims=template.dims, coords=template.coords,
            attrs={'description': f'Absolute difference for {var}'}
        )
        result[f'{var}_pct_diff'] = xr.DataArray(
            pct_diff, dims=template.dims, coords=template.coords,
            attrs={'description': f'Percentage difference for {var}', 'units': '%'}
        )

    return result
