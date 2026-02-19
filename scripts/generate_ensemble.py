#!/usr/bin/env python
"""
Generate ensemble of Atlantis subsurface models for lithology uncertainty quantification.

This script:
1. Loads GeoTOP with kans probability data
2. Samples lithology from kans distributions (Holocene-only)
3. Creates N subsurface model realizations
4. Saves each realization and summary statistics

Usage:
    python scripts/generate_ensemble.py --n_realizations 10 --output_dir output/ensemble_test
"""

import argparse
from pathlib import Path
import time

import numpy as np
import xarray as xr

from atmod import AtlansParameters, build_atlantis_model
from atmod.bro_models import GeoTop
from atmod.uncertainty import (
    sample_lithology_from_kans,
    create_geotop_realization,
    generate_lithology_ensemble,
    compute_lithology_entropy,
    compute_mode_probability,
)


def create_synthetic_ahn(bbox, resolution=100):
    """
    Create synthetic AHN data for testing.

    In production, replace with real AHN data.
    """
    xmin, ymin, xmax, ymax = bbox

    nx = int((xmax - xmin) / resolution)
    ny = int((ymax - ymin) / resolution)

    x = np.linspace(xmin + resolution/2, xmax - resolution/2, nx)
    y = np.linspace(ymin + resolution/2, ymax - resolution/2, ny)

    xx, yy = np.meshgrid(x, y)

    # Low-lying polder landscape (0-2m NAP)
    elevation = (
        1.0 +
        0.5 * np.sin(xx/2000) +
        0.3 * np.cos(yy/2000) +
        0.1 * np.sin(xx/500) * np.cos(yy/500)
    )

    da = xr.DataArray(
        elevation.astype(np.float32),
        dims=['y', 'x'],
        coords={'x': x, 'y': y},
        name='surface'
    )
    da.attrs['crs'] = 'EPSG:28992'

    return da


def create_synthetic_glg(ahn, depth_below_surface=1.2):
    """
    Create synthetic GLG (mean lowest groundwater) data.

    In production, replace with real GLG data.
    """
    glg = ahn - depth_below_surface
    glg.name = 'phreatic_level'
    return glg


def main():
    parser = argparse.ArgumentParser(
        description='Generate ensemble of Atlantis models for uncertainty quantification'
    )
    parser.add_argument(
        '--n_realizations', type=int, default=10,
        help='Number of realizations to generate (default: 10)'
    )
    parser.add_argument(
        '--base_seed', type=int, default=42,
        help='Base random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='output/ensemble',
        help='Output directory for ensemble files'
    )
    parser.add_argument(
        '--bbox', type=float, nargs=4,
        default=[127_000, 448_000, 132_000, 453_000],
        help='Bounding box: xmin ymin xmax ymax (default: 5x5km near Utrecht)'
    )
    parser.add_argument(
        '--lazy', action='store_true',
        help='Use lazy loading for GeoTOP (slower but less memory)'
    )

    args = parser.parse_args()

    bbox = tuple(args.bbox)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Lithology Uncertainty Ensemble Generation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Realizations: {args.n_realizations}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Bounding box: {bbox}")
    print(f"  Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # Step 1: Load GeoTOP with kans data
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 1: Loading GeoTOP with kans probability data")
    print("-" * 50)

    t0 = time.time()
    print("Connecting to OPeNDAP server...")
    print("Note: This downloads kans_1-9 in addition to strat/lithok")

    try:
        geotop = GeoTop.from_opendap(
            bbox=bbox,
            include_kans=True,  # Load probability distributions
            lazy=args.lazy
        )
        print(f"✓ GeoTOP loaded in {time.time() - t0:.1f}s")
        print(f"  Dimensions: {dict(geotop.ds.sizes)}")
        print(f"  Variables: {list(geotop.ds.data_vars)}")
        print(f"  Has kans: {geotop.has_kans}")
    except Exception as e:
        print(f"✗ Error loading GeoTOP: {e}")
        return 1

    # -------------------------------------------------------------------------
    # Step 2: Compute uncertainty metrics
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 2: Computing uncertainty metrics")
    print("-" * 50)

    print("Computing lithology entropy (higher = more uncertain)...")
    entropy = compute_lithology_entropy(geotop)

    print("Computing mode probability (lower = more uncertain)...")
    mode_prob = compute_mode_probability(geotop)

    # Save metrics
    metrics = xr.Dataset({
        'entropy': entropy,
        'mode_probability': mode_prob
    })
    metrics_file = output_dir / 'uncertainty_metrics.nc'
    metrics.to_netcdf(metrics_file)
    print(f"✓ Saved uncertainty metrics to {metrics_file}")

    # Print summary
    valid_entropy = entropy.values[~np.isnan(entropy.values)]
    valid_mode = mode_prob.values[~np.isnan(mode_prob.values)]
    if len(valid_entropy) > 0:
        print(f"\nEntropy statistics:")
        print(f"  Mean: {np.mean(valid_entropy):.3f} nats")
        print(f"  Max:  {np.max(valid_entropy):.3f} nats (max possible: {np.log(9):.3f})")
        print(f"\nMode probability statistics:")
        print(f"  Mean: {np.mean(valid_mode):.1f}%")
        print(f"  Min:  {np.min(valid_mode):.1f}% (most uncertain voxels)")

    # -------------------------------------------------------------------------
    # Step 3: Create synthetic AHN and GLG
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 3: Preparing surface and groundwater data")
    print("-" * 50)

    print("Creating synthetic AHN (surface elevation)...")
    ahn = create_synthetic_ahn(bbox, resolution=100)
    print(f"  Grid: {ahn.sizes['x']} x {ahn.sizes['y']}")
    print(f"  Elevation range: {float(ahn.min()):.2f} to {float(ahn.max()):.2f} m NAP")

    print("Creating synthetic GLG (phreatic level)...")
    glg = create_synthetic_glg(ahn, depth_below_surface=1.2)

    # -------------------------------------------------------------------------
    # Step 4: Generate lithology ensemble
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print(f"Step 4: Generating {args.n_realizations} lithology realizations")
    print("-" * 50)

    t0 = time.time()
    print("Sampling lithology from kans distributions (Holocene-only)...")

    ensemble = generate_lithology_ensemble(
        geotop,
        n_realizations=args.n_realizations,
        holocene_only=True,
        base_seed=args.base_seed
    )

    print(f"✓ Ensemble generated in {time.time() - t0:.1f}s")
    print(f"  Shape: {ensemble.shape} (realizations, z, y, x)")

    # Save raw ensemble (convert bool attrs to int for NetCDF compatibility)
    ensemble_file = output_dir / 'lithology_ensemble.nc'
    ensemble_to_save = ensemble.copy()
    if 'holocene_only' in ensemble_to_save.attrs:
        ensemble_to_save.attrs['holocene_only'] = int(ensemble_to_save.attrs['holocene_only'])
    ensemble_to_save.to_netcdf(ensemble_file)
    print(f"✓ Saved lithology ensemble to {ensemble_file}")

    # -------------------------------------------------------------------------
    # Step 5: Build Atlantis models for each realization
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print(f"Step 5: Building {args.n_realizations} Atlantis models")
    print("-" * 50)

    parameters = AtlansParameters()
    models = []

    for i in range(args.n_realizations):
        print(f"\n  Realization {i+1}/{args.n_realizations} (seed={args.base_seed + i})...")
        t0 = time.time()

        # Get sampled lithology for this realization
        sampled_lith = ensemble.isel(realization=i)

        # Create GeoTop with sampled lithology
        geotop_i = create_geotop_realization(geotop, sampled_lith)

        # Build Atlantis model
        try:
            model_i = build_atlantis_model(
                ahn=ahn,
                geotop=geotop_i,
                glg=glg,
                parameters=parameters
            )

            # Add metadata (use int for bools for NetCDF compatibility)
            model_i.attrs['realization'] = i
            model_i.attrs['seed'] = args.base_seed + i
            model_i.attrs['holocene_only_sampling'] = 1

            # Save individual model
            model_file = output_dir / f'atlantis_realization_{i:03d}.nc'
            model_i.to_netcdf(model_file)

            models.append(model_i)
            print(f"    ✓ Built and saved in {time.time() - t0:.1f}s")

        except Exception as e:
            print(f"    ✗ Error building model: {e}")
            continue

    # -------------------------------------------------------------------------
    # Step 6: Compute ensemble statistics
    # -------------------------------------------------------------------------
    print("\n" + "-" * 50)
    print("Step 6: Computing ensemble statistics")
    print("-" * 50)

    if len(models) > 0:
        # Stack key variables across realizations
        vars_to_analyze = ['lithology', 'mass_fraction_organic', 'rho_bulk']

        stats = xr.Dataset()

        for var in vars_to_analyze:
            if var in models[0].data_vars:
                # Stack across realizations
                stacked = np.stack([m[var].values for m in models], axis=0)

                # Compute statistics
                stats[f'{var}_mean'] = xr.DataArray(
                    np.nanmean(stacked, axis=0),
                    dims=models[0][var].dims,
                    coords=models[0][var].coords
                )
                stats[f'{var}_std'] = xr.DataArray(
                    np.nanstd(stacked, axis=0),
                    dims=models[0][var].dims,
                    coords=models[0][var].coords
                )

                print(f"  {var}:")
                valid_std = stats[f'{var}_std'].values[~np.isnan(stats[f'{var}_std'].values)]
                if len(valid_std) > 0:
                    print(f"    Mean std: {np.mean(valid_std):.4f}")
                    print(f"    Max std:  {np.max(valid_std):.4f}")

        # Save statistics
        stats_file = output_dir / 'ensemble_statistics.nc'
        stats.to_netcdf(stats_file)
        print(f"\n✓ Saved ensemble statistics to {stats_file}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Ensemble Generation Complete!")
    print("=" * 70)

    print(f"\nOutput files in {output_dir}/:")
    print(f"  • uncertainty_metrics.nc    - Entropy and mode probability maps")
    print(f"  • lithology_ensemble.nc     - Raw sampled lithology ({args.n_realizations} realizations)")
    print(f"  • atlantis_realization_*.nc - Individual Atlantis models")
    print(f"  • ensemble_statistics.nc    - Mean and std across ensemble")

    print(f"\nNext steps:")
    print(f"  1. Inspect uncertainty_metrics.nc to see where uncertainty is highest")
    print(f"  2. Run each atlantis_realization_*.nc through Atlantis Julia")
    print(f"  3. Analyze spread in subsidence predictions")
    print(f"  4. If spread is large, consider increasing N or adding spatial correlation")

    return 0


if __name__ == "__main__":
    exit(main())
