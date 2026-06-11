# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-19

First consolidated release of the `atmod` package. This release groups a large
body of feature work — parcel-based modeling, lithology uncertainty
quantification, performance optimization, and compatibility fixes — into a single
published baseline.

### Added

#### Parcel-based modeling
- New `src/atmod/parcels/` module enabling subsurface models at parcel locations
  via a 1×N virtual pseudo-grid that Atlans.jl processes unchanged (exploiting
  column independence):
  - `data.py` — `ParcelData` / `ParcelModelConfig` dataclasses.
  - `aggregation.py` — centroid, mode, area-weighted, and probability aggregation.
  - `extractor.py` — `ParcelExtractor` for sampling subsurface data per parcel.
  - `virtual_grid.py` — `VirtualGridBuilder` for the Atlans.jl-compatible grid.
  - `remapper.py` — `ParcelResultMapper` to convert output back to parcel format.
  - `build.py` — `build_parcel_model`, `build_parcel_forcing`, `remap_parcel_results`.
- Unit and integration test coverage for the parcels module and build entry points.

#### Lithology uncertainty quantification
- Uncertainty sampling module for lithology UQ.
- Ensemble build function for generating uncertainty-aware model realizations.
- GeoTOP extended to load `kans_1`–`kans_9` probability data, enabling
  probability-based most-likely-lithology selection.
- Comprehensive test suite for the lithology uncertainty workflow.

### Changed

#### Performance
- Mode aggregation optimized for an ~18× speedup.

#### Compatibility
- Integrated ArcGIS compatibility work into the `src/atmod` structure.

### Fixed
- Bodemkaart V2024 `maparea_id` format compatibility.
- Y-axis orientation in visualization functions (with added smoke tests and
  enhanced fixtures).

### Housekeeping
- `.gitignore` updated to exclude development artifacts.
- Lint cleanups: import sorting and removal of unused variables / unnecessary
  f-strings across the new modules.
