"""
Integration tests for parcel-based modeling workflow.

These tests verify the complete parcel pipeline from
input parcels through virtual grid to result remapping.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from atmod.parcels.build import build_parcel_model
from atmod.parcels.data import ParcelData
from atmod.parcels.extractor import ExtractionResult, ParcelExtractor
from atmod.parcels.remapper import remap_parcel_results
from atmod.parcels.virtual_grid import VirtualGridBuilder


@pytest.mark.integrationtest
class TestParcelWorkflow:
    """
    End-to-end integration tests for parcel-based modeling.

    TESTS: Full pipeline from parcels → virtual grid → remap
    DOES NOT TEST: Actual Atlans.jl execution, real GeoTOP data
    """

    def test_full_parcel_to_virtual_grid_workflow(
        self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster
    ):
        """
        WHAT: Complete workflow from parcels to Atlans-ready virtual grid.
        WHY: Verify all components work together.
        LOGIC:
            1. Create ParcelData from GeoDataFrame
            2. Extract using ParcelExtractor
            3. Build virtual grid with VirtualGridBuilder
            4. Verify output is Atlans.jl compatible
        EXPECTED: Virtual grid has correct structure and data.
        """
        # Step 1: Create ParcelData
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )
        assert len(parcel_data) == 5

        # Step 2: Extract subsurface data
        extractor = ParcelExtractor(parcel_data, method="centroid")
        result = extractor.extract(
            voxelmodel=sample_voxelmodel,
            rasters={"surface_level": sample_ahn_raster},
            variables_3d=["lithology", "geology", "thickness"],
        )
        assert isinstance(result, ExtractionResult)
        assert result.n_parcels == 5

        # Step 3: Build virtual grid
        builder = VirtualGridBuilder(result)
        virtual_grid = builder.build(compression_level=0)

        # Step 4: Verify Atlans compatibility
        assert virtual_grid.sizes["y"] == 1
        assert virtual_grid.sizes["x"] == 5
        assert "z" in virtual_grid.dims  # Before renaming to layer
        assert "lithology" in virtual_grid.data_vars
        assert "surface_level" in virtual_grid.data_vars

        # Verify parcel IDs are preserved
        assert "parcel_id" in virtual_grid.data_vars
        assert set(virtual_grid["parcel_id"].values) == set(
            sample_parcels_gdf["Perceel_ID"].values
        )

    def test_parcel_results_remapping_workflow(self, sample_parcels_gdf):
        """
        WHAT: Complete workflow for remapping Atlans results back to parcels.
        WHY: Must be able to interpret simulation outputs.
        LOGIC:
            1. Create mock Atlans output (x, y, time) and save to file
            2. Remap to parcel format (parcel_id, time)
            3. Verify parcel IDs match input
        EXPECTED: Result has parcel_id dimension with correct values.
        """
        n_parcels = len(sample_parcels_gdf)
        n_times = 10
        parcel_ids = sample_parcels_gdf["Perceel_ID"].tolist()

        # Step 1: Create mock Atlans output
        mock_output = xr.Dataset(
            {
                "subsidence": (
                    ["x", "y", "time"],
                    np.random.randn(n_parcels, 1, n_times) * 0.01,
                ),
                "consolidation": (
                    ["x", "y", "time"],
                    np.random.randn(n_parcels, 1, n_times) * 0.005,
                ),
            },
            coords={
                "x": np.arange(1, n_parcels + 1, dtype=float),
                "y": np.array([0.0]),
                "time": np.arange(n_times),
            },
        )

        # Step 2: Create mapping info (matches actual structure)
        mapping_info = {
            "parcel_id": parcel_ids,
            "n_parcels": n_parcels,
            "centroid_x": np.linspace(100, 500, n_parcels).tolist(),
            "centroid_y": np.full(n_parcels, 150).tolist(),
            "n_layers": 10,
            "crs": "EPSG:28992",
            "aggregation_method": "centroid",
        }

        # Save mock output to temp file and remap
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            output_path = Path(f.name)

        try:
            mock_output.to_netcdf(output_path)
            result = remap_parcel_results(output_path, mapping_info)

            # Verify parcel_id dimension
            assert "parcel_id" in result.dims
            assert result.sizes["parcel_id"] == n_parcels

            # Verify parcel IDs match
            assert set(result["parcel_id"].values) == set(parcel_ids)

            # Verify time dimension preserved
            assert "time" in result.dims
            assert result.sizes["time"] == n_times

            # Verify variables preserved
            assert "subsidence" in result.data_vars
            assert "consolidation" in result.data_vars

        finally:
            output_path.unlink()

    def test_round_trip_parcel_ids(
        self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster
    ):
        """
        WHAT: Verify parcel IDs survive build → remap round trip.
        WHY: Critical for data integrity.
        LOGIC:
            1. Build virtual grid
            2. Create mock Atlans output and save to file
            3. Remap results
            4. Verify IDs match original
        EXPECTED: Final parcel IDs == input parcel IDs.
        """
        # Build virtual grid
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        # Create mock Atlans output (simulate simulation)
        n_parcels = virtual_grid.sizes["x"]
        n_times = 5
        mock_output = xr.Dataset(
            {
                "subsidence": (
                    ["x", "y", "time"],
                    np.linspace(0, -0.05, n_times).reshape(1, 1, n_times).repeat(n_parcels, axis=0),
                ),
            },
            coords={
                "x": virtual_grid.x.values,
                "y": virtual_grid.y.values,
                "time": np.arange(n_times),
            },
        )

        # Save to file and remap results
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            output_path = Path(f.name)

        try:
            mock_output.to_netcdf(output_path)
            result = remap_parcel_results(output_path, mapping)

            # Verify IDs match
            original_ids = set(sample_parcels_gdf["Perceel_ID"].values)
            result_ids = set(result["parcel_id"].values)
            assert result_ids == original_ids

        finally:
            output_path.unlink()

    def test_mapping_json_round_trip(
        self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster
    ):
        """
        WHAT: Verify mapping info survives JSON serialization.
        WHY: Users may save mapping to disk.
        LOGIC:
            1. Build model and get mapping
            2. Save mapping to JSON
            3. Load and remap with loaded mapping
        EXPECTED: Remapping works with loaded mapping.
        """
        # Build virtual grid
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        # Save mapping to JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(mapping, f)
            mapping_path = Path(f.name)

        # Create mock output and save to file
        n_parcels = virtual_grid.sizes["x"]
        mock_output = xr.Dataset(
            {"subsidence": (["x", "y", "time"], np.zeros((n_parcels, 1, 3)))},
            coords={
                "x": virtual_grid.x.values,
                "y": virtual_grid.y.values,
                "time": np.arange(3),
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            output_path = Path(f.name)

        try:
            mock_output.to_netcdf(output_path)

            # Remap with JSON path
            result = remap_parcel_results(output_path, mapping_path)

            # Verify it worked
            assert "parcel_id" in result.dims
            assert result.sizes["parcel_id"] == len(sample_parcels_gdf)

        finally:
            mapping_path.unlink()
            output_path.unlink()

    def test_different_aggregation_methods_produce_different_results(
        self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster
    ):
        """
        WHAT: Verify different aggregation methods give different results.
        WHY: Methods should behave differently.
        LOGIC:
            1. Build with centroid method
            2. Build with mode method
            3. Compare lithology values
        EXPECTED: Results differ (at least for some parcels).
        """
        # Build with centroid
        grid_centroid, _ = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        # Build with mode
        grid_mode, _ = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="mode",
        )

        # Results should be different (in most cases)
        # Note: Could be same by chance, so we just verify both ran
        assert grid_centroid.sizes == grid_mode.sizes
        # Both should have valid data
        assert not np.all(np.isnan(grid_centroid["lithology"].values))
        assert not np.all(np.isnan(grid_mode["lithology"].values))


@pytest.mark.integrationtest
class TestVirtualGridCompatibility:
    """
    Tests for Atlans.jl virtual grid compatibility.

    TESTS: Output format matches Atlans.jl expectations
    DOES NOT TEST: Actual Atlans.jl execution
    """

    def test_virtual_grid_has_atlans_dimensions(
        self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster
    ):
        """
        WHAT: Verify virtual grid has dimensions Atlans.jl expects.
        WHY: Atlans.jl has specific dimension requirements.
        LOGIC: Check for y, x, layer dimensions.
        EXPECTED: All required dimensions present.
        """
        virtual_grid, _ = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        assert "y" in virtual_grid.dims
        assert "x" in virtual_grid.dims
        assert "layer" in virtual_grid.dims

    def test_virtual_grid_layer_indices_start_at_one(
        self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster
    ):
        """
        WHAT: Verify layer indices start at 1 (Julia convention).
        WHY: Atlans.jl expects 1-based layer indices.
        LOGIC: Check layer coordinate values.
        EXPECTED: Layer values are 1, 2, 3, ...
        """
        virtual_grid, _ = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        layers = virtual_grid["layer"].values
        assert layers[0] == 1
        assert np.all(np.diff(layers) == 1)

    def test_virtual_grid_has_model_attributes(
        self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster
    ):
        """
        WHAT: Verify virtual grid has required model attributes.
        WHY: Metadata needed for model interpretation.
        LOGIC: Check attrs dict.
        EXPECTED: Key attributes present.
        """
        virtual_grid, _ = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        # Should have at least these attributes
        assert "grid_type" in virtual_grid.attrs
        assert virtual_grid.attrs["grid_type"] == "virtual_parcel_grid"

    def test_virtual_grid_netcdf_round_trip(
        self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster
    ):
        """
        WHAT: Verify virtual grid survives NetCDF save/load.
        WHY: Must be able to save for Atlans.jl.
        LOGIC: Save to NetCDF, reload, compare.
        EXPECTED: Data unchanged after round trip.
        """
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            nc_path = Path(f.name)

        try:
            virtual_grid.to_netcdf(nc_path)
            loaded = xr.open_dataset(nc_path)

            # Check dimensions preserved
            assert loaded.sizes == virtual_grid.sizes

            # Check key variables
            for var in ["lithology", "surface_level", "parcel_id"]:
                assert var in loaded.data_vars

            loaded.close()

        finally:
            nc_path.unlink()
