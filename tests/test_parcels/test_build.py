"""
Test suite for parcel build entry points.

These tests verify that build_parcel_model and remap_parcel_results
entry points work correctly.
"""

import numpy as np
import pytest
import xarray as xr

from atmod.parcels.build import build_parcel_model, build_parcel_forcing
from atmod.parcels.data import ParcelData, ParcelModelConfig


@pytest.mark.unittest
class TestBuildParcelModel:
    """
    Tests for build_parcel_model() entry point.

    TESTS: Basic model building, output structure, parcel ID preservation
    DOES NOT TEST: Actual Atlans.jl compatibility, performance with large datasets
    """

    def test_build_parcel_model_basic(self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster):
        """
        WHAT: Verify basic parcel model building returns valid Dataset.
        WHY: Core functionality must work.
        LOGIC: Build model with minimal inputs, check output type.
        EXPECTED: Returns tuple of (xr.Dataset, dict).
        """
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        assert isinstance(virtual_grid, xr.Dataset)
        assert isinstance(mapping, dict)

    def test_build_parcel_model_preserves_ids(self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster):
        """
        WHAT: Verify parcel IDs are preserved in output.
        WHY: IDs needed to map results back to parcels.
        LOGIC: Check parcel_id variable matches input.
        EXPECTED: parcel_id in dataset with correct values.
        """
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        assert "parcel_id" in virtual_grid.data_vars
        # Check that IDs match input
        output_ids = set(virtual_grid["parcel_id"].values)
        input_ids = set(sample_parcels_gdf["Perceel_ID"].values)
        assert output_ids == input_ids

    def test_build_parcel_model_with_config(self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster):
        """
        WHAT: Verify custom ParcelModelConfig is applied.
        WHY: Users need to customize model building.
        LOGIC: Pass config with specific options, verify applied.
        EXPECTED: Config settings reflected in output.
        """
        config = ParcelModelConfig(
            aggregation_method="mode",
            use_probability=False,
            compression_level=0,
        )

        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            config=config,
        )

        assert isinstance(virtual_grid, xr.Dataset)
        # Verify config was applied by checking metadata
        assert "aggregation_method" in virtual_grid.attrs

    def test_build_parcel_model_virtual_grid_dimensions(self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster):
        """
        WHAT: Verify output has correct virtual grid dimensions.
        WHY: Atlans.jl requires specific dimension structure.
        LOGIC: Check dims are (y=1, x=N, layer=K).
        EXPECTED: y=1, x=number of parcels, layer=number of layers.
        """
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        assert virtual_grid.sizes["y"] == 1
        assert virtual_grid.sizes["x"] == len(sample_parcels_gdf)
        assert "layer" in virtual_grid.dims

    def test_build_parcel_model_has_required_variables(self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster):
        """
        WHAT: Verify output contains Atlans-required variables.
        WHY: Atlans.jl needs specific variables to run.
        LOGIC: Check for lithology, thickness, surface_level, etc.
        EXPECTED: All required variables present.
        """
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        required_3d = ["lithology", "geology", "thickness"]
        required_2d = ["surface_level"]

        for var in required_3d:
            assert var in virtual_grid.data_vars, f"Missing required 3D variable: {var}"

        for var in required_2d:
            assert var in virtual_grid.data_vars, f"Missing required 2D variable: {var}"

    def test_build_parcel_model_from_parcel_data(self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster):
        """
        WHAT: Verify ParcelData input type works.
        WHY: Users may pre-process parcels.
        LOGIC: Pass ParcelData object instead of GeoDataFrame.
        EXPECTED: Builds successfully.
        """
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        virtual_grid, mapping = build_parcel_model(
            parcels=parcel_data,
            parcel_id_column="Perceel_ID",  # Ignored when ParcelData
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        assert isinstance(virtual_grid, xr.Dataset)
        assert len(virtual_grid.x) == len(sample_parcels_gdf)

    def test_build_parcel_model_with_glg(self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster, sample_glg_raster):
        """
        WHAT: Verify GLG raster is extracted to phreatic_level.
        WHY: Groundwater level needed for subsidence calculation.
        LOGIC: Pass GLG raster, check phreatic_level in output.
        EXPECTED: phreatic_level variable present.
        """
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            glg=sample_glg_raster,
            aggregation_method="centroid",
        )

        assert "phreatic_level" in virtual_grid.data_vars

    def test_build_parcel_model_mapping_info(self, sample_parcels_gdf, sample_voxelmodel, sample_ahn_raster):
        """
        WHAT: Verify mapping_info contains required fields.
        WHY: Mapping needed to remap results back to parcels.
        LOGIC: Check mapping dict structure.
        EXPECTED: Contains parcel_id, n_parcels, centroids.
        """
        virtual_grid, mapping = build_parcel_model(
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            ahn=sample_ahn_raster,
            geotop=sample_voxelmodel,
            aggregation_method="centroid",
        )

        assert "parcel_id" in mapping
        assert "n_parcels" in mapping
        assert "centroid_x" in mapping
        assert "centroid_y" in mapping
        assert mapping["n_parcels"] == len(sample_parcels_gdf)
        assert len(mapping["parcel_id"]) == len(sample_parcels_gdf)


@pytest.mark.unittest
class TestBuildParcelForcing:
    """
    Tests for build_parcel_forcing() entry point.

    TESTS: Basic forcing conversion, temporal preservation
    DOES NOT TEST: Real forcing data, large datasets
    """

    @pytest.fixture
    def sample_gridded_forcing(self, sample_voxelmodel):
        """Create sample time-varying gridded forcing."""
        x = sample_voxelmodel.ds.x.values
        y = sample_voxelmodel.ds.y.values
        times = np.arange(5)  # 5 timesteps

        # Create time-varying groundwater level
        phreatic = np.zeros((len(times), len(y), len(x)))
        for t in range(len(times)):
            phreatic[t] = -0.5 - 0.1 * np.sin(2 * np.pi * t / 5)  # Seasonal variation

        return xr.Dataset(
            {"phreatic_level": (["time", "y", "x"], phreatic)},
            coords={"time": times, "y": y, "x": x},
        )

    def test_build_parcel_forcing_basic(self, sample_gridded_forcing, sample_parcels_gdf):
        """
        WHAT: Verify basic forcing conversion works.
        WHY: Core functionality for time-varying inputs.
        LOGIC: Convert gridded forcing to parcel format.
        EXPECTED: Returns Dataset with correct dimensions.
        """
        result = build_parcel_forcing(
            gridded_forcing=sample_gridded_forcing,
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            aggregation_method="centroid",
        )

        assert isinstance(result, xr.Dataset)
        assert result.sizes["time"] == len(sample_gridded_forcing.time)
        assert result.sizes["x"] == len(sample_parcels_gdf)
        assert result.sizes["y"] == 1

    def test_build_parcel_forcing_preserves_time(self, sample_gridded_forcing, sample_parcels_gdf):
        """
        WHAT: Verify time coordinates are preserved.
        WHY: Temporal dynamics must be maintained.
        LOGIC: Compare time coordinates.
        EXPECTED: Same time values as input.
        """
        result = build_parcel_forcing(
            gridded_forcing=sample_gridded_forcing,
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            aggregation_method="centroid",
        )

        np.testing.assert_array_equal(
            result.time.values,
            sample_gridded_forcing.time.values
        )

    def test_build_parcel_forcing_has_parcel_ids(self, sample_gridded_forcing, sample_parcels_gdf):
        """
        WHAT: Verify parcel IDs are included in output.
        WHY: IDs needed for result interpretation.
        LOGIC: Check parcel_id variable.
        EXPECTED: parcel_id matches input.
        """
        result = build_parcel_forcing(
            gridded_forcing=sample_gridded_forcing,
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            aggregation_method="centroid",
        )

        assert "parcel_id" in result.data_vars
        output_ids = set(result["parcel_id"].values)
        input_ids = set(sample_parcels_gdf["Perceel_ID"].values)
        assert output_ids == input_ids

    def test_build_parcel_forcing_with_parcel_data(self, sample_gridded_forcing, sample_parcels_gdf):
        """
        WHAT: Verify ParcelData input type works.
        WHY: Consistent with build_parcel_model.
        LOGIC: Pass ParcelData object.
        EXPECTED: Builds successfully.
        """
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = build_parcel_forcing(
            gridded_forcing=sample_gridded_forcing,
            parcels=parcel_data,
            aggregation_method="centroid",
        )

        assert isinstance(result, xr.Dataset)
        assert result.sizes["x"] == len(sample_parcels_gdf)

    def test_build_parcel_forcing_multiple_variables(self, sample_voxelmodel, sample_parcels_gdf):
        """
        WHAT: Verify multiple forcing variables are handled.
        WHY: May have multiple time-varying inputs.
        LOGIC: Create forcing with multiple variables.
        EXPECTED: All variables in output.
        """
        x = sample_voxelmodel.ds.x.values
        y = sample_voxelmodel.ds.y.values
        times = np.arange(3)

        forcing = xr.Dataset(
            {
                "phreatic_level": (["time", "y", "x"], np.random.rand(3, len(y), len(x))),
                "surcharge": (["time", "y", "x"], np.random.rand(3, len(y), len(x))),
            },
            coords={"time": times, "y": y, "x": x},
        )

        result = build_parcel_forcing(
            gridded_forcing=forcing,
            parcels=sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            aggregation_method="centroid",
        )

        assert "phreatic_level" in result.data_vars
        assert "surcharge" in result.data_vars
