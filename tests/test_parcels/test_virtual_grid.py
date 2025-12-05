"""Tests for VirtualGridBuilder class."""

import numpy as np
import pytest
import xarray as xr
import tempfile
from pathlib import Path

from atmod.parcels.data import ParcelData
from atmod.parcels.extractor import ParcelExtractor
from atmod.parcels.virtual_grid import VirtualGridBuilder, create_virtual_grid


class TestVirtualGridBuilder:
    """Tests for VirtualGridBuilder class."""

    @pytest.fixture
    def extraction_result(
        self, sample_parcels_gdf, sample_voxelmodel, sample_raster
    ):
        """Create extraction result for tests."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )
        extractor = ParcelExtractor(parcel_data, method="centroid")
        return extractor.extract(
            voxelmodel=sample_voxelmodel,
            rasters={"surface_level": sample_raster},
            variables_3d=["lithology", "geology", "thickness"],
        )

    def test_builder_init(self, extraction_result):
        """Test builder initialization."""
        builder = VirtualGridBuilder(extraction_result)

        assert builder.result == extraction_result
        assert builder.ds is None

    def test_build_creates_dataset(self, extraction_result):
        """Test that build creates xarray Dataset."""
        builder = VirtualGridBuilder(extraction_result)
        ds = builder.build()

        assert isinstance(ds, xr.Dataset)
        assert builder.ds is not None

    def test_virtual_grid_dimensions(self, extraction_result):
        """Test that virtual grid has correct dimensions."""
        builder = VirtualGridBuilder(extraction_result)
        ds = builder.build()

        assert ds.dims["y"] == 1
        assert ds.dims["x"] == 5  # 5 parcels
        assert ds.dims["z"] == extraction_result.n_layers

    def test_virtual_grid_coordinates(self, extraction_result):
        """Test that coordinates are correctly assigned."""
        builder = VirtualGridBuilder(extraction_result)
        ds = builder.build()

        # y should be [0.0]
        assert len(ds.y) == 1
        assert ds.y.values[0] == 0.0

        # x should be [1, 2, 3, 4, 5]
        assert len(ds.x) == 5
        np.testing.assert_array_equal(ds.x.values, [1, 2, 3, 4, 5])

    def test_virtual_grid_has_parcel_mapping(self, extraction_result):
        """Test that parcel mapping variables are included."""
        builder = VirtualGridBuilder(extraction_result)
        ds = builder.build()

        assert "parcel_id" in ds.data_vars
        assert "centroid_x" in ds.data_vars
        assert "centroid_y" in ds.data_vars
        assert "parcel_area" in ds.data_vars

    def test_virtual_grid_has_3d_variables(self, extraction_result):
        """Test that 3D variables are correctly shaped."""
        builder = VirtualGridBuilder(extraction_result)
        ds = builder.build()

        assert "lithology" in ds.data_vars
        assert ds["lithology"].dims == ("y", "x", "z")
        assert ds["lithology"].shape == (1, 5, extraction_result.n_layers)

    def test_virtual_grid_has_2d_variables(self, extraction_result):
        """Test that 2D variables are correctly shaped."""
        builder = VirtualGridBuilder(extraction_result)
        ds = builder.build()

        assert "surface_level" in ds.data_vars
        assert ds["surface_level"].dims == ("y", "x")
        assert ds["surface_level"].shape == (1, 5)

    def test_virtual_grid_attributes(self, extraction_result):
        """Test that dataset has correct attributes."""
        builder = VirtualGridBuilder(extraction_result)
        ds = builder.build()

        assert ds.attrs["grid_type"] == "virtual_parcel_grid"
        assert ds.attrs["n_parcels"] == 5
        assert ds.attrs["aggregation_method"] == "centroid"
        assert "crs" in ds.attrs

    def test_get_mapping_info(self, extraction_result):
        """Test that mapping info is correctly generated."""
        builder = VirtualGridBuilder(extraction_result)
        builder.build()

        mapping = builder.get_mapping_info()

        assert "parcel_id" in mapping
        assert "centroid_x" in mapping
        assert "centroid_y" in mapping
        assert "n_parcels" in mapping
        assert mapping["n_parcels"] == 5

    def test_get_mapping_info_before_build_raises(self, extraction_result):
        """Test that getting mapping before build raises error."""
        builder = VirtualGridBuilder(extraction_result)

        with pytest.raises(ValueError, match="build"):
            builder.get_mapping_info()

    def test_to_netcdf(self, extraction_result):
        """Test saving to NetCDF file."""
        builder = VirtualGridBuilder(extraction_result)
        builder.build()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = Path(f.name)

        try:
            builder.to_netcdf(path)
            assert path.exists()

            # Verify file can be opened
            with xr.open_dataset(path) as ds:
                assert ds.dims["y"] == 1
                assert ds.dims["x"] == 5
        finally:
            path.unlink()

    def test_save_mapping(self, extraction_result):
        """Test saving mapping info to JSON."""
        builder = VirtualGridBuilder(extraction_result)
        builder.build()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            builder.save_mapping(path)
            assert path.exists()

            # Verify file can be loaded
            import json
            with open(path) as f:
                mapping = json.load(f)
            assert mapping["n_parcels"] == 5
        finally:
            path.unlink()


class TestCreateVirtualGrid:
    """Tests for create_virtual_grid convenience function."""

    def test_create_virtual_grid(
        self, sample_parcels_gdf, sample_voxelmodel, sample_raster
    ):
        """Test convenience function."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )
        extractor = ParcelExtractor(parcel_data, method="centroid")
        result = extractor.extract(
            voxelmodel=sample_voxelmodel,
            rasters={"surface_level": sample_raster},
        )

        ds, mapping = create_virtual_grid(result)

        assert isinstance(ds, xr.Dataset)
        assert isinstance(mapping, dict)
        assert ds.dims["x"] == 5

    def test_create_virtual_grid_with_output(
        self, sample_parcels_gdf, sample_voxelmodel, sample_raster
    ):
        """Test convenience function with file output."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )
        extractor = ParcelExtractor(parcel_data, method="centroid")
        result = extractor.extract(
            voxelmodel=sample_voxelmodel,
            rasters={"surface_level": sample_raster},
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = Path(f.name)

        try:
            ds, mapping = create_virtual_grid(result, output_path=path)
            assert path.exists()
        finally:
            path.unlink()
