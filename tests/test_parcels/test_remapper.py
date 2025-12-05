"""Tests for ParcelResultMapper class."""

import numpy as np
import pytest
import xarray as xr
import tempfile
from pathlib import Path

from atmod.parcels.remapper import ParcelResultMapper, remap_parcel_results


class TestParcelResultMapper:
    """Tests for ParcelResultMapper class."""

    @pytest.fixture
    def mapping_info(self):
        """Create sample mapping info."""
        return {
            "parcel_id": ["P1", "P2", "P3", "P4", "P5"],
            "centroid_x": [140.0, 240.0, 340.0, 440.0, 540.0],
            "centroid_y": [140.0, 140.0, 140.0, 140.0, 140.0],
            "parcel_area": [6400.0, 6400.0, 6400.0, 6400.0, 6400.0],
            "n_parcels": 5,
            "n_layers": 12,
            "crs": "EPSG:28992",
            "aggregation_method": "centroid",
        }

    @pytest.fixture
    def atlans_output(self, mapping_info):
        """Create mock Atlans.jl output in native format.

        Atlans.jl outputs with dimension order ("x", "y", "time").
        """
        n_parcels = mapping_info["n_parcels"]
        n_times = 10

        # Create output in Atlans.jl native format: ("x", "y", "time")
        ds = xr.Dataset(
            {
                "subsidence": (
                    ["x", "y", "time"],
                    np.random.rand(n_parcels, 1, n_times) * 0.1,
                ),
                "consolidation": (
                    ["x", "y", "time"],
                    np.random.rand(n_parcels, 1, n_times) * 0.05,
                ),
                "oxidation": (
                    ["x", "y", "time"],
                    np.random.rand(n_parcels, 1, n_times) * 0.03,
                ),
            },
            coords={
                "x": np.arange(1, n_parcels + 1, dtype=float),
                "y": [0.0],
                "time": np.arange(n_times),
            },
        )
        return ds

    def test_mapper_init(self, mapping_info):
        """Test mapper initialization."""
        mapper = ParcelResultMapper(mapping_info)

        assert mapper.mapping_info == mapping_info

    def test_mapper_init_missing_fields(self):
        """Test that missing required fields raises error."""
        with pytest.raises(ValueError, match="missing required"):
            ParcelResultMapper({"parcel_id": [1, 2, 3]})

    def test_from_json(self, mapping_info):
        """Test creating mapper from JSON file."""
        import json

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(mapping_info, f)
            path = Path(f.name)

        try:
            mapper = ParcelResultMapper.from_json(path)
            assert mapper.mapping_info["n_parcels"] == 5
        finally:
            path.unlink()

    def test_remap(self, mapping_info, atlans_output):
        """Test remapping Atlans output."""
        # Save atlans output to temp file
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = Path(f.name)
        atlans_output.to_netcdf(path)

        try:
            mapper = ParcelResultMapper(mapping_info)
            result = mapper.remap(path)

            # Check dimensions
            assert "parcel_id" in result.dims
            assert "time" in result.dims
            assert result.dims["parcel_id"] == 5
            assert result.dims["time"] == 10

            # Check variables
            assert "subsidence" in result.data_vars
            assert "consolidation" in result.data_vars
            assert "centroid_x" in result.data_vars
        finally:
            path.unlink()

    def test_remap_parcel_id_preserved(self, mapping_info, atlans_output):
        """Test that parcel IDs are correctly preserved."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = Path(f.name)
        atlans_output.to_netcdf(path)

        try:
            mapper = ParcelResultMapper(mapping_info)
            result = mapper.remap(path)

            # Parcel IDs should match
            np.testing.assert_array_equal(
                result.parcel_id.values, mapping_info["parcel_id"]
            )
        finally:
            path.unlink()

    def test_remap_with_selected_variables(self, mapping_info, atlans_output):
        """Test remapping with selected output variables."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = Path(f.name)
        atlans_output.to_netcdf(path)

        try:
            mapper = ParcelResultMapper(mapping_info)
            result = mapper.remap(path, output_variables=["subsidence"])

            assert "subsidence" in result.data_vars
            # consolidation and oxidation should not be included
            assert "consolidation" not in result.data_vars
        finally:
            path.unlink()

    def test_remap_validates_dimensions(self, mapping_info):
        """Test that dimension mismatch raises error."""
        # Create output with wrong number of parcels
        ds = xr.Dataset(
            {
                "subsidence": (["time", "y", "x"], np.zeros((5, 1, 3))),
            },
            coords={
                "time": np.arange(5),
                "y": [0.0],
                "x": [1.0, 2.0, 3.0],
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = Path(f.name)
        ds.to_netcdf(path)

        try:
            mapper = ParcelResultMapper(mapping_info)
            with pytest.raises(ValueError, match="expected"):
                mapper.remap(path)
        finally:
            path.unlink()


class TestRemapParcelResults:
    """Tests for remap_parcel_results convenience function."""

    @pytest.fixture
    def mapping_info(self):
        """Create sample mapping info."""
        return {
            "parcel_id": [101, 102, 103],
            "centroid_x": [100.0, 200.0, 300.0],
            "centroid_y": [100.0, 100.0, 100.0],
            "n_parcels": 3,
            "crs": "EPSG:28992",
        }

    @pytest.fixture
    def atlans_output_file(self, mapping_info):
        """Create mock Atlans output file in native format.

        Atlans.jl outputs with dimension order ("x", "y", "time").
        """
        ds = xr.Dataset(
            {
                "subsidence": (
                    ["x", "y", "time"],
                    np.random.rand(3, 1, 5) * 0.1,
                ),
            },
            coords={
                "x": [1.0, 2.0, 3.0],
                "y": [0.0],
                "time": np.arange(5),
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = Path(f.name)
        ds.to_netcdf(path)

        yield path
        path.unlink()

    def test_remap_with_dict(self, atlans_output_file, mapping_info):
        """Test remap with mapping dict."""
        result = remap_parcel_results(atlans_output_file, mapping_info)

        assert "parcel_id" in result.dims
        assert result.dims["parcel_id"] == 3

    def test_remap_with_json_path(self, atlans_output_file, mapping_info):
        """Test remap with JSON file path."""
        import json

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(mapping_info, f)
            json_path = Path(f.name)

        try:
            result = remap_parcel_results(atlans_output_file, json_path)
            assert result.dims["parcel_id"] == 3
        finally:
            json_path.unlink()

    def test_remap_with_output_path(self, atlans_output_file, mapping_info):
        """Test remap with output file path."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = remap_parcel_results(
                atlans_output_file, mapping_info, output_path=output_path
            )

            assert output_path.exists()

            # Verify output file
            with xr.open_dataset(output_path) as ds:
                assert "parcel_id" in ds.dims
        finally:
            output_path.unlink()
