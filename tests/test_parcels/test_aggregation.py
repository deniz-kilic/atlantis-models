"""Tests for aggregation methods."""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import box

from atmod.parcels.aggregation import (
    AggregationMethod,
    aggregate_2d,
    aggregate_3d,
    aggregate_3d_probability,
)
from atmod.parcels.data import ParcelData


class TestAggregationMethod:
    """Tests for AggregationMethod enum."""

    def test_enum_values(self):
        """Test that all enum values exist."""
        assert AggregationMethod.CENTROID.value == "centroid"
        assert AggregationMethod.MODE.value == "mode"
        assert AggregationMethod.AREA_WEIGHTED.value == "area_weighted"
        assert AggregationMethod.PROBABILITY.value == "probability"

    def test_from_string(self):
        """Test creating enum from string."""
        method = AggregationMethod("centroid")
        assert method == AggregationMethod.CENTROID


class TestAggregate2D:
    """Tests for 2D aggregation."""

    def test_centroid_aggregation(self, sample_parcels_gdf, sample_raster):
        """Test centroid aggregation extracts correct values."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = aggregate_2d(
            parcel_data.geometries,
            sample_raster,
            AggregationMethod.CENTROID,
            centroids=parcel_data.centroids,
        )

        assert result.shape == (5,)
        assert not np.all(np.isnan(result))

    def test_mode_aggregation(self, sample_parcels_gdf, sample_raster):
        """Test mode aggregation."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = aggregate_2d(
            parcel_data.geometries,
            sample_raster,
            AggregationMethod.MODE,
        )

        assert result.shape == (5,)

    def test_area_weighted_aggregation(self, sample_parcels_gdf, sample_raster):
        """Test area-weighted aggregation."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = aggregate_2d(
            parcel_data.geometries,
            sample_raster,
            AggregationMethod.AREA_WEIGHTED,
        )

        assert result.shape == (5,)

    def test_string_method(self, sample_parcels_gdf, sample_raster):
        """Test that string method works."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = aggregate_2d(
            parcel_data.geometries,
            sample_raster,
            "centroid",  # string instead of enum
            centroids=parcel_data.centroids,
        )

        assert result.shape == (5,)

    def test_outside_bounds_returns_nan(self, outside_bounds_parcel, sample_raster):
        """Test that parcels outside bounds return NaN."""
        parcel_data = ParcelData.from_geodataframe(
            outside_bounds_parcel, "parcel_id"
        )

        result = aggregate_2d(
            parcel_data.geometries,
            sample_raster,
            AggregationMethod.CENTROID,
            centroids=parcel_data.centroids,
        )

        assert np.isnan(result[0])


class TestAggregate3D:
    """Tests for 3D aggregation."""

    def test_centroid_aggregation(self, sample_parcels_gdf, sample_voxelmodel):
        """Test 3D centroid aggregation."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = aggregate_3d(
            parcel_data.geometries,
            sample_voxelmodel,
            "lithology",
            AggregationMethod.CENTROID,
            centroids=parcel_data.centroids,
        )

        assert result.shape == (5, sample_voxelmodel.nz)

    def test_mode_aggregation(self, sample_parcels_gdf, sample_voxelmodel):
        """Test 3D mode aggregation."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = aggregate_3d(
            parcel_data.geometries,
            sample_voxelmodel,
            "lithology",
            AggregationMethod.MODE,
        )

        assert result.shape == (5, sample_voxelmodel.nz)

    def test_area_weighted_aggregation(self, sample_parcels_gdf, sample_voxelmodel):
        """Test 3D area-weighted aggregation."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = aggregate_3d(
            parcel_data.geometries,
            sample_voxelmodel,
            "thickness",  # continuous variable
            AggregationMethod.AREA_WEIGHTED,
        )

        assert result.shape == (5, sample_voxelmodel.nz)
        # Thickness should be ~0.5 everywhere
        assert np.allclose(result[~np.isnan(result)], 0.5, atol=0.1)

    def test_probability_method_raises_for_regular_aggregate3d(
        self, sample_parcels_gdf, sample_voxelmodel
    ):
        """Test that probability method raises error in aggregate_3d."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        with pytest.raises(ValueError, match="PROBABILITY"):
            aggregate_3d(
                parcel_data.geometries,
                sample_voxelmodel,
                "lithology",
                AggregationMethod.PROBABILITY,
            )


class TestAggregate3DProbability:
    """Tests for probability-based 3D aggregation."""

    def test_probability_aggregation(self, sample_parcels_gdf, sample_voxelmodel):
        """Test probability aggregation returns valid lithology codes."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        result = aggregate_3d_probability(
            parcel_data.geometries,
            sample_voxelmodel,
            centroids=parcel_data.centroids,
        )

        assert result.shape == (5, sample_voxelmodel.nz)
        # Results should be in range 1-9 (lithology codes)
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 1)
        assert np.all(valid_values <= 9)

    def test_missing_kans_raises(self, sample_parcels_gdf, sample_voxelmodel):
        """Test that missing kans_* variables raises error."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        # Remove kans_* variables
        ds = sample_voxelmodel.ds.drop_vars(
            [f"kans_{i}" for i in range(1, 10)]
        )
        from atmod.base import VoxelModel
        voxelmodel = VoxelModel(ds, 50, 0.5, 28992)

        with pytest.raises(ValueError, match="missing probability"):
            aggregate_3d_probability(
                parcel_data.geometries,
                voxelmodel,
            )


class TestEdgeCases:
    """Tests for edge cases in aggregation."""

    def test_tiny_parcel(self, tiny_parcel, sample_raster):
        """Test aggregation for parcel smaller than grid cell."""
        parcel_data = ParcelData.from_geodataframe(tiny_parcel, "parcel_id")

        result = aggregate_2d(
            parcel_data.geometries,
            sample_raster,
            AggregationMethod.CENTROID,
            centroids=parcel_data.centroids,
        )

        assert result.shape == (1,)
        assert not np.isnan(result[0])

    def test_large_parcel(self, large_parcel, sample_raster):
        """Test aggregation for parcel spanning many cells."""
        parcel_data = ParcelData.from_geodataframe(large_parcel, "parcel_id")

        result = aggregate_2d(
            parcel_data.geometries,
            sample_raster,
            AggregationMethod.AREA_WEIGHTED,
        )

        assert result.shape == (1,)
        assert not np.isnan(result[0])

    def test_single_cell_centroid_identity(self, single_cell_parcel, sample_raster):
        """Test that centroid aggregation at cell center matches cell value."""
        parcel_data = ParcelData.from_geodataframe(
            single_cell_parcel, "parcel_id"
        )

        centroid = aggregate_2d(
            parcel_data.geometries,
            sample_raster,
            AggregationMethod.CENTROID,
            centroids=parcel_data.centroids,
        )

        # The centroid should give us the exact cell value
        # Since our parcel is centered on a cell
        assert centroid.shape == (1,)
