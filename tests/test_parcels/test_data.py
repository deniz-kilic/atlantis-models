"""Tests for ParcelData class."""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS

from atmod.parcels.data import ParcelData, ParcelModelConfig
from atmod.parcels.exceptions import (
    DuplicateParcelIdError,
    ValidationError,
)

# Create CRS object for Dutch RD New
RD_NEW_CRS = CRS.from_proj4(
    "+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 "
    "+k=0.9999079 +x_0=155000 +y_0=463000 +ellps=bessel "
    "+towgs84=565.417,50.3319,465.552,-0.398957,0.343988,-1.8774,4.0725 "
    "+units=m +no_defs"
)


class TestParcelData:
    """Tests for ParcelData class."""

    def test_from_geodataframe_basic(self, sample_parcels_gdf):
        """Test basic creation from GeoDataFrame."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
        )

        assert len(parcel_data) == 5
        assert parcel_data.crs == "EPSG:28992"
        assert parcel_data.centroids.shape == (5, 2)
        assert len(parcel_data.areas) == 5

    def test_from_geodataframe_with_attributes(self, sample_parcels_gdf):
        """Test creation with attribute columns."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf,
            parcel_id_column="Perceel_ID",
            attribute_columns=["soil_type", "area_m2"],
        )

        assert "soil_type" in parcel_data.attributes
        assert "area_m2" in parcel_data.attributes
        assert len(parcel_data.attributes["soil_type"]) == 5

    def test_from_geodataframe_numeric_id(self, sample_parcels_numeric_id):
        """Test creation with numeric parcel IDs."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_numeric_id,
            parcel_id_column="parcel_num",
        )

        assert len(parcel_data) == 5
        assert parcel_data.parcel_id[0] == 101

    def test_missing_column_raises(self, sample_parcels_gdf):
        """Test that missing parcel_id column raises ValidationError."""
        with pytest.raises(ValidationError, match="not found"):
            ParcelData.from_geodataframe(
                sample_parcels_gdf,
                parcel_id_column="nonexistent",
            )

    def test_duplicate_ids_raises(self, sample_parcels_gdf):
        """Test that duplicate parcel IDs raise error."""
        gdf = sample_parcels_gdf.copy()
        gdf["Perceel_ID"] = ["P1", "P1", "P3", "P4", "P5"]

        with pytest.raises(DuplicateParcelIdError, match="duplicate"):
            ParcelData.from_geodataframe(gdf, "Perceel_ID")

    def test_missing_crs_raises(self, sample_parcels_gdf):
        """Test that missing CRS raises ValidationError."""
        gdf = sample_parcels_gdf.copy()
        gdf = gdf.set_crs(None, allow_override=True)

        with pytest.raises(ValidationError, match="CRS"):
            ParcelData.from_geodataframe(gdf, "Perceel_ID")

    def test_crs_transformation(self):
        """Test automatic CRS transformation to EPSG:28992."""
        # Create parcels in WGS84 using proj4 string to avoid EPSG lookup
        wgs84 = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
        geom = box(4.5, 52.0, 4.6, 52.1)
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[geom],
            crs=wgs84,
        )

        parcel_data = ParcelData.from_geodataframe(gdf, "id")

        # Check that CRS was transformed (should contain RD projection info)
        assert "28992" in parcel_data.crs or "sterea" in str(parcel_data.crs).lower()
        # Centroid should be in RD coordinates (roughly 100000-300000 range)
        assert parcel_data.centroids[0, 0] > 50000

    def test_bounds_property(self, sample_parcels_gdf):
        """Test bounds property returns correct values."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        bounds = parcel_data.bounds
        assert len(bounds) == 4
        assert bounds[0] == parcel_data.xmin
        assert bounds[1] == parcel_data.ymin
        assert bounds[2] == parcel_data.xmax
        assert bounds[3] == parcel_data.ymax

    def test_subset(self, sample_parcels_gdf):
        """Test subset method."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        subset = parcel_data.subset([0, 2, 4])

        assert len(subset) == 3
        assert list(subset.parcel_id) == ["P1", "P3", "P5"]

    def test_to_geodataframe(self, sample_parcels_gdf):
        """Test conversion back to GeoDataFrame."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf,
            "Perceel_ID",
            attribute_columns=["soil_type"],
        )

        gdf = parcel_data.to_geodataframe()

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert "parcel_id" in gdf.columns
        assert "centroid_x" in gdf.columns
        assert "area_m2" in gdf.columns
        assert "soil_type" in gdf.columns
        assert len(gdf) == 5

    def test_repr(self, sample_parcels_gdf):
        """Test string representation."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        repr_str = repr(parcel_data)
        assert "ParcelData" in repr_str
        assert "n_parcels=5" in repr_str


class TestParcelModelConfig:
    """Tests for ParcelModelConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ParcelModelConfig()

        assert config.aggregation_method == "centroid"
        assert config.use_probability is True
        assert config.chunk_size == 1000
        assert config.compression_level == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = ParcelModelConfig(
            aggregation_method="mode",
            use_probability=False,
            chunk_size=500,
        )

        assert config.aggregation_method == "mode"
        assert config.use_probability is False
        assert config.chunk_size == 500

    def test_invalid_aggregation_method(self):
        """Test that invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="Invalid aggregation_method"):
            ParcelModelConfig(aggregation_method="invalid")

    def test_invalid_compression_level(self):
        """Test that invalid compression level raises error."""
        with pytest.raises(ValueError, match="compression_level"):
            ParcelModelConfig(compression_level=10)
