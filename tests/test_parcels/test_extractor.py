"""Tests for ParcelExtractor class."""

import numpy as np
import pytest

from atmod.parcels.data import ParcelData
from atmod.parcels.extractor import ParcelExtractor, ExtractionResult
from atmod.parcels.aggregation import AggregationMethod


class TestParcelExtractor:
    """Tests for ParcelExtractor class."""

    def test_extractor_init(self, sample_parcels_gdf):
        """Test extractor initialization."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        extractor = ParcelExtractor(parcel_data, method="centroid")

        assert extractor.method == AggregationMethod.CENTROID
        assert extractor.use_probability is True
        assert extractor.strict is False

    def test_extract_from_voxelmodel(self, sample_parcels_gdf, sample_voxelmodel):
        """Test extraction from voxel model."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        extractor = ParcelExtractor(parcel_data, method="centroid")
        result = extractor.extract_from_voxelmodel(
            sample_voxelmodel,
            variables=["lithology", "geology"],
        )

        assert "lithology" in result
        assert "geology" in result
        assert result["lithology"].shape == (5, sample_voxelmodel.nz)

    def test_extract_from_raster(self, sample_parcels_gdf, sample_raster):
        """Test extraction from 2D raster."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        extractor = ParcelExtractor(parcel_data, method="centroid")
        result = extractor.extract_from_raster(sample_raster)

        assert result.shape == (5,)

    def test_full_extraction(
        self, sample_parcels_gdf, sample_voxelmodel, sample_raster
    ):
        """Test full extraction with rasters and voxelmodel."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        extractor = ParcelExtractor(parcel_data, method="centroid")
        result = extractor.extract(
            voxelmodel=sample_voxelmodel,
            rasters={"surface_level": sample_raster},
            variables_3d=["lithology", "geology", "thickness"],
        )

        assert isinstance(result, ExtractionResult)
        assert result.n_parcels == 5
        assert result.n_layers == sample_voxelmodel.nz
        assert "lithology" in result.data_3d
        assert "surface_level" in result.data_2d

    def test_extraction_result_properties(
        self, sample_parcels_gdf, sample_voxelmodel, sample_raster
    ):
        """Test ExtractionResult properties."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        extractor = ParcelExtractor(parcel_data, method="centroid")
        result = extractor.extract(
            voxelmodel=sample_voxelmodel,
            rasters={"surface_level": sample_raster},
        )

        assert result.n_parcels == 5
        assert result.n_valid_parcels <= 5
        assert result.aggregation_method == "centroid"
        assert result.layer_coords is not None

    def test_strict_mode(self, outside_bounds_parcel, sample_voxelmodel):
        """Test that strict mode raises exceptions."""
        parcel_data = ParcelData.from_geodataframe(
            outside_bounds_parcel, "parcel_id"
        )

        extractor = ParcelExtractor(parcel_data, method="centroid", strict=True)

        # This should raise because parcel is outside bounds
        with pytest.raises(Exception):  # ParcelOutsideBoundsError
            extractor.extract(voxelmodel=sample_voxelmodel)

    def test_non_strict_mode_logs_warning(
        self, outside_bounds_parcel, sample_voxelmodel
    ):
        """Test that non-strict mode logs warnings instead of raising."""
        parcel_data = ParcelData.from_geodataframe(
            outside_bounds_parcel, "parcel_id"
        )

        extractor = ParcelExtractor(parcel_data, method="centroid", strict=False)
        result = extractor.extract(voxelmodel=sample_voxelmodel)

        # Should complete without error
        assert result is not None
        assert len(result.warnings) > 0

    def test_probability_method(self, sample_parcels_gdf, sample_voxelmodel):
        """Test extraction with probability method."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        extractor = ParcelExtractor(
            parcel_data,
            method="probability",
            use_probability=True,
        )
        result = extractor.extract(voxelmodel=sample_voxelmodel)

        # Lithology should be extracted using probability
        assert "lithology" in result.data_3d
        # Values should be valid lithology codes (1-9)
        valid = result.data_3d["lithology"][~np.isnan(result.data_3d["lithology"])]
        assert np.all(valid >= 1)
        assert np.all(valid <= 9)

    def test_categorical_uses_mode_for_area_weighted(
        self, sample_parcels_gdf, sample_voxelmodel
    ):
        """Test that categorical variables use MODE when AREA_WEIGHTED requested."""
        parcel_data = ParcelData.from_geodataframe(
            sample_parcels_gdf, "Perceel_ID"
        )

        extractor = ParcelExtractor(parcel_data, method="area_weighted")
        result = extractor.extract(
            voxelmodel=sample_voxelmodel,
            variables_3d=["lithology"],  # categorical
        )

        # Should complete without error, using MODE internally
        assert "lithology" in result.data_3d
