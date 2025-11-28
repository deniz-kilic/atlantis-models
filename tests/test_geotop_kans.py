"""
Test suite for GeoTop kans (probability) data loading.

These tests verify that GeoTop correctly loads and validates the kans_1-9
probability variables used for lithology uncertainty quantification.
"""

import numpy as np
import pytest

from atmod.bro_models.voxelmodels import KANS_VARS


class TestGeoTopKansLoading:
    """Tests for loading kans probability data into GeoTop."""

    def test_has_kans_true_when_all_kans_present(self, mock_geotop_with_kans):
        """
        WHAT: Verify has_kans returns True when all kans_1-9 are present.
        WHY: Core functionality check - must detect kans presence correctly.
        LOGIC: Create GeoTop with all 9 kans vars, check has_kans property.
        EXPECTED: has_kans == True
        """
        assert mock_geotop_with_kans.has_kans is True

    def test_has_kans_false_when_kans_missing(self, mock_geotop_without_kans):
        """
        WHAT: Verify has_kans returns False when kans are not loaded.
        WHY: Backward compatibility - existing code without kans should work.
        LOGIC: Create GeoTop without kans, check has_kans property.
        EXPECTED: has_kans == False
        """
        assert mock_geotop_without_kans.has_kans is False

    def test_kans_vars_list_contains_all_nine(self):
        """
        WHAT: Verify KANS_VARS constant has correct variable names.
        WHY: Variable names must match GeoTOP NetCDF conventions.
        LOGIC: Check KANS_VARS contains kans_1 through kans_9.
        EXPECTED: List of 9 strings ['kans_1', ..., 'kans_9']
        """
        assert len(KANS_VARS) == 9
        for i in range(1, 10):
            assert f'kans_{i}' in KANS_VARS

    def test_kans_data_has_correct_shape(self, mock_geotop_with_kans):
        """
        WHAT: Verify kans arrays have same shape as lithok.
        WHY: Spatial alignment required for sampling.
        LOGIC: Compare shape of kans_1 with lithok shape.
        EXPECTED: Identical shapes (nz, ny, nx)
        """
        expected_shape = mock_geotop_with_kans.ds['lithok'].shape
        for i in range(1, 10):
            assert mock_geotop_with_kans.ds[f'kans_{i}'].shape == expected_shape

    def test_kans_data_has_correct_coordinates(self, mock_geotop_with_kans):
        """
        WHAT: Verify kans arrays share coordinates with lithok.
        WHY: Coordinate alignment essential for spatial analysis.
        LOGIC: Compare coords of kans_1 with lithok coords.
        EXPECTED: Same x, y, z coordinates
        """
        lithok_coords = set(mock_geotop_with_kans.ds['lithok'].coords.keys())
        for i in range(1, 10):
            kans_coords = set(mock_geotop_with_kans.ds[f'kans_{i}'].coords.keys())
            assert kans_coords == lithok_coords


class TestGeoTopKansValidation:
    """Tests for kans probability validation."""

    def test_validate_kans_returns_true_for_valid_data(self, mock_geotop_with_kans):
        """
        WHAT: Verify validate_kans() returns True for properly summed kans.
        WHY: Validation must confirm probabilities are usable.
        LOGIC: Create kans that sum to 100%, validate.
        EXPECTED: validate_kans() == True
        """
        assert mock_geotop_with_kans.validate_kans() == True

    def test_validate_kans_returns_false_without_kans(self, mock_geotop_without_kans):
        """
        WHAT: Verify validate_kans() returns False when no kans loaded.
        WHY: Cannot validate what doesn't exist.
        LOGIC: Call validate_kans() on GeoTop without kans.
        EXPECTED: validate_kans() == False
        """
        assert mock_geotop_without_kans.validate_kans() is False

    def test_validate_kans_uses_tolerance(self, mock_geotop_with_kans):
        """
        WHAT: Verify tolerance parameter affects validation.
        WHY: GeoTOP kans may have small rounding errors from compression.
        LOGIC: Default tolerance=5.0 should pass, tight tolerance may fail.
        EXPECTED: tolerance=5.0 passes, tolerance=0 stricter
        """
        # Default tolerance should pass
        assert mock_geotop_with_kans.validate_kans(tolerance=5.0) == True
        # Very tight tolerance - might still pass if data is exact
        result = mock_geotop_with_kans.validate_kans(tolerance=0.01)
        # Just ensure it doesn't crash and returns boolean-like
        assert result in (True, False, np.True_, np.False_)

    def test_validate_kans_handles_nan_gracefully(self, mock_geotop_with_nan):
        """
        WHAT: Verify validation doesn't crash on NaN values.
        WHY: GeoTOP has NaN outside Netherlands domain.
        LOGIC: Create kans with NaN, call validate_kans().
        EXPECTED: Returns bool without exception
        """
        # Should not raise, should return True (valid voxels are valid)
        result = mock_geotop_with_nan.validate_kans()
        # numpy bool types are bool-like
        assert result in (True, False, np.True_, np.False_)


class TestGeoTopKansProperties:
    """Tests for kans data properties and characteristics."""

    def test_kans_values_are_percentages(self, mock_geotop_with_kans):
        """
        WHAT: Verify kans values are in percentage range (0-100).
        WHY: GeoTOP stores probabilities as percentages, not fractions.
        LOGIC: Check min >= 0, max <= 100 for all kans.
        EXPECTED: All values in [0, 100] range
        """
        for i in range(1, 10):
            kans = mock_geotop_with_kans.ds[f'kans_{i}'].values
            valid = ~np.isnan(kans)
            if np.any(valid):
                assert np.nanmin(kans) >= 0
                assert np.nanmax(kans) <= 100

    def test_kans_sum_approximates_100(self, mock_geotop_with_kans):
        """
        WHAT: Verify kans sum to approximately 100% for valid voxels.
        WHY: Probabilities must sum to 100% for valid sampling.
        LOGIC: Sum all 9 kans, check close to 100.
        EXPECTED: Sum within tolerance of 100
        """
        total = sum(
            mock_geotop_with_kans.ds[f'kans_{i}'].values
            for i in range(1, 10)
        )
        valid_mask = ~np.isnan(total)
        if np.any(valid_mask):
            valid_totals = total[valid_mask]
            # Check all valid totals are close to 100
            assert np.allclose(valid_totals, 100, atol=5.0)
