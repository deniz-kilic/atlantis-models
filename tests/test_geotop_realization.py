"""
Test suite for GeoTop realization creation.

These tests verify that we can create valid GeoTop objects with sampled
lithology that work with the existing build pipeline.
"""

import numpy as np
import pytest

from atmod.uncertainty import (
    create_geotop_realization,
    sample_lithology_from_kans,
)
from atmod.bro_models.voxelmodels import GeoTop


class TestRealizationCreation:
    """Tests for creating GeoTop realizations."""

    def test_realization_replaces_lithok(self, mock_geotop_with_kans):
        """
        WHAT: Created realization has sampled lithology as lithok.
        WHY: Build pipeline uses lithok; must be replaced for ensemble.
        LOGIC: Sample, create realization, compare lithok.
        EXPECTED: realization.ds['lithok'] == sampled
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        np.testing.assert_array_equal(
            realization.ds['lithok'].values,
            sampled.values
        )

    def test_realization_preserves_strat(self, mock_geotop_with_kans):
        """
        WHAT: Stratigraphy is unchanged in realization.
        WHY: Only lithology varies; strat defines Holocene/Older.
        LOGIC: Create realization, compare strat to original.
        EXPECTED: strat identical.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        np.testing.assert_array_equal(
            realization.ds['strat'].values,
            mock_geotop_with_kans.ds['strat'].values
        )

    def test_realization_drops_kans_by_default(self, mock_geotop_with_kans):
        """
        WHAT: Kans data is dropped by default in realization.
        WHY: Kans is only needed for sampling, not for Atlantis runs.
        LOGIC: Create realization with default settings, check kans removed.
        EXPECTED: No kans variables in realization.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        for i in range(1, 10):
            assert f'kans_{i}' not in realization.ds

    def test_realization_preserves_kans_when_requested(self, mock_geotop_with_kans):
        """
        WHAT: Kans data is preserved when drop_kans=False.
        WHY: May want to keep kans for debugging or analysis.
        LOGIC: Create realization with drop_kans=False, check kans_1-9 exist.
        EXPECTED: All kans variables preserved.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(
            mock_geotop_with_kans, sampled, drop_kans=False
        )

        for i in range(1, 10):
            assert f'kans_{i}' in realization.ds
            np.testing.assert_array_equal(
                realization.ds[f'kans_{i}'].values,
                mock_geotop_with_kans.ds[f'kans_{i}'].values
            )


class TestRealizationProperties:
    """Tests for realization object properties."""

    def test_realization_is_geotop_instance(self, mock_geotop_with_kans):
        """
        WHAT: Realization is a GeoTop instance.
        WHY: Must be compatible with build pipeline expectations.
        LOGIC: Create realization, check type.
        EXPECTED: isinstance(realization, GeoTop)
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        assert isinstance(realization, GeoTop)

    def test_realization_preserves_cellsize(self, mock_geotop_with_kans):
        """
        WHAT: Realization has same cellsize as original.
        WHY: Resolution must be preserved for spatial calculations.
        LOGIC: Create realization, compare cellsize.
        EXPECTED: Cellsizes match.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        assert realization.cellsize == mock_geotop_with_kans.cellsize

    def test_realization_preserves_dz(self, mock_geotop_with_kans):
        """
        WHAT: Realization has same vertical resolution (dz).
        WHY: Vertical resolution must be preserved.
        LOGIC: Create realization, compare dz.
        EXPECTED: dz values match.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        assert realization.dz == mock_geotop_with_kans.dz

    def test_realization_preserves_epsg(self, mock_geotop_with_kans):
        """
        WHAT: Realization has same EPSG as original.
        WHY: Coordinate reference system must be preserved.
        LOGIC: Create realization, compare epsg.
        EXPECTED: EPSG values match.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        assert realization.epsg == mock_geotop_with_kans.epsg

    def test_realization_preserves_coordinates(self, mock_geotop_with_kans):
        """
        WHAT: Spatial coordinates unchanged in realization.
        WHY: Coordinate mismatch would break spatial alignment.
        LOGIC: Create realization, compare x, y, z coords.
        EXPECTED: All coordinates identical.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        np.testing.assert_array_equal(
            realization.ds.coords['x'].values,
            mock_geotop_with_kans.ds.coords['x'].values
        )
        np.testing.assert_array_equal(
            realization.ds.coords['y'].values,
            mock_geotop_with_kans.ds.coords['y'].values
        )
        np.testing.assert_array_equal(
            realization.ds.coords['z'].values,
            mock_geotop_with_kans.ds.coords['z'].values
        )


class TestRealizationDataIntegrity:
    """Tests for data integrity in realizations."""

    def test_realization_is_independent_of_original(self, mock_geotop_with_kans):
        """
        WHAT: Modifying realization doesn't affect original.
        WHY: Deep copy must be made to avoid side effects.
        LOGIC: Create realization, modify it, check original unchanged.
        EXPECTED: Original GeoTop unmodified.
        """
        original_lithok = mock_geotop_with_kans.ds['lithok'].values.copy()

        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        # Modify realization
        realization.ds['lithok'].values[:] = 9

        # Original should be unchanged
        np.testing.assert_array_equal(
            mock_geotop_with_kans.ds['lithok'].values,
            original_lithok
        )

    def test_realization_has_kans_false_by_default(self, mock_geotop_with_kans):
        """
        WHAT: Realization has has_kans=False by default (kans dropped).
        WHY: Kans is dropped by default to reduce memory for Atlantis runs.
        LOGIC: Create realization with default settings, check has_kans.
        EXPECTED: has_kans == False
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)

        assert realization.has_kans is False

    def test_realization_has_kans_true_when_preserved(self, mock_geotop_with_kans):
        """
        WHAT: Realization has has_kans=True when drop_kans=False.
        WHY: Can preserve kans for debugging or analysis.
        LOGIC: Create realization with drop_kans=False, check has_kans.
        EXPECTED: has_kans == True
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        realization = create_geotop_realization(
            mock_geotop_with_kans, sampled, drop_kans=False
        )

        assert realization.has_kans is True
