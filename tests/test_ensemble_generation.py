"""
Test suite for lithology ensemble generation.

These tests verify that ensemble generation produces the correct number of
distinct, reproducible realizations suitable for Monte Carlo analysis.
"""

import numpy as np
import pytest

from atmod.uncertainty import generate_lithology_ensemble


class TestEnsembleSize:
    """Tests for ensemble dimension and size."""

    def test_ensemble_has_correct_number_of_realizations(self, mock_geotop_with_kans):
        """
        WHAT: Ensemble has requested number of realizations.
        WHY: User expects N realizations for uncertainty quantification.
        LOGIC: Generate ensemble with n=50, check first dimension.
        EXPECTED: ensemble.shape[0] == 50
        """
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=50
        )
        assert ensemble.shape[0] == 50

    def test_ensemble_spatial_shape_matches_input(self, mock_geotop_with_kans):
        """
        WHAT: Ensemble has correct spatial dimensions.
        WHY: Spatial structure must be preserved for each realization.
        LOGIC: Check that (z, y, x) dimensions match lithok.
        EXPECTED: ensemble.shape[1:] == lithok.shape
        """
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10
        )
        lithok_shape = mock_geotop_with_kans.ds['lithok'].shape
        assert ensemble.shape[1:] == lithok_shape

    def test_ensemble_has_realization_dimension(self, mock_geotop_with_kans):
        """
        WHAT: Ensemble DataArray has 'realization' as first dimension.
        WHY: Needed for xarray indexing and analysis.
        LOGIC: Generate ensemble, check 'realization' in dims.
        EXPECTED: 'realization' is first dimension.
        """
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10
        )
        assert 'realization' in ensemble.dims
        assert ensemble.dims[0] == 'realization'

    def test_realization_coordinate_values(self, mock_geotop_with_kans):
        """
        WHAT: Realization coordinate has correct values (0 to N-1).
        WHY: Coordinate values should be sequential integers.
        LOGIC: Generate N=10, check coord values.
        EXPECTED: realization coords == [0, 1, 2, ..., 9]
        """
        n = 10
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=n
        )
        expected = list(range(n))
        assert list(ensemble.coords['realization'].values) == expected


class TestEnsembleDistinctness:
    """Tests for distinctness of realizations."""

    def test_realizations_are_distinct(self, mock_geotop_with_kans):
        """
        WHAT: Each realization in ensemble is different from others.
        WHY: Identical realizations would defeat purpose of ensemble.
        LOGIC: Compare each pair of realizations.
        EXPECTED: No two realizations are identical.
        """
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10, base_seed=42
        )

        for i in range(10):
            for j in range(i + 1, 10):
                r_i = ensemble.isel(realization=i).values
                r_j = ensemble.isel(realization=j).values
                assert not np.array_equal(r_i, r_j), \
                    f"Realizations {i} and {j} are identical"

    def test_single_realization_equals_direct_sample(self, mock_geotop_with_kans):
        """
        WHAT: First realization equals direct sample with same seed.
        WHY: Ensemble should use base_seed + i for each realization.
        LOGIC: Compare ensemble[0] with sample(seed=base_seed).
        EXPECTED: Arrays are identical.
        """
        from atmod.uncertainty import sample_lithology_from_kans

        base_seed = 42
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=5, base_seed=base_seed
        )
        direct_sample = sample_lithology_from_kans(
            mock_geotop_with_kans, seed=base_seed
        )

        np.testing.assert_array_equal(
            ensemble.isel(realization=0).values,
            direct_sample.values
        )


class TestEnsembleReproducibility:
    """Tests for reproducibility of ensembles."""

    def test_same_base_seed_produces_identical_ensembles(self, mock_geotop_with_kans):
        """
        WHAT: Same base_seed produces identical ensembles.
        WHY: Reproducibility for scientific validation.
        LOGIC: Generate two ensembles with same base_seed, compare.
        EXPECTED: Byte-for-byte identical.
        """
        e1 = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10, base_seed=42
        )
        e2 = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10, base_seed=42
        )
        np.testing.assert_array_equal(e1.values, e2.values)

    def test_different_base_seed_produces_different_ensembles(self, mock_geotop_with_kans):
        """
        WHAT: Different base_seed produces different ensembles.
        WHY: Confirms randomness is properly parameterized.
        LOGIC: Generate ensembles with base_seed=42 and base_seed=100.
        EXPECTED: Ensembles differ.
        """
        e1 = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10, base_seed=42
        )
        e2 = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10, base_seed=100
        )
        assert not np.array_equal(e1.values, e2.values)


class TestEnsembleAttributes:
    """Tests for ensemble metadata attributes."""

    def test_ensemble_has_metadata_attributes(self, mock_geotop_with_kans):
        """
        WHAT: Ensemble DataArray has descriptive attributes.
        WHY: Attributes help document the data provenance.
        LOGIC: Generate ensemble, check attrs dict.
        EXPECTED: Contains n_realizations, base_seed, holocene_only.
        """
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10, base_seed=42, holocene_only=True
        )

        assert 'n_realizations' in ensemble.attrs
        assert ensemble.attrs['n_realizations'] == 10
        assert 'base_seed' in ensemble.attrs
        assert ensemble.attrs['base_seed'] == 42
        assert 'holocene_only' in ensemble.attrs
        assert ensemble.attrs['holocene_only'] is True

    def test_ensemble_has_descriptive_name(self, mock_geotop_with_kans):
        """
        WHAT: Ensemble has meaningful name.
        WHY: Helps identify data in xarray operations.
        LOGIC: Check name attribute.
        EXPECTED: name == 'lithology_ensemble'
        """
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=5
        )
        assert ensemble.name == 'lithology_ensemble'


class TestEnsembleHoloceneOnly:
    """Tests for holocene_only parameter in ensemble generation."""

    def test_holocene_only_propagates_to_all_realizations(self, mock_geotop_with_kans):
        """
        WHAT: holocene_only=True affects all realizations consistently.
        WHY: All realizations should respect the same sampling mask.
        LOGIC: Generate ensemble with holocene_only=True, check older voxels.
        EXPECTED: All realizations have unchanged older voxels.
        """
        from atmod.uncertainty import HOLOCENE_UNITS

        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=5, holocene_only=True
        )
        original_lithok = mock_geotop_with_kans.ds['lithok'].values
        strat = mock_geotop_with_kans.ds['strat'].values
        older_mask = ~np.isin(strat, HOLOCENE_UNITS)

        for i in range(5):
            realization = ensemble.isel(realization=i).values
            np.testing.assert_array_equal(
                realization[older_mask],
                original_lithok[older_mask],
                err_msg=f"Realization {i} has modified older voxels"
            )


class TestEnsembleValidity:
    """Tests for validity of ensemble values."""

    def test_all_ensemble_values_in_valid_range(self, mock_geotop_with_kans):
        """
        WHAT: All values in all realizations are valid lithology codes.
        WHY: Invalid codes would break downstream processing.
        LOGIC: Check min/max across entire ensemble.
        EXPECTED: All values in [1, 9].
        """
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=10
        )
        assert ensemble.values.min() >= 1
        assert ensemble.values.max() <= 9
