"""
Test suite for lithology sampling from kans distributions.

These tests verify that the sampling functions produce valid, reproducible,
and probabilistically correct lithology realizations.
"""

import numpy as np
import pytest

from atmod.uncertainty import (
    sample_lithology_from_kans,
    HOLOCENE_UNITS,
    compute_lithology_entropy,
    compute_mode_probability,
)


class TestSampleReproducibility:
    """Tests for reproducibility with random seeds."""

    def test_same_seed_produces_identical_results(self, mock_geotop_with_kans):
        """
        WHAT: Same seed produces identical samples.
        WHY: Reproducibility is essential for scientific validity.
        LOGIC: Sample twice with seed=42, compare byte-for-byte.
        EXPECTED: Arrays are exactly equal.
        """
        s1 = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        s2 = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        np.testing.assert_array_equal(s1.values, s2.values)

    def test_different_seeds_produce_different_results(self, mock_geotop_with_kans):
        """
        WHAT: Different seeds produce different samples.
        WHY: Confirms randomness is working properly.
        LOGIC: Sample with seed=42 and seed=43, compare.
        EXPECTED: Arrays differ in at least some locations.
        """
        s1 = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        s2 = sample_lithology_from_kans(mock_geotop_with_kans, seed=43)
        assert not np.array_equal(s1.values, s2.values)

    def test_none_seed_is_random(self, mock_geotop_with_kans):
        """
        WHAT: seed=None produces different results on multiple calls.
        WHY: Ensure true randomness when reproducibility not needed.
        LOGIC: Sample twice without seed, compare.
        EXPECTED: Very likely to differ (astronomical odds of identical).
        """
        s1 = sample_lithology_from_kans(mock_geotop_with_kans, seed=None)
        s2 = sample_lithology_from_kans(mock_geotop_with_kans, seed=None)
        # Statistically, these should differ (very unlikely to be equal)
        assert not np.array_equal(s1.values, s2.values)


class TestSampleValidity:
    """Tests for validity of sampled lithology values."""

    def test_sampled_values_in_valid_range(self, mock_geotop_with_kans):
        """
        WHAT: All sampled values are valid lithology codes (1-9).
        WHY: Invalid codes would break downstream processing.
        LOGIC: Sample and check min/max values.
        EXPECTED: All values in range [1, 9].
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        assert sampled.values.min() >= 1
        assert sampled.values.max() <= 9

    def test_output_shape_matches_input(self, mock_geotop_with_kans):
        """
        WHAT: Output array has same shape as input lithok.
        WHY: Shape mismatch would break downstream merge/build.
        LOGIC: Sample and compare shapes.
        EXPECTED: sampled.shape == lithok.shape
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        assert sampled.shape == mock_geotop_with_kans.ds['lithok'].shape

    def test_output_preserves_coordinates(self, mock_geotop_with_kans):
        """
        WHAT: Output DataArray has same coordinates as input.
        WHY: Coordinate alignment essential for spatial analysis.
        LOGIC: Sample and compare coordinates.
        EXPECTED: x, y, z coordinates identical.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        lithok = mock_geotop_with_kans.ds['lithok']

        np.testing.assert_array_equal(sampled.coords['x'].values, lithok.coords['x'].values)
        np.testing.assert_array_equal(sampled.coords['y'].values, lithok.coords['y'].values)
        np.testing.assert_array_equal(sampled.coords['z'].values, lithok.coords['z'].values)

    def test_output_has_descriptive_name(self, mock_geotop_with_kans):
        """
        WHAT: Output DataArray has meaningful name attribute.
        WHY: Helps identify data in xarray operations.
        LOGIC: Check name attribute.
        EXPECTED: name == 'sampled_lithology'
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        assert sampled.name == 'sampled_lithology'


class TestSampleProbabilityDistribution:
    """Tests for correctness of probability-based sampling."""

    def test_deterministic_kans_produces_single_class(self, mock_geotop_deterministic_kans):
        """
        WHAT: 100% probability for one class always produces that class.
        WHY: Edge case - deterministic data should stay deterministic.
        LOGIC: Sample from kans with 100% clay, check all values = 2.
        EXPECTED: All sampled values == 2 (clay).
        """
        sampled = sample_lithology_from_kans(mock_geotop_deterministic_kans, seed=42)
        np.testing.assert_array_equal(sampled.values, 2)

    def test_high_probability_class_dominates(self, mock_geotop_with_kans):
        """
        WHAT: Class with highest probability appears most frequently.
        WHY: Sampling must follow the distribution.
        LOGIC: In Holocene, kans_2=80% should dominate samples.
        EXPECTED: Class 2 appears in majority of Holocene samples.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        strat = mock_geotop_with_kans.ds['strat'].values
        holocene_mask = np.isin(strat, HOLOCENE_UNITS)

        holocene_samples = sampled.values[holocene_mask]
        class_2_count = np.sum(holocene_samples == 2)
        total_count = len(holocene_samples)

        # With 80% probability, class 2 should appear >60% of the time
        # (allowing statistical variation)
        assert class_2_count / total_count > 0.6

    def test_zero_probability_class_never_appears(self, mock_geotop_with_kans):
        """
        WHAT: Classes with 0% probability never get sampled.
        WHY: Zero probability must mean never selected.
        LOGIC: In Holocene, kans_7 through kans_9 are 0%, shouldn't appear.
        EXPECTED: Classes 7, 8, 9 never appear in Holocene samples.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        strat = mock_geotop_with_kans.ds['strat'].values
        holocene_mask = np.isin(strat, HOLOCENE_UNITS)

        holocene_samples = sampled.values[holocene_mask]
        # Classes 7, 8, 9 have 0% probability in Holocene
        assert np.sum(holocene_samples == 7) == 0
        assert np.sum(holocene_samples == 8) == 0
        assert np.sum(holocene_samples == 9) == 0


class TestHoloceneOnlySampling:
    """Tests for Holocene-only sampling mode."""

    def test_holocene_only_preserves_older_materials(self, mock_geotop_with_kans):
        """
        WHAT: With holocene_only=True, non-Holocene voxels keep original lithok.
        WHY: We only want uncertainty in Holocene (subsidence-relevant) materials.
        LOGIC: Sample with holocene_only=True, compare older voxels to lithok.
        EXPECTED: Older materials unchanged.
        """
        original_lithok = mock_geotop_with_kans.ds['lithok'].values
        strat = mock_geotop_with_kans.ds['strat'].values
        older_mask = ~np.isin(strat, HOLOCENE_UNITS)

        sampled = sample_lithology_from_kans(
            mock_geotop_with_kans, seed=42, holocene_only=True
        )

        # Older voxels should be unchanged
        np.testing.assert_array_equal(
            sampled.values[older_mask],
            original_lithok[older_mask]
        )

    def test_holocene_only_samples_holocene(self, mock_geotop_with_kans):
        """
        WHAT: With holocene_only=True, Holocene voxels are sampled.
        WHY: Holocene materials are the target of uncertainty.
        LOGIC: Sample with holocene_only=True, verify Holocene differs from lithok.
        EXPECTED: At least some Holocene samples differ from lithok.
        """
        original_lithok = mock_geotop_with_kans.ds['lithok'].values
        strat = mock_geotop_with_kans.ds['strat'].values
        holocene_mask = np.isin(strat, HOLOCENE_UNITS)

        sampled = sample_lithology_from_kans(
            mock_geotop_with_kans, seed=42, holocene_only=True
        )

        # Some Holocene voxels should differ (given kans has multiple classes)
        holocene_diff = sampled.values[holocene_mask] != original_lithok[holocene_mask]
        assert np.any(holocene_diff)

    def test_holocene_only_false_samples_everything(self, mock_geotop_with_kans):
        """
        WHAT: With holocene_only=False, all voxels are sampled.
        WHY: User may want full uncertainty (research flexibility).
        LOGIC: Sample with holocene_only=False, check older voxels differ.
        EXPECTED: Older voxels potentially different from original.
        """
        original_lithok = mock_geotop_with_kans.ds['lithok'].values
        strat = mock_geotop_with_kans.ds['strat'].values
        older_mask = ~np.isin(strat, HOLOCENE_UNITS)

        sampled = sample_lithology_from_kans(
            mock_geotop_with_kans, seed=42, holocene_only=False
        )

        # Older voxels might differ if kans != 100% for lithok
        # (Our mock data has 90% fine sand, so ~10% should differ)
        older_diff = sampled.values[older_mask] != original_lithok[older_mask]
        assert np.any(older_diff)


class TestNaNHandling:
    """Tests for handling NaN values."""

    def test_sampling_handles_nan_without_crash(self, mock_geotop_with_nan):
        """
        WHAT: Sampling doesn't crash on NaN values.
        WHY: GeoTOP has NaN outside Netherlands domain.
        LOGIC: Sample from GeoTop with NaN, should not raise.
        EXPECTED: Returns DataArray without exception.
        """
        sampled = sample_lithology_from_kans(mock_geotop_with_nan, seed=42)
        assert sampled is not None
        assert hasattr(sampled, 'values')


class TestErrorHandling:
    """Tests for error handling."""

    def test_raises_error_without_kans(self, mock_geotop_without_kans):
        """
        WHAT: Raises ValueError when kans data not loaded.
        WHY: Cannot sample without probability data.
        LOGIC: Try to sample from GeoTop without kans.
        EXPECTED: ValueError with helpful message.
        """
        with pytest.raises(ValueError, match="kans probability data"):
            sample_lithology_from_kans(mock_geotop_without_kans, seed=42)


class TestEntropyCalculation:
    """Tests for lithology entropy computation."""

    def test_uniform_distribution_has_max_entropy(self, mock_geotop_uniform_kans):
        """
        WHAT: Uniform distribution produces maximum entropy.
        WHY: Maximum uncertainty when all classes equally likely.
        LOGIC: Compute entropy for uniform kans, check near log(9).
        EXPECTED: Entropy close to log(9) ≈ 2.197.
        """
        entropy = compute_lithology_entropy(mock_geotop_uniform_kans)
        max_entropy = np.log(9)  # log of number of classes

        valid_mask = ~np.isnan(entropy.values)
        if np.any(valid_mask):
            mean_entropy = np.mean(entropy.values[valid_mask])
            assert np.isclose(mean_entropy, max_entropy, atol=0.01)

    def test_deterministic_distribution_has_zero_entropy(self, mock_geotop_deterministic_kans):
        """
        WHAT: Deterministic (100% one class) has zero entropy.
        WHY: No uncertainty when one class guaranteed.
        LOGIC: Compute entropy for 100% clay kans, check ~0.
        EXPECTED: Entropy close to 0.
        """
        entropy = compute_lithology_entropy(mock_geotop_deterministic_kans)

        valid_mask = ~np.isnan(entropy.values)
        if np.any(valid_mask):
            mean_entropy = np.mean(entropy.values[valid_mask])
            assert np.isclose(mean_entropy, 0.0, atol=0.01)

    def test_entropy_raises_without_kans(self, mock_geotop_without_kans):
        """
        WHAT: Entropy computation raises error without kans.
        WHY: Cannot compute uncertainty without probability data.
        LOGIC: Try to compute entropy on GeoTop without kans.
        EXPECTED: ValueError raised.
        """
        with pytest.raises(ValueError, match="kans probability data"):
            compute_lithology_entropy(mock_geotop_without_kans)


class TestModeProbability:
    """Tests for mode probability computation."""

    def test_deterministic_has_100_percent_mode(self, mock_geotop_deterministic_kans):
        """
        WHAT: 100% probability gives mode_probability = 100.
        WHY: Maximum confidence when one class certain.
        LOGIC: Compute mode probability for 100% clay kans.
        EXPECTED: mode_probability == 100 everywhere valid.
        """
        mode_prob = compute_mode_probability(mock_geotop_deterministic_kans)

        valid_mask = ~np.isnan(mode_prob.values)
        if np.any(valid_mask):
            np.testing.assert_array_equal(mode_prob.values[valid_mask], 100.0)

    def test_uniform_has_low_mode_probability(self, mock_geotop_uniform_kans):
        """
        WHAT: Uniform distribution has mode_probability ≈ 11.1%.
        WHY: Each of 9 classes has equal probability.
        LOGIC: Compute mode probability for uniform kans.
        EXPECTED: mode_probability ≈ 100/9 ≈ 11.1%.
        """
        mode_prob = compute_mode_probability(mock_geotop_uniform_kans)

        valid_mask = ~np.isnan(mode_prob.values)
        if np.any(valid_mask):
            expected = 100.0 / 9.0
            mean_mode_prob = np.mean(mode_prob.values[valid_mask])
            assert np.isclose(mean_mode_prob, expected, atol=0.1)

    def test_mode_probability_raises_without_kans(self, mock_geotop_without_kans):
        """
        WHAT: Mode probability raises error without kans.
        WHY: Cannot compute without probability data.
        LOGIC: Try on GeoTop without kans.
        EXPECTED: ValueError raised.
        """
        with pytest.raises(ValueError, match="kans probability data"):
            compute_mode_probability(mock_geotop_without_kans)
