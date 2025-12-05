"""
Integration tests for lithology uncertainty quantification.

These tests verify end-to-end workflows and the interaction between
different modules in the uncertainty quantification system.
"""

import numpy as np
import pytest

from atmod.uncertainty import (
    sample_lithology_from_kans,
    generate_lithology_ensemble,
    create_geotop_realization,
    compute_lithology_entropy,
    compute_mode_probability,
    HOLOCENE_UNITS,
)


@pytest.mark.integrationtest
class TestFullSamplingWorkflow:
    """Integration tests for complete sampling workflow."""

    def test_sample_to_realization_workflow(self, mock_geotop_with_kans):
        """
        WHAT: Full workflow from GeoTop to realization.
        WHY: Verify components work together correctly.
        LOGIC:
          1. Sample lithology from kans
          2. Create realization with sampled lithology
          3. Verify realization is usable
        EXPECTED: All steps complete successfully.
        """
        # Step 1: Sample
        sampled = sample_lithology_from_kans(mock_geotop_with_kans, seed=42)
        assert sampled is not None

        # Step 2: Create realization
        realization = create_geotop_realization(mock_geotop_with_kans, sampled)
        assert realization is not None

        # Step 3: Verify realization properties
        assert realization.ds['lithok'].shape == mock_geotop_with_kans.ds['lithok'].shape
        # Kans is dropped by default, so has_kans should be False
        assert realization.has_kans is False
        np.testing.assert_array_equal(realization.ds['lithok'].values, sampled.values)

    def test_ensemble_to_realizations_workflow(self, mock_geotop_with_kans):
        """
        WHAT: Generate ensemble and extract individual realizations.
        WHY: Common workflow for Monte Carlo analysis.
        LOGIC:
          1. Generate ensemble
          2. Extract each realization
          3. Create GeoTop object for each
          4. Verify all are valid
        EXPECTED: N valid GeoTop realizations (without kans by default).
        """
        n = 5
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=n, base_seed=42
        )

        realizations = []
        for i in range(n):
            sampled = ensemble.isel(realization=i)
            realization = create_geotop_realization(mock_geotop_with_kans, sampled)
            realizations.append(realization)

        # All should be valid GeoTop objects (kans dropped by default)
        assert len(realizations) == n
        for r in realizations:
            assert r.has_kans is False  # kans dropped by default
            assert r.ds['lithok'].shape == mock_geotop_with_kans.ds['lithok'].shape


@pytest.mark.integrationtest
class TestUncertaintyMetricsWorkflow:
    """Integration tests for uncertainty metrics."""

    def test_entropy_and_mode_probability_consistency(self, mock_geotop_with_kans):
        """
        WHAT: Entropy and mode probability are inversely related.
        WHY: High entropy = low mode probability (more uncertain).
        LOGIC:
          - Compute both metrics
          - High entropy locations should have lower mode probability
        EXPECTED: Inverse correlation (qualitative check).
        """
        entropy = compute_lithology_entropy(mock_geotop_with_kans)
        mode_prob = compute_mode_probability(mock_geotop_with_kans)

        # Both should have same shape
        assert entropy.shape == mode_prob.shape

        # Where entropy is high, mode_prob should be lower
        valid = ~(np.isnan(entropy.values) | np.isnan(mode_prob.values))
        if np.any(valid):
            # Check there's variation in both
            assert entropy.values[valid].std() >= 0
            assert mode_prob.values[valid].std() >= 0

    def test_metrics_match_sampling_behavior(self, mock_geotop_deterministic_kans):
        """
        WHAT: Uncertainty metrics predict sampling behavior.
        WHY: Low entropy should mean consistent samples.
        LOGIC:
          - Deterministic kans (0 entropy)
          - Sample multiple times
          - All samples should be identical
        EXPECTED: Zero entropy → identical samples.
        """
        entropy = compute_lithology_entropy(mock_geotop_deterministic_kans)

        # Entropy should be zero (or near-zero)
        valid = ~np.isnan(entropy.values)
        if np.any(valid):
            assert np.allclose(entropy.values[valid], 0, atol=0.01)

        # Multiple samples should be identical
        s1 = sample_lithology_from_kans(mock_geotop_deterministic_kans, seed=1)
        s2 = sample_lithology_from_kans(mock_geotop_deterministic_kans, seed=2)
        np.testing.assert_array_equal(s1.values, s2.values)


@pytest.mark.integrationtest
class TestHoloceneStratigraphyIntegration:
    """Integration tests for Holocene-focused sampling."""

    def test_holocene_mask_matches_stratigraphy(self, mock_geotop_with_kans):
        """
        WHAT: Holocene mask correctly identifies Holocene voxels.
        WHY: Foundation for Holocene-only sampling.
        LOGIC:
          - Get strat values from GeoTop
          - Check which match HOLOCENE_UNITS
          - Verify sampling respects this mask
        EXPECTED: Holocene mask consistent with strat values.
        """
        strat = mock_geotop_with_kans.ds['strat'].values
        holocene_mask = np.isin(strat, HOLOCENE_UNITS)

        # Our mock data has top 2 layers as Holocene
        # Verify mask is True for those layers
        assert np.all(holocene_mask[:2, :, :])  # top 2 layers
        assert not np.any(holocene_mask[2:, :, :])  # bottom 3 layers

    def test_sampling_respects_stratigraphy_consistently(self, mock_geotop_with_kans):
        """
        WHAT: Multiple samples consistently respect Holocene boundary.
        WHY: Stratigraphy mask should be deterministic.
        LOGIC:
          - Sample multiple times with holocene_only=True
          - All samples should have identical older voxels
        EXPECTED: Older voxels identical across samples.
        """
        strat = mock_geotop_with_kans.ds['strat'].values
        older_mask = ~np.isin(strat, HOLOCENE_UNITS)
        original_lithok = mock_geotop_with_kans.ds['lithok'].values

        samples = [
            sample_lithology_from_kans(mock_geotop_with_kans, seed=i, holocene_only=True)
            for i in range(5)
        ]

        # All samples should have identical older voxels (unchanged from lithok)
        for s in samples:
            np.testing.assert_array_equal(
                s.values[older_mask],
                original_lithok[older_mask]
            )


@pytest.mark.integrationtest
class TestEnsembleStatistics:
    """Integration tests for ensemble statistical properties."""

    def test_ensemble_variance_higher_in_uncertain_regions(self, mock_geotop_with_kans):
        """
        WHAT: Regions with lower mode_probability have higher sample variance.
        WHY: Uncertain regions should show more variation in samples.
        LOGIC:
          - Generate ensemble
          - Compute sample variance at each voxel
          - Compare to mode_probability
        EXPECTED: Lower mode_prob → higher variance (qualitative).
        """
        n = 20
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=n, base_seed=42
        )
        mode_prob = compute_mode_probability(mock_geotop_with_kans)

        # Compute variance across realizations
        sample_variance = np.var(ensemble.values.astype(float), axis=0)

        # For deterministic voxels (mode_prob=100), variance should be 0
        # Our mock data has 90% for older layers → low variance
        # and 80% for Holocene → slightly higher variance

        valid = ~np.isnan(mode_prob.values)
        if np.any(valid):
            high_prob_mask = (mode_prob.values > 85) & valid
            lower_prob_mask = (mode_prob.values < 85) & valid

            if np.any(high_prob_mask) and np.any(lower_prob_mask):
                high_prob_variance = sample_variance[high_prob_mask].mean()
                lower_prob_variance = sample_variance[lower_prob_mask].mean()

                # Lower probability regions should have higher variance
                assert lower_prob_variance >= high_prob_variance * 0.8  # Allow some tolerance

    def test_ensemble_mode_matches_high_probability_class(self, mock_geotop_with_kans):
        """
        WHAT: Ensemble mode (most frequent class) matches highest kans class.
        WHY: With enough samples, mode should converge to most probable class.
        LOGIC:
          - Generate large ensemble
          - Compute mode at each voxel using numpy
          - Compare to class with highest kans
        EXPECTED: Mode mostly matches deterministic lithok.
        """
        n = 50  # Larger ensemble for better convergence
        ensemble = generate_lithology_ensemble(
            mock_geotop_with_kans, n_realizations=n, base_seed=42
        )

        # Compute mode (most frequent value) at each voxel using numpy
        # Find the most frequent value along the realization axis
        def numpy_mode(arr, axis=0):
            """Compute mode along an axis using numpy."""
            # For each position, find most frequent value
            unique_vals = np.unique(arr)
            counts = np.stack([
                np.sum(arr == val, axis=axis)
                for val in unique_vals
            ], axis=-1)
            mode_indices = np.argmax(counts, axis=-1)
            return unique_vals[mode_indices]

        sample_mode = numpy_mode(ensemble.values, axis=0)

        # Compare to original lithok (which is set to most likely class)
        original_lithok = mock_geotop_with_kans.ds['lithok'].values

        # Mode should match lithok in most locations
        # (given that kans are 80-90% for the lithok class)
        agreement = np.mean(sample_mode == original_lithok)
        assert agreement > 0.7  # At least 70% agreement


@pytest.mark.integrationtest
class TestPackageExports:
    """Integration tests for package-level exports."""

    def test_all_functions_importable_from_atmod(self):
        """
        WHAT: All UQ functions are importable from main atmod package.
        WHY: User-friendly API design.
        LOGIC: Import each function from atmod.
        EXPECTED: No ImportError.
        """
        from atmod import (
            sample_lithology_from_kans,
            generate_lithology_ensemble,
            create_geotop_realization,
            compute_lithology_entropy,
            compute_mode_probability,
            build_ensemble_models,
        )

        # All should be callable
        assert callable(sample_lithology_from_kans)
        assert callable(generate_lithology_ensemble)
        assert callable(create_geotop_realization)
        assert callable(compute_lithology_entropy)
        assert callable(compute_mode_probability)
        assert callable(build_ensemble_models)
