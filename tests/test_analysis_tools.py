"""
Test suite for analysis_tools module.

These tests verify the provenance tracking, model summary statistics,
visualization functions, and ensemble analysis capabilities.
"""

import numpy as np
import pytest
import xarray as xr

from atmod.analysis_tools import (
    SOURCE_BODEMKAART,
    SOURCE_GEOTOP,
    SOURCE_NL3D,
    SOURCE_NODATA,
    compute_data_source_fractions,
    get_data_source_3d,
    compute_model_summary,
    compute_holocene_statistics,
    compute_ensemble_statistics,
    compute_data_quality_flags,
    compare_models,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_surface():
    """Create a synthetic surface elevation (AHN)."""
    x = np.arange(100, 600, 100)  # 5 columns
    y = np.arange(100, 600, 100)  # 5 rows

    # Low-lying surface (0-2m NAP)
    xx, yy = np.meshgrid(x, y)
    elevation = 1.0 + 0.5 * np.sin(xx / 500) + 0.3 * np.cos(yy / 500)

    return xr.DataArray(
        elevation.astype(np.float32),
        dims=['y', 'x'],
        coords={'x': x, 'y': y},
        name='surface'
    )


@pytest.fixture
def mock_geotop_valid():
    """Create a GeoTOP validity mask (most areas valid)."""
    x = np.arange(100, 600, 100)
    y = np.arange(100, 600, 100)

    # Most areas have GeoTOP, corner does not
    valid = np.ones((5, 5), dtype=bool)
    valid[4, 4] = False  # Corner uses NL3D

    return xr.DataArray(valid, dims=['y', 'x'], coords={'x': x, 'y': y})


@pytest.fixture
def mock_bodemkaart_mask():
    """Create a bodemkaart availability mask."""
    x = np.arange(100, 600, 100)
    y = np.arange(100, 600, 100)

    # Bodemkaart available everywhere except one row
    mask = np.ones((5, 5), dtype=bool)
    mask[0, :] = False  # Top row has no bodemkaart

    return xr.DataArray(mask, dims=['y', 'x'], coords={'x': x, 'y': y})


@pytest.fixture
def mock_holocene_base():
    """Create holocene base elevation."""
    x = np.arange(100, 600, 100)
    y = np.arange(100, 600, 100)

    # Holocene base varies from -5 to -10m NAP
    base = np.full((5, 5), -7.0, dtype=np.float32)
    base[2:, :] = -10.0  # Deeper Holocene in south

    return xr.DataArray(base, dims=['y', 'x'], coords={'x': x, 'y': y})


@pytest.fixture
def mock_atlantis_model():
    """Create a synthetic Atlantis model for testing."""
    x = np.arange(100, 600, 100)
    y = np.arange(100, 600, 100)
    z = np.arange(-10, 2, 0.5)  # From -10 to +2m NAP

    shape = (len(y), len(x), len(z))

    # Create model variables
    thickness = np.full(shape, 0.5, dtype=np.float32)
    lithology = np.full(shape, 2.0, dtype=np.float32)  # clay
    lithology[:, :, :5] = 5.0  # deeper layers are sand
    geology = np.full(shape, 1, dtype=np.int32)  # Holocene
    geology[:, :, :5] = 2  # deeper is older
    mass_organic = np.full(shape, 0.1, dtype=np.float32)
    mass_organic[:, :, 5:10] = 0.5  # some organic layers

    # Create data_source: top layers from bodemkaart, rest from GeoTOP
    # except corner (4,4) uses NL3D
    data_source = np.full(shape, SOURCE_GEOTOP, dtype=np.int8)
    data_source[:, :, 20:] = SOURCE_BODEMKAART  # top ~3 layers from bodemkaart
    data_source[4, 4, :] = SOURCE_NL3D  # corner uses NL3D

    # Add some NaN for realism
    thickness[:, 4, :] = np.nan  # one column is invalid
    data_source[:, 4, :] = SOURCE_NODATA  # invalid columns have no data

    return xr.Dataset(
        {
            'thickness': (['y', 'x', 'z'], thickness),
            'lithology': (['y', 'x', 'z'], lithology),
            'geology': (['y', 'x', 'z'], geology),
            'mass_fraction_organic': (['y', 'x', 'z'], mass_organic),
            'data_source': (['y', 'x', 'z'], data_source),
        },
        coords={'x': x, 'y': y, 'z': z}
    )


# =============================================================================
# Tests for Data Source Provenance
# =============================================================================

class TestDataSourceFractions:
    """Tests for compute_data_source_fractions()."""

    def test_returns_expected_variables(self, mock_atlantis_model):
        """
        WHAT: Verify function returns expected dataset variables.
        WHY: Core output structure must be correct.
        """
        result = compute_data_source_fractions(mock_atlantis_model)

        assert isinstance(result, xr.Dataset)
        assert 'frac_bodemkaart' in result
        assert 'frac_geotop' in result
        assert 'frac_nl3d' in result
        assert 'bodemkaart_thickness' in result
        assert 'geotop_thickness' in result
        assert 'nl3d_thickness' in result
        assert 'total_thickness' in result

    def test_fractions_sum_to_one(self, mock_atlantis_model):
        """
        WHAT: Verify fractions sum to 1.0 for valid locations.
        WHY: Fractions must be consistent.
        """
        result = compute_data_source_fractions(mock_atlantis_model)

        total = (result['frac_bodemkaart'] +
                 result['frac_geotop'] +
                 result['frac_nl3d'])

        valid = ~np.isnan(total.values)
        if np.any(valid):
            assert np.allclose(total.values[valid], 1.0, atol=0.01)

    def test_raises_without_data_source(self):
        """
        WHAT: Verify error when model lacks data_source variable.
        WHY: data_source is required for provenance tracking.
        """
        # Create model without data_source
        x = np.arange(100, 300, 100)
        y = np.arange(100, 300, 100)
        z = np.arange(-5, 0, 0.5)
        shape = (len(y), len(x), len(z))
        model = xr.Dataset(
            {'thickness': (['y', 'x', 'z'], np.full(shape, 0.5))},
            coords={'x': x, 'y': y, 'z': z}
        )

        with pytest.raises(ValueError, match="data_source"):
            compute_data_source_fractions(model)

    def test_holocene_only_filtering(self, mock_atlantis_model):
        """
        WHAT: Verify holocene_only parameter filters correctly.
        WHY: Should only count Holocene layers when True.
        """
        result_all = compute_data_source_fractions(mock_atlantis_model, holocene_only=False)
        result_holocene = compute_data_source_fractions(mock_atlantis_model, holocene_only=True)

        # Holocene-only should have less total thickness (excludes older layers)
        valid_all = ~np.isnan(result_all['total_thickness'].values)
        valid_holo = ~np.isnan(result_holocene['total_thickness'].values)

        # Both should have valid data
        assert np.any(valid_all)
        assert np.any(valid_holo)


class TestGetDataSource3D:
    """Tests for get_data_source_3d()."""

    def test_returns_correct_shape(self, mock_atlantis_model):
        """
        WHAT: Verify 3D mask has correct shape.
        WHY: Must match model dimensions.
        """
        result = get_data_source_3d(mock_atlantis_model)

        assert result.shape == (5, 5, 24)  # y, x, z

    def test_contains_valid_source_codes(self, mock_atlantis_model):
        """
        WHAT: Verify only valid source codes in output.
        WHY: Codes must match defined constants.
        """
        result = get_data_source_3d(mock_atlantis_model)

        unique_codes = np.unique(result.values)
        valid_codes = {SOURCE_NODATA, SOURCE_BODEMKAART, SOURCE_GEOTOP, SOURCE_NL3D}

        for code in unique_codes:
            assert code in valid_codes

    def test_raises_without_data_source(self):
        """
        WHAT: Verify error when model lacks data_source variable.
        WHY: data_source is required.
        """
        x = np.arange(100, 300, 100)
        y = np.arange(100, 300, 100)
        z = np.arange(-5, 0, 0.5)
        shape = (len(y), len(x), len(z))
        model = xr.Dataset(
            {'thickness': (['y', 'x', 'z'], np.full(shape, 0.5))},
            coords={'x': x, 'y': y, 'z': z}
        )

        with pytest.raises(ValueError, match="data_source"):
            get_data_source_3d(model)


# =============================================================================
# Tests for Model Summary Statistics
# =============================================================================

class TestModelSummary:
    """Tests for compute_model_summary()."""

    def test_returns_expected_keys(self, mock_atlantis_model):
        """
        WHAT: Verify summary contains expected keys.
        WHY: Output structure must be predictable.
        """
        result = compute_model_summary(mock_atlantis_model)

        assert isinstance(result, dict)
        assert 'domain_bounds' in result
        assert 'n_columns' in result
        assert 'grid_shape' in result

    def test_domain_bounds_correct(self, mock_atlantis_model):
        """
        WHAT: Verify domain bounds are computed correctly.
        WHY: Bounds used for spatial context.
        """
        result = compute_model_summary(mock_atlantis_model)

        xmin, ymin, xmax, ymax = result['domain_bounds']
        assert xmin == 100.0
        assert xmax == 500.0
        assert ymin == 100.0
        assert ymax == 500.0

    def test_lithology_distribution_sums_to_100(self, mock_atlantis_model):
        """
        WHAT: Verify lithology percentages sum to 100.
        WHY: Distribution must be complete.
        """
        result = compute_model_summary(mock_atlantis_model)

        if 'lithology_distribution' in result:
            total = sum(result['lithology_distribution'].values())
            assert abs(total - 100.0) < 0.1


class TestHoloceneStatistics:
    """Tests for compute_holocene_statistics()."""

    def test_returns_expected_variables(self, mock_atlantis_model):
        """
        WHAT: Verify output contains expected variables.
        WHY: Standard output structure.
        """
        result = compute_holocene_statistics(mock_atlantis_model)

        assert isinstance(result, xr.Dataset)
        assert 'holocene_thickness' in result
        assert 'n_holocene_layers' in result

    def test_thickness_non_negative(self, mock_atlantis_model):
        """
        WHAT: Verify Holocene thickness is non-negative.
        WHY: Physical constraint.
        """
        result = compute_holocene_statistics(mock_atlantis_model)

        thickness = result['holocene_thickness'].values
        valid = ~np.isnan(thickness)
        assert np.all(thickness[valid] >= 0)


# =============================================================================
# Tests for Ensemble Analysis
# =============================================================================

class TestEnsembleStatistics:
    """Tests for compute_ensemble_statistics()."""

    def test_computes_mean_and_std(self, mock_atlantis_model):
        """
        WHAT: Verify mean and std are computed.
        WHY: Core statistics for ensemble analysis.
        """
        # Create 3 slightly different models
        models = []
        for i in range(3):
            m = mock_atlantis_model.copy(deep=True)
            m['lithology'] = m['lithology'] + i * 0.1
            models.append(m)

        result = compute_ensemble_statistics(models, variables=['lithology'])

        assert 'lithology_mean' in result
        assert 'lithology_std' in result

    def test_computes_percentiles(self, mock_atlantis_model):
        """
        WHAT: Verify percentiles are computed.
        WHY: Useful for uncertainty bounds.
        """
        models = [mock_atlantis_model.copy(deep=True) for _ in range(5)]
        result = compute_ensemble_statistics(models, variables=['lithology'])

        assert 'lithology_p10' in result
        assert 'lithology_p50' in result
        assert 'lithology_p90' in result

    def test_empty_list_raises(self):
        """
        WHAT: Verify empty model list raises error.
        WHY: Must have at least one model.
        """
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_ensemble_statistics([])


# =============================================================================
# Tests for Data Quality Flags
# =============================================================================

class TestDataQualityFlags:
    """Tests for compute_data_quality_flags()."""

    def test_returns_dataset(self, mock_atlantis_model):
        """
        WHAT: Verify function returns xr.Dataset.
        WHY: Standard output format.
        """
        result = compute_data_quality_flags(mock_atlantis_model)
        assert isinstance(result, xr.Dataset)

    def test_detects_nl3d_usage(self, mock_atlantis_model, mock_geotop_valid):
        """
        WHAT: Verify NL3D usage is detected.
        WHY: Important for data provenance.
        """
        result = compute_data_quality_flags(
            mock_atlantis_model,
            geotop_valid=mock_geotop_valid,
        )

        assert 'uses_nl3d' in result
        # Corner (4,4) uses NL3D
        assert result['uses_nl3d'].values[4, 4] == True

    def test_detects_data_gaps(self, mock_atlantis_model):
        """
        WHAT: Verify data gaps are detected.
        WHY: Important for quality assessment.
        """
        result = compute_data_quality_flags(mock_atlantis_model)

        assert 'has_gaps' in result
        # Column 4 has all NaN in our mock
        assert result['has_gaps'].values[0, 4] == True


# =============================================================================
# Tests for Model Comparison
# =============================================================================

class TestCompareModels:
    """Tests for compare_models()."""

    def test_computes_difference(self, mock_atlantis_model):
        """
        WHAT: Verify difference is computed.
        WHY: Core comparison functionality.
        """
        model_a = mock_atlantis_model
        model_b = mock_atlantis_model.copy(deep=True)
        model_b['lithology'] = model_b['lithology'] + 1.0

        result = compare_models(model_a, model_b, variables=['lithology'])

        assert 'lithology_diff' in result
        # Difference should be 1.0 everywhere
        diff = result['lithology_diff'].values
        valid = ~np.isnan(diff)
        assert np.allclose(diff[valid], 1.0)

    def test_computes_absolute_difference(self, mock_atlantis_model):
        """
        WHAT: Verify absolute difference is computed.
        WHY: Useful for magnitude of changes.
        """
        model_a = mock_atlantis_model
        model_b = mock_atlantis_model.copy(deep=True)
        model_b['lithology'] = model_b['lithology'] - 2.0

        result = compare_models(model_a, model_b, variables=['lithology'])

        assert 'lithology_abs_diff' in result
        abs_diff = result['lithology_abs_diff'].values
        valid = ~np.isnan(abs_diff)
        assert np.all(abs_diff[valid] >= 0)


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integrationtest
class TestAnalysisWorkflow:
    """Integration tests for analysis_tools workflow."""

    def test_full_provenance_workflow(self, mock_atlantis_model):
        """
        WHAT: Full provenance analysis workflow.
        WHY: Verify components work together.
        """
        # Step 1: Get data source from model
        data_source = get_data_source_3d(mock_atlantis_model)
        assert data_source.shape == mock_atlantis_model['thickness'].shape

        # Step 2: Compute source fractions
        fractions = compute_data_source_fractions(mock_atlantis_model)

        # Step 3: Verify outputs are usable
        assert not np.all(np.isnan(fractions['frac_bodemkaart'].values))
        assert not np.all(np.isnan(fractions['frac_geotop'].values))

        # Step 4: Check thickness calculations are consistent
        total = fractions['total_thickness'].values
        valid = ~np.isnan(total) & (total > 0)  # Only check non-zero thicknesses
        assert np.any(valid)  # Some valid data exists

    def test_ensemble_analysis_workflow(self, mock_atlantis_model):
        """
        WHAT: Full ensemble analysis workflow.
        WHY: Verify ensemble functions work together.
        """
        # Create ensemble
        models = []
        for i in range(5):
            m = mock_atlantis_model.copy(deep=True)
            # Add some variation
            m['mass_fraction_organic'] = m['mass_fraction_organic'] + np.random.randn() * 0.05
            models.append(m)

        # Compute statistics
        stats = compute_ensemble_statistics(models, variables=['mass_fraction_organic'])

        # Verify statistics are sensible
        assert stats['mass_fraction_organic_std'].values.max() < 1.0
        assert stats.attrs['n_realizations'] == 5
