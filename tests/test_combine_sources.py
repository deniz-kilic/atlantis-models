import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

from atmod.merge import (
    _combine_with_soilprofile,
    _fill_anthropogenic,
    _get_top_voxel_idx,
    _shift_voxel_surface_down,
    _shift_voxel_surface_up,
    SOURCE_BODEMKAART,
    SOURCE_GEOTOP,
)


class TestCombineColumns:
    @pytest.fixture
    def test_voxel_thickness(self):
        thickness = np.repeat([0.5, np.nan], [10, 5])
        return thickness

    @pytest.fixture
    def test_voxel_geology(self):
        geology = np.repeat([2, 1, np.nan], [7, 3, 5])
        return geology

    @pytest.fixture
    def test_voxel_lithology(self):
        lithology = np.repeat([4, 2, 1, 3, np.nan], [3, 2, 3, 2, 5])
        return lithology

    @pytest.fixture
    def test_voxel_organic(self):
        organic = np.repeat([50, np.nan], [10, 5])
        return organic

    @pytest.fixture
    def test_voxel_data_source(self):
        # All voxels start from GeoTOP
        data_source = np.full(15, SOURCE_GEOTOP, dtype=np.int8)
        return data_source

    @pytest.fixture
    def test_soil_thickness(self):
        return np.array([0.2, 0.3, 0.2, 0.5])

    @pytest.fixture
    def test_soil_lithology(self):
        return np.array([1, 3, 3, 2])

    @pytest.fixture
    def test_soil_organic(self):
        return np.array([0.2, 0.3, 0.3, 0.1])

    @pytest.mark.unittest
    def test_voxel_columns_match(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
    ):
        assert len(test_voxel_thickness) == 15
        assert len(test_voxel_geology) == 15
        assert len(test_voxel_lithology) == 15
        assert len(test_voxel_organic) == 15

    @pytest.mark.unittest
    def test_sum_soil_thickness(self, test_soil_thickness):
        assert np.sum(test_soil_thickness) == 1.2

    @pytest.mark.unittest
    def test_get_top_voxel_idx(self, test_voxel_thickness):
        top_idx = _get_top_voxel_idx(test_voxel_thickness)
        assert top_idx == 9

    @pytest.mark.unittest
    def test_fill_anthropogenic(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
    ):
        difference = 3.1
        vt, vg, vl, vo, vs = _fill_anthropogenic(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            difference,
        )
        filled_index = 10
        assert vt[filled_index] == difference
        assert vg[filled_index] == 1.0
        assert vl[filled_index] == 0.0
        assert vo[filled_index] == 0.0
        # Data source should be inherited from layer below
        assert vs[filled_index] == vs[filled_index - 1]

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vl, vo]])
        assert_array_equal(remaining_nans, 4)

    @pytest.mark.unittest
    def test_shift_voxel_surface_up_small_thickness(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
    ):
        modelbase = -5.0
        surface = 0.05

        thickness_to_shift = surface - (modelbase + np.nansum(test_voxel_thickness))
        assert_almost_equal(thickness_to_shift, 0.05)

        vt, _, _, _, _ = _shift_voxel_surface_up(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            surface,
            modelbase,
        )
        changed_voxel_idx = 9
        assert_almost_equal(vt[changed_voxel_idx], 0.55)

        new_surface_level = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_up(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
    ):
        modelbase = -5.0
        surface = 0.15

        thickness_to_shift = surface - (modelbase + np.nansum(test_voxel_thickness))
        assert_almost_equal(thickness_to_shift, 0.15)

        vt, vg, vl, vo, vs = _shift_voxel_surface_up(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            surface,
            modelbase,
        )
        changed_voxel_idx = 10

        assert_almost_equal(vt[changed_voxel_idx], 0.15)
        assert vg[changed_voxel_idx] == vg[changed_voxel_idx - 1]
        assert vl[changed_voxel_idx] == vl[changed_voxel_idx - 1]
        assert vo[changed_voxel_idx] == vo[changed_voxel_idx - 1]
        assert vs[changed_voxel_idx] == vs[changed_voxel_idx - 1]

        new_surface_level = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_down_small_thickness(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
    ):
        modelbase = -5.0
        surface = -0.45

        vt, vg, vl, vo, vs = _shift_voxel_surface_down(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            surface,
            modelbase,
        )
        changed_idx = 8
        assert_almost_equal(vt[changed_idx], 0.55)
        assert np.all(np.isnan(vt[changed_idx + 1 :]))
        assert np.all(np.isnan(vg[changed_idx + 1 :]))
        assert np.all(np.isnan(vl[changed_idx + 1 :]))
        assert np.all(np.isnan(vo[changed_idx + 1 :]))

        new_surface_level = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface_level, surface)

    @pytest.mark.unittest
    def test_shift_voxel_surface_down(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
    ):
        modelbase = -5.0
        surface = -0.25

        vt, _, _, _, _ = _shift_voxel_surface_down(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            surface,
            modelbase,
        )
        changed_idx = 9
        assert_almost_equal(vt[changed_idx], 0.25)

        new_surface_level = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface_level, surface)

    @pytest.mark.unittest
    def test_combine_fitting_soilprofile(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 1.2

        vt, vg, vl, vo, vs = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )
        min_idx_soil = 10
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        top_idx_geology = 9

        assert_equal(vt[min_idx_soil:max_idx_soil], test_soil_thickness)
        assert_equal(vg[min_idx_soil:max_idx_soil], test_voxel_geology[top_idx_geology])
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)
        # Soilprofile layers should be marked as BODEMKAART source
        assert np.all(vs[min_idx_soil:max_idx_soil] == SOURCE_BODEMKAART)

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vg, vl, vo]])
        assert_array_equal(remaining_nans, 1)

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)

    @pytest.mark.unittest
    def test_combine_overlapping_soilprofile(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 0.45

        vt, vg, vl, vo, vs = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )

        min_idx_soil = 9
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        top_idx_geology = 9

        assert_equal(vt[min_idx_soil - 1], 0.25)
        assert_equal(vt[min_idx_soil:max_idx_soil], test_soil_thickness)
        assert_equal(vg[min_idx_soil:max_idx_soil], test_voxel_geology[top_idx_geology])
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)
        # Soilprofile layers should be marked as BODEMKAART source
        assert np.all(vs[min_idx_soil:max_idx_soil] == SOURCE_BODEMKAART)

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vg, vl, vo]])
        assert_array_equal(remaining_nans, 2)

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)

    @pytest.mark.unittest
    def test_combine_soilprofile_above(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 1.3

        vt, vg, vl, vo, vs = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )

        min_idx_soil = 11
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        top_idx_geology = 9

        assert_almost_equal(vt[min_idx_soil - 1], 0.1)
        assert_equal(vt[min_idx_soil:max_idx_soil], test_soil_thickness)
        assert_equal(vg[min_idx_soil:max_idx_soil], test_voxel_geology[top_idx_geology])
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)
        # Soilprofile layers should be marked as BODEMKAART source
        assert np.all(vs[min_idx_soil:max_idx_soil] == SOURCE_BODEMKAART)

        no_remaining_nans = ~np.any(np.isnan(np.array([vt, vg, vl, vo])))
        assert no_remaining_nans

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)

    @pytest.mark.unittest
    def test_combine_soilprofile_above_small_thickness(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 1.204

        vt, vg, vl, vo, vs = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )

        min_idx_soil = 10
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        top_idx_geology = 9

        assert_almost_equal(vt[min_idx_soil], 0.204)
        assert_equal(vg[min_idx_soil:max_idx_soil], test_voxel_geology[top_idx_geology])
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)
        # Soilprofile layers should be marked as BODEMKAART source
        assert np.all(vs[min_idx_soil:max_idx_soil] == SOURCE_BODEMKAART)

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vg, vl, vo]])
        assert_array_equal(remaining_nans, 1)

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)

    @pytest.mark.unittest
    def test_combine_soilprofile_below_small_thickness(
        self,
        test_voxel_thickness,
        test_voxel_geology,
        test_voxel_lithology,
        test_voxel_organic,
        test_voxel_data_source,
        test_soil_thickness,
        test_soil_lithology,
        test_soil_organic,
    ):
        modelbase = -5
        surface = 0.703

        vt, vg, vl, vo, vs = _combine_with_soilprofile(
            test_voxel_thickness,
            test_voxel_geology,
            test_voxel_lithology,
            test_voxel_organic,
            test_voxel_data_source,
            test_soil_thickness,
            test_soil_lithology,
            test_soil_organic,
            surface,
            modelbase,
        )

        min_idx_soil = 9
        max_idx_soil = min_idx_soil + len(test_soil_thickness)

        top_idx_geology = 9

        assert_almost_equal(vt[min_idx_soil], 0.203)
        assert_equal(vg[min_idx_soil:max_idx_soil], test_voxel_geology[top_idx_geology])
        assert_equal(vl[min_idx_soil:max_idx_soil], test_soil_lithology)
        assert_equal(vo[min_idx_soil:max_idx_soil], test_soil_organic)
        # Soilprofile layers should be marked as BODEMKAART source
        assert np.all(vs[min_idx_soil:max_idx_soil] == SOURCE_BODEMKAART)

        remaining_nans = np.array([np.sum(np.isnan(v)) for v in [vt, vl, vo]])
        assert_array_equal(remaining_nans, 2)

        new_surface = modelbase + np.nansum(vt)
        assert_almost_equal(new_surface, surface)
