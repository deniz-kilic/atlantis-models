"""
Benchmark tests for parcel aggregation optimization.

These tests measure performance of aggregation methods and compare
optimization levels. Uses real Krimpenerwaard data (500 parcels, 282 layers).

Run with: pytest tests/test_parcels/test_benchmark.py -v -s
"""

import time
import numpy as np
import pytest

from atmod.parcels.aggregation import (
    AggregationMethod,
    aggregate_3d,
    _aggregate_3d_mode_level1,
    _aggregate_3d_mode_level2,
    _aggregate_3d_mode_level3,
)


class TestBaselineBenchmark:
    """Baseline performance measurements with current implementation."""

    @pytest.mark.benchmark
    def test_mode_baseline_timing(self, benchmark_data, benchmark_voxelmodel):
        """
        Measure current mode aggregation performance.

        This establishes the baseline timing that optimizations improve upon.
        Expected: ~30-60 sec for 500 parcels × 282 layers.
        """
        parcels = benchmark_data["parcels"]
        voxelmodel = benchmark_voxelmodel
        n_parcels = benchmark_data["n_parcels"]
        n_layers = benchmark_data["n_layers"]

        print(f"\n{'='*60}")
        print(f"BASELINE BENCHMARK: Mode Aggregation")
        print(f"{'='*60}")
        print(f"Parcels: {n_parcels}")
        print(f"Layers: {n_layers}")
        print(f"Total operations: {n_parcels * n_layers:,}")

        # Time the mode aggregation
        start = time.perf_counter()
        result = aggregate_3d(
            geometries=parcels.geometry,
            voxelmodel=voxelmodel,
            variable="lithology",
            method=AggregationMethod.MODE,
        )
        elapsed = time.perf_counter() - start

        print(f"\nResult shape: {result.shape}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Per-parcel: {elapsed / n_parcels * 1000:.2f} ms")
        print(f"Per-operation: {elapsed / (n_parcels * n_layers) * 1e6:.2f} µs")
        print(f"{'='*60}")

        # Verify result shape
        assert result.shape == (n_parcels, n_layers)
        assert not np.all(np.isnan(result)), "All values are NaN"

        # Store timing for comparison
        return {"method": "baseline", "time": elapsed, "n_parcels": n_parcels}

    @pytest.mark.benchmark
    def test_centroid_baseline_timing(self, benchmark_data, benchmark_voxelmodel):
        """
        Measure centroid aggregation performance (fast baseline).

        Centroid is already fast - this shows the target for optimization.
        Expected: ~1-5 sec for 500 parcels × 282 layers.
        """
        parcels = benchmark_data["parcels"]
        voxelmodel = benchmark_voxelmodel
        n_parcels = benchmark_data["n_parcels"]
        n_layers = benchmark_data["n_layers"]

        # Pre-compute centroids
        centroids = np.column_stack([
            parcels.geometry.centroid.x,
            parcels.geometry.centroid.y,
        ])

        print(f"\n{'='*60}")
        print(f"CENTROID BENCHMARK (reference)")
        print(f"{'='*60}")

        start = time.perf_counter()
        result = aggregate_3d(
            geometries=parcels.geometry,
            voxelmodel=voxelmodel,
            variable="lithology",
            method=AggregationMethod.CENTROID,
            centroids=centroids,
        )
        elapsed = time.perf_counter() - start

        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Per-parcel: {elapsed / n_parcels * 1000:.2f} ms")
        print(f"{'='*60}")

        assert result.shape == (n_parcels, n_layers)

        return {"method": "centroid", "time": elapsed, "n_parcels": n_parcels}

    @pytest.mark.benchmark
    def test_area_weighted_baseline_timing(self, benchmark_data, benchmark_voxelmodel):
        """
        Measure area-weighted aggregation performance.

        Expected: ~30-60 sec for 500 parcels × 282 layers.
        """
        parcels = benchmark_data["parcels"]
        voxelmodel = benchmark_voxelmodel
        n_parcels = benchmark_data["n_parcels"]
        n_layers = benchmark_data["n_layers"]

        print(f"\n{'='*60}")
        print(f"AREA-WEIGHTED BENCHMARK")
        print(f"{'='*60}")

        start = time.perf_counter()
        result = aggregate_3d(
            geometries=parcels.geometry,
            voxelmodel=voxelmodel,
            variable="lithology",  # Using lithology, will use mode internally
            method=AggregationMethod.AREA_WEIGHTED,
        )
        elapsed = time.perf_counter() - start

        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Per-parcel: {elapsed / n_parcels * 1000:.2f} ms")
        print(f"{'='*60}")

        assert result.shape == (n_parcels, n_layers)

        return {"method": "area_weighted", "time": elapsed, "n_parcels": n_parcels}


class TestOptimizedBenchmark:
    """
    Performance measurements for optimized implementations.
    """

    @pytest.mark.benchmark
    def test_mode_level1_numpy(self, benchmark_data, benchmark_voxelmodel):
        """
        Level 1: Numpy indexing + bincount.

        Expected speedup: ~6x (from ~20 sec to ~3 sec).
        """
        parcels = benchmark_data["parcels"]
        voxelmodel = benchmark_voxelmodel
        n_parcels = benchmark_data["n_parcels"]
        n_layers = benchmark_data["n_layers"]

        print(f"\n{'='*60}")
        print(f"LEVEL 1 BENCHMARK: Numpy + Bincount")
        print(f"{'='*60}")

        start = time.perf_counter()
        result = _aggregate_3d_mode_level1(
            geometries=parcels.geometry,
            voxelmodel=voxelmodel,
            variable="lithology",
        )
        elapsed = time.perf_counter() - start

        print(f"Result shape: {result.shape}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Per-parcel: {elapsed / n_parcels * 1000:.2f} ms")
        print(f"Speedup vs baseline (20 sec): {20.0 / elapsed:.1f}x")
        print(f"{'='*60}")

        assert result.shape == (n_parcels, n_layers)

    @pytest.mark.benchmark
    def test_mode_level2_rasterized(self, benchmark_data, benchmark_voxelmodel):
        """
        Level 2: Single rasterization for all parcels.

        Expected speedup: ~40x (from ~20 sec to ~0.5 sec).
        """
        parcels = benchmark_data["parcels"]
        voxelmodel = benchmark_voxelmodel
        n_parcels = benchmark_data["n_parcels"]
        n_layers = benchmark_data["n_layers"]

        print(f"\n{'='*60}")
        print(f"LEVEL 2 BENCHMARK: Single Rasterization")
        print(f"{'='*60}")

        start = time.perf_counter()
        result = _aggregate_3d_mode_level2(
            geometries=parcels.geometry,
            voxelmodel=voxelmodel,
            variable="lithology",
        )
        elapsed = time.perf_counter() - start

        print(f"Result shape: {result.shape}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Per-parcel: {elapsed / n_parcels * 1000:.2f} ms")
        print(f"Speedup vs baseline (20 sec): {20.0 / elapsed:.1f}x")
        print(f"{'='*60}")

        assert result.shape == (n_parcels, n_layers)

    @pytest.mark.benchmark
    def test_mode_level3_parallel(self, benchmark_data, benchmark_voxelmodel):
        """
        Level 3: Parallel processing with 4 workers.

        Expected speedup: ~120x (from ~20 sec to ~0.2 sec).
        """
        parcels = benchmark_data["parcels"]
        voxelmodel = benchmark_voxelmodel
        n_parcels = benchmark_data["n_parcels"]
        n_layers = benchmark_data["n_layers"]

        print(f"\n{'='*60}")
        print(f"LEVEL 3 BENCHMARK: Parallel (4 workers)")
        print(f"{'='*60}")

        start = time.perf_counter()
        result = _aggregate_3d_mode_level3(
            geometries=parcels.geometry,
            voxelmodel=voxelmodel,
            variable="lithology",
            n_workers=4,
        )
        elapsed = time.perf_counter() - start

        print(f"Result shape: {result.shape}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Per-parcel: {elapsed / n_parcels * 1000:.2f} ms")
        print(f"Speedup vs baseline (20 sec): {20.0 / elapsed:.1f}x")
        print(f"{'='*60}")

        assert result.shape == (n_parcels, n_layers)


class TestOptimizationComparison:
    """
    Compare all optimization levels side-by-side.
    """

    @pytest.mark.skip(reason="Full comparison not yet available")
    @pytest.mark.benchmark
    def test_full_comparison(self, benchmark_data, benchmark_voxelmodel):
        """
        Run all optimization levels and compare timings.

        Produces a comparison table:
        | Level | Time | Speedup |
        |-------|------|---------|
        | Baseline | X sec | 1x |
        | Level 1 | Y sec | Nx |
        | Level 2 | Z sec | Mx |
        | Level 3 | W sec | Px |
        """
        # TODO: Implement after all levels are ready
        pass
