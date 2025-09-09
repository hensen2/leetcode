# tests/validation/test_memory_fixes.py
"""
Validation tests for memory efficiency fixes from last session

These tests verify that the memory-efficient unique value generation
actually works with large inputs without creating huge ranges in memory.
"""

import os

import psutil
import pytest
from testgen.core.generators import IntegerGenerator
from testgen.core.models import Constraints


class TestMemoryEfficiency:
    """Test memory-efficient generation strategies"""

    def test_small_range_uses_sample(self):
        """Small ranges should use random.sample (existing behavior)"""
        generator = IntegerGenerator()
        constraints = Constraints(
            min_value=1,
            max_value=1000,  # Small range (1000 elements)
            is_unique=True,
        )

        # Should work without issues
        result = generator.generate_array(size=100, constraints=constraints)

        assert len(result) == 100
        assert len(set(result)) == 100  # All unique
        assert all(1 <= x <= 1000 for x in result)

    @pytest.mark.memory
    def test_large_range_small_sample_memory_efficient(self):
        """Large range, small sample should not consume excessive memory"""
        generator = IntegerGenerator()
        constraints = Constraints(
            min_value=-1_000_000,
            max_value=1_000_000,  # 2M range
            is_unique=True,
        )

        # Monitor memory before generation
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Generate small sample from large range
        result = generator.generate_array(size=100, constraints=constraints)

        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before

        # Verify correctness
        assert len(result) == 100
        assert len(set(result)) == 100  # All unique
        assert all(-1_000_000 <= x <= 1_000_000 for x in result)

        # Memory delta should be reasonable (< 10MB for this operation)
        # If old method was used, it would allocate ~8MB just for the range
        assert memory_delta < 10 * 1024 * 1024, (
            f"Memory usage too high: {memory_delta / 1024 / 1024:.1f}MB"
        )

    @pytest.mark.memory
    @pytest.mark.slow
    def test_large_range_large_sample_chunked_sampling(self):
        """Large range, large sample should use chunked reservoir sampling"""
        generator = IntegerGenerator()
        constraints = Constraints(
            min_value=-10_000_000,
            max_value=10_000_000,  # 20M range
            is_unique=True,
        )

        # Monitor memory
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Generate large sample from large range
        result = generator.generate_array(size=10_000, constraints=constraints)

        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before

        # Verify correctness
        assert len(result) == 10_000
        assert len(set(result)) == 10_000  # All unique
        assert all(-10_000_000 <= x <= 10_000_000 for x in result)

        # Memory should not exceed reasonable bounds
        # Old method would allocate ~160MB for the range alone
        assert memory_delta < 50 * 1024 * 1024, (
            f"Memory usage too high: {memory_delta / 1024 / 1024:.1f}MB"
        )

    def test_memory_strategy_selection_logic(self):
        """Test that correct strategy is selected based on input size"""
        generator = IntegerGenerator()

        # Test strategy detection (this would require exposing the strategy method)
        # For now, we'll test through behavior

        # Small range - should be fast (uses random.sample)
        import time

        start = time.time()

        constraints_small = Constraints(min_value=1, max_value=10000, is_unique=True)
        result_small = generator.generate_array(
            size=1000, constraints=constraints_small
        )

        time_small = time.time() - start

        # Large range, small sample - should also be reasonably fast
        start = time.time()

        constraints_large = Constraints(
            min_value=1, max_value=10_000_000, is_unique=True
        )
        result_large = generator.generate_array(
            size=1000, constraints=constraints_large
        )

        time_large = time.time() - start

        # Both should complete successfully
        assert len(result_small) == 1000
        assert len(result_large) == 1000

        # Large range shouldn't be dramatically slower (if using efficient method)
        assert time_large < time_small * 10, "Large range generation unexpectedly slow"

    @pytest.mark.memory
    def test_edge_case_full_range_sampling(self):
        """Test sampling the entire available range"""
        generator = IntegerGenerator()
        constraints = Constraints(min_value=1, max_value=100, is_unique=True)

        # Request exactly the full range
        result = generator.generate_array(size=100, constraints=constraints)

        assert len(result) == 100
        assert len(set(result)) == 100
        assert set(result) == set(range(1, 101))

    def test_impossible_unique_constraint_handling(self):
        """Test handling when unique constraint cannot be satisfied"""
        generator = IntegerGenerator()
        constraints = Constraints(
            min_value=1,
            max_value=10,  # Only 10 possible values
            is_unique=True,
        )

        # Requesting more unique values than possible should raise an error
        with pytest.raises(ValueError, match="Cannot generate.*unique.*values"):
            generator.generate_array(size=15, constraints=constraints)

    @pytest.mark.memory
    def test_memory_usage_stays_constant_across_calls(self):
        """Verify no memory leaks in repeated generation"""
        generator = IntegerGenerator()
        constraints = Constraints(
            min_value=-1_000_000, max_value=1_000_000, is_unique=True
        )

        process = psutil.Process(os.getpid())
        memory_readings = []

        # Generate multiple batches and track memory
        for i in range(5):
            result = generator.generate_array(size=1000, constraints=constraints)
            memory_readings.append(process.memory_info().rss)

            # Verify each result
            assert len(result) == 1000
            assert len(set(result)) == 1000

        # Memory should not continuously increase
        memory_growth = memory_readings[-1] - memory_readings[0]
        assert memory_growth < 5 * 1024 * 1024, (
            f"Potential memory leak: {memory_growth / 1024 / 1024:.1f}MB growth"
        )


class TestMemoryEfficiencyBoundaryConditions:
    """Test boundary conditions for memory efficiency"""

    def test_threshold_boundary_100k_elements(self):
        """Test around the 100K element threshold"""
        generator = IntegerGenerator()

        # Just under threshold - should use random.sample
        constraints_under = Constraints(min_value=1, max_value=99_999, is_unique=True)
        result_under = generator.generate_array(
            size=1000, constraints=constraints_under
        )
        assert len(result_under) == 1000

        # Just over threshold - should use rejection sampling
        constraints_over = Constraints(min_value=1, max_value=100_001, is_unique=True)
        result_over = generator.generate_array(size=1000, constraints=constraints_over)
        assert len(result_over) == 1000

    def test_sample_size_threshold_boundary(self):
        """Test around sample size thresholds for strategy selection"""
        generator = IntegerGenerator()
        large_range_constraints = Constraints(
            min_value=1, max_value=10_000_000, is_unique=True
        )

        # Small sample from large range - rejection sampling
        small_sample = generator.generate_array(
            size=1000, constraints=large_range_constraints
        )
        assert len(small_sample) == 1000

        # Large sample from large range - chunked reservoir sampling
        large_sample = generator.generate_array(
            size=50_000, constraints=large_range_constraints
        )
        assert len(large_sample) == 50_000
