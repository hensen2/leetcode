"""
Performance analyzer module for algorithm complexity analysis
Measures and estimates time complexity of functions
"""

import math
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.config import Config
from ..core.generators import IntegerGenerator


class PerformanceAnalyzer:
    """Analyze algorithm performance and estimate complexity"""

    def __init__(self, generator: Optional[IntegerGenerator] = None):
        """
        Initialize performance analyzer

        Args:
            generator: Integer generator for creating test inputs
        """
        self.generator = generator or IntegerGenerator()

    def analyze_complexity(
        self,
        func: Callable,
        sizes: Optional[List[int]] = None,
        iterations: int = Config.PERF_DEFAULT_ITERATIONS,
        input_generator: Optional[Callable[[int], Any]] = None,
        warmup: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze time complexity of a function

        Args:
            func: Function to analyze
            sizes: Input sizes to test
            iterations: Number of iterations per size
            input_generator: Custom input generator (size -> input)
            warmup: Whether to perform warmup runs

        Returns:
            Dictionary with results and estimated complexity
        """
        if sizes is None:
            sizes = Config.PERF_DEFAULT_SIZES

        if input_generator is None:
            input_generator = self._default_input_generator

        # Warmup if requested
        if warmup and sizes:
            self._warmup(func, input_generator, min(sizes))

        # Measure performance
        results = {}
        for size in sizes:
            times = self._measure_performance(func, input_generator(size), iterations)

            results[size] = {
                "avg_time": statistics.mean(times) if times else float("inf"),
                "min_time": min(times) if times else float("inf"),
                "max_time": max(times) if times else float("inf"),
                "median_time": statistics.median(times) if times else float("inf"),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            }

        # Estimate complexity
        complexity = self._estimate_complexity(results)

        return {
            "results": results,
            "estimated_complexity": complexity,
            "sizes_tested": sizes,
            "iterations": iterations,
        }

    def compare_algorithms(
        self,
        algorithms: List[Tuple[str, Callable]],
        sizes: Optional[List[int]] = None,
        iterations: int = Config.PERF_DEFAULT_ITERATIONS,
        input_generator: Optional[Callable[[int], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compare performance of multiple algorithms

        Args:
            algorithms: List of (name, function) tuples
            sizes: Input sizes to test
            iterations: Number of iterations per size
            input_generator: Custom input generator

        Returns:
            Comparison results
        """
        if sizes is None:
            sizes = Config.PERF_DEFAULT_SIZES

        if input_generator is None:
            input_generator = self._default_input_generator

        results = {}

        for algo_name, algo_func in algorithms:
            results[algo_name] = self.analyze_complexity(
                algo_func, sizes, iterations, input_generator, warmup=True
            )

        # Find the best algorithm for each size
        best_by_size = {}
        for size in sizes:
            best_time = float("inf")
            best_algo = None

            for algo_name in results:
                avg_time = results[algo_name]["results"][size]["avg_time"]
                if avg_time < best_time:
                    best_time = avg_time
                    best_algo = algo_name

            best_by_size[size] = {"algorithm": best_algo, "time": best_time}

        return {
            "algorithms": results,
            "best_by_size": best_by_size,
            "recommendation": self._get_recommendation(results, sizes),
        }

    def measure_memory_usage(self, func: Callable, test_input: Any) -> Dict[str, Any]:
        """
        Measure memory usage of a function (placeholder)

        Note: Actual memory profiling would require tracemalloc or memory_profiler

        Args:
            func: Function to measure
            test_input: Input for the function

        Returns:
            Memory usage statistics
        """
        # This is a placeholder - actual implementation would use
        # tracemalloc or memory_profiler library
        return {
            "peak_memory": None,
            "average_memory": None,
            "note": "Memory profiling requires additional libraries",
        }

    def _default_input_generator(self, size: int) -> List[int]:
        """Default input generator - creates integer array"""
        return self.generator.generate_array(size)

    def _warmup(
        self, func: Callable, input_generator: Callable, size: int, runs: int = 3
    ) -> None:
        """Perform warmup runs to stabilize performance"""
        test_input = (
            input_generator(size) if callable(input_generator) else input_generator
        )

        for _ in range(runs):
            try:
                if isinstance(test_input, tuple):
                    func(*test_input)
                elif isinstance(test_input, dict):
                    func(**test_input)
                else:
                    func(test_input)
            except Exception:
                pass  # Ignore errors during warmup

    def _measure_performance(
        self, func: Callable, test_input: Any, iterations: int
    ) -> List[float]:
        """Measure execution time for multiple iterations"""
        times = []

        for _ in range(iterations):
            # Create a copy of input if it's mutable
            input_copy = self._copy_input(test_input)

            start = time.perf_counter()
            try:
                if isinstance(input_copy, tuple):
                    func(*input_copy)
                elif isinstance(input_copy, dict):
                    func(**input_copy)
                else:
                    func(input_copy)

                elapsed = time.perf_counter() - start
                times.append(elapsed)
            except Exception:
                # Skip failed iterations
                continue

        return times

    def _copy_input(self, test_input: Any) -> Any:
        """Create a copy of the input to avoid mutation issues"""
        import copy

        if isinstance(test_input, (list, dict)):
            return copy.deepcopy(test_input)
        elif isinstance(test_input, tuple):
            return tuple(copy.deepcopy(item) for item in test_input)
        else:
            return test_input

    def _estimate_complexity(self, results: Dict[int, Dict]) -> str:
        """
        Estimate time complexity from benchmark results

        Args:
            results: Performance results by input size

        Returns:
            Estimated complexity string
        """
        sizes = sorted(results.keys())
        if len(sizes) < 2:
            return "Insufficient data"

        times = [results[s]["avg_time"] for s in sizes]

        # Skip if any time is infinity
        if any(t == float("inf") for t in times):
            return "Cannot determine (errors in execution)"

        # Calculate growth ratios
        ratios = []
        for i in range(1, len(sizes)):
            if times[i - 1] > 0:
                size_ratio = sizes[i] / sizes[i - 1]
                time_ratio = times[i] / times[i - 1]
                ratios.append((size_ratio, time_ratio))

        if not ratios:
            return "Cannot determine"

        # Average ratios
        avg_size_ratio = sum(r[0] for r in ratios) / len(ratios)
        avg_time_ratio = sum(r[1] for r in ratios) / len(ratios)

        # Estimate complexity based on growth pattern
        tolerance = Config.PERF_COMPLEXITY_TOLERANCE

        # Try to fit different complexity models
        complexities = []

        # O(1) - Constant
        if avg_time_ratio < 1 + tolerance:
            complexities.append(("O(1)", 1.0))

        # O(log n) - Logarithmic
        expected_log_ratio = (
            math.log(sizes[-1]) / math.log(sizes[0]) if sizes[0] > 1 else 1
        )
        if abs(avg_time_ratio - expected_log_ratio) < tolerance:
            complexities.append(("O(log n)", 0.9))

        # O(n) - Linear
        if abs(avg_time_ratio - avg_size_ratio) < avg_size_ratio * tolerance:
            complexities.append(("O(n)", 0.8))

        # O(n log n) - Linearithmic
        expected_nlogn_ratio = (
            avg_size_ratio * (math.log(sizes[-1]) / math.log(sizes[0]))
            if sizes[0] > 1
            else avg_size_ratio
        )
        if (
            abs(avg_time_ratio - expected_nlogn_ratio)
            < expected_nlogn_ratio * tolerance
        ):
            complexities.append(("O(n log n)", 0.7))

        # O(n²) - Quadratic
        if abs(avg_time_ratio - avg_size_ratio**2) < (avg_size_ratio**2) * tolerance:
            complexities.append(("O(n²)", 0.6))

        # O(n³) - Cubic
        if abs(avg_time_ratio - avg_size_ratio**3) < (avg_size_ratio**3) * tolerance:
            complexities.append(("O(n³)", 0.5))

        # O(2ⁿ) - Exponential
        if avg_time_ratio > avg_size_ratio**3:
            complexities.append(("O(2ⁿ) or worse", 0.4))

        # Return the best match
        if complexities:
            complexities.sort(key=lambda x: x[1], reverse=True)
            return f"{complexities[0][0]} - {self._get_complexity_description(complexities[0][0])}"

        return "Cannot determine complexity"

    def _get_complexity_description(self, complexity: str) -> str:
        """Get human-readable description of complexity"""
        descriptions = {
            "O(1)": "Constant",
            "O(log n)": "Logarithmic",
            "O(n)": "Linear",
            "O(n log n)": "Linearithmic",
            "O(n²)": "Quadratic",
            "O(n³)": "Cubic",
            "O(2ⁿ) or worse": "Exponential",
        }
        return descriptions.get(complexity, "Unknown")

    def _get_recommendation(self, results: Dict[str, Any], sizes: List[int]) -> str:
        """
        Get recommendation based on performance analysis

        Args:
            results: Algorithm performance results
            sizes: Sizes tested

        Returns:
            Recommendation string
        """
        if not results:
            return "No algorithms to compare"

        # Find overall best performer
        best_algo = None
        best_score = float("inf")

        for algo_name, algo_results in results.items():
            # Calculate average normalized time across all sizes
            total_time = sum(
                algo_results["results"][size]["avg_time"]
                for size in sizes
                if size in algo_results["results"]
            )

            if total_time < best_score:
                best_score = total_time
                best_algo = algo_name

        if best_algo:
            complexity = results[best_algo]["estimated_complexity"]
            return f"Recommended: {best_algo} (complexity: {complexity})"

        return "Unable to determine best algorithm"


class ComplexityValidator:
    """Validate if a function has expected complexity"""

    @staticmethod
    def validate_complexity(
        func: Callable,
        expected_complexity: str,
        sizes: Optional[List[int]] = None,
        tolerance: float = 0.5,
    ) -> Tuple[bool, str]:
        """
        Validate if function has expected complexity

        Args:
            func: Function to validate
            expected_complexity: Expected complexity (e.g., "O(n)")
            sizes: Sizes to test
            tolerance: Tolerance for validation

        Returns:
            Tuple of (is_valid, message)
        """
        analyzer = PerformanceAnalyzer()
        results = analyzer.analyze_complexity(func, sizes)

        estimated = results["estimated_complexity"]

        # Extract just the big-O notation
        estimated_o = estimated.split(" - ")[0] if " - " in estimated else estimated

        is_valid = expected_complexity.upper() in estimated_o.upper()

        if is_valid:
            message = f"✅ Complexity matches expected {expected_complexity}"
        else:
            message = f"❌ Expected {expected_complexity}, got {estimated_o}"

        return is_valid, message
