"""
Test runner module for executing and analyzing test cases
Handles test execution, timeout management, and result collection
"""

import signal
import time
from contextlib import contextmanager
from typing import Any, Callable, List, Optional

from ..core.config import Config
from ..core.models import TestRunResult, TestSuiteResult, TreeNode
from ..error_handling.handlers import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    ExecutionError,
    TimeoutError,
)


class CustomComparators:
    """Collection of custom comparator functions for common scenarios"""

    @staticmethod
    def unordered_list(a: List, b: List) -> bool:
        """Compare lists ignoring order"""
        return sorted(a) == sorted(b)

    @staticmethod
    def float_tolerance(tolerance: float = 1e-9) -> Callable:
        """Create comparator for floating-point comparison with tolerance"""

        def compare(a: float, b: float) -> bool:
            return abs(a - b) < tolerance

        return compare

    @staticmethod
    def set_equality(a: Any, b: Any) -> bool:
        """Compare as sets (unique elements, order doesn't matter)"""
        return set(a) == set(b)

    @staticmethod
    def any_valid(valid_outputs: List[Any]) -> Callable:
        """Create comparator that accepts any of the valid outputs"""

        def compare(actual: Any, expected: Any) -> bool:
            return actual in valid_outputs

        return compare

    @staticmethod
    def custom_tree_equality(a: TreeNode, b: TreeNode) -> bool:
        """Compare trees by structure and values"""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return (
            a.val == b.val
            and CustomComparators.custom_tree_equality(a.left, b.left)
            and CustomComparators.custom_tree_equality(a.right, b.right)
        )


class TestRunner:
    """Test runner with integrated timeout and sophisticated error handling"""

    def __init__(self, timeout: float = Config.DEFAULT_TIMEOUT):
        """
        Initialize test runner with error handling integration

        Args:
            timeout: Maximum execution time per test in seconds
        """
        self.timeout = min(timeout, Config.MAX_TIMEOUT)
        self.results: List[TestRunResult] = []

        # INTEGRATION: Use sophisticated error handling
        self.error_handler = ErrorHandler(max_retries=2, retry_delay=0.5)

        # Track error statistics
        self.error_stats = {
            "total_errors": 0,
            "timeout_errors": 0,
            "execution_errors": 0,
            "recovered_errors": 0,
        }

    def run(
        self,
        func: Callable,
        test_input: Any,
        expected_output: Optional[Any] = None,
        custom_comparator: Optional[Callable[[Any, Any], bool]] = None,
        test_name: str = "unnamed_test",
    ) -> TestRunResult:
        """
        Run a single test case with enhanced error handling

        Args:
            func: Function to test
            test_input: Input for the function
            expected_output: Expected output (if any)
            custom_comparator: Custom function to compare outputs
            test_name: Custom test name string


        Returns:
            TestRunResult object with test execution details
        """
        # Create rich error context
        error_context = ErrorContext(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            component="TestRunner",
            operation="run_test",
            test_case=test_input,
            additional_info={
                "function_name": func.__name__
                if hasattr(func, "__name__")
                else str(func),
                "test_name": test_name,
                "expected_output": expected_output,
            },
        )

        result = TestRunResult(
            input=test_input,
            expected=expected_output,
            actual=None,
            passed=False,
            error=None,
            execution_time=None,
        )

        def execute_test():
            """Internal function for test execution with timeout"""
            with self._timeout_context(self.timeout):
                start_time = time.perf_counter()
                actual_output = self._execute_function_safely(
                    func, test_input, error_context
                )
                execution_time = time.perf_counter() - start_time
                return actual_output, execution_time

        try:
            # Execute with sophisticated error handling
            execution_result = self.error_handler.handle_error(
                Exception("dummy"),  # Will be replaced by actual execution
                context=error_context,
                retry_operation=execute_test,
            )

            if execution_result is None:
                # Try direct execution (no retry needed)
                actual_output, execution_time = execute_test()
            else:
                actual_output, execution_time = execution_result

            # Populate successful result
            result.actual = actual_output
            result.execution_time = execution_time

            # Validate result
            if expected_output is not None:
                if custom_comparator:
                    result.passed = self._safe_compare(
                        custom_comparator, actual_output, expected_output, error_context
                    )
                else:
                    result.passed = actual_output == expected_output
            else:
                result.passed = True  # No expected output means success if no error

        except TimeoutError as e:
            self._handle_timeout_error(e, result, error_context)
        except ExecutionError as e:
            self._handle_execution_error(e, result, error_context)
        except Exception as e:
            self._handle_generic_error(e, result, error_context)

        # Record result and update stats
        self.results.append(result)
        self._update_error_stats(result)

        return result

    def run_test_suite(
        self,
        func: Callable,
        test_cases: List[Any],
        expected_outputs: Optional[List[Any]] = None,
        custom_comparator: Optional[Callable[[Any, Any], bool]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        test_names: Optional[List[str]] = None,
    ) -> TestSuiteResult:
        """
        Run multiple test cases with enhanced error reporting

        Args:
            func: Function to test
            test_cases: List of test inputs
            expected_outputs: List of expected outputs
            custom_comparator: Custom comparison function
            progress_callback: Callback for progress updates (current, total)
            test_names: List of custom test name strings

        Returns:
            TestSuiteResult with summary statistics
        """
        if expected_outputs is None:
            expected_outputs = [None] * len(test_cases)

        if test_names is None:
            test_names = [f"test_{i + 1}" for i in range(len(test_cases))]

        if len(test_cases) != len(expected_outputs):
            raise ValueError(
                f"Mismatch between test cases ({len(test_cases)}) "
                f"and expected outputs ({len(expected_outputs)})"
            )

        # Reset error stats for this suite
        self.error_stats = {key: 0 for key in self.error_stats}

        results = []
        passed = failed = errors = timeouts = 0
        total_time = 0.0

        print(f"🚀 Running test suite with {len(test_cases)} test cases...")

        for i, (test_input, expected, test_name) in enumerate(
            zip(test_cases, expected_outputs, test_names)
        ):
            if progress_callback:
                progress_callback(i + 1, len(test_cases))

            # Show progress
            if i % max(1, len(test_cases) // 10) == 0:
                print(f"📊 Progress: {i + 1}/{len(test_cases)} tests")

            result = self.run_test(
                func, test_input, expected, custom_comparator, test_name
            )
            results.append(result)

            # Update counters
            if result.execution_time:
                total_time += result.execution_time

            if result.error and "TIMEOUT" in result.error:
                timeouts += 1
            elif result.error:
                errors += 1
            elif result.passed:
                passed += 1
            else:
                failed += 1

        # Generate comprehensive result with error analysis
        suite_result = TestSuiteResult(
            total=len(test_cases),
            passed=passed,
            failed=failed,
            errors=errors,
            timeout=timeouts,
            results=results,
            total_time=total_time,
        )

        # Add error summary to result
        self._add_error_summary_to_result(suite_result)

        return suite_result

    def _execute_function_safely(
        self, func: Callable, test_input: Any, context: ErrorContext
    ) -> Any:
        """
        Execute function with proper input handling and detailed error context

        Args:
            func: Function to execute
            test_input: Input data
            context: Context information for an error

        Returns:
            Function output
        """
        try:
            if isinstance(test_input, tuple):
                return func(*test_input)
            elif isinstance(test_input, dict):
                return func(**test_input)
            else:
                return func(test_input)
        except Exception as e:
            # Convert to rich error with context
            raise ExecutionError(
                f"Function execution failed: {str(e)}",
                function=context.additional_info.get("function_name", "unknown"),
                operation=context.operation,
                test_case=test_input,
                additional_info={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def _safe_compare(
        self, comparator: Callable, actual: Any, expected: Any, context: ErrorContext
    ) -> bool:
        """Safely execute custom comparator with error handling"""
        try:
            return comparator(actual, expected)
        except Exception as e:
            # Log comparison error but don't fail the test
            comparison_error = ExecutionError(
                f"Custom comparator failed: {str(e)}",
                function="custom_comparator",
                operation="compare_outputs",
                additional_info={
                    "actual": str(actual)[:100],
                    "expected": str(expected)[:100],
                    "comparator_error": str(e),
                },
            )
            self.error_handler.handle_error(comparison_error, context)

            # Fallback to simple equality
            return actual == expected

    @contextmanager
    def _timeout_context(self, seconds: float):
        """Enhanced timeout context with better error reporting"""

        def timeout_occurred(signum, frame):
            raise TimeoutError(
                f"Test execution timed out after {seconds} seconds",
                timeout_duration=seconds,
                component="TestRunner",
                operation="test_execution",
            )

        old_handler = signal.signal(signal.SIGALRM, timeout_occurred)
        signal.setitimer(signal.ITIMER_REAL, seconds)

        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def _handle_timeout_error(
        self, error: TimeoutError, result: TestRunResult, context: ErrorContext
    ):
        """Handle timeout errors with detailed logging"""
        self.error_handler.handle_error(error, context)
        result.error = f"TIMEOUT ({error.context.additional_info.get('timeout_duration', 'unknown')}s)"
        self.error_stats["timeout_errors"] += 1
        self.error_stats["total_errors"] += 1

    def _handle_execution_error(
        self, error: ExecutionError, result: TestRunResult, context: ErrorContext
    ):
        """Handle execution errors with recovery attempts"""
        recovery_result = self.error_handler.handle_error(error, context)

        if recovery_result:
            # Recovery successful
            result.actual = recovery_result
            result.passed = True
            result.error = f"RECOVERED: {str(error)}"
            self.error_stats["recovered_errors"] += 1
        else:
            # Recovery failed
            result.error = f"EXECUTION_ERROR: {str(error)}"
            self.error_stats["execution_errors"] += 1
            self.error_stats["total_errors"] += 1

    def _handle_generic_error(
        self, error: Exception, result: TestRunResult, context: ErrorContext
    ):
        """Handle unexpected errors"""
        # Convert to structured error
        execution_error = ExecutionError(
            f"Unexpected error: {str(error)}",
            function=context.additional_info.get("function_name", "unknown"),
            operation=context.operation,
            additional_info={
                "original_error": str(error),
                "error_type": type(error).__name__,
            },
        )

        self.error_handler.handle_error(execution_error, context)
        result.error = f"UNEXPECTED_ERROR: {str(error)}"
        self.error_stats["total_errors"] += 1

    def _update_error_stats(self, result: TestRunResult):
        """Update error statistics"""
        if result.error and "RECOVERED" in result.error:
            # Don't count recovered errors as failures
            pass
        elif result.error:
            # Error already counted in specific handlers
            pass

    def _add_error_summary_to_result(self, suite_result: TestSuiteResult):
        """Add detailed error summary to test suite result"""
        # Add error summary as additional data
        error_summary = self.error_handler.get_error_summary()

        # Store in the result for later analysis
        if hasattr(suite_result, "error_analysis"):
            suite_result.error_analysis = error_summary
        else:
            # Monkey patch for backward compatibility
            suite_result.error_analysis = error_summary

    def print_enhanced_summary(self, result: TestSuiteResult) -> None:
        """
        Print enhanced test execution summary with error analysis

        Args:
            result: TestSuiteResult to summarize
        """
        print("\n" + "=" * 70)
        print("🧪 ENHANCED TEST SUMMARY")
        print("=" * 70)

        # Basic stats
        print(f"📊 Total Tests: {result.total}")
        print(f"✅ Passed: {result.passed} ({result.pass_rate:.1f}%)")
        print(f"❌ Failed: {result.failed}")
        print(f"🚫 Errors: {result.errors}")
        print(f"⏰ Timeouts: {result.timeout}")
        print(f"⚡ Total Time: {result.total_time:.3f}s")

        # Enhanced error breakdown
        if self.error_stats["total_errors"] > 0:
            print("\n🔍 ERROR BREAKDOWN:")
            print(f"   💥 Execution Errors: {self.error_stats['execution_errors']}")
            print(f"   ⏰ Timeout Errors: {self.error_stats['timeout_errors']}")
            print(f"   🔄 Recovered Errors: {self.error_stats['recovered_errors']}")

            # Show error summary from sophisticated system
            error_summary = self.error_handler.get_error_summary()
            if error_summary.get("errors_by_category"):
                print("\n📋 ERRORS BY CATEGORY:")
                for category, count in error_summary["errors_by_category"].items():
                    print(f"   {category}: {count}")

        # Show problematic test cases
        if result.failed > 0 or result.errors > 0:
            print("\n🚨 PROBLEMATIC TEST CASES:")
            problem_count = 0
            for i, test_result in enumerate(result.results):
                if not test_result.passed or test_result.error:
                    if problem_count < 5:  # Limit output
                        error_msg = test_result.error or "Wrong output"
                        print(f"   Test {i + 1}: {error_msg}")
                        problem_count += 1
                    else:
                        remaining = sum(
                            1 for r in result.results[i:] if not r.passed or r.error
                        )
                        if remaining > 0:
                            print(f"   ... and {remaining} more issues")
                        break

        # Recovery statistics
        recovery_stats = self.error_handler.get_error_summary().get(
            "recovery_stats", {}
        )
        if recovery_stats.get("attempted", 0) > 0:
            print("\n🔄 RECOVERY STATISTICS:")
            print(f"   Attempts: {recovery_stats['attempted']}")
            print(f"   Successful: {recovery_stats['successful']}")
            success_rate = (
                recovery_stats["successful"] / recovery_stats["attempted"]
            ) * 100
            print(f"   Success Rate: {success_rate:.1f}%")

        print("=" * 70)

    def get_error_report(self, format: str = "text") -> str:
        """
        Get detailed error report from the sophisticated error system

        Args:
            format: Format for error report (defaults to text)

        Returns:
            Error report string
        """
        from error_handling import ErrorReporter

        reporter = ErrorReporter(self.error_handler)
        return reporter.generate_report(format)

    def clear_results(self) -> None:
        """Clear stored test results and error history"""
        self.results = []
        self.error_handler.clear_errors()
        self.error_stats = {key: 0 for key in self.error_stats}


class ParallelTestRunner(TestRunner):
    """Test runner with parallel execution support"""

    def __init__(self, timeout: float = Config.DEFAULT_TIMEOUT, max_workers: int = 4):
        """
        Initialize parallel test runner

        Args:
            timeout: Maximum execution time per test
            max_workers: Maximum number of parallel workers
        """
        super().__init__(timeout)
        self.max_workers = max_workers

    def run_test_suite(
        self,
        func: Callable,
        test_cases: List[Any],
        expected_outputs: Optional[List[Any]] = None,
        custom_comparator: Optional[Callable[[Any, Any], bool]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TestSuiteResult:
        """
        Run test suite in parallel

        Note: This is a placeholder for parallel execution.
        Actual implementation would use multiprocessing or threading.
        """
        # For now, fall back to sequential execution
        # Parallel execution would require careful handling of:
        # - Process/thread pool management
        # - Shared state and synchronization
        # - Serialization of test inputs/outputs
        return super().run_test_suite(
            func, test_cases, expected_outputs, custom_comparator, progress_callback
        )
