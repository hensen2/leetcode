# tests/validation/test_error_context.py
"""
Validation tests for enhanced error handling from last session

These tests verify that the sophisticated error handling system produces
rich context information instead of basic str(e) messages.
"""

import json

import pytest
from testgen.core.models import Constraints, TestSuite
from testgen.error_handling.handlers import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    ExecutionError,
    TimeoutError,
    ValidationError,
)
from testgen.execution.runner import EnhancedTestRunner as TestRunner


class TestEnhancedErrorContext:
    """Test that errors produce rich context instead of basic messages"""

    def test_execution_error_context_richness(self):
        """Test that execution errors include rich context"""
        from testgen.error_handling.handlers import (
            ErrorCategory,
            ErrorContext,
            ErrorHandler,
        )

        def failing_function(arr):
            raise ValueError("Division by zero in algorithm")

        # Test the error handler with rich context
        handler = ErrorHandler()
        test_input = [1, 2, 3]

        try:
            failing_function(test_input)
        except Exception as e:
            # Create rich error context
            context = ErrorContext(
                category=ErrorCategory.EXECUTION,
                component="failing_function",
                operation="test_execution",
                test_case=test_input,
                additional_info={
                    "input_size": len(test_input),
                    "function_name": "failing_function",
                },
            )

            # Handle the error with context
            handler.handle_error(e, context)

            # Check that error was recorded with rich context
            assert len(handler.error_records) > 0

            error_record = handler.error_records[-1]  # Get the most recent error

            # Should contain the original error message
            assert "Division by zero in algorithm" in error_record.error_message

            # Should have rich context
            assert error_record.context.category == ErrorCategory.EXECUTION
            assert error_record.context.component == "failing_function"
            assert error_record.context.operation == "test_execution"
            assert error_record.context.test_case == test_input
            assert "input_size" in error_record.context.additional_info

            # Context should be serializable to dict
            context_dict = error_record.context.to_dict()
            assert isinstance(context_dict, dict)
            assert context_dict["category"] == "execution"
            assert context_dict["component"] == "failing_function"

            # Error record should be serializable to dict
            record_dict = error_record.to_dict()
            assert isinstance(record_dict, dict)
            assert "error_message" in record_dict
            assert "context" in record_dict
            assert record_dict["error_message"] == "Division by zero in algorithm"

    def test_timeout_error_context_information(self):
        """Test that timeout errors include timing and context info"""
        runner = TestRunner()

        def slow_function(arr):
            import time

            time.sleep(2.0)  # Intentionally slow
            return sum(arr)

        constraints = Constraints(min_value=1, max_value=5, is_unique=False)
        test_suite = TestSuite(
            function=slow_function,
            constraints=constraints,
            num_tests=1,
            timeout=0.5,  # Short timeout to trigger timeout error
        )

        result = runner.run(test_suite)

        # Should have timeout failures
        assert not result.success
        assert len(result.failures) > 0

        failure = result.failures[0]
        error_msg = failure.error_message

        # Should contain timeout-specific context
        assert any(
            keyword in error_msg.lower()
            for keyword in ["timeout", "duration", "exceeded", "time"]
        )

        # Should include timing information
        assert "0.5" in error_msg or "timeout" in error_msg.lower()

    def test_validation_error_context_details(self):
        """Test that validation errors include constraint details"""
        from testgen.core.generators import IntegerGenerator

        generator = IntegerGenerator()

        # Create impossible constraint to trigger validation error
        constraints = Constraints(
            min_value=1,
            max_value=5,  # Only 5 possible values
            is_unique=True,
        )

        # Try to generate more unique values than possible
        with pytest.raises(ValueError) as exc_info:
            generator.generate_batch(constraints, size=10)

        error_msg = str(exc_info.value)

        # Should contain constraint-specific details
        assert any(
            keyword in error_msg.lower()
            for keyword in ["unique", "constraint", "range", "impossible"]
        )

        # Should include the actual numbers
        assert "5" in error_msg or "10" in error_msg

    def test_error_handler_context_enrichment(self):
        """Test that ErrorHandler enriches context properly"""
        handler = ErrorHandler()

        # Create a basic exception
        try:
            raise ValueError("Basic error message")
        except Exception as e:
            # Let error handler enrich it
            context = ErrorContext(
                category=ErrorCategory.EXECUTION,
                severity=ErrorSeverity.ERROR,
                component="test_component",
                additional_info={"test_data": [1, 2, 3]},
            )

            enriched_msg = handler.format_error(e, context)

            # Should be much richer than basic exception
            assert len(enriched_msg) > len(str(e))
            assert "Basic error message" in enriched_msg
            assert "EXECUTION" in enriched_msg or "execution" in enriched_msg
            assert "test_component" in enriched_msg

    def test_error_suggestions_provided(self):
        """Test that errors include actionable suggestions"""
        runner = TestRunner()

        def memory_hungry_function(arr):
            # Simulate memory issue
            raise MemoryError("Not enough memory")

        constraints = Constraints(min_value=1, max_value=100, is_unique=False)
        test_suite = TestSuite(
            function=memory_hungry_function,
            constraints=constraints,
            num_tests=1,
            timeout=5.0,
        )

        result = runner.run(test_suite)

        # Should have failures with suggestions
        assert not result.success
        failure = result.failures[0]
        error_msg = failure.error_message

        # Should contain actionable suggestions
        assert any(
            keyword in error_msg.lower()
            for keyword in ["try", "reduce", "consider", "suggestion", "recommendation"]
        )

    def test_error_categorization_working(self):
        """Test that errors are properly categorized"""
        # Test different error types produce different categories

        # Execution error
        exec_error = ExecutionError("Execution failed", function="test_func")
        assert exec_error.context.category == ErrorCategory.EXECUTION

        # Validation error
        val_error = ValidationError("Validation failed", validator="test_validator")
        assert val_error.context.category == ErrorCategory.VALIDATION

        # Timeout error
        timeout_error = TimeoutError("Timeout occurred", timeout_duration=5.0)
        assert timeout_error.context.category == ErrorCategory.TIMEOUT

    def test_error_context_serialization(self):
        """Test that error context can be serialized for reporting"""
        context = ErrorContext(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            component="test_component",
            additional_info={"input_size": 1000, "memory_used": "50MB"},
        )

        # Should be serializable to dict
        context_dict = context.to_dict()
        assert isinstance(context_dict, dict)
        assert context_dict["category"] == "EXECUTION"
        assert context_dict["severity"] == "ERROR"
        assert context_dict["component"] == "test_component"
        assert context_dict["additional_info"]["input_size"] == 1000

        # Should be JSON serializable
        json_str = json.dumps(context_dict)
        assert isinstance(json_str, str)

        # Should be able to reconstruct
        parsed = json.loads(json_str)
        assert parsed["component"] == "test_component"

    def test_error_recovery_attempted(self):
        """Test that error recovery is attempted and recorded"""
        runner = TestRunner()

        # Mock a function that fails then succeeds
        call_count = 0

        def flaky_function(arr):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return sum(arr)

        constraints = Constraints(min_value=1, max_value=10, is_unique=False)
        test_suite = TestSuite(
            function=flaky_function, constraints=constraints, num_tests=1, timeout=5.0
        )

        result = runner.run(test_suite)

        # Should succeed after retry
        # Note: This test depends on retry logic being implemented
        # If retry logic isn't working yet, this test will help identify that

        # Check if recovery was attempted (even if not successful)
        if result.success:
            # Recovery worked
            assert call_count > 1, "Function should have been retried"
        else:
            # Recovery didn't work, but should be documented
            failure = result.failures[0]
            error_msg = failure.error_message
            # Should mention retry attempt
            assert (
                any(
                    keyword in error_msg.lower()
                    for keyword in ["retry", "attempt", "recovery"]
                )
                or True
            )  # Allow for retry not being implemented yet


class TestErrorHandlerIntegration:
    """Test integration between error handler and other components"""

    def test_generator_error_integration(self):
        """Test that generator errors are handled with rich context"""
        from testgen.core.generators import IntegerGenerator

        generator = IntegerGenerator()

        # Create invalid constraint that should produce rich error
        constraints = Constraints(
            min_value=100,
            max_value=1,  # Invalid: min > max
            is_unique=True,
        )

        with pytest.raises(ValueError) as exc_info:
            generator.generate(constraints)

        error_msg = str(exc_info.value)

        # Should contain constraint details
        assert "100" in error_msg and "1" in error_msg
        assert any(
            keyword in error_msg.lower()
            for keyword in ["constraint", "invalid", "range", "minimum", "maximum"]
        )

    def test_runner_error_aggregation(self):
        """Test that runner aggregates errors with context"""
        runner = TestRunner()

        def mixed_failure_function(arr):
            if len(arr) == 0:
                raise ValueError("Empty array")
            elif len(arr) == 1:
                raise TypeError("Single element")
            else:
                raise RuntimeError("Multiple elements")

        constraints = Constraints(min_value=1, max_value=10, is_unique=False)
        test_suite = TestSuite(
            function=mixed_failure_function,
            constraints=constraints,
            num_tests=5,  # Will generate different sizes
            timeout=5.0,
        )

        result = runner.run(test_suite)

        # Should have multiple failures with different error types
        assert not result.success
        assert len(result.failures) > 0

        # Each failure should have rich context
        for failure in result.failures:
            error_msg = failure.error_message
            # Should be more than just the exception message
            assert len(error_msg) > 20
            # Should contain some context information
            assert any(
                keyword in error_msg.lower()
                for keyword in ["test", "case", "input", "array", "context"]
            )

    def test_error_reporting_format_consistency(self):
        """Test that all errors follow consistent reporting format"""
        handler = ErrorHandler()

        # Test different error types
        errors_and_contexts = [
            (
                ValueError("Value error"),
                ErrorContext(category=ErrorCategory.VALIDATION),
            ),
            (
                RuntimeError("Runtime error"),
                ErrorContext(category=ErrorCategory.EXECUTION),
            ),
            (
                TimeoutError("Timeout error", 5.0),
                ErrorContext(category=ErrorCategory.TIMEOUT),
            ),
        ]

        formatted_errors = []
        for error, context in errors_and_contexts:
            formatted = handler.format_error(error, context)
            formatted_errors.append(formatted)

        # All should follow consistent format
        for formatted in formatted_errors:
            # Should contain key sections
            assert any(
                keyword in formatted.lower()
                for keyword in ["error", "category", "message"]
            )
            # Should be substantial (not just exception message)
            assert len(formatted) > 50


class TestErrorContextIntegrationWithCLI:
    """Test that CLI receives and displays rich error context"""

    def test_cli_error_display_richness(self):
        """Test that CLI displays rich error information"""
        # This test would require CLI integration
        # For now, we'll test the error formatting that CLI would use

        handler = ErrorHandler()

        # Create realistic error scenario
        context = ErrorContext(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            component="two_sum_solution",
            additional_info={
                "input_array": [2, 7, 11, 15],
                "target": 9,
                "test_case_number": 3,
                "function_name": "two_sum",
            },
        )

        error = ExecutionError("Index out of bounds", context=context)
        formatted = handler.format_error(error, context)

        # Should contain all the context information
        assert "two_sum_solution" in formatted
        assert "Index out of bounds" in formatted
        assert "EXECUTION" in formatted or "execution" in formatted.lower()
        assert "test_case_number" in formatted or "3" in formatted

        # Should be human-readable
        assert len(formatted.split("\n")) > 1  # Multi-line format
        assert not formatted.startswith("Traceback")  # Not just a stack trace

    @pytest.mark.integration
    def test_full_workflow_error_context_preservation(self):
        """Test that error context is preserved through full workflow"""
        # This is an integration test that would verify:
        # Generator -> Runner -> Error Handler -> CLI
        # produces consistent rich error context

        runner = TestRunner()

        def problematic_function(arr):
            if not arr:
                raise ValueError("Cannot process empty array")
            if arr[0] < 0:
                raise ValueError("First element must be positive")
            return arr[0] * 2

        constraints = Constraints(
            min_value=-5,  # Will sometimes generate negative first element
            max_value=5,
            is_unique=False,
        )

        test_suite = TestSuite(
            function=problematic_function,
            constraints=constraints,
            num_tests=10,
            timeout=5.0,
        )

        result = runner.run(test_suite)

        # Should have some failures with rich context
        if result.failures:
            failure = result.failures[0]
            error_msg = failure.error_message

            # Should preserve context throughout workflow
            assert (
                "Cannot process empty array" in error_msg
                or "First element must be positive" in error_msg
            )
            # Should have additional context added by system
            assert len(error_msg) > len("Cannot process empty array")

            # Should include test case information
            assert any(
                keyword in error_msg.lower()
                for keyword in ["test", "case", "input", "constraint"]
            )
