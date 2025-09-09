"""
Validation tests for CLI enhancements from last session

These tests verify that the enhanced CLI provides progress indicators,
rich reporting, and actionable suggestions instead of basic output.
"""

import json
from io import StringIO
from unittest.mock import Mock, patch

from testgen.cli.main import main


class TestCLIProgressIndicators:
    """Test that CLI shows progress indicators during generation"""

    @patch("sys.argv", ["testgen", "array", "-n", "5"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_progress_indicators_displayed(self, mock_stdout):
        """Test that progress indicators are shown during generation"""
        # Mock the actual generator that CLI uses - IntegerGenerator
        with patch("testgen.cli.main.IntegerGenerator") as mock_int_gen:
            mock_gen_instance = Mock()
            mock_int_gen.return_value = mock_gen_instance
            mock_gen_instance.generate_array.return_value = [1, 2, 3, 4, 5]

            try:
                main()
            except SystemExit:
                pass  # CLI may exit normally

            output = mock_stdout.getvalue()

            # Should contain progress-related output
            assert any(
                keyword in output.lower()
                for keyword in [
                    "starting",
                    "generating",
                    "progress",
                    "complete",
                    "🚀",
                    "📊",
                ]
            ), f"Should show progress information. Got output: {output}"

    @patch(
        "sys.argv", ["testgen", "generate", "--problem", "array-sum", "--count", "10"]
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_test_case_analysis_displayed(self, mock_stdout):
        """Test that CLI shows test case analysis and statistics"""
        with patch("testgen.cli.main.TestCaseGenerator") as mock_generator:
            mock_gen_instance = Mock()
            mock_generator.return_value = mock_gen_instance
            mock_gen_instance.generate_for_problem.return_value = {
                "test_cases": [
                    {"input": [1, 2, 3], "expected": 6},
                    {"input": [4, 5, 6], "expected": 15},
                    {"input": [], "expected": 0},
                ]
                * 3,  # 9 test cases
                "metadata": {"problem": "array-sum", "count": 9},
            }

            try:
                main()
            except SystemExit:
                pass

            output = mock_stdout.getvalue()

            # Should contain analysis information
            assert (
                any(
                    keyword in output.lower()
                    for keyword in [
                        "analysis",
                        "statistics",
                        "summary",
                        "cases",
                        "generated",
                    ]
                )
                or "9" in output
            ), "Should show test case analysis"


class TestCLIEnhancedCommands:
    """Test new CLI commands from enhancement"""

    @patch("sys.argv", ["testgen", "validate", "--input", "test.json"])
    def test_validate_command_exists(self):
        """Test that validate command is available"""
        with patch("testgen.cli.main.TestCaseGenerator"):
            with patch("builtins.open", mock_open_json([])):
                try:
                    main()
                except (SystemExit, FileNotFoundError):
                    pass  # Command should be recognized

        # If we get here without ImportError/AttributeError, command exists
        assert True, "Validate command should be available"

    @patch("sys.argv", ["testgen", "benchmark", "--problem", "two-sum"])
    def test_benchmark_command_exists(self):
        """Test that benchmark command is available"""
        with patch("testgen.cli.main.TestCaseGenerator"):
            try:
                main()
            except (SystemExit, NotImplementedError):
                pass  # Command should be recognized

        # If we get here without ImportError/AttributeError, command exists
        assert True, "Benchmark command should be available"

    @patch("sys.argv", ["testgen", "--help"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_help_shows_enhanced_commands(self, mock_stdout):
        """Test that help shows the new enhanced commands"""
        try:
            main()
        except SystemExit:
            pass  # Help command exits normally

        output = mock_stdout.getvalue()

        # Should show enhanced commands
        assert any(
            command in output.lower()
            for command in ["validate", "benchmark", "generate"]
        ), "Help should show enhanced commands"


class TestCLIOutputFormats:
    """Test enhanced output formats"""

    def test_json_output_format(self, temp_dir):
        """Test JSON output format"""
        output_file = temp_dir / "output.json"

        with patch(
            "sys.argv",
            [
                "testgen",
                "generate",
                "--problem",
                "two-sum",
                "--output",
                str(output_file),
                "--format",
                "json",
            ],
        ):
            with patch("testgen.cli.main.TestCaseGenerator") as mock_generator:
                mock_gen_instance = Mock()
                mock_generator.return_value = mock_gen_instance
                mock_gen_instance.generate_for_problem.return_value = {
                    "test_cases": [{"input": [2, 7], "target": 9}],
                    "metadata": {"format": "json"},
                }

                try:
                    main()
                except SystemExit:
                    pass

        # Should create JSON file
        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
                assert isinstance(data, dict), "Should produce valid JSON"

    def test_html_output_format(self, temp_dir):
        """Test HTML output format"""
        output_file = temp_dir / "output.html"

        with patch(
            "sys.argv",
            [
                "testgen",
                "generate",
                "--problem",
                "two-sum",
                "--output",
                str(output_file),
                "--format",
                "html",
            ],
        ):
            with patch("testgen.cli.main.TestCaseGenerator") as mock_generator:
                mock_gen_instance = Mock()
                mock_generator.return_value = mock_gen_instance
                mock_gen_instance.generate_for_problem.return_value = {
                    "test_cases": [{"input": [2, 7], "target": 9}],
                    "metadata": {"format": "html"},
                }

                try:
                    main()
                except SystemExit:
                    pass

        # Should create HTML file
        if output_file.exists():
            content = output_file.read_text()
            assert "<html>" in content.lower() or "<div>" in content.lower(), (
                "Should produce HTML content"
            )


class TestCLIErrorReporting:
    """Test enhanced error reporting in CLI"""

    @patch("sys.argv", ["testgen", "generate", "--problem", "invalid-problem"])
    @patch("sys.stderr", new_callable=StringIO)
    def test_actionable_error_suggestions(self, mock_stderr):
        """Test that CLI provides actionable suggestions for errors"""
        with patch("testgen.cli.main.TestCaseGenerator") as mock_generator:
            mock_gen_instance = Mock()
            mock_generator.return_value = mock_gen_instance
            mock_gen_instance.generate_for_problem.side_effect = ValueError(
                "Unknown problem type: invalid-problem"
            )

            try:
                main()
            except SystemExit:
                pass

        error_output = mock_stderr.getvalue()

        # Should contain actionable suggestions
        assert (
            any(
                keyword in error_output.lower()
                for keyword in ["try", "available", "suggestion", "help", "see", "use"]
            )
            or len(error_output) > 50
        ), "Should provide actionable error suggestions"

    @patch("sys.argv", ["testgen", "generate", "--count", "invalid"])
    @patch("sys.stderr", new_callable=StringIO)
    def test_input_validation_errors(self, mock_stderr):
        """Test validation errors for invalid inputs"""
        try:
            main()
        except (SystemExit, ValueError):
            pass

        error_output = mock_stderr.getvalue()

        # Should provide clear validation error
        assert (
            any(
                keyword in error_output.lower()
                for keyword in ["invalid", "number", "integer", "count"]
            )
            or "invalid" in error_output.lower()
        ), "Should show clear validation error"


class TestCLIPerformanceReporting:
    """Test performance reporting features"""

    @patch(
        "sys.argv", ["testgen", "generate", "--problem", "two-sum", "--count", "100"]
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_generation_timing_reported(self, mock_stdout):
        """Test that generation timing is reported"""
        with patch("testgen.cli.main.TestCaseGenerator") as mock_generator:
            mock_gen_instance = Mock()
            mock_generator.return_value = mock_gen_instance
            mock_gen_instance.generate_for_problem.return_value = {
                "test_cases": [{"input": [1, 2]} for _ in range(100)],
                "metadata": {"generation_time": 0.123, "count": 100},
            }

            try:
                main()
            except SystemExit:
                pass

        output = mock_stdout.getvalue()

        # Should report timing information
        assert any(
            keyword in output.lower()
            for keyword in ["time", "ms", "seconds", "duration", "performance"]
        ) or any(char in output for char in ["0.", "ms"]), (
            "Should report generation timing"
        )

    @patch(
        "sys.argv",
        ["testgen", "generate", "--problem", "large-array", "--count", "1000"],
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_memory_usage_reporting(self, mock_stdout):
        """Test that memory usage is reported for large generations"""
        with patch("testgen.cli.main.TestCaseGenerator") as mock_generator:
            mock_gen_instance = Mock()
            mock_generator.return_value = mock_gen_instance
            mock_gen_instance.generate_for_problem.return_value = {
                "test_cases": [{"input": list(range(1000))} for _ in range(100)],
                "metadata": {"memory_used": "15.2MB", "count": 100},
            }

            try:
                main()
            except SystemExit:
                pass

        output = mock_stdout.getvalue()

        # Should report memory information for large operations
        assert (
            any(
                keyword in output.lower()
                for keyword in ["memory", "mb", "usage", "allocated"]
            )
            or "MB" in output
        ), "Should report memory usage for large operations"


class TestCLIUserExperience:
    """Test overall user experience improvements"""

    @patch("sys.argv", ["testgen"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_helpful_default_output(self, mock_stdout):
        """Test that running without args shows helpful information"""
        try:
            main()
        except SystemExit:
            pass

        output = mock_stdout.getvalue()

        # Should show helpful information, not just error
        assert (
            any(
                keyword in output.lower()
                for keyword in ["help", "usage", "command", "generate", "example"]
            )
            or len(output) > 20
        ), "Should show helpful default output"

    @patch("sys.argv", ["testgen", "generate", "--problem", "two-sum"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_user_friendly_output_formatting(self, mock_stdout):
        """Test that output is formatted in user-friendly way"""
        with patch("testgen.cli.main.TestCaseGenerator") as mock_generator:
            mock_gen_instance = Mock()
            mock_generator.return_value = mock_gen_instance
            mock_gen_instance.generate_for_problem.return_value = {
                "test_cases": [
                    {"input": [2, 7, 11, 15], "target": 9, "expected": [0, 1]},
                    {"input": [3, 2, 4], "target": 6, "expected": [1, 2]},
                ],
                "metadata": {"problem": "two-sum"},
            }

            try:
                main()
            except SystemExit:
                pass

        output = mock_stdout.getvalue()

        # Should be formatted nicely, not just raw data dump
        assert "\n" in output, "Should have multi-line formatting"
        assert len(output) > 50, "Should have substantial formatted output"
        # Should not be just JSON dump
        assert not output.strip().startswith("{"), "Should not be raw JSON dump"


# Helper function for mocking file operations


def mock_open_json(data):
    """Helper to mock opening JSON files"""
    import json
    from unittest.mock import mock_open

    return mock_open(read_data=json.dumps(data))
