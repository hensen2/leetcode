# tests/conftest.py
"""
Shared pytest fixtures and configuration for testgen testing

Provides common test data, mock objects, and setup/teardown for all tests.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from testgen.core.generators import IntegerGenerator, StringGenerator, TreeGenerator

# Import testgen components for fixtures
from testgen.core.models import Constraints, GraphProperties, TreeProperties
from testgen.error_handling.handlers import ErrorHandler
from testgen.execution.runner import EnhancedTestRunner as TestRunner
from testgen.plugins import PluginRegistry, get_registry

# ============== Basic Fixtures ==============


@pytest.fixture
def sample_constraints():
    """Basic constraints for testing"""
    return Constraints(min_value=1, max_value=100, is_unique=False, is_sorted=False)


@pytest.fixture
def strict_constraints():
    """Constraints for testing edge cases"""
    return Constraints(min_value=-1000, max_value=1000, is_unique=True, is_sorted=True)


@pytest.fixture
def tree_properties():
    """Tree properties for testing"""
    return TreeProperties(
        min_nodes=1,
        max_nodes=15,
        is_binary=True,
        is_balanced=False,
        allow_duplicates=True,
    )


@pytest.fixture
def graph_properties():
    """Graph properties for testing"""
    return GraphProperties(
        min_nodes=3,
        max_nodes=10,
        min_edges=2,
        max_edges=20,
        is_directed=False,
        is_connected=True,
        allow_self_loops=False,
        allow_multi_edges=False,
    )


# ============== Generator Fixtures ==============


@pytest.fixture
def integer_generator():
    """Integer generator instance"""
    return IntegerGenerator()


@pytest.fixture
def string_generator():
    """String generator instance"""
    return StringGenerator()


@pytest.fixture
def tree_generator():
    """Tree generator instance"""
    return TreeGenerator()


@pytest.fixture
def test_runner():
    """Test runner instance"""
    return TestRunner()


@pytest.fixture
def error_handler():
    """Error handler instance"""
    return ErrorHandler()


# ============== Memory Testing Fixtures ==============


@pytest.fixture
def large_constraints():
    """Constraints for memory testing - large ranges"""
    return Constraints(
        min_value=-1_000_000, max_value=1_000_000, is_unique=True, is_sorted=False
    )


@pytest.fixture
def memory_stress_constraints():
    """Constraints for stress testing memory efficiency"""
    return Constraints(
        min_value=-10_000_000, max_value=10_000_000, is_unique=True, is_sorted=False
    )


# ============== File System Fixtures ==============


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_output_file(temp_dir):
    """Temporary output file for testing"""
    return temp_dir / "test_output.json"


# ============== Mock Fixtures ==============


@pytest.fixture
def mock_function():
    """Mock function for testing execution"""

    def test_func(arr):
        """Simple test function that returns sum"""
        return sum(arr)

    return test_func


@pytest.fixture
def slow_function():
    """Mock slow function for timeout testing"""
    import time

    def slow_func(arr):
        time.sleep(0.1)  # Simulate slow operation
        return len(arr)

    return slow_func


@pytest.fixture
def failing_function():
    """Mock function that always fails"""

    def fail_func(arr):
        raise ValueError("Intentional test failure")

    return fail_func


# ============== Plugin Testing Fixtures ==============


@pytest.fixture
def clean_plugin_registry():
    """Clean plugin registry for testing"""
    registry = PluginRegistry()
    yield registry
    registry.clear()  # Cleanup after test


@pytest.fixture
def mock_plugin():
    """Mock plugin for testing"""
    plugin = Mock()
    plugin.name = "test_plugin"
    plugin.version = "1.0.0"
    plugin.plugin_type = "generator"
    plugin.initialize.return_value = None
    plugin.cleanup.return_value = None
    return plugin


# ============== Test Data Fixtures ==============


@pytest.fixture
def sample_arrays():
    """Sample arrays for testing"""
    return {
        "empty": [],
        "single": [42],
        "small": [1, 2, 3],
        "medium": list(range(1, 101)),
        "duplicates": [1, 2, 2, 3, 3, 3],
        "negatives": [-5, -2, 0, 3, 7],
        "large_numbers": [999999, 1000000, 1000001],
    }


@pytest.fixture
def sample_strings():
    """Sample strings for testing"""
    return {
        "empty": "",
        "single_char": "a",
        "lowercase": "hello",
        "uppercase": "WORLD",
        "mixed_case": "Hello World",
        "numbers": "12345",
        "special_chars": "!@#$%",
        "unicode": "café naïve résumé",
    }


@pytest.fixture
def sample_tree_data():
    """Sample tree data for testing"""
    return {
        "empty": None,
        "single": {"val": 1, "left": None, "right": None},
        "small_balanced": {
            "val": 2,
            "left": {"val": 1, "left": None, "right": None},
            "right": {"val": 3, "left": None, "right": None},
        },
        "unbalanced": {
            "val": 1,
            "left": None,
            "right": {
                "val": 2,
                "left": None,
                "right": {"val": 3, "left": None, "right": None},
            },
        },
    }


# ============== Performance Testing Fixtures ==============


@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests"""
    import os
    import time

    import psutil

    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process(os.getpid())

        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss

        def stop(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            return {
                "duration": end_time - self.start_time,
                "memory_delta": end_memory - self.start_memory,
                "peak_memory": self.process.memory_info().rss,
            }

    return PerformanceTracker()


# ============== Test Configuration ==============


def pytest_configure(config):
    """Pytest configuration hook"""
    # Add custom markers
    config.addinivalue_line("markers", "memory: mark test as memory-intensive")
    config.addinivalue_line("markers", "slow: mark test as slow-running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection - add markers automatically"""
    for item in items:
        # Mark memory tests
        if "memory" in item.nodeid or "large" in item.nodeid:
            item.add_marker(pytest.mark.memory)

        # Mark slow tests
        if "stress" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before each test"""
    # Clear global plugin registry
    get_registry().clear()
    yield
    # Cleanup after test
    get_registry().clear()
