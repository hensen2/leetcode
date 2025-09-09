__version__ = "0.1.0"
__author__ = "Matt Hensen"

# Import main classes for easy access
from .core.generators import (
    GraphGenerator,
    IntegerGenerator,
    LinkedListGenerator,
    MatrixGenerator,
    StringGenerator,
    TreeGenerator,
)
from .core.models import (
    Constraints,
    GraphProperties,
    TestRunResult,
    TestSuite,
    TestSuiteResult,
    TreeProperties,
)
from .error_handling.handlers import ErrorHandler
from .execution.runner import EnhancedTestRunner as TestRunner

# Main facade class for easy usage
from .facade import TestCaseGenerator
from .patterns.edge_cases import EdgeCaseGenerator

# Plugin system
from .plugins.registry import PluginRegistry

__all__ = [
    # Core generators
    "IntegerGenerator",
    "StringGenerator",
    "TreeGenerator",
    "GraphGenerator",
    "MatrixGenerator",
    "LinkedListGenerator",
    # Models
    "Constraints",
    "TreeProperties",
    "GraphProperties",
    "TestSuite",
    "TestRunResult",
    "TestSuiteResult",
    # Execution
    "TestRunner",
    # Patterns
    "EdgeCaseGenerator",
    # Error handling
    "ErrorHandler",
    # Main interface
    "TestCaseGenerator",
    # Plugins
    "PluginRegistry",
]
