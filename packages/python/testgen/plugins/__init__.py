"""
Simplified plugin system for LeetCode test generator

Provides essential plugin functionality without over-engineering:
- Core protocols for generators, validators, reporters, comparators
- Simple plugin registry and factory
- Clean public interface

Usage:
    from testgen.plugins import GeneratorProtocol, register_plugin, get_plugin

    class MyGenerator(GeneratorProtocol):
        def generate(self, constraints):
            return "test_data"

    register_plugin(MyGenerator())
    generator = get_plugin("my_generator")
"""

# Core protocols - essential interfaces
from .base import (  # Optional base classes
    BaseGenerator,
    BasePlugin,
    BaseValidator,
    ComparatorProtocol,
    GeneratorProtocol,
    PluginProtocol,
    ReporterProtocol,
    ReportFormat,
    ValidatorProtocol,
)

# Registry and factory
from .registry import (  # Global convenience functions
    PluginFactory,
    PluginRegistry,
    get_factory,
    get_plugin,
    get_registry,
    list_plugins,
    register_plugin,
)

# Public API - what external code should use
__all__ = [
    # Essential protocols
    "GeneratorProtocol",
    "ValidatorProtocol",
    "ReporterProtocol",
    "ComparatorProtocol",
    "PluginProtocol",
    "ReportFormat",
    # Optional base classes
    "BaseGenerator",
    "BaseValidator",
    "BasePlugin",
    # Plugin management
    "PluginRegistry",
    "PluginFactory",
    # Global convenience functions (recommended interface)
    "register_plugin",
    "get_plugin",
    "list_plugins",
    "get_registry",
    "get_factory",
]
