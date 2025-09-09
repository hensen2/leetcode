"""
Simplified plugin registry and management

Focuses on essential plugin functionality without over-engineering.
Removed complex dependency resolution, dynamic loading, and enterprise features.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    ComparatorProtocol,
    GeneratorProtocol,
    PluginProtocol,
    ReporterProtocol,
    ValidatorProtocol,
)


class PluginRegistry:
    """Simple plugin registry - no over-engineering"""

    def __init__(self):
        # Simple dictionary storage
        self._plugins: Dict[str, PluginProtocol] = {}
        self._plugins_by_type: Dict[str, List[str]] = {
            "generator": [],
            "validator": [],
            "reporter": [],
            "comparator": [],
        }

    def register(self, plugin: PluginProtocol) -> None:
        """Register a plugin"""
        name = plugin.name
        plugin_type = plugin.plugin_type

        # Simple validation
        if not name:
            raise ValueError("Plugin name cannot be empty")

        if plugin_type not in self._plugins_by_type:
            raise ValueError(f"Unknown plugin type: {plugin_type}")

        # Register
        self._plugins[name] = plugin
        if name not in self._plugins_by_type[plugin_type]:
            self._plugins_by_type[plugin_type].append(name)

        # Initialize if needed
        try:
            plugin.initialize()
        except Exception as e:
            # Remove from registry if initialization fails
            self._plugins.pop(name, None)
            if name in self._plugins_by_type[plugin_type]:
                self._plugins_by_type[plugin_type].remove(name)
            raise RuntimeError(f"Failed to initialize plugin '{name}': {e}")

    def unregister(self, name: str) -> None:
        """Unregister a plugin by name"""
        plugin = self._plugins.get(name)
        if plugin:
            # Cleanup
            try:
                plugin.cleanup()
            except Exception:
                pass  # Don't fail on cleanup errors

            # Remove from registry
            plugin_type = plugin.plugin_type
            self._plugins.pop(name)
            if name in self._plugins_by_type[plugin_type]:
                self._plugins_by_type[plugin_type].remove(name)

    def get(self, name: str) -> Optional[PluginProtocol]:
        """Get plugin by name"""
        return self._plugins.get(name)

    def list_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """List registered plugins, optionally filtered by type"""
        if plugin_type:
            return self._plugins_by_type.get(plugin_type, []).copy()
        return list(self._plugins.keys())

    def get_by_type(self, plugin_type: str) -> List[PluginProtocol]:
        """Get all plugins of a specific type"""
        names = self._plugins_by_type.get(plugin_type, [])
        return [self._plugins[name] for name in names if name in self._plugins]

    def clear(self) -> None:
        """Clear all plugins (with cleanup)"""
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception:
                pass  # Don't fail on cleanup

        self._plugins.clear()
        for plugin_list in self._plugins_by_type.values():
            plugin_list.clear()

    def load_from_file(self, filepath: str) -> None:
        """Load plugins from Python file - simplified version"""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {filepath}")

        if path.suffix != ".py":
            raise ValueError(f"Plugin file must be Python (.py): {filepath}")

        # Dynamic import
        module_name = f"plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)

        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin from {filepath}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find plugin classes (simple detection)
        registered_count = 0
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # Look for classes that implement PluginProtocol
            if (
                isinstance(attr, type)
                and hasattr(attr, "__mro__")
                and any(
                    hasattr(base, "__annotations__") and "PluginProtocol" in str(base)
                    for base in attr.__mro__
                )
            ):
                try:
                    # Try to instantiate and register
                    plugin_instance = attr()
                    if isinstance(plugin_instance, PluginProtocol):
                        self.register(plugin_instance)
                        registered_count += 1
                except Exception:
                    continue  # Skip problematic plugins

        if registered_count == 0:
            raise ImportError(f"No valid plugins found in {filepath}")


class PluginFactory:
    """Simple factory for creating plugin instances"""

    def __init__(self, registry: PluginRegistry):
        self.registry = registry

    def create_generator(self, name: str, **kwargs) -> Optional[GeneratorProtocol]:
        """Create generator plugin instance"""
        plugin = self.registry.get(name)
        if plugin and plugin.plugin_type == "generator":
            return plugin
        return None

    def create_validator(self, name: str, **kwargs) -> Optional[ValidatorProtocol]:
        """Create validator plugin instance"""
        plugin = self.registry.get(name)
        if plugin and plugin.plugin_type == "validator":
            return plugin
        return None

    def create_reporter(self, name: str, **kwargs) -> Optional[ReporterProtocol]:
        """Create reporter plugin instance"""
        plugin = self.registry.get(name)
        if plugin and plugin.plugin_type == "reporter":
            return plugin
        return None

    def create_comparator(self, name: str, **kwargs) -> Optional[ComparatorProtocol]:
        """Create comparator plugin instance"""
        plugin = self.registry.get(name)
        if plugin and plugin.plugin_type == "comparator":
            return plugin
        return None

    def list_available(self, plugin_type: str) -> List[str]:
        """List available plugins of type"""
        return self.registry.list_plugins(plugin_type)


# ============== Global Registry Instance ==============

# Single global instance for simplicity
_global_registry = PluginRegistry()
_global_factory = PluginFactory(_global_registry)


def get_registry() -> PluginRegistry:
    """Get the global plugin registry"""
    return _global_registry


def get_factory() -> PluginFactory:
    """Get the global plugin factory"""
    return _global_factory


def register_plugin(plugin: PluginProtocol) -> None:
    """Convenience function to register a plugin globally"""
    _global_registry.register(plugin)


def get_plugin(name: str) -> Optional[PluginProtocol]:
    """Convenience function to get a plugin globally"""
    return _global_registry.get(name)


def list_plugins(plugin_type: Optional[str] = None) -> List[str]:
    """Convenience function to list plugins globally"""
    return _global_registry.list_plugins(plugin_type)
