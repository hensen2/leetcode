"""
Plugin system for extensible test case generation
Provides plugin loading, registration, and management
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from packages.python.testgen.plugins.base import (
    ComparatorPluginProtocol,
    FactoryProtocol,
    GeneratorPluginProtocol,
    PluginMetadata,
    PluginProtocol,
    PluginRegistryProtocol,
    ReporterPluginProtocol,
    ValidatorPluginProtocol,
)


class PluginRegistry(PluginRegistryProtocol):
    """Central registry for managing plugins"""

    def __init__(self):
        self._plugins: Dict[str, PluginProtocol] = {}
        self._plugin_types: Dict[str, List[str]] = {
            "generator": [],
            "validator": [],
            "comparator": [],
            "reporter": [],
        }

    def register_plugin(self, plugin: PluginProtocol) -> None:
        """
        Register a plugin

        Args:
            plugin: Plugin instance to register
        """
        if not isinstance(plugin, PluginProtocol):
            raise TypeError(f"Plugin must implement PluginProtocol, got {type(plugin)}")

        metadata = plugin.metadata
        plugin_name = metadata.name

        if plugin_name in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' is already registered")

        # Validate plugin environment
        if not plugin.validate_environment():
            raise RuntimeError(f"Plugin '{plugin_name}' failed environment validation")

        # Initialize plugin
        plugin.initialize()

        # Register plugin
        self._plugins[plugin_name] = plugin

        # Categorize by type
        plugin_type = metadata.type
        if plugin_type in self._plugin_types:
            self._plugin_types[plugin_type].append(plugin_name)

    def unregister_plugin(self, plugin_name: str) -> None:
        """
        Unregister a plugin

        Args:
            plugin_name: Name of plugin to unregister
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' is not registered")

        plugin = self._plugins[plugin_name]

        # Cleanup plugin
        plugin.cleanup()

        # Remove from registry
        del self._plugins[plugin_name]

        # Remove from type categories
        for plugin_list in self._plugin_types.values():
            if plugin_name in plugin_list:
                plugin_list.remove(plugin_name)

    def get_plugin(self, plugin_name: str) -> Optional[PluginProtocol]:
        """
        Get plugin by name

        Args:
            plugin_name: Name of plugin

        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(plugin_name)

    def list_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """
        List registered plugins

        Args:
            plugin_type: Optional filter by type

        Returns:
            List of plugin names
        """
        if plugin_type:
            return self._plugin_types.get(plugin_type, []).copy()
        return list(self._plugins.keys())

    def load_plugin_from_file(self, filepath: str) -> None:
        """
        Load plugin from Python file

        Args:
            filepath: Path to plugin file
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {filepath}")

        if not path.suffix == ".py":
            raise ValueError(f"Plugin file must be a Python file (.py): {filepath}")

        # Load module
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, filepath)

        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load plugin from {filepath}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find and register plugin classes
        self._register_plugins_from_module(module)

    def load_plugins_from_directory(self, directory: str) -> None:
        """
        Load all plugins from directory

        Args:
            directory: Path to plugins directory
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Plugin directory not found: {directory}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Load all Python files
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # Skip private files

            try:
                self.load_plugin_from_file(str(py_file))
            except Exception as e:
                print(f"Warning: Failed to load plugin from {py_file}: {e}")

    def _register_plugins_from_module(self, module: Any) -> None:
        """Register all plugin classes found in module"""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, PluginProtocol):
                # Skip abstract classes
                if inspect.isabstract(obj):
                    continue

                # Create instance and register
                try:
                    plugin_instance = obj()
                    self.register_plugin(plugin_instance)
                except Exception as e:
                    print(f"Warning: Failed to instantiate plugin {name}: {e}")


class PluginFactory(FactoryProtocol[PluginProtocol]):
    """Factory for creating plugin instances"""

    def __init__(self, registry: Optional[PluginRegistry] = None):
        self._registry = registry or PluginRegistry()
        self._creators: Dict[str, Type[PluginProtocol]] = {}

    def create(self, type_name: str, **kwargs) -> PluginProtocol:
        """
        Create plugin instance

        Args:
            type_name: Type of plugin to create
            **kwargs: Arguments for plugin initialization

        Returns:
            Plugin instance
        """
        if type_name not in self._creators:
            raise ValueError(f"Unknown plugin type: {type_name}")

        plugin_class = self._creators[type_name]
        plugin = plugin_class(**kwargs)

        # Auto-register if registry is available
        if self._registry:
            self._registry.register_plugin(plugin)

        return plugin

    def register(self, type_name: str, creator: Type[PluginProtocol]) -> None:
        """
        Register plugin creator

        Args:
            type_name: Name for this plugin type
            creator: Plugin class
        """
        if not issubclass(creator, PluginProtocol):
            raise TypeError("Creator must be a PluginProtocol subclass")

        self._creators[type_name] = creator

    def list_available(self) -> List[str]:
        """List available plugin types"""
        return list(self._creators.keys())


class PluginManager:
    """High-level plugin management interface"""

    def __init__(self):
        self.registry = PluginRegistry()
        self.factory = PluginFactory(self.registry)
        self._plugin_paths: List[Path] = []

    def add_plugin_path(self, path: Union[str, Path]) -> None:
        """
        Add path to search for plugins

        Args:
            path: Directory path to add
        """
        path = Path(path)
        if path.exists() and path.is_dir():
            self._plugin_paths.append(path)
            sys.path.insert(0, str(path))

    def discover_plugins(self) -> None:
        """Discover and load plugins from all configured paths"""
        for path in self._plugin_paths:
            try:
                self.registry.load_plugins_from_directory(str(path))
            except Exception as e:
                print(f"Error loading plugins from {path}: {e}")

    def get_generator(self, name: str) -> Optional[GeneratorPluginProtocol]:
        """Get generator plugin by name"""
        plugin = self.registry.get_plugin(name)
        if plugin and isinstance(plugin, GeneratorPluginProtocol):
            return plugin
        return None

    def get_validator(self, name: str) -> Optional[ValidatorPluginProtocol]:
        """Get validator plugin by name"""
        plugin = self.registry.get_plugin(name)
        if plugin and isinstance(plugin, ValidatorPluginProtocol):
            return plugin
        return None

    def get_comparator(self, name: str) -> Optional[ComparatorPluginProtocol]:
        """Get comparator plugin by name"""
        plugin = self.registry.get_plugin(name)
        if plugin and isinstance(plugin, ComparatorPluginProtocol):
            return plugin
        return None

    def get_reporter(self, name: str) -> Optional[ReporterPluginProtocol]:
        """Get reporter plugin by name"""
        plugin = self.registry.get_plugin(name)
        if plugin and isinstance(plugin, ReporterPluginProtocol):
            return plugin
        return None

    def list_generators(self) -> List[str]:
        """List available generator plugins"""
        return self.registry.list_plugins("generator")

    def list_validators(self) -> List[str]:
        """List available validator plugins"""
        return self.registry.list_plugins("validator")

    def list_comparators(self) -> List[str]:
        """List available comparator plugins"""
        return self.registry.list_plugins("comparator")

    def list_reporters(self) -> List[str]:
        """List available reporter plugins"""
        return self.registry.list_plugins("reporter")

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin"""
        plugin = self.registry.get_plugin(name)
        if plugin:
            metadata = plugin.metadata
            return {
                "name": metadata.name,
                "version": metadata.version,
                "author": metadata.author,
                "description": metadata.description,
                "type": metadata.type,
                "supported_data_types": metadata.supported_data_types,
                "dependencies": metadata.dependencies or [],
            }
        return None


# ============== Plugin Base Classes ==============


class BasePlugin(PluginProtocol):
    """Base class for plugins with common functionality"""

    def __init__(self, metadata: PluginMetadata):
        self._metadata = metadata
        self._config: Dict[str, Any] = {}
        self._initialized = False

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return self._metadata

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin with configuration"""
        if self._initialized:
            return

        if config:
            self._config.update(config)

        self._initialize_impl()
        self._initialized = True

    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        if not self._initialized:
            return

        self._cleanup_impl()
        self._initialized = False

    def validate_environment(self) -> bool:
        """Check if plugin can run in current environment"""
        # Check dependencies
        if self._metadata.dependencies:
            for dep in self._metadata.dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    return False

        # Custom validation
        return self._validate_environment_impl()

    # Methods for subclasses to override

    def _initialize_impl(self) -> None:
        """Implementation-specific initialization"""
        pass

    def _cleanup_impl(self) -> None:
        """Implementation-specific cleanup"""
        pass

    def _validate_environment_impl(self) -> bool:
        """Implementation-specific environment validation"""
        return True


class BaseGeneratorPlugin(BasePlugin, GeneratorPluginProtocol[Any]):
    """Base class for generator plugins"""

    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._constraints: Dict[str, Any] = {}

    def generate(self, **kwargs) -> Any:
        """Generate a single test case"""
        raise NotImplementedError

    def generate_batch(self, count: int, **kwargs) -> List[Any]:
        """Generate multiple test cases"""
        return [self.generate(**kwargs) for _ in range(count)]

    def get_edge_cases(self) -> List[Any]:
        """Get edge cases for this data type"""
        return []

    def validate_constraints(self, constraints: Dict[str, Any]) -> bool:
        """Validate if constraints are valid for this generator"""
        supported = self.get_supported_constraints()
        for key in constraints:
            if key not in supported:
                return False
        return True

    def get_supported_constraints(self) -> List[str]:
        """Get list of supported constraint names"""
        return []

    def get_default_constraints(self) -> Dict[str, Any]:
        """Get default constraint values"""
        return {}


class BaseValidatorPlugin(BasePlugin, ValidatorPluginProtocol):
    """Base class for validator plugins"""

    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._rules: Dict[str, Callable[[Any], bool]] = {}

    def validate(self, test_case: Any) -> bool:
        """Validate if test case meets requirements"""
        errors = self.get_validation_errors(test_case)
        return len(errors) == 0

    def get_validation_errors(self, test_case: Any) -> List[str]:
        """Get detailed validation errors"""
        errors = []
        for rule_name, rule_func in self._rules.items():
            try:
                if not rule_func(test_case):
                    errors.append(f"Failed rule: {rule_name}")
            except Exception as e:
                errors.append(f"Error in rule {rule_name}: {e}")
        return errors

    def validate_batch(self, test_cases: List[Any]) -> List[bool]:
        """Validate multiple test cases"""
        return [self.validate(tc) for tc in test_cases]

    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        return list(self._rules.keys())

    def add_custom_rule(self, name: str, rule: Callable[[Any], bool]) -> None:
        """Add custom validation rule"""
        self._rules[name] = rule


class BaseComparatorPlugin(BasePlugin, ComparatorPluginProtocol):
    """Base class for comparator plugins"""

    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._mode = "default"
        self._modes: Dict[str, Callable[[Any, Any], bool]] = {
            "default": self._default_compare
        }

    def compare(self, expected: Any, actual: Any) -> bool:
        """Compare two values for equality"""
        return self._modes[self._mode](expected, actual)

    def get_difference(self, expected: Any, actual: Any) -> Optional[str]:
        """Get human-readable difference description"""
        if self.compare(expected, actual):
            return None
        return f"Expected: {expected}, Actual: {actual}"

    def get_comparison_modes(self) -> List[str]:
        """Get available comparison modes"""
        return list(self._modes.keys())

    def set_mode(self, mode: str) -> None:
        """Set comparison mode"""
        if mode not in self._modes:
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode

    def _default_compare(self, expected: Any, actual: Any) -> bool:
        """Default comparison implementation"""
        return expected == actual


class BaseReporterPlugin(BasePlugin, ReporterPluginProtocol):
    """Base class for reporter plugins"""

    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._format = "text"
        self._verbosity = 1

    def report(self, results: Dict[str, Any]) -> None:
        """Generate and output report"""
        formatted = self.format_results(results)
        print(formatted)

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as string"""
        raise NotImplementedError

    def save_report(self, results: Dict[str, Any], filepath: str) -> None:
        """Save report to file"""
        formatted = self.format_results(results)
        with open(filepath, "w") as f:
            f.write(formatted)

    def set_format(self, format: str) -> None:
        """Set report format"""
        self._format = format

    def set_verbosity(self, level: int) -> None:
        """Set verbosity level"""
        self._verbosity = max(0, min(2, level))
