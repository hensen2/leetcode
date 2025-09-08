"""
Protocol definitions and interfaces for extensible test case generation
Defines contracts for generators, validators, comparators, and reporters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# Type variables for generic protocols
T = TypeVar("T")
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


# ============== Generator Protocols ==============


@runtime_checkable
class GeneratorProtocol(Protocol[T]):
    """Protocol for all test case generators"""

    def generate(self, **kwargs) -> T:
        """Generate a single test case"""
        ...

    def generate_batch(self, count: int, **kwargs) -> List[T]:
        """Generate multiple test cases"""
        ...

    def get_edge_cases(self) -> List[T]:
        """Get edge cases for this data type"""
        ...

    def validate_constraints(self, constraints: Dict[str, Any]) -> bool:
        """Validate if constraints are valid for this generator"""
        ...


@runtime_checkable
class ConfigurableGeneratorProtocol(GeneratorProtocol[T], Protocol):
    """Protocol for generators that support configuration"""

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the generator with settings"""
        ...

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        ...

    def reset_configuration(self) -> None:
        """Reset to default configuration"""
        ...


@runtime_checkable
class SeededGeneratorProtocol(GeneratorProtocol[T], Protocol):
    """Protocol for generators with reproducible output"""

    @property
    def seed(self) -> Optional[int]:
        """Get current seed"""
        ...

    @seed.setter
    def seed(self, value: Optional[int]) -> None:
        """Set seed for reproducibility"""
        ...

    def reset_seed(self) -> None:
        """Reset seed to random"""
        ...


# ============== Validator Protocols ==============


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for test case validators"""

    def validate(self, test_case: Any) -> bool:
        """Validate if test case meets requirements"""
        ...

    def get_validation_errors(self, test_case: Any) -> List[str]:
        """Get detailed validation errors"""
        ...

    def validate_batch(self, test_cases: List[Any]) -> List[bool]:
        """Validate multiple test cases"""
        ...


@runtime_checkable
class ConstraintValidatorProtocol(ValidatorProtocol, Protocol):
    """Protocol for constraint-based validators"""

    def set_constraints(self, constraints: Dict[str, Any]) -> None:
        """Set validation constraints"""
        ...

    def get_constraints(self) -> Dict[str, Any]:
        """Get current constraints"""
        ...

    def validate_with_constraints(
        self, test_case: Any, constraints: Dict[str, Any]
    ) -> bool:
        """Validate with specific constraints"""
        ...


# ============== Comparator Protocols ==============


@runtime_checkable
class ComparatorProtocol(Protocol):
    """Protocol for output comparison"""

    def compare(self, expected: Any, actual: Any) -> bool:
        """Compare two values for equality"""
        ...

    def get_difference(self, expected: Any, actual: Any) -> Optional[str]:
        """Get human-readable difference description"""
        ...


@runtime_checkable
class ToleranceComparatorProtocol(ComparatorProtocol, Protocol):
    """Protocol for comparators with tolerance/threshold"""

    @property
    def tolerance(self) -> float:
        """Get current tolerance"""
        ...

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set comparison tolerance"""
        ...


@runtime_checkable
class CustomComparatorProtocol(ComparatorProtocol, Protocol):
    """Protocol for custom comparison logic"""

    def set_comparison_function(self, func: Callable[[Any, Any], bool]) -> None:
        """Set custom comparison function"""
        ...

    def add_preprocessor(self, func: Callable[[Any], Any]) -> None:
        """Add preprocessing before comparison"""
        ...


# ============== Reporter Protocols ==============


class ReportFormat(Enum):
    """Supported report formats"""

    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    XML = "xml"


@runtime_checkable
class ReporterProtocol(Protocol):
    """Protocol for test result reporters"""

    def report(self, results: Dict[str, Any]) -> None:
        """Generate and output report"""
        ...

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as string"""
        ...

    def save_report(self, results: Dict[str, Any], filepath: str) -> None:
        """Save report to file"""
        ...


@runtime_checkable
class ConfigurableReporterProtocol(ReporterProtocol, Protocol):
    """Protocol for configurable reporters"""

    def set_format(self, format: ReportFormat) -> None:
        """Set report format"""
        ...

    def set_verbosity(self, level: int) -> None:
        """Set verbosity level (0=minimal, 1=normal, 2=detailed)"""
        ...

    def configure(self, **options) -> None:
        """Configure reporter options"""
        ...


@runtime_checkable
class StreamingReporterProtocol(ReporterProtocol, Protocol):
    """Protocol for streaming/progressive reporters"""

    def start_report(self, total_tests: int) -> None:
        """Initialize streaming report"""
        ...

    def update_progress(self, current: int, passed: int, failed: int) -> None:
        """Update progress during testing"""
        ...

    def finalize_report(self) -> None:
        """Finalize and close report"""
        ...


# ============== Serializer Protocols ==============


@runtime_checkable
class SerializerProtocol(Protocol[T]):
    """Protocol for data serialization"""

    def serialize(self, data: T) -> Any:
        """Serialize data to transferable format"""
        ...

    def deserialize(self, serialized: Any) -> T:
        """Deserialize back to original type"""
        ...

    def can_serialize(self, data: Any) -> bool:
        """Check if data can be serialized"""
        ...


@runtime_checkable
class BidirectionalSerializerProtocol(SerializerProtocol[T], Protocol):
    """Protocol for bidirectional serialization with validation"""

    def validate_serialized(self, serialized: Any) -> bool:
        """Validate serialized data"""
        ...

    def get_serialization_metadata(self, data: T) -> Dict[str, Any]:
        """Get metadata about serialization"""
        ...


# ============== Analyzer Protocols ==============


@runtime_checkable
class AnalyzerProtocol(Protocol):
    """Protocol for performance/complexity analyzers"""

    def analyze(self, func: Callable, test_data: List[Any]) -> Dict[str, Any]:
        """Analyze function with test data"""
        ...

    def estimate_complexity(self, measurements: Dict[int, float]) -> str:
        """Estimate algorithmic complexity"""
        ...


@runtime_checkable
class MemoryAnalyzerProtocol(AnalyzerProtocol, Protocol):
    """Protocol for memory usage analysis"""

    def measure_memory(self, func: Callable, test_data: Any) -> Dict[str, Any]:
        """Measure memory usage"""
        ...

    def detect_memory_leaks(self, func: Callable, test_data: List[Any]) -> bool:
        """Detect potential memory leaks"""
        ...


# ============== Plugin Protocols ==============


@dataclass
class PluginMetadata:
    """Metadata for plugins"""

    name: str
    version: str
    author: str
    description: str
    type: str  # 'generator', 'validator', 'comparator', 'reporter'
    supported_data_types: List[str]
    dependencies: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "type": self.type,
            "supported_data_types": self.supported_data_types,
            "dependencies": self.dependencies or [],
        }


@runtime_checkable
class PluginProtocol(Protocol):
    """Base protocol for all plugins"""

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin with configuration"""
        ...

    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        ...

    def validate_environment(self) -> bool:
        """Check if plugin can run in current environment"""
        ...


@runtime_checkable
class GeneratorPluginProtocol(PluginProtocol, GeneratorProtocol[T], Protocol):
    """Protocol for generator plugins"""

    def get_supported_constraints(self) -> List[str]:
        """Get list of supported constraint names"""
        ...

    def get_default_constraints(self) -> Dict[str, Any]:
        """Get default constraint values"""
        ...


@runtime_checkable
class ValidatorPluginProtocol(PluginProtocol, ValidatorProtocol, Protocol):
    """Protocol for validator plugins"""

    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules"""
        ...

    def add_custom_rule(self, name: str, rule: Callable[[Any], bool]) -> None:
        """Add custom validation rule"""
        ...


@runtime_checkable
class ComparatorPluginProtocol(PluginProtocol, ComparatorProtocol, Protocol):
    """Protocol for comparator plugins"""

    def get_comparison_modes(self) -> List[str]:
        """Get available comparison modes"""
        ...

    def set_mode(self, mode: str) -> None:
        """Set comparison mode"""
        ...


@runtime_checkable
class ReporterPluginProtocol(PluginProtocol, ReporterProtocol, Protocol):
    """Protocol for reporter plugins"""

    def get_supported_formats(self) -> List[ReportFormat]:
        """Get supported output formats"""
        ...

    def add_custom_section(self, name: str, content: Any) -> None:
        """Add custom section to report"""
        ...


# ============== Factory Protocols ==============


@runtime_checkable
class FactoryProtocol(Protocol[T]):
    """Protocol for factory classes"""

    def create(self, type_name: str, **kwargs) -> T:
        """Create instance of specified type"""
        ...

    def register(self, type_name: str, creator: Callable[..., T]) -> None:
        """Register new type creator"""
        ...

    def list_available(self) -> List[str]:
        """List available types"""
        ...


# ============== Registry Protocol ==============


@runtime_checkable
class PluginRegistryProtocol(Protocol):
    """Protocol for plugin registry"""

    def register_plugin(self, plugin: PluginProtocol) -> None:
        """Register a plugin"""
        ...

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin"""
        ...

    def get_plugin(self, plugin_name: str) -> Optional[PluginProtocol]:
        """Get plugin by name"""
        ...

    def list_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """List registered plugins"""
        ...

    def load_plugin_from_file(self, filepath: str) -> None:
        """Load plugin from file"""
        ...

    def load_plugins_from_directory(self, directory: str) -> None:
        """Load all plugins from directory"""
        ...


# ============== Composite Protocols ==============


@runtime_checkable
class TestSuiteProtocol(Protocol):
    """Protocol for test suite management"""

    def add_generator(self, name: str, generator: GeneratorProtocol) -> None:
        """Add generator to suite"""
        ...

    def add_validator(self, name: str, validator: ValidatorProtocol) -> None:
        """Add validator to suite"""
        ...

    def add_comparator(self, name: str, comparator: ComparatorProtocol) -> None:
        """Add comparator to suite"""
        ...

    def add_reporter(self, name: str, reporter: ReporterProtocol) -> None:
        """Add reporter to suite"""
        ...

    def run(self, test_function: Callable, **options) -> Dict[str, Any]:
        """Run test suite"""
        ...


# ============== Extension Points ==============


class ExtensionPoint(ABC):
    """Abstract base class for extension points"""

    @abstractmethod
    def extend(self, extension: Any) -> None:
        """Add extension"""
        pass

    @abstractmethod
    def get_extensions(self) -> List[Any]:
        """Get all extensions"""
        pass


class GeneratorExtensionPoint(ExtensionPoint):
    """Extension point for generators"""

    def __init__(self):
        self._extensions: List[GeneratorProtocol] = []

    def extend(self, extension: GeneratorProtocol) -> None:
        """Add generator extension"""
        if isinstance(extension, GeneratorProtocol):
            self._extensions.append(extension)

    def get_extensions(self) -> List[GeneratorProtocol]:
        """Get all generator extensions"""
        return self._extensions.copy()


# ============== Type Guards ==============


def is_generator(obj: Any) -> bool:
    """Check if object implements GeneratorProtocol"""
    return isinstance(obj, GeneratorProtocol)


def is_validator(obj: Any) -> bool:
    """Check if object implements ValidatorProtocol"""
    return isinstance(obj, ValidatorProtocol)


def is_comparator(obj: Any) -> bool:
    """Check if object implements ComparatorProtocol"""
    return isinstance(obj, ComparatorProtocol)


def is_reporter(obj: Any) -> bool:
    """Check if object implements ReporterProtocol"""
    return isinstance(obj, ReporterProtocol)


def is_plugin(obj: Any) -> bool:
    """Check if object implements PluginProtocol"""
    return isinstance(obj, PluginProtocol)
