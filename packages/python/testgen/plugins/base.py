"""
Essential protocols for the LeetCode test generator

This module contains only the core 3-4 protocols actually needed for basic functionality.
Removed over-engineered protocols that added complexity without clear value.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


# ============== Core Protocols (ESSENTIAL) ==============


@runtime_checkable
class GeneratorProtocol(Protocol[T]):
    """Essential protocol for test data generators"""

    def generate(self, constraints: Dict[str, Any]) -> T:
        """Generate test data based on constraints"""
        ...

    def generate_batch(self, constraints: Dict[str, Any], size: int) -> List[T]:
        """Generate batch of test data"""
        ...


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Essential protocol for test data validation"""

    def validate(self, test_case: Any) -> bool:
        """Validate a test case"""
        ...

    def get_validation_errors(self, test_case: Any) -> List[str]:
        """Get detailed validation errors for debugging"""
        ...


class ReportFormat(Enum):
    """Supported report formats - keep it simple"""

    TEXT = "text"
    JSON = "json"
    HTML = "html"


@runtime_checkable
class ReporterProtocol(Protocol):
    """Essential protocol for test result reporting"""

    def report(
        self, results: Dict[str, Any], format: ReportFormat = ReportFormat.TEXT
    ) -> str:
        """Generate report from results"""
        ...

    def save_report(
        self,
        results: Dict[str, Any],
        filepath: str,
        format: ReportFormat = ReportFormat.TEXT,
    ) -> None:
        """Save report to file"""
        ...


@runtime_checkable
class ComparatorProtocol(Protocol):
    """Essential protocol for output comparison"""

    def compare(self, expected: Any, actual: Any) -> bool:
        """Compare expected vs actual output"""
        ...

    def get_difference(self, expected: Any, actual: Any) -> Optional[str]:
        """Get human-readable difference description"""
        ...


# ============== Plugin System (SIMPLIFIED) ==============


@runtime_checkable
class PluginProtocol(Protocol):
    """Base protocol for all plugins - simplified version"""

    @property
    def name(self) -> str:
        """Plugin name"""
        ...

    @property
    def version(self) -> str:
        """Plugin version"""
        ...

    @property
    def plugin_type(self) -> str:
        """Plugin type: 'generator', 'validator', 'reporter', 'comparator'"""
        ...

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin with optional configuration"""
        ...

    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        ...


# ============== Base Implementation Classes (OPTIONAL) ==============


class BaseGenerator(ABC, GeneratorProtocol[T]):
    """Optional base class for generators - provides common functionality"""

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def generate(self, constraints: Dict[str, Any]) -> T:
        """Implement in subclass"""
        pass

    def generate_batch(self, constraints: Dict[str, Any], size: int) -> List[T]:
        """Default batch implementation"""
        return [self.generate(constraints) for _ in range(size)]


class BaseValidator(ABC, ValidatorProtocol):
    """Optional base class for validators"""

    def __init__(self, name: str = "base"):
        self.name = name

    @abstractmethod
    def validate(self, test_case: Any) -> bool:
        """Implement in subclass"""
        pass

    def get_validation_errors(self, test_case: Any) -> List[str]:
        """Default implementation - override for detailed errors"""
        if self.validate(test_case):
            return []
        return [f"Validation failed for {type(test_case).__name__}"]


class BasePlugin(ABC):
    """Optional base class for plugins"""

    def __init__(self, name: str, version: str = "1.0.0", plugin_type: str = "unknown"):
        self._name = name
        self._version = version
        self._plugin_type = plugin_type
        self._initialized = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def plugin_type(self) -> str:
        return self._plugin_type

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Base initialization - override in subclass"""
        if not self._initialized:
            self._initialized = True

    def cleanup(self) -> None:
        """Base cleanup - override in subclass"""
        if self._initialized:
            self._initialized = False
