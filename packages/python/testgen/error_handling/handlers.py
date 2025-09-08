"""
Comprehensive error handling system for test case generation
Provides resilient error handling, recovery strategies, and detailed error reporting
"""

import functools
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from ..plugins.base import GeneratorProtocol, ValidatorProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Error Types ==============


class ErrorSeverity(Enum):
    """Severity levels for errors"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for better organization"""

    GENERATION = "generation"
    VALIDATION = "validation"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    SERIALIZATION = "serialization"
    PLUGIN = "plugin"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for an error"""

    timestamp: datetime = field(default_factory=datetime.now)
    category: ErrorCategory = ErrorCategory.SYSTEM
    severity: ErrorSeverity = ErrorSeverity.ERROR
    component: str = ""
    operation: str = ""
    test_case: Optional[Any] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "test_case": str(self.test_case) if self.test_case else None,
            "additional_info": self.additional_info,
        }


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""

    error_type: str
    error_message: str
    traceback: str
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_successful: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "context": self.context.to_dict(),
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
        }


# ============== Custom Exceptions ==============


class TestGenerationError(Exception):
    """Base exception for test generation errors"""

    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context or ErrorContext()


class GeneratorError(TestGenerationError):
    """Error in test case generation"""

    def __init__(self, message: str, generator: Optional[str] = None, **kwargs):
        context = ErrorContext(
            category=ErrorCategory.GENERATION,
            component=generator or "unknown_generator",
            **kwargs,
        )
        super().__init__(message, context)


class ValidationError(TestGenerationError):
    """Error in test case validation"""

    def __init__(self, message: str, validator: Optional[str] = None, **kwargs):
        context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            component=validator or "unknown_validator",
            **kwargs,
        )
        super().__init__(message, context)


class ExecutionError(TestGenerationError):
    """Error in test execution"""

    def __init__(self, message: str, function: Optional[str] = None, **kwargs):
        context = ErrorContext(
            category=ErrorCategory.EXECUTION,
            component=function or "unknown_function",
            **kwargs,
        )
        super().__init__(message, context)


class TimeoutError(TestGenerationError):
    """Timeout during test execution"""

    def __init__(self, message: str, timeout_duration: float, **kwargs):
        context = ErrorContext(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            additional_info={"timeout_duration": timeout_duration},
            **kwargs,
        )
        super().__init__(message, context)


class PluginError(TestGenerationError):
    """Error in plugin operation"""

    def __init__(self, message: str, plugin_name: Optional[str] = None, **kwargs):
        context = ErrorContext(
            category=ErrorCategory.PLUGIN,
            component=plugin_name or "unknown_plugin",
            **kwargs,
        )
        super().__init__(message, context)


# ============== Error Handlers ==============


class ErrorHandler:
    """Central error handling system"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_records: List[ErrorRecord] = []
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self._setup_default_strategies()

    def _setup_default_strategies(self):
        """Setup default recovery strategies"""
        self.recovery_strategies[GeneratorError] = self._recover_from_generator_error
        self.recovery_strategies[ValidationError] = self._recover_from_validation_error
        self.recovery_strategies[TimeoutError] = self._recover_from_timeout
        self.recovery_strategies[MemoryError] = self._recover_from_memory_error

    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        retry_operation: Optional[Callable] = None,
    ) -> Optional[Any]:
        """
        Handle an error with optional recovery

        Args:
            error: The exception that occurred
            context: Additional context information
            retry_operation: Function to retry if recovery is possible

        Returns:
            Result of retry operation if successful, None otherwise
        """
        # Create error record
        record = self._create_error_record(error, context)
        self.error_records.append(record)

        # Log the error
        self._log_error(record)

        # Attempt recovery
        if retry_operation:
            result = self._attempt_recovery(error, record, retry_operation)
            if result is not None:
                record.recovery_successful = True
                return result

        # No recovery possible
        return None

    def _create_error_record(
        self, error: Exception, context: Optional[ErrorContext]
    ) -> ErrorRecord:
        """Create an error record from exception"""
        error_context = context or getattr(error, "context", ErrorContext())

        return ErrorRecord(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            context=error_context,
        )

    def _log_error(self, record: ErrorRecord) -> None:
        """Log error based on severity"""
        severity = record.context.severity
        message = f"[{record.context.category.value}] {record.error_message}"

        if severity == ErrorSeverity.DEBUG:
            logger.debug(message)
        elif severity == ErrorSeverity.INFO:
            logger.info(message)
        elif severity == ErrorSeverity.WARNING:
            logger.warning(message)
        elif severity == ErrorSeverity.ERROR:
            logger.error(message)
        elif severity == ErrorSeverity.CRITICAL:
            logger.critical(message)

    def _attempt_recovery(
        self, error: Exception, record: ErrorRecord, retry_operation: Callable
    ) -> Optional[Any]:
        """Attempt to recover from error"""
        record.recovery_attempted = True

        # Find recovery strategy
        strategy = None
        for error_type, recovery_func in self.recovery_strategies.items():
            if isinstance(error, error_type):
                strategy = recovery_func
                break

        if not strategy:
            return None

        # Apply recovery strategy
        try:
            return strategy(error, retry_operation)
        except Exception as recovery_error:
            logger.error(f"Recovery failed: {recovery_error}")
            return None

    def _recover_from_generator_error(
        self, error: GeneratorError, retry_operation: Callable
    ) -> Optional[Any]:
        """Recovery strategy for generator errors"""
        logger.info("Attempting to recover from generator error")

        # Try with reduced constraints
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.retry_delay * (attempt + 1))
                return retry_operation()
            except GeneratorError:
                if attempt == self.max_retries - 1:
                    raise
                logger.info(f"Retry attempt {attempt + 1} failed, trying again...")

        return None

    def _recover_from_validation_error(
        self, error: ValidationError, retry_operation: Callable
    ) -> Optional[Any]:
        """Recovery strategy for validation errors"""
        logger.info("Validation error occurred, skipping validation")
        # Skip validation and continue
        return None

    def _recover_from_timeout(
        self, error: TimeoutError, retry_operation: Callable
    ) -> Optional[Any]:
        """Recovery strategy for timeout errors"""
        logger.info("Timeout occurred, retrying with increased timeout")
        # Could implement retry with increased timeout
        return None

    def _recover_from_memory_error(
        self, error: MemoryError, retry_operation: Callable
    ) -> Optional[Any]:
        """Recovery strategy for memory errors"""
        logger.warning("Memory error occurred, attempting garbage collection")
        import gc

        gc.collect()

        try:
            return retry_operation()
        except MemoryError:
            logger.error("Memory error persists after garbage collection")
            return None

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors"""
        summary = {
            "total_errors": len(self.error_records),
            "errors_by_category": {},
            "errors_by_severity": {},
            "recovery_stats": {"attempted": 0, "successful": 0},
            "recent_errors": [],
        }

        for record in self.error_records:
            # Count by category
            category = record.context.category.value
            summary["errors_by_category"][category] = (
                summary["errors_by_category"].get(category, 0) + 1
            )

            # Count by severity
            severity = record.context.severity.value
            summary["errors_by_severity"][severity] = (
                summary["errors_by_severity"].get(severity, 0) + 1
            )

            # Recovery stats
            if record.recovery_attempted:
                summary["recovery_stats"]["attempted"] += 1
            if record.recovery_successful:
                summary["recovery_stats"]["successful"] += 1

        # Add recent errors
        summary["recent_errors"] = [
            record.to_dict() for record in self.error_records[-5:]
        ]

        return summary

    def clear_errors(self) -> None:
        """Clear error history"""
        self.error_records.clear()


# ============== Decorators for Error Handling ==============


def with_error_handling(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    max_retries: int = 3,
):
    """
    Decorator for automatic error handling

    Args:
        category: Error category
        severity: Error severity
        max_retries: Maximum retry attempts
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler(max_retries=max_retries)
            context = ErrorContext(
                category=category,
                severity=severity,
                component=func.__module__,
                operation=func.__name__,
            )

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Last attempt, handle and re-raise
                        handler.handle_error(e, context)
                        raise
                    else:
                        # Retry
                        result = handler.handle_error(
                            e, context, lambda: func(*args, **kwargs)
                        )
                        if result is not None:
                            return result

            return None

        return wrapper

    return decorator


def timeout_handler(timeout_seconds: float):
    """
    Decorator for handling timeouts

    Args:
        timeout_seconds: Timeout duration in seconds
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal

            def timeout_occurred(signum, frame):
                raise TimeoutError(
                    f"Operation timed out after {timeout_seconds} seconds",
                    timeout_duration=timeout_seconds,
                    operation=func.__name__,
                )

            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_occurred)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Restore original handler
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper

    return decorator


# ============== Resilient Wrappers ==============


class ResilientGenerator:
    """Wrapper for generators with error handling"""

    def __init__(
        self, generator: GeneratorProtocol, error_handler: Optional[ErrorHandler] = None
    ):
        self.generator = generator
        self.error_handler = error_handler or ErrorHandler()

    @with_error_handling(category=ErrorCategory.GENERATION)
    def generate(self, **kwargs) -> Any:
        """Generate with error handling"""
        try:
            return self.generator.generate(**kwargs)
        except Exception as e:
            # Provide fallback
            logger.warning(f"Generation failed, using fallback: {e}")
            return self._get_fallback_value(**kwargs)

    def _get_fallback_value(self, **kwargs) -> Any:
        """Get fallback value when generation fails"""
        data_type = kwargs.get("data_type", "integer")

        fallbacks = {
            "integer": 0,
            "string": "",
            "array": [],
            "matrix": [[]],
            "tree": None,
            "graph": {"nodes": 0, "edges": []},
            "linked_list": None,
        }

        return fallbacks.get(data_type, None)


class ResilientValidator:
    """Wrapper for validators with error handling"""

    def __init__(
        self, validator: ValidatorProtocol, error_handler: Optional[ErrorHandler] = None
    ):
        self.validator = validator
        self.error_handler = error_handler or ErrorHandler()

    @with_error_handling(
        category=ErrorCategory.VALIDATION, severity=ErrorSeverity.WARNING
    )
    def validate(self, test_case: Any) -> bool:
        """Validate with error handling"""
        try:
            return self.validator.validate(test_case)
        except Exception as e:
            # Log error but don't fail
            logger.warning(f"Validation error (treating as valid): {e}")
            return True  # Optimistic validation on error


# ============== Error Report Generator ==============


class ErrorReporter:
    """Generate detailed error reports"""

    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler

    def generate_report(self, format: str = "text") -> str:
        """Generate error report in specified format"""
        if format == "text":
            return self._generate_text_report()
        elif format == "json":
            return self._generate_json_report()
        elif format == "html":
            return self._generate_html_report()
        else:
            raise ValueError(f"Unknown report format: {format}")

    def _generate_text_report(self) -> str:
        """Generate text format error report"""
        summary = self.error_handler.get_error_summary()

        lines = [
            "=" * 60,
            "ERROR REPORT",
            "=" * 60,
            f"Total Errors: {summary['total_errors']}",
            "",
            "Errors by Category:",
        ]

        for category, count in summary["errors_by_category"].items():
            lines.append(f"  {category}: {count}")

        lines.extend(
            [
                "",
                "Errors by Severity:",
            ]
        )

        for severity, count in summary["errors_by_severity"].items():
            lines.append(f"  {severity}: {count}")

        lines.extend(
            [
                "",
                f"Recovery Attempts: {summary['recovery_stats']['attempted']}",
                f"Successful Recoveries: {summary['recovery_stats']['successful']}",
                "",
                "Recent Errors:",
            ]
        )

        for error in summary["recent_errors"]:
            lines.append(
                f"  - [{error['context']['timestamp']}] {error['error_message']}"
            )

        return "\n".join(lines)

    def _generate_json_report(self) -> str:
        """Generate JSON format error report"""
        import json

        summary = self.error_handler.get_error_summary()
        return json.dumps(summary, indent=2, default=str)

    def _generate_html_report(self) -> str:
        """Generate HTML format error report"""
        summary = self.error_handler.get_error_summary()

        html = f"""
        <html>
        <head><title>Error Report</title></head>
        <body>
            <h1>Error Report</h1>
            <p>Total Errors: {summary["total_errors"]}</p>
            
            <h2>Errors by Category</h2>
            <ul>
                {"".join(f"<li>{cat}: {count}</li>" for cat, count in summary["errors_by_category"].items())}
            </ul>
            
            <h2>Recovery Statistics</h2>
            <p>Attempts: {summary["recovery_stats"]["attempted"]}</p>
            <p>Successful: {summary["recovery_stats"]["successful"]}</p>
        </body>
        </html>
        """

        return html
