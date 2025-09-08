"""
Example plugin implementations demonstrating the protocol system
Shows how to create custom generators, validators, comparators, and reporters
"""

import json
import random
import re
from typing import Any, Dict, List, Optional

from packages.python.testgen.plugins.base import (
    ConfigurableGeneratorProtocol,
    ConfigurableReporterProtocol,
    ConstraintValidatorProtocol,
    PluginMetadata,
    ReportFormat,
    ToleranceComparatorProtocol,
)
from packages.python.testgen.plugins.registry import (
    BaseComparatorPlugin,
    BaseGeneratorPlugin,
    BaseReporterPlugin,
    BaseValidatorPlugin,
)

# ============== Generator Plugin Examples ==============


class PrimeNumberGeneratorPlugin(BaseGeneratorPlugin):
    """Plugin for generating prime numbers"""

    def __init__(self):
        metadata = PluginMetadata(
            name="prime_generator",
            version="1.0.0",
            author="TestGen",
            description="Generates prime numbers for testing",
            type="generator",
            supported_data_types=["integer", "array"],
        )
        super().__init__(metadata)
        self._primes_cache: List[int] = []

    def generate(self, **kwargs) -> int:
        """Generate a single prime number"""
        max_value = kwargs.get("max_value", 1000)
        primes = self._generate_primes_up_to(max_value)
        return random.choice(primes) if primes else 2

    def generate_batch(self, count: int, **kwargs) -> List[int]:
        """Generate multiple prime numbers"""
        max_value = kwargs.get("max_value", 1000)
        unique = kwargs.get("unique", True)

        primes = self._generate_primes_up_to(max_value)

        if unique:
            return random.sample(primes, min(count, len(primes)))
        else:
            return [random.choice(primes) for _ in range(count)]

    def get_edge_cases(self) -> List[int]:
        """Get edge case prime numbers"""
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def get_supported_constraints(self) -> List[str]:
        """Get supported constraints"""
        return ["max_value", "min_value", "unique", "count"]

    def _generate_primes_up_to(self, n: int) -> List[int]:
        """Generate all primes up to n using Sieve of Eratosthenes"""
        if n < 2:
            return []

        # Use cache if available
        if self._primes_cache and self._primes_cache[-1] >= n:
            return [p for p in self._primes_cache if p <= n]

        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False

        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        self._primes_cache = primes
        return primes


class RegexStringGeneratorPlugin(BaseGeneratorPlugin, ConfigurableGeneratorProtocol):
    """Plugin for generating strings matching regex patterns"""

    def __init__(self):
        metadata = PluginMetadata(
            name="regex_string_generator",
            version="1.0.0",
            author="TestGen",
            description="Generates strings matching regex patterns",
            type="generator",
            supported_data_types=["string"],
        )
        super().__init__(metadata)
        self._pattern = r"[a-z]{5,10}"
        self._charset_map = {
            r"\d": "0123456789",
            r"\w": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
            r"\s": " \t\n",
            r"[a-z]": "abcdefghijklmnopqrstuvwxyz",
            r"[A-Z]": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        }

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the generator"""
        if "pattern" in config:
            self._pattern = config["pattern"]
        self._config.update(config)

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {"pattern": self._pattern, **self._config}

    def reset_configuration(self) -> None:
        """Reset to default configuration"""
        self._pattern = r"[a-z]{5,10}"
        self._config = {}

    def generate(self, **kwargs) -> str:
        """Generate string matching pattern"""
        pattern = kwargs.get("pattern", self._pattern)
        return self._generate_from_pattern(pattern)

    def get_edge_cases(self) -> List[str]:
        """Get edge cases for regex strings"""
        return [
            "",  # Empty string
            "a",  # Single character
            "ab" * 50,  # Repeated pattern
            "test123",  # Alphanumeric
            "special!@#$%",  # Special characters
        ]

    def _generate_from_pattern(self, pattern: str) -> str:
        """Generate string from simplified regex pattern"""
        # This is a simplified implementation
        # A full implementation would use a regex parser

        # Handle character classes
        for char_class, chars in self._charset_map.items():
            if char_class in pattern:
                # Handle quantifiers
                if "{" in pattern:
                    # Extract min, max from {min,max}
                    match = re.search(r"\{(\d+),(\d+)\}", pattern)
                    if match:
                        min_len = int(match.group(1))
                        max_len = int(match.group(2))
                        length = random.randint(min_len, max_len)
                    else:
                        length = 5
                else:
                    length = 5

                return "".join(random.choices(chars, k=length))

        # Default: return the pattern itself if no match
        return pattern


# ============== Validator Plugin Examples ==============


class EmailValidatorPlugin(BaseValidatorPlugin, ConstraintValidatorProtocol):
    """Plugin for validating email addresses"""

    def __init__(self):
        metadata = PluginMetadata(
            name="email_validator",
            version="1.0.0",
            author="TestGen",
            description="Validates email address format",
            type="validator",
            supported_data_types=["string"],
        )
        super().__init__(metadata)
        self._constraints = {
            "allow_special_chars": True,
            "allow_subdomains": True,
            "max_length": 254,
        }
        self._setup_rules()

    def set_constraints(self, constraints: Dict[str, Any]) -> None:
        """Set validation constraints"""
        self._constraints.update(constraints)
        self._setup_rules()

    def get_constraints(self) -> Dict[str, Any]:
        """Get current constraints"""
        return self._constraints.copy()

    def validate_with_constraints(
        self, test_case: Any, constraints: Dict[str, Any]
    ) -> bool:
        """Validate with specific constraints"""
        old_constraints = self._constraints.copy()
        self._constraints.update(constraints)
        self._setup_rules()

        result = self.validate(test_case)

        self._constraints = old_constraints
        self._setup_rules()

        return result

    def _setup_rules(self) -> None:
        """Setup validation rules based on constraints"""
        self._rules = {
            "is_string": lambda x: isinstance(x, str),
            "not_empty": lambda x: len(x) > 0,
            "max_length": lambda x: len(x) <= self._constraints["max_length"],
            "has_at_symbol": lambda x: "@" in x,
            "has_domain": lambda x: "." in x.split("@")[-1] if "@" in x else False,
            "valid_format": self._is_valid_email_format,
        }

    def _is_valid_email_format(self, email: str) -> bool:
        """Check if email has valid format"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))


class RangeValidatorPlugin(BaseValidatorPlugin):
    """Plugin for validating numeric ranges"""

    def __init__(self):
        metadata = PluginMetadata(
            name="range_validator",
            version="1.0.0",
            author="TestGen",
            description="Validates if numbers are within specified range",
            type="validator",
            supported_data_types=["integer", "float", "array"],
        )
        super().__init__(metadata)
        self._min_value = float("-inf")
        self._max_value = float("inf")
        self._setup_rules()

    def configure_range(self, min_value: float, max_value: float) -> None:
        """Configure the valid range"""
        self._min_value = min_value
        self._max_value = max_value
        self._setup_rules()

    def _setup_rules(self) -> None:
        """Setup validation rules"""
        self._rules = {
            "is_numeric": lambda x: isinstance(x, (int, float))
            or (isinstance(x, list) and all(isinstance(i, (int, float)) for i in x)),
            "in_range": self._check_range,
        }

    def _check_range(self, value: Any) -> bool:
        """Check if value(s) are in range"""
        if isinstance(value, (int, float)):
            return self._min_value <= value <= self._max_value
        elif isinstance(value, list):
            return all(self._min_value <= v <= self._max_value for v in value)
        return False


# ============== Comparator Plugin Examples ==============


class FuzzyStringComparatorPlugin(BaseComparatorPlugin, ToleranceComparatorProtocol):
    """Plugin for fuzzy string comparison"""

    def __init__(self):
        metadata = PluginMetadata(
            name="fuzzy_string_comparator",
            version="1.0.0",
            author="TestGen",
            description="Compares strings with fuzzy matching",
            type="comparator",
            supported_data_types=["string"],
        )
        super().__init__(metadata)
        self._tolerance = 0.8  # Similarity threshold (0-1)
        self._setup_modes()

    @property
    def tolerance(self) -> float:
        """Get current tolerance"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set comparison tolerance"""
        self._tolerance = max(0.0, min(1.0, value))

    def _setup_modes(self) -> None:
        """Setup comparison modes"""
        self._modes = {
            "exact": lambda a, b: a == b,
            "case_insensitive": lambda a, b: a.lower() == b.lower(),
            "fuzzy": self._fuzzy_compare,
            "prefix": lambda a, b: a.startswith(b) or b.startswith(a),
            "contains": lambda a, b: a in b or b in a,
        }

    def _fuzzy_compare(self, expected: str, actual: str) -> bool:
        """Fuzzy string comparison using Levenshtein distance"""
        if not isinstance(expected, str) or not isinstance(actual, str):
            return False

        # Calculate similarity ratio
        similarity = self._calculate_similarity(expected, actual)
        return similarity >= self._tolerance

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings (0-1)"""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Simplified similarity: common characters / max length
        common = sum(1 for c in s1 if c in s2)
        return common / max(len1, len2)

    def get_difference(self, expected: Any, actual: Any) -> Optional[str]:
        """Get detailed difference description"""
        if self.compare(expected, actual):
            return None

        if isinstance(expected, str) and isinstance(actual, str):
            similarity = self._calculate_similarity(expected, actual)
            return (
                f"Strings differ (similarity: {similarity:.2%})\n"
                f"Expected: '{expected}'\n"
                f"Actual:   '{actual}'"
            )

        return super().get_difference(expected, actual)


class SetComparatorPlugin(BaseComparatorPlugin):
    """Plugin for comparing collections as sets"""

    def __init__(self):
        metadata = PluginMetadata(
            name="set_comparator",
            version="1.0.0",
            author="TestGen",
            description="Compares collections ignoring order and duplicates",
            type="comparator",
            supported_data_types=["array", "list", "set"],
        )
        super().__init__(metadata)
        self._setup_modes()

    def _setup_modes(self) -> None:
        """Setup comparison modes"""
        self._modes = {
            "default": self._set_compare,
            "subset": self._subset_compare,
            "superset": self._superset_compare,
            "intersection": self._intersection_compare,
        }

    def _set_compare(self, expected: Any, actual: Any) -> bool:
        """Compare as sets (exact equality)"""
        try:
            return set(expected) == set(actual)
        except TypeError:
            return False

    def _subset_compare(self, expected: Any, actual: Any) -> bool:
        """Check if actual is subset of expected"""
        try:
            return set(actual).issubset(set(expected))
        except TypeError:
            return False

    def _superset_compare(self, expected: Any, actual: Any) -> bool:
        """Check if actual is superset of expected"""
        try:
            return set(actual).issuperset(set(expected))
        except TypeError:
            return False

    def _intersection_compare(self, expected: Any, actual: Any) -> bool:
        """Check if there's any intersection"""
        try:
            return len(set(expected).intersection(set(actual))) > 0
        except TypeError:
            return False


# ============== Reporter Plugin Examples ==============


class MarkdownReporterPlugin(BaseReporterPlugin, ConfigurableReporterProtocol):
    """Plugin for generating Markdown reports"""

    def __init__(self):
        metadata = PluginMetadata(
            name="markdown_reporter",
            version="1.0.0",
            author="TestGen",
            description="Generates test reports in Markdown format",
            type="reporter",
            supported_data_types=["any"],
        )
        super().__init__(metadata)
        self._include_graphs = False
        self._include_details = True

    def configure(self, **options) -> None:
        """Configure reporter options"""
        self._include_graphs = options.get("include_graphs", False)
        self._include_details = options.get("include_details", True)
        if "verbosity" in options:
            self.set_verbosity(options["verbosity"])

    def get_supported_formats(self) -> List[ReportFormat]:
        """Get supported formats"""
        return [ReportFormat.MARKDOWN]

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as Markdown"""
        md_lines = []

        # Title
        md_lines.append("# Test Results Report")
        md_lines.append("")

        # Summary
        md_lines.append("## Summary")
        md_lines.append("")

        total = results.get("total", 0)
        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        pass_rate = (passed / total * 100) if total > 0 else 0

        md_lines.append(f"- **Total Tests**: {total}")
        md_lines.append(f"- **Passed**: {passed} ✅")
        md_lines.append(f"- **Failed**: {failed} ❌")
        md_lines.append(f"- **Pass Rate**: {pass_rate:.1f}%")
        md_lines.append("")

        # Details (if verbosity > 0)
        if self._verbosity > 0 and self._include_details:
            md_lines.append("## Test Details")
            md_lines.append("")

            test_results = results.get("results", [])
            if test_results:
                md_lines.append("| Test # | Status | Time (s) | Error |")
                md_lines.append("|--------|--------|----------|-------|")

                for i, test in enumerate(test_results[:20]):  # Limit to 20
                    status = "✅ Pass" if test.get("passed") else "❌ Fail"
                    time = test.get("execution_time", 0)
                    error = test.get("error", "-")[:50]  # Truncate errors

                    md_lines.append(f"| {i + 1} | {status} | {time:.4f} | {error} |")

        return "\n".join(md_lines)


class JSONReporterPlugin(BaseReporterPlugin):
    """Plugin for generating JSON reports"""

    def __init__(self):
        metadata = PluginMetadata(
            name="json_reporter",
            version="1.0.0",
            author="TestGen",
            description="Generates test reports in JSON format",
            type="reporter",
            supported_data_types=["any"],
        )
        super().__init__(metadata)
        self._indent = 2

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as JSON"""
        # Clean up non-serializable objects
        clean_results = self._clean_for_json(results)
        return json.dumps(clean_results, indent=self._indent)

    def _clean_for_json(self, obj: Any) -> Any:
        """Clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
