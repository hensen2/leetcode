"""
Data models and types for test case generation
Defines constraints, node types, and data structures
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

# Type aliases for clarity
TestCase: TypeAlias = Tuple[Any, Any]
TestSuite: TypeAlias = List[TestCase]
TestResult: TypeAlias = Dict[str, Any]
GraphData: TypeAlias = Dict[str, Any]


class DataType(Enum):
    """Supported data types for test generation"""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    MATRIX = "matrix"
    TREE = "tree"
    GRAPH = "graph"
    LINKED_LIST = "linked_list"


@dataclass
class Constraints:
    """Constraints for test case generation"""

    min_value: int = -(10**9)
    max_value: int = 10**9
    min_length: int = 0
    max_length: int = 10**4
    allow_duplicates: bool = True
    allow_negative: bool = True
    allow_zero: bool = True
    allow_empty: bool = True
    charset: str = field(default_factory=lambda: "abcdefghijklmnopqrstuvwxyz")
    is_sorted: bool = False
    is_unique: bool = False
    custom_validator: Optional[Callable] = None

    def validate(self) -> None:
        """Validate constraint consistency"""
        if self.min_value > self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) > max_value ({self.max_value})"
            )
        if self.min_length > self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) > max_length ({self.max_length})"
            )
        if self.min_length < 0:
            raise ValueError(f"min_length ({self.min_length}) cannot be negative")
        if not self.allow_negative and self.min_value < 0:
            self.min_value = 0
        if not self.allow_zero and self.min_value <= 0 <= self.max_value:
            if self.min_value == 0:
                self.min_value = 1

    def copy(self) -> "Constraints":
        """Create a copy of constraints"""
        import copy

        return copy.copy(self)


class TreeNode:
    """Binary tree node"""

    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return (
            self.val == other.val
            and self.left == other.left
            and self.right == other.right
        )

    def __hash__(self):
        return hash(self.val)


class ListNode:
    """Linked list node"""

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"

    def __eq__(self, other):
        if not isinstance(other, ListNode):
            return False
        return self.val == other.val

    def __hash__(self):
        return hash(self.val)


@dataclass
class GraphProperties:
    """Properties for graph generation"""

    num_nodes: int
    num_edges: Optional[int] = None
    directed: bool = False
    weighted: bool = False
    connected: bool = True
    allow_cycles: bool = True
    allow_self_loops: bool = False
    min_weight: int = 1
    max_weight: int = 100

    def validate(self) -> None:
        """Validate graph properties"""
        if self.num_nodes < 0:
            raise ValueError(f"Number of nodes cannot be negative: {self.num_nodes}")

        # Calculate edge limits
        max_edges = self.num_nodes * (self.num_nodes - 1)
        if not self.directed:
            max_edges //= 2
        if self.allow_self_loops:
            max_edges += self.num_nodes

        if self.connected and self.num_nodes > 1:
            min_edges = self.num_nodes - 1
        else:
            min_edges = 0

        if self.num_edges is not None:
            if self.num_edges < min_edges:
                raise ValueError(
                    f"Too few edges ({self.num_edges}) for connected graph "
                    f"with {self.num_nodes} nodes (minimum: {min_edges})"
                )
            if self.num_edges > max_edges:
                raise ValueError(
                    f"Too many edges ({self.num_edges}) for graph "
                    f"with {self.num_nodes} nodes (maximum: {max_edges})"
                )


@dataclass
class TreeProperties:
    """Properties for tree generation"""

    size: int
    balanced: bool = False
    complete: bool = False
    perfect: bool = False
    bst: bool = False
    skewed: Optional[str] = None  # 'left' or 'right'
    min_val: int = -(10**9)
    max_val: int = 10**9

    def validate(self) -> None:
        """Validate tree properties"""
        if self.size < 0:
            raise ValueError(f"Tree size cannot be negative: {self.size}")

        # Check for conflicting properties
        conflicting = sum(
            [self.balanced, self.complete, self.perfect, self.skewed is not None]
        )
        if conflicting > 1:
            raise ValueError("Conflicting tree properties specified")

        if self.perfect and self.size > 0:
            # Perfect tree must have 2^h - 1 nodes
            import math

            h = math.log2(self.size + 1)
            if not h.is_integer():
                raise ValueError(
                    f"Perfect tree must have 2^h - 1 nodes, got {self.size}"
                )


@dataclass
class TestRunResult:
    """Result from running a test case"""

    input: Any
    expected: Optional[Any]
    actual: Optional[Any]
    passed: bool
    error: Optional[str]
    execution_time: Optional[float]
    memory_usage: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "input": self.input,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "error": self.error,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
        }


@dataclass
class TestSuiteResult:
    """Summary of test suite execution"""

    total: int
    passed: int
    failed: int
    errors: int
    timeout: int
    results: List[TestRunResult]
    total_time: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate"""
        return self.passed / self.total * 100 if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "timeout": self.timeout,
            "pass_rate": self.pass_rate,
            "total_time": self.total_time,
            "results": [r.to_dict() for r in self.results],
        }
