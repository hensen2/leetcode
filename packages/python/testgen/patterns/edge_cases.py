"""
Edge case generator for comprehensive test coverage
Provides common edge cases for different data types and problem patterns
"""

import string
from typing import List, Optional

from ..core.config import Config
from ..core.generators import LinkedListGenerator, MatrixGenerator, TreeGenerator
from ..core.models import ListNode, TreeNode


class EdgeCaseGenerator:
    """Generate common edge cases for different problem types"""

    def __init__(self):
        self.tree_gen = TreeGenerator()
        self.list_gen = LinkedListGenerator()
        self.matrix_gen = MatrixGenerator()

    def get_array_edge_cases(self) -> List[List[int]]:
        """Generate common edge cases for array problems"""
        return [
            [],  # Empty array
            [0],  # Single zero
            [1],  # Single positive
            [-1],  # Single negative
            [Config.DEFAULT_MAX_VALUE],  # Single max value
            [Config.DEFAULT_MIN_VALUE],  # Single min value
            [1, 1, 1, 1],  # All same elements
            [1, 2, 3, 4, 5],  # Sorted ascending
            [5, 4, 3, 2, 1],  # Sorted descending
            list(range(100)),  # Large sorted ascending
            list(range(100, 0, -1)),  # Large sorted descending
            [Config.DEFAULT_MAX_VALUE, Config.DEFAULT_MIN_VALUE],  # Extreme values
            [0] * 100,  # All zeros
            [1, -1] * 50,  # Alternating pattern
            list(range(-50, 51)),  # Centered around zero
        ]

    def get_string_edge_cases(self) -> List[str]:
        """Generate common edge cases for string problems"""
        return [
            "",  # Empty string
            " ",  # Single space
            "a",  # Single character
            "A",  # Single uppercase
            "1",  # Single digit
            "!",  # Single special char
            "a" * 100,  # Repeated character
            "ab" * 50,  # Repeated pattern
            string.ascii_lowercase,  # All lowercase letters
            string.ascii_uppercase,  # All uppercase letters
            string.digits,  # All digits
            string.punctuation,  # All punctuation
            "".join([chr(i) for i in range(32, 127)]),  # All ASCII printable
            "   spaces   ",  # Leading/trailing spaces
            "MiXeD cAsE",  # Mixed case
            "hello\nworld",  # With newline
            "hello\tworld",  # With tab
        ]

    def get_matrix_edge_cases(self) -> List[List[List[int]]]:
        """Generate common edge cases for matrix problems"""
        return [
            [[]],  # Empty matrix
            [[1]],  # 1x1 matrix
            [[1, 2, 3]],  # Single row
            [[1], [2], [3]],  # Single column
            [[0] * 10 for _ in range(10)],  # All zeros
            [[1] * 10 for _ in range(10)],  # All ones
            [[i * 10 + j for j in range(10)] for i in range(10)],  # Sequential
            [[i for i in range(10)] for _ in range(10)],  # Same rows
            [[i] * 10 for i in range(10)],  # Same columns
            self.matrix_gen.generate_special(5, 5, "identity"),  # Identity matrix
            self.matrix_gen.generate_special(5, 5, "diagonal"),  # Diagonal matrix
            self.matrix_gen.generate_special(5, 5, "symmetric"),  # Symmetric matrix
        ]

    def get_tree_edge_cases(self) -> List[Optional[TreeNode]]:
        """Generate common edge cases for tree problems"""
        from packages.python.testgen.core.models import TreeProperties

        return [
            None,  # Empty tree
            TreeNode(1),  # Single node
            TreeNode(0),  # Single zero node
            TreeNode(-1),  # Single negative node
            self.tree_gen.generate(
                TreeProperties(size=10, skewed="left")
            ),  # Left skewed
            self.tree_gen.generate(
                TreeProperties(size=10, skewed="right")
            ),  # Right skewed
            self.tree_gen.generate(
                TreeProperties(size=7, complete=True)
            ),  # Complete tree
            self.tree_gen.generate(
                TreeProperties(size=7, perfect=True)
            ),  # Perfect tree
            self.tree_gen.generate(
                TreeProperties(size=15, balanced=True)
            ),  # Balanced tree
            self.tree_gen.generate(TreeProperties(size=15, bst=True)),  # BST
            self._create_two_level_tree(),  # Two level tree
            self._create_zigzag_tree(),  # Zigzag tree
        ]

    def get_linked_list_edge_cases(self) -> List[Optional[ListNode]]:
        """Generate common edge cases for linked list problems"""
        return [
            None,  # Empty list
            ListNode(1),  # Single node
            ListNode(0),  # Single zero node
            ListNode(-1),  # Single negative node
            self.list_gen.generate(2),  # Two nodes
            self.list_gen.generate(3),  # Three nodes
            self.list_gen.generate(100),  # Long list
            self.list_gen.generate(
                10, has_cycle=True, cycle_position=0
            ),  # Cycle at start
            self.list_gen.generate(
                10, has_cycle=True, cycle_position=5
            ),  # Cycle in middle
            self.list_gen.generate(
                10, has_cycle=True, cycle_position=9
            ),  # Cycle at end
            self._create_palindrome_list([1, 2, 3, 2, 1]),  # Palindrome list
        ]

    def get_graph_edge_cases(self) -> List[dict]:
        """Generate common edge cases for graph problems"""
        from packages.python.testgen.core.generators import GraphGenerator
        from packages.python.testgen.core.models import GraphProperties

        graph_gen = GraphGenerator()

        return [
            # Empty graph
            {
                "num_nodes": 0,
                "edges": [],
                "adjacency_list": {},
                "directed": False,
                "weighted": False,
            },
            # Single node
            {
                "num_nodes": 1,
                "edges": [],
                "adjacency_list": {0: []},
                "directed": False,
                "weighted": False,
            },
            # Two nodes connected
            graph_gen.generate(
                GraphProperties(num_nodes=2, num_edges=1, connected=True)
            ),
            # Two nodes disconnected
            graph_gen.generate(
                GraphProperties(num_nodes=2, num_edges=0, connected=False)
            ),
            # Complete graph (5 nodes)
            graph_gen.generate(
                GraphProperties(num_nodes=5, num_edges=10, connected=True)
            ),
            # Tree structure (10 nodes)
            graph_gen.generate(
                GraphProperties(num_nodes=10, num_edges=9, connected=True)
            ),
            # Dense graph
            graph_gen.generate(GraphProperties(num_nodes=10, num_edges=30)),
            # Sparse graph
            graph_gen.generate(
                GraphProperties(num_nodes=10, num_edges=5, connected=False)
            ),
            # Directed cycle
            self._create_directed_cycle(5),
            # Star graph
            self._create_star_graph(6),
        ]

    def get_number_edge_cases(self) -> List[int]:
        """Generate edge cases for single number problems"""
        return [
            0,  # Zero
            1,  # One
            -1,  # Negative one
            Config.DEFAULT_MAX_VALUE,  # Max value
            Config.DEFAULT_MIN_VALUE,  # Min value
            2**31 - 1,  # Max 32-bit int
            -(2**31),  # Min 32-bit int
            100,  # Round number
            -100,  # Negative round number
            999,  # Near thousand
            1000,  # Thousand
            1001,  # Just over thousand
        ]

    def get_boolean_edge_cases(self) -> List[bool]:
        """Generate edge cases for boolean problems"""
        return [True, False]

    def _create_two_level_tree(self) -> TreeNode:
        """Create a simple two-level tree"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        return root

    def _create_zigzag_tree(self) -> TreeNode:
        """Create a zigzag pattern tree"""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.left.right = TreeNode(3)
        root.left.right.left = TreeNode(4)
        root.left.right.left.right = TreeNode(5)
        return root

    def _create_palindrome_list(self, values: List[int]) -> ListNode:
        """Create a palindrome linked list"""
        if not values:
            return None

        dummy = ListNode(0)
        current = dummy
        for val in values:
            current.next = ListNode(val)
            current = current.next

        return dummy.next

    def _create_directed_cycle(self, size: int) -> dict:
        """Create a directed cycle graph"""
        edges = [(i, (i + 1) % size) for i in range(size)]
        adjacency_list = {i: [(i + 1) % size] for i in range(size)}

        return {
            "num_nodes": size,
            "edges": edges,
            "adjacency_list": adjacency_list,
            "directed": True,
            "weighted": False,
        }

    def _create_star_graph(self, size: int) -> dict:
        """Create a star graph (one central node connected to all others)"""
        if size < 2:
            return {
                "num_nodes": size,
                "edges": [],
                "adjacency_list": {i: [] for i in range(size)},
                "directed": False,
                "weighted": False,
            }

        edges = [(0, i) for i in range(1, size)]
        adjacency_list = {0: list(range(1, size))}
        for i in range(1, size):
            adjacency_list[i] = [0]

        return {
            "num_nodes": size,
            "edges": edges,
            "adjacency_list": adjacency_list,
            "directed": False,
            "weighted": False,
        }


class ProblemSpecificEdgeCases:
    """Edge cases for specific problem patterns"""

    @staticmethod
    def two_sum() -> List[tuple]:
        """Edge cases for Two Sum problem"""
        return [
            ([2, 7, 11, 15], 9),  # Basic case
            ([3, 3], 6),  # Duplicate elements
            ([1, 2], 10),  # No solution
            ([0, 4, 3, 0], 0),  # Target is zero
            ([-1, -2, -3, -4, -5], -8),  # All negative
            ([1000000, 500000, -1000000], 0),  # Large numbers
            ([1, 2], 3),  # Minimum size
        ]

    @staticmethod
    def binary_search() -> List[tuple]:
        """Edge cases for Binary Search problems"""
        return [
            ([1, 3, 5, 7, 9], 5),  # Target exists (middle)
            ([1, 3, 5, 7, 9], 1),  # Target exists (first)
            ([1, 3, 5, 7, 9], 9),  # Target exists (last)
            ([1, 3, 5, 7, 9], 0),  # Target too small
            ([1, 3, 5, 7, 9], 10),  # Target too large
            ([1, 3, 5, 7, 9], 4),  # Target doesn't exist (middle)
            ([1], 1),  # Single element (found)
            ([1], 2),  # Single element (not found)
            ([], 1),  # Empty array
            ([1, 1, 1, 1], 1),  # All duplicates
        ]

    @staticmethod
    def sliding_window() -> List[tuple]:
        """Edge cases for Sliding Window problems"""
        return [
            ([1, 2, 3, 4, 5], 2),  # Basic case
            ([1], 1),  # Single element
            ([1, 2, 3], 3),  # Window size equals array size
            ([1, 2, 3], 4),  # Window size larger than array
            ([5, 5, 5, 5], 2),  # All same elements
            ([], 1),  # Empty array
            ([-1, -2, -3], 2),  # Negative numbers
        ]

    @staticmethod
    def parentheses() -> List[str]:
        """Edge cases for Parentheses Matching problems"""
        return [
            "",  # Empty
            "()",  # Simple valid
            "()[]{}",  # Multiple types valid
            "((()))",  # Nested valid
            "(]",  # Invalid mix
            "([)]",  # Invalid interleaved
            "((((",  # Only opening
            "))))",  # Only closing
        ]
