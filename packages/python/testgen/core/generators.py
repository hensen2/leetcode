"""
Core generator classes for different data types
Focuses on pure data generation without serialization concerns
"""

import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from .config import Config
from .models import (
    Constraints,
    GraphData,
    GraphProperties,
    ListNode,
    TreeNode,
    TreeProperties,
)


class BaseGenerator:
    """Base class for all generators"""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility"""
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = None


class IntegerGenerator(BaseGenerator):
    """Generator for integer values and arrays"""

    def generate(self, constraints: Optional[Constraints] = None) -> int:
        """Generate a single integer based on constraints"""
        if constraints is None:
            constraints = Constraints()
        constraints.validate()

        return random.randint(constraints.min_value, constraints.max_value)

    def generate_array(
        self, size: Optional[int] = None, constraints: Optional[Constraints] = None
    ) -> List[int]:
        """Generate an array of integers"""
        if constraints is None:
            constraints = Constraints()
        constraints.validate()

        # Determine size
        if size is None:
            size = random.randint(constraints.min_length, constraints.max_length)

        # Handle empty array
        if size == 0:
            return []

        if constraints.is_unique:
            values = self._generate_unique_values(size, constraints)
        else:
            values = [self.generate(constraints) for _ in range(size)]

        # Sort if required
        if constraints.is_sorted:
            values.sort()

        return values

    def _generate_unique_values(self, size: int, constraints: Constraints) -> List[int]:
        """
        Memory-efficient unique value generation
        REPLACES the problematic range(min_value, max_value + 1) approach
        """
        value_range = constraints.max_value - constraints.min_value + 1

        if value_range < size:
            raise ValueError(
                f"Cannot generate {size} unique values in range "
                f"[{constraints.min_value}, {constraints.max_value}]"
            )

        # Strategy 1: Small ranges - use random.sample (existing approach works)
        if value_range <= 100_000:  # Reasonable memory limit
            values = random.sample(
                range(constraints.min_value, constraints.max_value + 1), size
            )
            return values

        # Strategy 2: Large ranges, small sample - use rejection sampling
        if size <= value_range * 0.1:  # Less than 10% of range
            return self._rejection_sampling(size, constraints)

        # Strategy 3: Large ranges, large sample - use shuffle algorithm
        return self._reservoir_sampling(size, constraints)

    def _rejection_sampling(self, size: int, constraints: Constraints) -> List[int]:
        """
        Rejection sampling for small samples from large ranges
        Efficient when size << range_size
        """
        values: Set[int] = set()
        max_attempts = size * 10  # Prevent infinite loops
        attempts = 0

        while len(values) < size and attempts < max_attempts:
            candidate = random.randint(constraints.min_value, constraints.max_value)
            values.add(candidate)
            attempts += 1

        if len(values) < size:
            # Fallback: fill remaining with guaranteed unique values
            remaining = size - len(values)
            # Find unused values systematically
            used_values = values
            for candidate in range(constraints.min_value, constraints.max_value + 1):
                if candidate not in used_values:
                    values.add(candidate)
                    remaining -= 1
                    if remaining == 0:
                        break

        return list(values)

    def _reservoir_sampling(self, size: int, constraints: Constraints) -> List[int]:
        """
        Modified reservoir sampling for large samples
        Memory-efficient for large ranges and large samples
        """
        # Generate a random permutation of indices
        # This is more complex but avoids creating the full range

        # For very large ranges, use a hybrid approach:
        # 1. Divide range into chunks
        # 2. Sample from each chunk proportionally

        range_size = constraints.max_value - constraints.min_value + 1
        chunk_size = min(50_000, range_size // 10)  # Reasonable chunk size
        num_chunks = (range_size + chunk_size - 1) // chunk_size

        values: Set[int] = set()
        per_chunk = size // num_chunks
        remainder = size % num_chunks

        for i in range(num_chunks):
            chunk_start = constraints.min_value + (i * chunk_size)
            chunk_end = min(chunk_start + chunk_size - 1, constraints.max_value)

            # Sample from this chunk
            chunk_sample_size = per_chunk + (1 if i < remainder else 0)
            chunk_range_size = chunk_end - chunk_start + 1

            if chunk_sample_size >= chunk_range_size:
                # Take all values from this chunk
                chunk_values = list(range(chunk_start, chunk_end + 1))
            else:
                # Sample from chunk
                chunk_values = random.sample(
                    range(chunk_start, chunk_end + 1), chunk_sample_size
                )

            values.update(chunk_values)

            if len(values) >= size:
                break

        # Ensure we have exactly the right number
        result = list(values)
        if len(result) > size:
            result = random.sample(result, size)
        elif len(result) < size:
            # Fill remaining with rejection sampling
            remaining = size - len(result)
            additional = self._rejection_sampling(remaining, constraints)
            # Remove any duplicates
            for val in additional:
                if val not in values:
                    result.append(val)
                    if len(result) >= size:
                        break

        return result[:size]


class StringGenerator(BaseGenerator):
    """Generator for strings and text data"""

    def generate(
        self, length: Optional[int] = None, constraints: Optional[Constraints] = None
    ) -> str:
        """Generate a random string"""
        if constraints is None:
            constraints = Constraints()

        if (
            length is None
            or length < constraints.min_length
            or length > constraints.max_length
        ):
            length = random.randint(
                max(0, constraints.min_length), constraints.max_length
            )

        if length == 0:
            return ""

        return "".join(random.choices(constraints.charset, k=length))

    def generate_palindrome(self, length: int, charset: Optional[str] = None) -> str:
        """Generate a palindrome string"""
        if charset is None:
            charset = Config.DEFAULT_CHARSET

        if length == 0:
            return ""

        half_length = length // 2
        first_half = "".join(random.choices(charset, k=half_length))

        if length % 2 == 0:
            return first_half + first_half[::-1]
        else:
            middle = random.choice(charset)
            return first_half + middle + first_half[::-1]

    def generate_with_pattern(self, pattern: str, charset: Optional[str] = None) -> str:
        """Generate string matching a pattern (e.g., 'a*b*c')"""
        if charset is None:
            charset = Config.DEFAULT_CHARSET

        result = []
        for char in pattern:
            if char == "*":
                # Add 0-5 random characters
                count = random.randint(0, 5)
                result.extend(random.choices(charset, k=count))
            else:
                result.append(char)

        return "".join(result)


class MatrixGenerator(BaseGenerator):
    """Generator for 2D matrices"""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.int_gen = IntegerGenerator(seed)

    def generate(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        constraints: Optional[Constraints] = None,
    ) -> List[List[int]]:
        """Generate a 2D matrix"""
        if constraints is None:
            constraints = Constraints()

        if rows is None:
            rows = random.randint(
                1, min(Config.DEFAULT_MATRIX_MAX_SIZE, constraints.max_length)
            )
        if cols is None:
            cols = random.randint(
                1, min(Config.DEFAULT_MATRIX_MAX_SIZE, constraints.max_length)
            )

        if rows == 0 or cols == 0:
            return [[]]

        matrix = []
        for _ in range(rows):
            row = self.int_gen.generate_array(cols, constraints)
            matrix.append(row)

        return matrix

    def generate_special(
        self, rows: int, cols: int, matrix_type: str = "random"
    ) -> List[List[int]]:
        """Generate special types of matrices"""
        if matrix_type == "identity":
            matrix = [[1 if i == j else 0 for j in range(cols)] for i in range(rows)]
        elif matrix_type == "diagonal":
            matrix = [
                [random.randint(1, 10) if i == j else 0 for j in range(cols)]
                for i in range(rows)
            ]
        elif matrix_type == "upper_triangular":
            matrix = [
                [random.randint(1, 10) if j >= i else 0 for j in range(cols)]
                for i in range(rows)
            ]
        elif matrix_type == "lower_triangular":
            matrix = [
                [random.randint(1, 10) if j <= i else 0 for j in range(cols)]
                for i in range(rows)
            ]
        elif matrix_type == "symmetric":
            # Generate upper triangle and mirror
            matrix = [[0] * cols for _ in range(rows)]
            for i in range(min(rows, cols)):
                for j in range(i, min(rows, cols)):
                    val = random.randint(1, 10)
                    matrix[i][j] = val
                    matrix[j][i] = val
        elif matrix_type == "zero":
            matrix = [[0 for j in range(cols)] for i in range(rows)]
        else:  # random
            matrix = [[random.randint(1, 10) for _ in range(cols)] for _ in range(rows)]

        return matrix


class TreeGenerator(BaseGenerator):
    """Generator for binary trees"""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.int_gen = IntegerGenerator(seed)

    def generate(
        self,
        properties: Optional[TreeProperties] = None,
        constraints: Optional[Constraints] = None,
    ) -> Optional[TreeNode]:
        """Generate a binary tree based on properties"""
        if properties is None:
            properties = TreeProperties(size=10)
        properties.validate()

        if properties.size <= 0:
            return None

        if constraints is None:
            constraints = Constraints(
                min_value=properties.min_val, max_value=properties.max_val
            )

        values = self.int_gen.generate_array(properties.size, constraints)

        # Generate based on tree type
        if properties.bst:
            return self._build_bst(values)
        elif properties.balanced:
            values.sort()
            return self._build_balanced(values, 0, len(values) - 1)
        elif properties.complete:
            return self._build_complete(values)
        elif properties.perfect:
            return self._build_perfect(values)
        elif properties.skewed == "left":
            return self._build_left_skewed(values)
        elif properties.skewed == "right":
            return self._build_right_skewed(values)
        else:
            return self._build_random(values)

    def _build_balanced(
        self, values: List[int], start: int, end: int
    ) -> Optional[TreeNode]:
        """Build a balanced binary tree from sorted values"""
        if start > end:
            return None

        mid = (start + end) // 2
        node = TreeNode(values[mid])
        node.left = self._build_balanced(values, start, mid - 1)
        node.right = self._build_balanced(values, mid + 1, end)

        return node

    def _build_random(self, values: List[int]) -> Optional[TreeNode]:
        """Build a random structure binary tree"""
        if not values:
            return None

        nodes = [TreeNode(val) for val in values]
        root = nodes[0]

        for i in range(1, len(nodes)):
            # Find a random parent that has space for children
            possible_parents = []
            for j in range(i):
                if nodes[j].left is None or nodes[j].right is None:
                    possible_parents.append(j)

            if not possible_parents:
                break

            parent_idx = random.choice(possible_parents)
            parent = nodes[parent_idx]

            if parent.left is None and parent.right is None:
                if random.random() < 0.5:
                    parent.left = nodes[i]
                else:
                    parent.right = nodes[i]
            elif parent.left is None:
                parent.left = nodes[i]
            else:
                parent.right = nodes[i]

        return root

    def _build_bst(self, values: List[int]) -> Optional[TreeNode]:
        """Build a valid Binary Search Tree"""
        if not values:
            return None

        # Ensure unique values for valid BST
        values = list(set(values))
        values.sort()

        return self._build_balanced(values, 0, len(values) - 1)

    def _build_left_skewed(self, values: List[int]) -> Optional[TreeNode]:
        """Build a left-skewed tree"""
        if not values:
            return None

        root = TreeNode(values[0])
        current = root
        for val in values[1:]:
            current.left = TreeNode(val)
            current = current.left

        return root

    def _build_right_skewed(self, values: List[int]) -> Optional[TreeNode]:
        """Build a right-skewed tree"""
        if not values:
            return None

        root = TreeNode(values[0])
        current = root
        for val in values[1:]:
            current.right = TreeNode(val)
            current = current.right

        return root

    def _build_complete(self, values: List[int]) -> Optional[TreeNode]:
        """Build a complete binary tree using level-order insertion"""
        if not values:
            return None

        root = TreeNode(values[0])
        queue = deque([root])
        idx = 1

        while queue and idx < len(values):
            node = queue.popleft()

            # Add left child
            if idx < len(values):
                node.left = TreeNode(values[idx])
                queue.append(node.left)
                idx += 1

            # Add right child
            if idx < len(values):
                node.right = TreeNode(values[idx])
                queue.append(node.right)
                idx += 1

        return root

    def _build_perfect(self, values: List[int]) -> Optional[TreeNode]:
        """Build a perfect binary tree"""
        import math

        if not values:
            return None

        # Perfect tree has 2^h - 1 nodes
        h = int(math.log2(len(values) + 1))
        size = 2**h - 1

        # Use only the required number of values
        return self._build_complete(values[:size])


class GraphGenerator(BaseGenerator):
    """Generator for graphs"""

    def generate(self, properties: Optional[GraphProperties] = None) -> GraphData:
        """Generate a graph with specified properties"""
        if properties is None:
            properties = GraphProperties(num_nodes=10)
        properties.validate()

        if properties.num_nodes <= 0:
            return {
                "num_nodes": 0,
                "edges": [],
                "adjacency_list": {},
                "directed": properties.directed,
                "weighted": properties.weighted,
            }

        # Calculate edge limits
        max_edges = properties.num_nodes * (properties.num_nodes - 1)
        if not properties.directed:
            max_edges //= 2
        if properties.allow_self_loops:
            max_edges += properties.num_nodes

        if properties.connected and properties.num_nodes > 1:
            min_edges = properties.num_nodes - 1
        else:
            min_edges = 0

        if properties.num_edges is None:
            properties.num_edges = random.randint(
                min_edges, min(max_edges, properties.num_nodes * 2)
            )
        else:
            properties.num_edges = max(min_edges, min(properties.num_edges, max_edges))

        edges = []
        edge_set = set()

        # Create spanning tree if connected graph required
        if properties.connected and properties.num_nodes > 1:
            edges, edge_set = self._create_spanning_tree(
                properties.num_nodes, properties.directed
            )

        # Add remaining random edges
        edges = self._add_random_edges(edges, edge_set, properties)

        # Add weights if needed
        if properties.weighted:
            edges = self._add_weights(edges, properties)

        # Build adjacency list
        adj_list = self._build_adjacency_list(edges, properties)

        return {
            "num_nodes": properties.num_nodes,
            "edges": edges,
            "adjacency_list": adj_list,
            "directed": properties.directed,
            "weighted": properties.weighted,
        }

    def _create_spanning_tree(
        self, num_nodes: int, directed: bool
    ) -> Tuple[List[tuple], set]:
        """Create a spanning tree to ensure connectivity"""
        edges = []
        edge_set = set()

        for i in range(1, num_nodes):
            parent = random.randint(0, i - 1)
            edge = (parent, i) if directed else tuple(sorted([parent, i]))
            edges.append(edge)
            edge_set.add(edge)

        return edges, edge_set

    def _add_random_edges(
        self, edges: List[tuple], edge_set: set, properties: GraphProperties
    ) -> List[tuple]:
        """Add random edges to the graph"""
        attempts = 0
        max_attempts = properties.num_edges * Config.GRAPH_MAX_ATTEMPTS_MULTIPLIER

        while len(edges) < properties.num_edges and attempts < max_attempts:
            attempts += 1
            u = random.randint(0, properties.num_nodes - 1)
            v = random.randint(0, properties.num_nodes - 1)

            if not properties.allow_self_loops and u == v:
                continue

            edge = (u, v) if properties.directed else tuple(sorted([u, v]))

            if edge not in edge_set:
                edges.append(edge)
                edge_set.add(edge)

        return edges

    def _add_weights(
        self, edges: List[tuple], properties: GraphProperties
    ) -> List[tuple]:
        """Add weights to edges"""
        weighted_edges = []
        for edge in edges:
            weight = random.randint(properties.min_weight, properties.max_weight)
            weighted_edges.append((*edge, weight))
        return weighted_edges

    def _build_adjacency_list(
        self, edges: List[tuple], properties: GraphProperties
    ) -> Dict[int, List]:
        """Build adjacency list from edges"""
        adj_list = defaultdict(list)

        for edge in edges:
            if properties.weighted:
                u, v, w = edge
                adj_list[u].append((v, w))
                if not properties.directed:
                    adj_list[v].append((u, w))
            else:
                u, v = edge
                adj_list[u].append(v)
                if not properties.directed:
                    adj_list[v].append(u)

        # Ensure all nodes are in adjacency list
        for i in range(properties.num_nodes):
            if i not in adj_list:
                adj_list[i] = []

        return dict(adj_list)


class LinkedListGenerator(BaseGenerator):
    """Generator for linked lists"""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.int_gen = IntegerGenerator(seed)

    def generate(
        self,
        size: int,
        constraints: Optional[Constraints] = None,
        has_cycle: bool = False,
        cycle_position: Optional[int] = None,
    ) -> Optional[ListNode]:
        """Generate a linked list, optionally with a cycle"""
        if size <= 0:
            return None

        if constraints is None:
            constraints = Constraints()

        values = self.int_gen.generate_array(size, constraints)

        # Create nodes
        dummy = ListNode(0)
        current = dummy
        nodes = []

        for val in values:
            node = ListNode(val)
            current.next = node
            current = node
            nodes.append(node)

        # Add cycle if requested
        if has_cycle and len(nodes) > 1:
            if cycle_position is None:
                cycle_position = random.randint(0, len(nodes) - 2)
            if 0 <= cycle_position < len(nodes):
                nodes[-1].next = nodes[cycle_position]
        elif has_cycle and len(nodes) == 1:
            nodes[0].next = nodes[0]

        return dummy.next
