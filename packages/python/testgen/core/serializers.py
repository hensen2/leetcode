"""
Serialization module for converting data structures to/from various formats
Handles trees, linked lists, graphs, and other complex structures
"""

from collections import deque
from typing import Any, Dict, List, Optional, Protocol


class TreeNode(Protocol):
    """Protocol for tree nodes"""

    val: Any
    left: Optional["TreeNode"]
    right: Optional["TreeNode"]


class ListNode(Protocol):
    """Protocol for linked list nodes"""

    val: Any
    next: Optional["ListNode"]


class TreeSerializer:
    """Handles tree serialization and deserialization"""

    @staticmethod
    def to_array(root: Optional[TreeNode]) -> List[Optional[int]]:
        """
        Convert tree to array representation (level-order)

        Args:
            root: Root node of the tree

        Returns:
            Array representation of the tree
        """
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            node = queue.popleft()
            if node:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append(None)

        # Remove trailing None values
        while result and result[-1] is None:
            result.pop()

        return result

    @staticmethod
    def from_array(arr: List[Optional[int]], node_class: type) -> Optional[TreeNode]:
        """
        Convert array to tree (level-order)

        Args:
            arr: Array representation of tree
            node_class: Class to use for creating nodes

        Returns:
            Root node of reconstructed tree
        """
        if not arr or arr[0] is None:
            return None

        root = node_class(arr[0])
        queue = deque([root])
        i = 1

        while queue and i < len(arr):
            node = queue.popleft()

            # Process left child
            if i < len(arr) and arr[i] is not None:
                node.left = node_class(arr[i])
                queue.append(node.left)
            i += 1

            # Process right child
            if i < len(arr) and arr[i] is not None:
                node.right = node_class(arr[i])
                queue.append(node.right)
            i += 1

        return root

    @staticmethod
    def to_preorder(root: Optional[TreeNode]) -> List[int]:
        """Convert tree to preorder traversal list"""
        if not root:
            return []

        result = []

        def traverse(node):
            if node:
                result.append(node.val)
                traverse(node.left)
                traverse(node.right)

        traverse(root)
        return result

    @staticmethod
    def to_inorder(root: Optional[TreeNode]) -> List[int]:
        """Convert tree to inorder traversal list"""
        if not root:
            return []

        result = []

        def traverse(node):
            if node:
                traverse(node.left)
                result.append(node.val)
                traverse(node.right)

        traverse(root)
        return result

    @staticmethod
    def to_postorder(root: Optional[TreeNode]) -> List[int]:
        """Convert tree to postorder traversal list"""
        if not root:
            return []

        result = []

        def traverse(node):
            if node:
                traverse(node.left)
                traverse(node.right)
                result.append(node.val)

        traverse(root)
        return result


class LinkedListSerializer:
    """Handles linked list serialization and deserialization"""

    CYCLE_MARKER = "CYCLE_DETECTED"

    @staticmethod
    def to_array(head: Optional[ListNode], detect_cycle: bool = True) -> List[Any]:
        """
        Convert linked list to array (with cycle detection)

        Args:
            head: Head node of linked list
            detect_cycle: Whether to detect cycles

        Returns:
            Array representation of linked list
        """
        if not head:
            return []

        result = []

        if detect_cycle:
            # Use Floyd's algorithm to detect cycle
            slow = fast = head

            while fast and fast.next:
                result.append(slow.val)
                slow = slow.next
                fast = fast.next.next

                if slow == fast:
                    # Cycle detected
                    result.append(slow.val)
                    result.append(LinkedListSerializer.CYCLE_MARKER)
                    break
            else:
                # No cycle, continue adding remaining elements
                while slow:
                    result.append(slow.val)
                    slow = slow.next
        else:
            # Simple traversal without cycle detection
            current = head
            while current:
                result.append(current.val)
                current = current.next

        return result

    @staticmethod
    def from_array(
        arr: List[Any], node_class: type, create_cycle_at: Optional[int] = None
    ) -> Optional[ListNode]:
        """
        Convert array to linked list

        Args:
            arr: Array of values
            node_class: Class to use for creating nodes
            create_cycle_at: Index to create cycle at (if specified)

        Returns:
            Head node of linked list
        """
        if not arr:
            return None

        dummy = node_class(0)
        current = dummy
        nodes = []

        for val in arr:
            if isinstance(val, str) and val == LinkedListSerializer.CYCLE_MARKER:
                break
            new_node = node_class(val)
            current.next = new_node
            current = new_node
            nodes.append(new_node)

        # Create cycle if requested
        if create_cycle_at is not None and 0 <= create_cycle_at < len(nodes):
            nodes[-1].next = nodes[create_cycle_at]

        return dummy.next


class GraphSerializer:
    """Handles graph serialization and deserialization"""

    @staticmethod
    def to_adjacency_list(
        edges: List[tuple], num_nodes: int, directed: bool = False
    ) -> Dict[int, List]:
        """
        Convert edge list to adjacency list

        Args:
            edges: List of edges (tuples)
            num_nodes: Number of nodes in graph
            directed: Whether graph is directed

        Returns:
            Adjacency list representation
        """
        adj_list = {i: [] for i in range(num_nodes)}

        for edge in edges:
            if len(edge) == 2:  # Unweighted
                u, v = edge
                adj_list[u].append(v)
                if not directed:
                    adj_list[v].append(u)
            elif len(edge) == 3:  # Weighted
                u, v, w = edge
                adj_list[u].append((v, w))
                if not directed:
                    adj_list[v].append((u, w))

        return adj_list

    @staticmethod
    def to_adjacency_matrix(
        edges: List[tuple], num_nodes: int, directed: bool = False
    ) -> List[List[int]]:
        """
        Convert edge list to adjacency matrix

        Args:
            edges: List of edges
            num_nodes: Number of nodes
            directed: Whether graph is directed

        Returns:
            Adjacency matrix representation
        """
        matrix = [[0] * num_nodes for _ in range(num_nodes)]

        for edge in edges:
            if len(edge) == 2:  # Unweighted
                u, v = edge
                matrix[u][v] = 1
                if not directed:
                    matrix[v][u] = 1
            elif len(edge) == 3:  # Weighted
                u, v, w = edge
                matrix[u][v] = w
                if not directed:
                    matrix[v][u] = w

        return matrix

    @staticmethod
    def from_adjacency_list(
        adj_list: Dict[int, List], weighted: bool = False
    ) -> List[tuple]:
        """
        Convert adjacency list to edge list

        Args:
            adj_list: Adjacency list representation
            weighted: Whether edges have weights

        Returns:
            List of edges
        """
        edges = []
        seen = set()

        for u, neighbors in adj_list.items():
            for neighbor in neighbors:
                if weighted and isinstance(neighbor, tuple):
                    v, w = neighbor
                    edge_key = tuple(sorted([u, v]))
                    if edge_key not in seen:
                        edges.append((u, v, w))
                        seen.add(edge_key)
                else:
                    v = neighbor
                    edge_key = tuple(sorted([u, v]))
                    if edge_key not in seen:
                        edges.append((u, v))
                        seen.add(edge_key)

        return edges

    @staticmethod
    def from_adjacency_matrix(
        matrix: List[List[int]], directed: bool = False
    ) -> List[tuple]:
        """
        Convert adjacency matrix to edge list

        Args:
            matrix: Adjacency matrix
            directed: Whether graph is directed

        Returns:
            List of edges
        """
        edges = []
        n = len(matrix)

        for i in range(n):
            start_j = 0 if directed else i + 1
            for j in range(start_j, n):
                if matrix[i][j] != 0:
                    if matrix[i][j] == 1:
                        edges.append((i, j))
                    else:
                        edges.append((i, j, matrix[i][j]))

        return edges


class MatrixSerializer:
    """Handles matrix serialization and special matrix formats"""

    @staticmethod
    def to_flat_array(matrix: List[List[Any]]) -> List[Any]:
        """Flatten a 2D matrix to 1D array"""
        if not matrix or not matrix[0]:
            return []
        return [element for row in matrix for element in row]

    @staticmethod
    def from_flat_array(arr: List[Any], rows: int, cols: int) -> List[List[Any]]:
        """Convert flat array to 2D matrix"""
        if not arr or rows == 0 or cols == 0:
            return [[]]

        matrix = []
        for i in range(rows):
            row = arr[i * cols : (i + 1) * cols]
            matrix.append(row)

        return matrix

    @staticmethod
    def to_sparse_representation(matrix: List[List[Any]]) -> List[tuple]:
        """Convert matrix to sparse representation (row, col, value)"""
        sparse = []
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val != 0:  # Only store non-zero values
                    sparse.append((i, j, val))
        return sparse

    @staticmethod
    def from_sparse_representation(
        sparse: List[tuple], rows: int, cols: int
    ) -> List[List[Any]]:
        """Convert sparse representation back to full matrix"""
        matrix = [[0] * cols for _ in range(rows)]
        for i, j, val in sparse:
            matrix[i][j] = val
        return matrix
