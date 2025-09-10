"""
Comprehensive tests for TreeSerializer and LinkedListSerializer

These serializers are critical for CLI functionality but have zero test coverage.
Tests cover serialization, deserialization, edge cases, and CLI integration.
"""

import json
from typing import Optional

import pytest
from testgen.core.models import ListNode, TreeNode


class TestTreeSerializer:
    """Test TreeSerializer functionality"""

    def test_tree_serializer_import(self):
        """Test that TreeSerializer can be imported"""

        from testgen.core.serializers import TreeSerializer

        assert TreeSerializer is not None

    def test_simple_tree_to_array(self):
        """Test conversion of simple tree to array"""
        from testgen.core.serializers import TreeSerializer

        # Create simple tree: 1 -> 2, 3
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)

        array_repr = TreeSerializer.to_array(root)

        assert isinstance(array_repr, list), "Should return list"
        assert len(array_repr) > 0, "Should not be empty for non-null tree"
        assert len(array_repr) == 3, "Should not be empty for non-null tree"
        assert 1 in array_repr, "Should contain root value"

    def test_null_tree_serialization(self):
        """Test serialization of null tree"""
        from testgen.core.serializers import TreeSerializer

        array_repr = TreeSerializer.to_array(None)

        # Should handle null gracefully
        assert array_repr == [] or array_repr is None, (
            "Null tree should serialize to empty array or None"
        )

    def test_single_node_tree_serialization(self):
        """Test serialization of single node tree"""
        from testgen.core.serializers import TreeSerializer

        root = TreeNode(42)
        array_repr = TreeSerializer.to_array(root)

        assert isinstance(array_repr, list), "Should return list"
        assert 42 in array_repr, "Should contain the single value"

    def test_complete_tree_serialization(self):
        """Test serialization of complete binary tree"""
        from testgen.core.serializers import TreeSerializer

        # Create complete tree:     1
        #                         /   \
        #                        2     3
        #                       / \   / \
        #                      4   5 6   7
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)

        array_repr = TreeSerializer.to_array(root)

        assert isinstance(array_repr, list), "Should return list"
        assert len(array_repr) >= 7, (
            "Should contain at least 7 elements for 7-node tree"
        )

        # Should contain all values
        for val in [1, 2, 3, 4, 5, 6, 7]:
            assert val in array_repr, f"Should contain value {val}"

    def test_unbalanced_tree_serialization(self):
        """Test serialization of unbalanced tree"""
        from testgen.core.serializers import TreeSerializer

        # Create unbalanced tree (right-skewed)
        root = TreeNode(1)
        root.right = TreeNode(2)
        root.right.right = TreeNode(3)
        root.right.right.right = TreeNode(4)

        array_repr = TreeSerializer.to_array(root)

        assert isinstance(array_repr, list), "Should return list"
        assert all(val in array_repr for val in [1, 2, 3, 4]), (
            "Should contain all values from unbalanced tree"
        )

    def test_tree_with_null_children_serialization(self):
        """Test serialization of tree with null children"""
        from testgen.core.serializers import TreeSerializer

        # Create tree with gaps:  1
        #                        / \
        #                       2   null
        #                      / \
        #                   null  3
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.left.right = TreeNode(3)
        # root.right is None
        # root.left.left is None

        array_repr = TreeSerializer.to_array(root)

        assert isinstance(array_repr, list), "Should return list"
        assert (
            1 in array_repr
            and 2 in array_repr
            and 3 in array_repr
            and None in array_repr
        ), "Should contain all values, including None"


class TestTreeSerializerRoundTrip:
    """Test round-trip serialization (to_array -> from_array)"""

    def test_round_trip_simple_tree(self):
        """Test round-trip serialization of simple tree"""
        from testgen.core.serializers import TreeSerializer

        # Create original tree
        original = TreeNode(1)
        original.left = TreeNode(2)
        original.right = TreeNode(3)

        # Serialize and deserialize
        array_repr = TreeSerializer.to_array(original)
        reconstructed = TreeSerializer.from_array(array_repr)

        # Should be equivalent
        assert self._trees_equal(original, reconstructed), (
            "Round-trip should preserve tree structure"
        )

    def test_round_trip_complex_tree(self):
        """Test round-trip serialization of complex tree"""
        from testgen.core.serializers import TreeSerializer

        # Create complex tree with various structures
        original = TreeNode(10)
        original.left = TreeNode(5)
        original.right = TreeNode(15)
        original.left.left = TreeNode(3)
        original.left.right = TreeNode(7)
        original.right.right = TreeNode(18)

        # Round-trip
        array_repr = TreeSerializer.to_array(original)
        reconstructed = TreeSerializer.from_array(array_repr)

        assert self._trees_equal(original, reconstructed), (
            "Round-trip should preserve complex tree structure"
        )

    def test_round_trip_null_tree(self):
        """Test round-trip serialization of null tree"""
        from testgen.core.serializers import TreeSerializer

        # Serialize and deserialize null
        array_repr = TreeSerializer.to_array(None)
        reconstructed = TreeSerializer.from_array(array_repr)

        assert reconstructed is None, "Round-trip null should remain null"

    def _trees_equal(
        self, tree1: Optional[TreeNode], tree2: Optional[TreeNode]
    ) -> bool:
        """Helper to check if two trees are structurally equal"""
        if tree1 is None and tree2 is None:
            return True
        if tree1 is None or tree2 is None:
            return False
        return (
            tree1.val == tree2.val
            and self._trees_equal(tree1.left, tree2.left)
            and self._trees_equal(tree1.right, tree2.right)
        )


class TestLinkedListSerializer:
    """Test LinkedListSerializer functionality"""

    def test_linked_list_serializer_import(self):
        """Test that LinkedListSerializer can be imported"""
        from testgen.core.serializers import LinkedListSerializer

        assert LinkedListSerializer is not None

    def test_simple_list_to_array(self):
        """Test conversion of simple linked list to array"""
        from testgen.core.serializers import LinkedListSerializer

        # Create simple list: 1 -> 2 -> 3
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)

        array_repr = LinkedListSerializer.to_array(head)

        assert isinstance(array_repr, list), "Should return list"
        assert array_repr == [1, 2, 3], "Should convert to simple array"

    def test_null_list_serialization(self):
        """Test serialization of null list"""
        from testgen.core.serializers import LinkedListSerializer

        array_repr = LinkedListSerializer.to_array(None)

        assert array_repr == [] or array_repr is None, (
            "Null list should serialize to empty array or None"
        )

    def test_single_node_list_serialization(self):
        """Test serialization of single node list"""
        from testgen.core.serializers import LinkedListSerializer

        head = ListNode(42)
        array_repr = LinkedListSerializer.to_array(head)

        assert array_repr == [42], (
            "Single node should serialize to single-element array"
        )

    def test_long_list_serialization(self):
        """Test serialization of longer linked list"""
        from testgen.core.serializers import LinkedListSerializer

        # Create list: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
        head = ListNode(1)
        current = head
        for i in range(2, 11):
            current.next = ListNode(i)
            current = current.next

        array_repr = LinkedListSerializer.to_array(head)

        expected = list(range(1, 11))
        assert array_repr == expected, f"Should serialize to {expected}"


class TestLinkedListSerializerCycles:
    """Test LinkedListSerializer with cyclic lists"""

    def test_cycle_detection(self):
        """Test detection and handling of cycles in linked lists"""
        from testgen.core.serializers import LinkedListSerializer

        # Create list with cycle: 1 -> 2 -> 3 -> 2 (cycle back to 2)
        head = ListNode(1)
        node2 = ListNode(2)
        node3 = ListNode(3)

        head.next = node2
        node2.next = node3
        node3.next = node2  # Creates cycle

        # Should handle cycle gracefully (not infinite loop)
        try:
            array_repr = LinkedListSerializer.to_array(head)

            # Should either detect cycle or limit output length
            assert isinstance(array_repr, list), "Should return list for cyclic list"
            assert len(array_repr) < 1000, "Should not create infinite array"

        except Exception as e:
            # Cycle detection might raise an exception - that's also valid
            assert "cycle" in str(e).lower(), "Exception should mention cycle"

    def test_self_referencing_node(self):
        """Test handling of self-referencing single node"""
        from testgen.core.serializers import LinkedListSerializer

        # Create self-referencing node
        head = ListNode(42)
        head.next = head  # Points to itself

        array_repr = LinkedListSerializer.to_array(head)

        # Should handle gracefully
        assert isinstance(array_repr, list), "Should return list"
        assert 42 in array_repr, "Should contain the node value"
        assert len(array_repr) < 100, "Should not create infinite array"


class TestLinkedListSerializerRoundTrip:
    """Test round-trip serialization for linked lists"""

    def test_round_trip_simple_list(self):
        """Test round-trip serialization of simple list"""
        from testgen.core.serializers import LinkedListSerializer

        # Create original list
        original = ListNode(1)
        original.next = ListNode(2)
        original.next.next = ListNode(3)

        # Round-trip
        array_repr = LinkedListSerializer.to_array(original)
        reconstructed = LinkedListSerializer.from_array(array_repr)

        # Should be equivalent
        assert self._lists_equal(original, reconstructed), (
            "Round-trip should preserve list structure"
        )

    def test_round_trip_empty_list(self):
        """Test round-trip serialization of empty list"""
        from testgen.core.serializers import LinkedListSerializer

        # Round-trip empty list
        array_repr = LinkedListSerializer.to_array(None)
        reconstructed = LinkedListSerializer.from_array(array_repr)

        assert reconstructed is None, "Round-trip empty should remain None"

    def _lists_equal(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> bool:
        """Helper to check if two linked lists are equal"""
        current1, current2 = list1, list2

        while current1 and current2:
            if current1.val != current2.val:
                return False
            current1 = current1.next
            current2 = current2.next

        return current1 is None and current2 is None


class TestSerializationCLIIntegration:
    """Test serialization integration with CLI functionality"""

    def test_json_serialization_compatibility(self):
        """Test that serialized output is JSON compatible"""
        from testgen.core.serializers import TreeSerializer

        # Create tree
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)

        # Serialize
        array_repr = TreeSerializer.to_array(root)

        # Should be JSON serializable (for CLI output)
        json_str = json.dumps(array_repr)
        assert isinstance(json_str, str), "Should be JSON serializable"

        # Should be able to deserialize
        parsed_back = json.loads(json_str)
        assert parsed_back == array_repr, "JSON round-trip should preserve data"

    def test_linked_list_json_compatibility(self):
        """Test that linked list serialization is JSON compatible"""
        from testgen.core.serializers import LinkedListSerializer

        # Create list
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)

        # Serialize
        array_repr = LinkedListSerializer.to_array(head)

        # Should be JSON serializable
        json_str = json.dumps(array_repr)
        assert isinstance(json_str, str), "Should be JSON serializable"

        parsed_back = json.loads(json_str)
        assert parsed_back == array_repr, "JSON round-trip should preserve data"

    def test_cli_output_format_consistency(self):
        """Test that serialization produces consistent CLI output format"""
        from testgen.core.serializers import LinkedListSerializer, TreeSerializer

        # Test tree output format
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)

        tree_array = TreeSerializer.to_array(root)

        # CLI expects array format for display
        assert isinstance(tree_array, list), "Tree should serialize to array for CLI"
        assert all(isinstance(x, (int, type(None))) for x in tree_array), (
            "Tree array should contain only integers and nulls"
        )

        # Test linked list output format
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)

        list_array = LinkedListSerializer.to_array(head)

        # CLI expects simple array for linked lists
        assert isinstance(list_array, list), "List should serialize to array for CLI"
        assert all(isinstance(x, int) for x in list_array), (
            "List array should contain only integers"
        )

    def test_large_data_structure_serialization(self):
        """Test serialization of large data structures for CLI performance"""
        from testgen.core.serializers import LinkedListSerializer, TreeSerializer

        # Create large tree (100 nodes)
        large_tree = TreeNode(1)
        current_level = [large_tree]
        node_count = 1

        while node_count < 100:
            next_level = []
            for node in current_level:
                if node_count < 100:
                    node.left = TreeNode(node_count + 1)
                    next_level.append(node.left)
                    node_count += 1
                if node_count < 100:
                    node.right = TreeNode(node_count + 1)
                    next_level.append(node.right)
                    node_count += 1
            current_level = next_level

        # Should serialize large tree efficiently
        tree_array = TreeSerializer.to_array(large_tree)
        assert len(tree_array) >= 100, "Should contain all nodes"

        # Create large linked list (1000 nodes)
        large_list = ListNode(1)
        current = large_list
        for i in range(2, 1001):
            current.next = ListNode(i)
            current = current.next

        # Should serialize large list efficiently
        list_array = LinkedListSerializer.to_array(large_list)
        assert len(list_array) == 1000, "Should contain all 1000 nodes"


class TestSerializationEdgeCases:
    """Test edge cases and error conditions for serialization"""

    def test_malformed_tree_handling(self):
        """Test handling of malformed tree structures"""
        from testgen.core.serializers import TreeSerializer

        # Test tree with unusual values
        root = TreeNode(0)  # Zero value
        root.left = TreeNode(-1)  # Negative value
        root.right = TreeNode(999999)  # Large value

        array_repr = TreeSerializer.to_array(root)

        assert 0 in array_repr, "Should handle zero values"
        assert -1 in array_repr, "Should handle negative values"
        assert 999999 in array_repr, "Should handle large values"

    def test_malformed_linked_list_handling(self):
        """Test handling of malformed linked list structures"""
        from testgen.core.serializers import LinkedListSerializer

        # Test list with unusual values
        head = ListNode(0)
        head.next = ListNode(-100)
        head.next.next = ListNode(999999)

        array_repr = LinkedListSerializer.to_array(head)

        expected = [0, -100, 999999]
        assert array_repr == expected, "Should handle unusual values correctly"

    def test_memory_efficiency_large_structures(self):
        """Test memory efficiency with large data structures"""
        import os

        import psutil
        from testgen.core.serializers import TreeSerializer

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create and serialize large tree
        large_tree = TreeNode(1)
        current = large_tree
        for i in range(2, 5001):  # 5000 node tree
            current.left = TreeNode(i)
            current = current.left

        tree_array = TreeSerializer.to_array(large_tree)

        mid_memory = process.memory_info().rss
        memory_growth = mid_memory - initial_memory

        # Should not use excessive memory
        assert memory_growth < 50 * 1024 * 1024, (
            f"Tree serialization used too much memory: {memory_growth / 1024 / 1024:.1f}MB"
        )

    def test_concurrent_serialization_safety(self):
        """Test that serialization is safe for concurrent use"""
        import threading
        import time

        from testgen.core.serializers import TreeSerializer

        # Create test tree
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)

        results = []
        errors = []

        def serialize_tree():
            try:
                for _ in range(100):
                    array_repr = TreeSerializer.to_array(root)
                    results.append(array_repr)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=serialize_tree)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should not have errors
        assert len(errors) == 0, f"Concurrent serialization had errors: {errors}"

        # Should have consistent results
        if results:
            first_result = results[0]
            assert all(result == first_result for result in results), (
                "Concurrent serialization should produce consistent results"
            )


class TestSerializationIntegration:
    """Test serialization integration with the broader system"""

    def test_generator_serialization_workflow(self):
        """Test complete workflow from generation to serialization"""
        from testgen.core.generators import LinkedListGenerator, TreeGenerator
        from testgen.core.models import Constraints, TreeProperties
        from testgen.core.serializers import LinkedListSerializer, TreeSerializer

        # Generate tree
        tree_gen = TreeGenerator(seed=42)
        props = TreeProperties(size=10, balanced=True)
        constraints = Constraints(min_value=1, max_value=100)

        tree = tree_gen.generate(props, constraints)

        # Serialize generated tree
        tree_array = TreeSerializer.to_array(tree)

        assert isinstance(tree_array, list), "Generated tree should serialize to array"
        assert len(tree_array) >= 10, "Serialized tree should contain all nodes"

        # Generate linked list
        list_gen = LinkedListGenerator(seed=42)
        linked_list = list_gen.generate(size=15, constraints=constraints)

        # Serialize generated list
        list_array = LinkedListSerializer.to_array(linked_list)

        assert isinstance(list_array, list), "Generated list should serialize to array"
        assert len(list_array) == 15, "Serialized list should contain all nodes"

    def test_error_handling_integration(self):
        """Test serialization error handling integration"""
        from testgen.core.serializers import TreeSerializer

        # Test with invalid input
        try:
            TreeSerializer.to_array("invalid_input")
        # If no error, should handle gracefully
        except Exception as e:
            # Error should be informative
            error_msg = str(e)
            assert len(error_msg) > 5, "Error message should be informative"

    def test_performance_benchmarking(self):
        """Test serialization performance for benchmarking"""
        import time

        from testgen.core.serializers import LinkedListSerializer, TreeSerializer

        # Create medium-sized structures
        tree = TreeNode(1)
        current_level = [tree]
        for level in range(6):  # Creates tree with ~63 nodes
            next_level = []
            for node in current_level:
                node.left = TreeNode(level * 10 + 1)
                node.right = TreeNode(level * 10 + 2)
                next_level.extend([node.left, node.right])
            current_level = next_level

        # Benchmark tree serialization
        start_time = time.time()
        for _ in range(1000):
            TreeSerializer.to_array(tree)
        tree_time = time.time() - start_time

        # Create linked list
        head = ListNode(1)
        current = head
        for i in range(2, 101):  # 100 nodes
            current.next = ListNode(i)
            current = current.next

        # Benchmark list serialization
        start_time = time.time()
        for _ in range(1000):
            LinkedListSerializer.to_array(head)
        list_time = time.time() - start_time

        # Should be reasonably fast
        assert tree_time < 5.0, "Tree serialization should be reasonably fast"
        assert list_time < 5.0, "List serialization should be reasonably fast"


if __name__ == "__main__":
    # Can be run directly for quick testing
    pytest.main([__file__, "-v"])
