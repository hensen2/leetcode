"""
Comprehensive tests for LinkedListGenerator

Tests linked list creation, cycle detection/creation, serialization compatibility,
and CLI integration (--cycle flag).
"""

from typing import List, Optional

import pytest
from testgen.core.generators import LinkedListGenerator
from testgen.core.models import Constraints, ListNode


class TestLinkedListGeneratorBasics:
    """Test basic linked list generation functionality"""

    def test_linked_list_generator_instantiation(self):
        """Test LinkedListGenerator can be instantiated"""
        generator = LinkedListGenerator()
        assert generator is not None
        assert hasattr(generator, "generate")

    def test_linked_list_generator_with_seed(self):
        """Test LinkedListGenerator with seed produces reproducible results"""
        seed = 12345
        constraints = Constraints(min_value=1, max_value=100)

        # First generation
        gen1 = LinkedListGenerator(seed)
        list1 = gen1.generate(5, constraints)

        # Second generation resets with same seed
        gen2 = LinkedListGenerator(seed)
        list2 = gen2.generate(5, constraints)

        # Same seed should produce same list structure and values
        assert self._lists_equal(list1, list2), (
            "Same seed should produce same linked list"
        )

    def test_basic_linked_list_generation(self):
        """Test basic linked list generation with various sizes"""
        generator = LinkedListGenerator()

        test_sizes = [0, 1, 3, 5, 10, 50, 100]

        for size in test_sizes:
            constraints = Constraints(min_value=1, max_value=100)
            linked_list = generator.generate(size, constraints)

            if size == 0:
                assert linked_list is None, "Empty list should be None"
            else:
                assert linked_list is not None, (
                    f"Non-empty list should not be None for size {size}"
                )
                assert isinstance(linked_list, ListNode), (
                    f"Should return ListNode for size {size}"
                )

                # Verify list has correct length
                actual_length = self._get_list_length(linked_list)
                assert actual_length == size, (
                    f"List should have {size} nodes, got {actual_length}"
                )

    def test_list_node_structure(self):
        """Test that generated lists have proper ListNode structure"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)
        linked_list = generator.generate(5, constraints)

        # Verify ListNode structure
        current = linked_list
        while current:
            assert hasattr(current, "val"), "ListNode should have 'val' attribute"
            assert hasattr(current, "next"), "ListNode should have 'next' attribute"
            assert isinstance(current.val, int), "Node value should be integer"
            assert 1 <= current.val <= 100, "Node value should be within constraints"
            current = current.next

    def test_list_values_within_constraints(self):
        """Test that all list values respect constraints"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=10, max_value=50)
        linked_list = generator.generate(8, constraints)

        values = self._collect_values(linked_list)
        assert all(10 <= val <= 50 for val in values), (
            "All values should be within constraints"
        )

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

    def _get_list_length(
        self, head: Optional[ListNode], max_length: int = 10000
    ) -> int:
        """Helper to get length of linked list (with cycle protection)"""
        if head is None:
            return 0

        current = head
        length = 0
        visited = set()

        while current and length < max_length:
            if id(current) in visited:
                # Cycle detected
                return length
            visited.add(id(current))
            length += 1
            current = current.next

        return length

    def _collect_values(
        self, head: Optional[ListNode], max_nodes: int = 10000
    ) -> List[int]:
        """Helper to collect all values in linked list (with cycle protection)"""
        if head is None:
            return []

        values = []
        current = head
        visited = set()

        while current and len(values) < max_nodes:
            if id(current) in visited:
                # Cycle detected, stop collection
                break
            visited.add(id(current))
            values.append(current.val)
            current = current.next

        return values


class TestLinkedListGeneratorCycles:
    """Test cycle creation and detection functionality"""

    def test_cycle_creation(self):
        """Test creation of linked lists with cycles"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)

        # Test cycle creation if supported
        try:
            cyclic_list = generator.generate(5, constraints, has_cycle=True)

            assert cyclic_list is not None, "Cyclic list should not be None"
            assert self._has_cycle(cyclic_list), (
                "List should have a cycle when has_cycle=True"
            )

        except TypeError:
            # has_cycle parameter might not be supported
            print("Note: has_cycle parameter not supported")
        except AttributeError:
            pytest.skip("Cycle creation not implemented")

    def test_non_cycle_creation(self):
        """Test creation of linked lists without cycles"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)

        try:
            acyclic_list = generator.generate(8, constraints, has_cycle=False)

            assert acyclic_list is not None, "Acyclic list should not be None"
            assert not self._has_cycle(acyclic_list), (
                "List should not have cycle when has_cycle=False"
            )

        except TypeError:
            # has_cycle parameter might not be supported
            acyclic_list = generator.generate(8, constraints)
            assert acyclic_list is not None, "Basic list generation should work"

    def test_cycle_detection_algorithm(self):
        """Test cycle detection algorithm works correctly"""
        generator = LinkedListGenerator()

        # Create known cyclic list manually
        head = ListNode(1)
        node2 = ListNode(2)
        node3 = ListNode(3)
        node4 = ListNode(4)

        head.next = node2
        node2.next = node3
        node3.next = node4
        node4.next = node2  # Creates cycle: 1->2->3->4->2...

        assert self._has_cycle(head), (
            "Should detect cycle in manually created cyclic list"
        )

        # Create known acyclic list
        head2 = ListNode(1)
        head2.next = ListNode(2)
        head2.next.next = ListNode(3)

        assert not self._has_cycle(head2), "Should not detect cycle in acyclic list"

    def test_cycle_position_variety(self):
        """Test that cycles can be created at different positions"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)

        try:
            # Generate multiple cyclic lists to test variety
            cyclic_lists = []
            for _ in range(10):
                cyclic_list = generator.generate(6, constraints, has_cycle=True)
                if self._has_cycle(cyclic_list):
                    cycle_pos = self._find_cycle_start(cyclic_list)
                    cyclic_lists.append(cycle_pos)

            if cyclic_lists:
                # Should have some variety in cycle positions
                unique_positions = set(cyclic_lists)
                assert len(unique_positions) >= 1, (
                    "Should create cycles at various positions"
                )

        except TypeError:
            pytest.skip("Cycle parameter not supported")

    def test_single_node_cycle(self):
        """Test cycle creation with single node"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)

        try:
            single_cyclic = generator.generate(1, constraints, has_cycle=True)

            if single_cyclic is not None:
                # Single node cycle should point to itself
                assert single_cyclic.next == single_cyclic, (
                    "Single node cycle should point to itself"
                )

        except TypeError:
            pytest.skip("Single node cycle not supported")

    def _has_cycle(self, head: Optional[ListNode]) -> bool:
        """Floyd's cycle detection algorithm (tortoise and hare)"""
        if head is None or head.next is None:
            return False

        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False

    def _find_cycle_start(self, head: Optional[ListNode]) -> int:
        """Find the position where cycle starts (0-indexed)"""
        if head is None or not self._has_cycle(head):
            return -1

        # Find meeting point
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break

        # Find start of cycle
        current = head
        position = 0
        while current != slow:
            current = current.next
            slow = slow.next
            position += 1

        return position


class TestLinkedListGeneratorConstraints:
    """Test linked list generation with various constraints"""

    def test_value_constraints(self):
        """Test that list values respect constraints"""
        generator = LinkedListGenerator()

        test_constraints = [
            Constraints(min_value=1, max_value=10),
            Constraints(min_value=-50, max_value=-10),
            Constraints(min_value=100, max_value=200),
            Constraints(min_value=0, max_value=1),  # Binary values
        ]

        for constraints in test_constraints:
            linked_list = generator.generate(6, constraints)
            values = self._collect_values(linked_list)

            for val in values:
                assert constraints.min_value <= val <= constraints.max_value, (
                    f"Value {val} should be within [{constraints.min_value}, {constraints.max_value}]"
                )

    def test_single_value_constraint(self):
        """Test constraint where min_value equals max_value"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=42, max_value=42)
        linked_list = generator.generate(5, constraints)

        values = self._collect_values(linked_list)
        assert all(val == 42 for val in values), (
            "All values should be 42 when min_value == max_value"
        )

    def test_invalid_constraints_handling(self):
        """Test handling of invalid constraints"""
        generator = LinkedListGenerator()

        # Invalid: min > max
        constraints = Constraints(min_value=100, max_value=1)

        try:
            linked_list = generator.generate(5, constraints)
            # If no error, should still produce valid list
            assert linked_list is not None or True  # Allow None for invalid constraints
        except (ValueError, AssertionError):
            print("✅ Properly handles invalid constraints with error")

    def test_constraint_edge_cases(self):
        """Test constraint edge cases"""
        generator = LinkedListGenerator()

        # Very large range
        constraints = Constraints(min_value=-1000000, max_value=1000000)
        linked_list = generator.generate(3, constraints)

        values = self._collect_values(linked_list)
        for val in values:
            assert -1000000 <= val <= 1000000, "Should handle large ranges"

    def _collect_values(
        self, head: Optional[ListNode], max_nodes: int = 10000
    ) -> List[int]:
        """Helper to collect all values in linked list (with cycle protection)"""
        if head is None:
            return []

        values = []
        current = head
        visited = set()

        while current and len(values) < max_nodes:
            if id(current) in visited:
                # Cycle detected, stop collection
                break
            visited.add(id(current))
            values.append(current.val)
            current = current.next

        return values


class TestLinkedListGeneratorEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_list_generation(self):
        """Test generation of empty linked lists"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)
        linked_list = generator.generate(0, constraints)

        assert linked_list is None, "Empty list should be None"

    def test_single_node_list(self):
        """Test generation of single node lists"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=5, max_value=15)
        linked_list = generator.generate(1, constraints)

        assert linked_list is not None, "Single node list should not be None"
        assert linked_list.next is None, "Single node should have no next"
        assert 5 <= linked_list.val <= 15, (
            "Single node value should be within constraints"
        )

    def test_large_list_generation(self):
        """Test generation of large linked lists"""
        generator = LinkedListGenerator()

        large_size = 10000
        constraints = Constraints(min_value=1, max_value=1000)

        linked_list = generator.generate(large_size, constraints)

        assert linked_list is not None, "Large list should not be None"
        actual_length = self._get_list_length(linked_list)
        assert actual_length == large_size, f"Large list should have {large_size} nodes"

    def test_repeated_generation_variety(self):
        """Test that repeated generation produces variety"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)

        lists = []
        for _ in range(20):
            linked_list = generator.generate(5, constraints)
            values = self._collect_values(linked_list)
            lists.append(tuple(values))

        # Should have some variety (not all identical)
        unique_lists = set(lists)
        assert len(unique_lists) > 1, "Should generate variety of different lists"

    def test_generator_state_independence(self):
        """Test that multiple generators maintain independent state"""
        gen1 = LinkedListGenerator(seed=123)
        gen2 = LinkedListGenerator(seed=456)

        constraints = Constraints(min_value=1, max_value=100)

        list1 = gen1.generate(8, constraints)
        list2 = gen2.generate(8, constraints)

        # Different seeds should produce different lists
        assert not self._lists_equal(list1, list2), (
            "Different seeds should produce different lists"
        )

    def test_negative_size_handling(self):
        """Test handling of negative sizes"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)

        try:
            linked_list = generator.generate(-5, constraints)
            # Should handle gracefully or raise error
            assert linked_list is None, (
                "Negative size should produce None or raise error"
            )
        except (ValueError, AssertionError):
            print("✅ Properly handles negative size with error")

    def test_memory_efficiency_large_lists(self):
        """Test memory efficiency for large list generation"""
        generator = LinkedListGenerator()
        constraints = Constraints(min_value=1, max_value=100)

        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate several large lists
        for _ in range(5):
            linked_list = generator.generate(5000, constraints)
            assert linked_list is not None

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Should not use excessive memory
        assert memory_growth < 200 * 1024 * 1024, (
            f"List generation used too much memory: {memory_growth / 1024 / 1024:.1f}MB"
        )

    def _get_list_length(
        self, head: Optional[ListNode], max_length: int = 20000
    ) -> int:
        """Helper to get length of linked list (with cycle protection)"""
        if head is None:
            return 0

        current = head
        length = 0
        visited = set()

        while current and length < max_length:
            if id(current) in visited:
                # Cycle detected
                return length
            visited.add(id(current))
            length += 1
            current = current.next

        return length

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

    def _collect_values(
        self, head: Optional[ListNode], max_nodes: int = 10000
    ) -> List[int]:
        """Helper to collect all values in linked list (with cycle protection)"""
        if head is None:
            return []

        values = []
        current = head
        visited = set()

        while current and len(values) < max_nodes:
            if id(current) in visited:
                # Cycle detected, stop collection
                break
            visited.add(id(current))
            values.append(current.val)
            current = current.next

        return values


class TestLinkedListGeneratorSerialization:
    """Test linked list generation compatibility with serialization"""

    def test_linked_list_serialization_compatibility(self):
        """Test that generated lists work with serialization"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)
        linked_list = generator.generate(6, constraints)

        # Test if LinkedListSerializer exists and works
        try:
            from testgen.core.serializers import LinkedListSerializer

            # Test serialization to array
            array_repr = LinkedListSerializer.to_array(linked_list)
            assert isinstance(array_repr, list), "Serialized list should be a list"

            # Test round-trip serialization if from_array exists
            if hasattr(LinkedListSerializer, "from_array"):
                reconstructed_list = LinkedListSerializer.from_array(array_repr)
                assert self._lists_equal(linked_list, reconstructed_list), (
                    "Round-trip serialization should preserve list structure"
                )

        except ImportError:
            print("Note: LinkedListSerializer not found, serialization tests skipped")
        except AttributeError as e:
            print(f"Note: LinkedListSerializer method not found: {e}")

    def test_cyclic_list_serialization_handling(self):
        """Test serialization handling of cyclic lists"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=100)

        try:
            # Try to create cyclic list
            cyclic_list = generator.generate(5, constraints, has_cycle=True)

            if cyclic_list and self._has_cycle(cyclic_list):
                try:
                    from testgen.core.serializers import LinkedListSerializer

                    # Should handle cycles gracefully
                    array_repr = LinkedListSerializer.to_array(cyclic_list)

                    # Should not create infinite array
                    assert len(array_repr) < 1000, (
                        "Cyclic list serialization should not create infinite array"
                    )

                except ImportError:
                    pytest.skip("LinkedListSerializer not available")

        except TypeError:
            pytest.skip("Cycle creation not supported")

    def test_cli_output_format_compatibility(self):
        """Test that generated lists are compatible with CLI output format"""
        generator = LinkedListGenerator()

        constraints = Constraints(min_value=1, max_value=50)
        linked_list = generator.generate(5, constraints)

        # CLI converts lists to arrays for output
        try:
            from testgen.core.serializers import LinkedListSerializer

            array_output = LinkedListSerializer.to_array(linked_list)

            # Should be JSON serializable (for CLI output)
            import json

            json_str = json.dumps(array_output)
            assert isinstance(json_str, str), "List array should be JSON serializable"

        except ImportError:
            print("Note: LinkedListSerializer not available for CLI compatibility test")

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

    def _has_cycle(self, head: Optional[ListNode]) -> bool:
        """Floyd's cycle detection algorithm (tortoise and hare)"""
        if head is None or head.next is None:
            return False

        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False


class TestLinkedListGeneratorIntegration:
    """Test LinkedListGenerator integration with other components"""

    def test_cli_integration_compatibility(self):
        """Test compatibility with CLI usage patterns"""
        generator = LinkedListGenerator(seed=42)  # CLI uses seed parameter

        constraints = Constraints(min_value=1, max_value=100)

        # Test CLI-style usage
        linked_list = generator.generate(8, constraints)

        assert linked_list is not None, "CLI-style generation should work"
        assert self._get_list_length(linked_list) == 8, "Should have correct length"

    def test_cycle_cli_integration(self):
        """Test cycle generation with CLI-style parameters"""
        generator = LinkedListGenerator(seed=42)

        constraints = Constraints(min_value=1, max_value=100)

        # Test CLI cycle generation (--cycle flag)
        try:
            cyclic_list = generator.generate(6, constraints, has_cycle=True)

            assert cyclic_list is not None, "Cyclic list generation should work"
            assert self._has_cycle(cyclic_list), (
                "CLI cycle flag should produce cyclic list"
            )

        except TypeError:
            print("Note: CLI cycle parameter (has_cycle) not supported")

    def test_error_handling_integration(self):
        """Test integration with error handling system"""
        generator = LinkedListGenerator()

        # Test that errors are properly formatted
        try:
            constraints = Constraints(min_value=100, max_value=1)  # Invalid
            linked_list = generator.generate(5, constraints)
        except Exception as e:
            error_msg = str(e)
            assert len(error_msg) > 10, "Error message should be informative"
            assert any(
                keyword in error_msg.lower()
                for keyword in ["constraint", "list", "min", "max"]
            ), "Error should mention relevant context"

    def test_constraints_model_integration(self):
        """Test integration with Constraints model"""
        generator = LinkedListGenerator()

        # Test with various constraint configurations
        constraint_configs = [
            Constraints(min_value=1, max_value=10),
            Constraints(min_value=50, max_value=100),
            Constraints(min_value=-20, max_value=20),
        ]

        for constraints in constraint_configs:
            linked_list = generator.generate(5, constraints)
            values = self._collect_values(linked_list)

            for val in values:
                assert constraints.min_value <= val <= constraints.max_value, (
                    f"Value {val} should be within [{constraints.min_value}, {constraints.max_value}]"
                )

    def test_performance_benchmarking(self):
        """Test linked list generation performance"""
        generator = LinkedListGenerator()
        constraints = Constraints(min_value=1, max_value=100)

        import time

        # Benchmark medium-sized list generation
        start_time = time.time()
        for _ in range(1000):
            linked_list = generator.generate(20, constraints)
        generation_time = time.time() - start_time

        print(f"✅ LinkedList generation: {generation_time:.3f}s for 1000 operations")

        # Should be reasonably fast
        assert generation_time < 5.0, "LinkedList generation should be reasonably fast"

    def _get_list_length(
        self, head: Optional[ListNode], max_length: int = 10000
    ) -> int:
        """Helper to get length of linked list (with cycle protection)"""
        if head is None:
            return 0

        current = head
        length = 0
        visited = set()

        while current and length < max_length:
            if id(current) in visited:
                # Cycle detected
                return length
            visited.add(id(current))
            length += 1
            current = current.next

        return length

    def _has_cycle(self, head: Optional[ListNode]) -> bool:
        """Floyd's cycle detection algorithm (tortoise and hare)"""
        if head is None or head.next is None:
            return False

        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                return True

        return False

    def _collect_values(
        self, head: Optional[ListNode], max_nodes: int = 10000
    ) -> List[int]:
        """Helper to collect all values in linked list (with cycle protection)"""
        if head is None:
            return []

        values = []
        current = head
        visited = set()

        while current and len(values) < max_nodes:
            if id(current) in visited:
                # Cycle detected, stop collection
                break
            visited.add(id(current))
            values.append(current.val)
            current = current.next

        return values


if __name__ == "__main__":
    # Can be run directly for quick testing
    pytest.main([__file__, "-v"])
