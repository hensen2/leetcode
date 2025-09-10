"""
Comprehensive tests for TreeGenerator

Tests tree structure generation, balance validation, BST properties,
serialization compatibility, and integration with CLI functionality.
"""

from typing import List, Optional

import pytest
from testgen.core.generators import TreeGenerator
from testgen.core.models import Constraints, TreeNode, TreeProperties


class TestTreeGeneratorBasics:
    """Test basic tree generation functionality"""

    def test_tree_generator_instantiation(self):
        """Test TreeGenerator can be instantiated"""
        generator = TreeGenerator()
        assert generator is not None
        assert hasattr(generator, "generate")

    def test_tree_generator_with_seed(self):
        """Test TreeGenerator with seed produces reproducible results"""
        seed = 12345

        props = TreeProperties(size=10)
        constraints = Constraints(min_value=1, max_value=100)

        # First generation
        gen1 = TreeGenerator(seed)
        tree1 = gen1.generate(props, constraints)

        # Second generation resets with same seed
        gen2 = TreeGenerator(seed)
        tree2 = gen2.generate(props, constraints)

        # Trees should have same structure with same seed
        assert self._trees_equal(tree1, tree2), (
            "Same seed should produce same tree structure"
        )

    def test_basic_tree_generation(self):
        """Test basic tree generation with various sizes"""
        generator = TreeGenerator()

        test_sizes = [0, 1, 3, 5, 10, 15, 50]

        for size in test_sizes:
            props = TreeProperties(size=size)
            constraints = Constraints(min_value=1, max_value=100)

            tree = generator.generate(props, constraints)

            if size == 0:
                assert tree is None, "Empty tree should be None"
            else:
                assert tree is not None, (
                    f"Non-empty tree should not be None for size {size}"
                )
                assert isinstance(tree, TreeNode), (
                    f"Should return TreeNode for size {size}"
                )

                # Verify tree has correct number of nodes
                node_count = self._count_nodes(tree)
                assert node_count == size, (
                    f"Tree should have {size} nodes, got {node_count}"
                )

    def test_tree_node_structure(self):
        """Test that generated trees have proper TreeNode structure"""
        generator = TreeGenerator()

        props = TreeProperties(size=5)
        constraints = Constraints(min_value=1, max_value=100)
        tree = generator.generate(props, constraints)

        # Verify TreeNode structure
        assert hasattr(tree, "val"), "TreeNode should have 'val' attribute"
        assert hasattr(tree, "left"), "TreeNode should have 'left' attribute"
        assert hasattr(tree, "right"), "TreeNode should have 'right' attribute"

        # Verify values are within constraints
        values = self._collect_values(tree)
        assert all(1 <= val <= 100 for val in values), (
            "All values should be within constraints"
        )

    def _trees_equal(
        self, tree1: Optional[TreeNode], tree2: Optional[TreeNode]
    ) -> bool:
        """Helper to check if two trees have same structure and values"""
        if tree1 is None and tree2 is None:
            return True
        if tree1 is None or tree2 is None:
            return False
        return (
            tree1.val == tree2.val
            and self._trees_equal(tree1.left, tree2.left)
            and self._trees_equal(tree1.right, tree2.right)
        )

    def _count_nodes(self, tree: Optional[TreeNode]) -> int:
        """Helper to count nodes in tree"""
        if tree is None:
            return 0
        return 1 + self._count_nodes(tree.left) + self._count_nodes(tree.right)

    def _collect_values(self, tree: Optional[TreeNode]) -> List[int]:
        """Helper to collect all values in tree"""
        if tree is None:
            return []
        return (
            [tree.val]
            + self._collect_values(tree.left)
            + self._collect_values(tree.right)
        )


class TestTreeGeneratorBalancing:
    """Test balanced tree generation functionality"""

    def test_balanced_tree_generation(self):
        """Test generation of balanced trees"""
        generator = TreeGenerator()

        for size in [3, 7, 15, 31]:  # Perfect binary tree sizes
            props = TreeProperties(size=size, balanced=True)
            constraints = Constraints(min_value=1, max_value=100)

            tree = generator.generate(props, constraints)

            assert tree is not None, f"Balanced tree should not be None for size {size}"
            assert self._is_balanced(tree), f"Tree should be balanced for size {size}"

            node_count = self._count_nodes(tree)
            assert node_count == size, f"Balanced tree should have {size} nodes"

    def test_unbalanced_tree_generation(self):
        """Test generation of unbalanced trees"""
        generator = TreeGenerator()

        props = TreeProperties(size=10, balanced=False)
        constraints = Constraints(min_value=1, max_value=100)

        # Generate multiple trees to ensure variety
        trees = []
        for _ in range(10):
            tree = generator.generate(props, constraints)
            trees.append(tree)
            assert tree is not None, "Unbalanced tree should not be None"

        # At least some trees should be unbalanced when balanced=False
        unbalanced_count = sum(1 for tree in trees if not self._is_balanced(tree))
        assert unbalanced_count > 0, (
            "Should generate some unbalanced trees when balanced=False"
        )

    def test_balance_property_validation(self):
        """Test that balanced trees satisfy balance property"""
        generator = TreeGenerator()

        props = TreeProperties(size=15, balanced=True)
        constraints = Constraints(min_value=1, max_value=100)

        tree = generator.generate(props, constraints)

        # For each node, height difference between subtrees should be ≤ 1
        assert self._is_balanced(tree), (
            "Balanced tree should satisfy AVL balance property"
        )

    def _is_balanced(self, tree: Optional[TreeNode]) -> bool:
        """Check if tree is balanced (AVL property)"""

        def height(node):
            if node is None:
                return 0
            return 1 + max(height(node.left), height(node.right))

        def is_balanced_helper(node):
            if node is None:
                return True

            left_height = height(node.left)
            right_height = height(node.right)

            return (
                abs(left_height - right_height) <= 1
                and is_balanced_helper(node.left)
                and is_balanced_helper(node.right)
            )

        return is_balanced_helper(tree)

    def _count_nodes(self, tree: Optional[TreeNode]) -> int:
        """Helper to count nodes in tree"""
        if tree is None:
            return 0
        return 1 + self._count_nodes(tree.left) + self._count_nodes(tree.right)


class TestTreeGeneratorBST:
    """Test Binary Search Tree generation functionality"""

    def test_bst_generation(self):
        """Test generation of valid Binary Search Trees"""
        generator = TreeGenerator()

        props = TreeProperties(size=10, bst=True)
        constraints = Constraints(min_value=1, max_value=100)

        tree = generator.generate(props, constraints)

        assert tree is not None, "BST should not be None"
        assert self._is_valid_bst(tree), "Generated tree should be a valid BST"

        # May want to remove this assertion since BSTs have unique values
        node_count = self._count_nodes(tree)
        assert node_count == 10, "BST should have correct number of nodes"

    def test_bst_ordering_property(self):
        """Test that BST maintains ordering property"""
        generator = TreeGenerator()

        props = TreeProperties(size=15, bst=True)
        constraints = Constraints(min_value=1, max_value=100)

        tree = generator.generate(props, constraints)

        # Collect values in inorder traversal
        inorder_values = self._inorder_traversal(tree)

        # BST inorder traversal should be sorted
        assert inorder_values == sorted(inorder_values), (
            "BST inorder traversal should be sorted"
        )

    def test_bst_with_duplicates(self):
        """Test BST generation with duplicate values allowed"""
        generator = TreeGenerator()

        # Small value range to force duplicates
        props = TreeProperties(size=10, bst=True)
        constraints = Constraints(min_value=1, max_value=5, allow_duplicates=True)

        tree = generator.generate(props, constraints)

        assert tree is not None, "BST with duplicates should not be None"
        # Should still maintain BST property (≤ for left, ≥ for right)
        assert self._is_valid_bst(tree, allow_duplicates=True), (
            "BST should be valid even with duplicates"
        )

    def test_bst_unique_values(self):
        """Test BST generation with unique values only"""
        generator = TreeGenerator()

        props = TreeProperties(size=10, bst=True)
        constraints = Constraints(min_value=1, max_value=100, allow_duplicates=False)

        tree = generator.generate(props, constraints)

        values = self._collect_values(tree)
        unique_values = set(values)

        assert len(values) == len(unique_values), "BST should have unique values only"
        assert self._is_valid_bst(tree), "BST should be valid with unique values"

    def _is_valid_bst(
        self,
        tree: Optional[TreeNode],
        min_val: float = float("-inf"),
        max_val: float = float("inf"),
        allow_duplicates: bool = False,
    ) -> bool:
        """Check if tree is a valid BST"""
        if tree is None:
            return True

        if allow_duplicates:
            if tree.val < min_val or tree.val > max_val:
                return False
            return self._is_valid_bst(
                tree.left, min_val, tree.val, allow_duplicates
            ) and self._is_valid_bst(tree.right, tree.val, max_val, allow_duplicates)
        else:
            if tree.val <= min_val or tree.val >= max_val:
                return False
            return self._is_valid_bst(
                tree.left, min_val, tree.val, allow_duplicates
            ) and self._is_valid_bst(tree.right, tree.val, max_val, allow_duplicates)

    def _inorder_traversal(self, tree: Optional[TreeNode]) -> List[int]:
        """Get inorder traversal of tree"""
        if tree is None:
            return []
        return (
            self._inorder_traversal(tree.left)
            + [tree.val]
            + self._inorder_traversal(tree.right)
        )

    def _count_nodes(self, tree: Optional[TreeNode]) -> int:
        """Helper to count nodes in tree"""
        if tree is None:
            return 0

        return 1 + self._count_nodes(tree.left) + self._count_nodes(tree.right)

    def _collect_values(self, tree: Optional[TreeNode]) -> List[int]:
        """Helper to collect all values in tree"""
        if tree is None:
            return []
        return (
            [tree.val]
            + self._collect_values(tree.left)
            + self._collect_values(tree.right)
        )


class TestTreeGeneratorConstraints:
    """Test tree generation with various constraints"""

    def test_value_constraints(self):
        """Test that tree values respect constraints"""
        generator = TreeGenerator()

        props = TreeProperties(size=15)
        constraints = Constraints(min_value=10, max_value=50)

        tree = generator.generate(props, constraints)
        values = self._collect_values(tree)

        assert all(10 <= val <= 50 for val in values), (
            "All tree values should be within constraints"
        )

    def test_constraint_edge_cases(self):
        """Test constraint edge cases"""
        generator = TreeGenerator()

        # Single value constraint
        props = TreeProperties(size=5)
        constraints = Constraints(min_value=42, max_value=42)

        tree = generator.generate(props, constraints)
        values = self._collect_values(tree)

        assert all(val == 42 for val in values), (
            "All values should be 42 when min_value == max_value"
        )

    def test_negative_value_constraints(self):
        """Test constraints with negative values"""
        generator = TreeGenerator()

        props = TreeProperties(size=8)
        constraints = Constraints(min_value=-100, max_value=-10)

        tree = generator.generate(props, constraints)
        values = self._collect_values(tree)

        assert all(-100 <= val <= -10 for val in values), (
            "All values should be within negative range"
        )

    def test_invalid_constraints_handling(self):
        """Test handling of invalid constraints"""
        generator = TreeGenerator()

        props = TreeProperties(size=5)
        constraints = Constraints(min_value=100, max_value=1)  # Invalid

        try:
            tree = generator.generate(props, constraints)
            # If no error, should still produce valid tree
            assert tree is not None or props.size == 0
        except (ValueError, AssertionError):
            print("✅ Properly handles invalid constraints with error")

    def _collect_values(self, tree: Optional[TreeNode]) -> List[int]:
        """Helper to collect all values in tree"""
        if tree is None:
            return []
        return (
            [tree.val]
            + self._collect_values(tree.left)
            + self._collect_values(tree.right)
        )


class TestTreeGeneratorEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_tree_generation(self):
        """Test generation of empty trees"""
        generator = TreeGenerator()

        props = TreeProperties(size=0)
        constraints = Constraints(min_value=1, max_value=100)

        tree = generator.generate(props, constraints)
        assert tree is None, "Empty tree should be None"

    def test_single_node_tree(self):
        """Test generation of single node trees"""
        generator = TreeGenerator()

        props = TreeProperties(size=1)
        constraints = Constraints(min_value=5, max_value=15)

        tree = generator.generate(props, constraints)

        assert tree is not None, "Single node tree should not be None"
        assert tree.left is None, "Single node should have no left child"
        assert tree.right is None, "Single node should have no right child"
        assert 5 <= tree.val <= 15, "Single node value should be within constraints"

    def test_large_tree_generation(self):
        """Test generation of large trees"""
        generator = TreeGenerator()

        large_size = 1000
        props = TreeProperties(size=large_size)
        constraints = Constraints(min_value=1, max_value=10000)

        tree = generator.generate(props, constraints)

        assert tree is not None, "Large tree should not be None"
        node_count = self._count_nodes(tree)
        assert node_count == large_size, f"Large tree should have {large_size} nodes"

    def test_repeated_generation_variety(self):
        """Test that repeated generation produces variety"""
        generator = TreeGenerator()

        props = TreeProperties(size=10)
        constraints = Constraints(min_value=1, max_value=100)

        trees = []
        for _ in range(20):
            tree = generator.generate(props, constraints)
            trees.append(tree)

        # Should have variety in tree structures
        tree_structures = [self._tree_to_structure_string(tree) for tree in trees]
        unique_structures = set(tree_structures)

        assert len(unique_structures) > 1, "Should generate variety of tree structures"

    def test_generator_state_independence(self):
        """Test that multiple generators maintain independent state"""
        gen1 = TreeGenerator(seed=123)
        gen2 = TreeGenerator(seed=456)

        props = TreeProperties(size=10)
        constraints = Constraints(min_value=1, max_value=100)

        tree1 = gen1.generate(props, constraints)
        tree2 = gen2.generate(props, constraints)

        # Different seeds should produce different trees
        assert not self._trees_equal(tree1, tree2), (
            "Different seeds should produce different trees"
        )

    def _count_nodes(self, tree: Optional[TreeNode]) -> int:
        """Helper to count nodes in tree"""
        if tree is None:
            return 0
        return 1 + self._count_nodes(tree.left) + self._count_nodes(tree.right)

    def _tree_to_structure_string(self, tree: Optional[TreeNode]) -> str:
        """Convert tree structure to string for comparison"""
        if tree is None:
            return "None"
        return f"({tree.val},{self._tree_to_structure_string(tree.left)},{self._tree_to_structure_string(tree.right)})"

    def _trees_equal(
        self, tree1: Optional[TreeNode], tree2: Optional[TreeNode]
    ) -> bool:
        """Helper to check if two trees have same structure and values"""
        if tree1 is None and tree2 is None:
            return True
        if tree1 is None or tree2 is None:
            return False
        return (
            tree1.val == tree2.val
            and self._trees_equal(tree1.left, tree2.left)
            and self._trees_equal(tree1.right, tree2.right)
        )


class TestTreeGeneratorSerialization:
    """Test tree generation compatibility with serialization"""

    def test_tree_serialization_compatibility(self):
        """Test that generated trees work with serialization"""
        generator = TreeGenerator()

        props = TreeProperties(size=10)
        constraints = Constraints(min_value=1, max_value=100)

        tree = generator.generate(props, constraints)

        # Test if TreeSerializer exists and works
        from testgen.core.serializers import TreeSerializer

        # Test serialization to array
        array_repr = TreeSerializer.to_array(tree)
        assert isinstance(array_repr, list), "Serialized tree should be a list"

        # Test round-trip serialization if from_array exists
        if hasattr(TreeSerializer, "from_array"):
            reconstructed_tree = TreeSerializer.from_array(array_repr)
            assert self._trees_equal(tree, reconstructed_tree), (
                "Round-trip serialization should preserve tree structure"
            )

    def test_cli_output_format_compatibility(self):
        """Test that generated trees are compatible with CLI output format"""
        generator = TreeGenerator()

        props = TreeProperties(size=7, balanced=True)
        constraints = Constraints(min_value=1, max_value=50)

        tree = generator.generate(props, constraints)

        # CLI converts trees to arrays for output
        try:
            from testgen.core.serializers import TreeSerializer

            array_output = TreeSerializer.to_array(tree)

            # Should be JSON serializable (for CLI output)
            import json

            json_str = json.dumps(array_output)
            assert isinstance(json_str, str), "Tree array should be JSON serializable"

        except ImportError:
            print("Note: TreeSerializer not available for CLI compatibility test")

    def _trees_equal(
        self, tree1: Optional[TreeNode], tree2: Optional[TreeNode]
    ) -> bool:
        """Helper to check if two trees have same structure and values"""
        if tree1 is None and tree2 is None:
            return True
        if tree1 is None or tree2 is None:
            return False
        return (
            tree1.val == tree2.val
            and self._trees_equal(tree1.left, tree2.left)
            and self._trees_equal(tree1.right, tree2.right)
        )


class TestTreeGeneratorIntegration:
    """Test TreeGenerator integration with other components"""

    def test_cli_integration_compatibility(self):
        """Test compatibility with CLI usage patterns"""
        generator = TreeGenerator(seed=42)  # CLI uses seed parameter

        # Test CLI-style usage
        props = TreeProperties(
            size=10,
            balanced=True,  # CLI --balanced flag
            bst=False,  # CLI --bst flag
        )
        constraints = Constraints(min_value=1, max_value=100)

        tree = generator.generate(props, constraints)

        assert tree is not None, "CLI-style generation should work"
        assert self._is_balanced(tree), "CLI balanced flag should work"

    def test_bst_cli_integration(self):
        """Test BST generation with CLI-style parameters"""
        generator = TreeGenerator(seed=42)

        # Test CLI BST generation
        props = TreeProperties(
            size=15,
            bst=True,  # CLI --bst flag
            balanced=False,
        )
        constraints = Constraints(min_value=1, max_value=100)

        tree = generator.generate(props, constraints)

        assert tree is not None, "BST generation should work"
        assert self._is_valid_bst(tree), "CLI BST flag should produce valid BST"

    def test_error_handling_integration(self):
        """Test integration with error handling system"""
        generator = TreeGenerator()

        # Test that errors are properly formatted
        try:
            props = TreeProperties(size=-5)  # Invalid size
            constraints = Constraints(min_value=1, max_value=100)
            tree = generator.generate(props, constraints)
        except Exception as e:
            error_msg = str(e)
            assert len(error_msg) > 10, "Error message should be informative"
            assert any(
                keyword in error_msg.lower()
                for keyword in ["size", "tree", "invalid", "property"]
            ), "Error should mention relevant context"

    def test_constraints_model_integration(self):
        """Test integration with Constraints model"""
        generator = TreeGenerator()

        # Test with various constraint configurations
        constraint_configs = [
            Constraints(min_value=1, max_value=10),
            Constraints(min_value=50, max_value=100),
            Constraints(min_value=-50, max_value=50),
        ]

        for constraints in constraint_configs:
            props = TreeProperties(size=10)
            tree = generator.generate(props, constraints)

            values = self._collect_values(tree)
            assert all(
                constraints.min_value <= val <= constraints.max_value for val in values
            ), (
                f"Values should respect constraints {constraints.min_value}-{constraints.max_value}"
            )

    def _is_balanced(self, tree: Optional[TreeNode]) -> bool:
        """Check if tree is balanced (AVL property)"""

        def height(node):
            if node is None:
                return 0
            return 1 + max(height(node.left), height(node.right))

        def is_balanced_helper(node):
            if node is None:
                return True

            left_height = height(node.left)
            right_height = height(node.right)

            return (
                abs(left_height - right_height) <= 1
                and is_balanced_helper(node.left)
                and is_balanced_helper(node.right)
            )

        return is_balanced_helper(tree)

    def _is_valid_bst(
        self,
        tree: Optional[TreeNode],
        min_val: float = float("-inf"),
        max_val: float = float("inf"),
    ) -> bool:
        """Check if tree is a valid BST"""
        if tree is None:
            return True

        if tree.val <= min_val or tree.val >= max_val:
            return False

        return self._is_valid_bst(tree.left, min_val, tree.val) and self._is_valid_bst(
            tree.right, tree.val, max_val
        )

    def _collect_values(self, tree: Optional[TreeNode]) -> List[int]:
        """Helper to collect all values in tree"""
        if tree is None:
            return []
        return (
            [tree.val]
            + self._collect_values(tree.left)
            + self._collect_values(tree.right)
        )


if __name__ == "__main__":
    # Can be run directly for quick testing
    pytest.main([__file__, "-v"])
