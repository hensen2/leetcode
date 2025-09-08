"""
Main facade class that provides simple interface to all functionality
This replaces the current test_case_generator.py
"""


class TestCaseGenerator:
    """
    Main interface for test case generation - simplified and focused
    """

    def __init__(self, seed=None):
        """Initialize with optional seed for reproducibility"""
        from .core.generators import (
            GraphGenerator,
            IntegerGenerator,
            LinkedListGenerator,
            MatrixGenerator,
            StringGenerator,
            TreeGenerator,
        )
        from .execution.runner import EnhancedTestRunner
        from .patterns.edge_cases import EdgeCaseGenerator

        # Initialize generators
        self.int_gen = IntegerGenerator(seed)
        self.str_gen = StringGenerator(seed)
        self.tree_gen = TreeGenerator(seed)
        self.graph_gen = GraphGenerator(seed)
        self.matrix_gen = MatrixGenerator(seed)
        self.list_gen = LinkedListGenerator(seed)
        self.edge_gen = EdgeCaseGenerator()

        # Initialize runner
        self.runner = EnhancedTestRunner()

        self.seed = seed

    # Simple generation methods
    def generate_array(self, size=None, **kwargs):
        """Generate integer array with optional constraints"""
        from .core.models import Constraints

        constraints = Constraints(**kwargs) if kwargs else None
        return self.int_gen.generate_array(size, constraints)

    def generate_string(self, length=None, **kwargs):
        """Generate string with optional constraints"""
        from .core.models import Constraints

        constraints = Constraints(**kwargs) if kwargs else None
        return self.str_gen.generate(length, constraints)

    def generate_tree(self, size=10, **kwargs):
        """Generate binary tree"""
        from .core.models import Constraints, TreeProperties

        props = TreeProperties(
            size=size,
            **{k: v for k, v in kwargs.items() if k in ["balanced", "bst", "complete"]},
        )
        constraints = Constraints(
            **{k: v for k, v in kwargs.items() if k in ["min_value", "max_value"]}
        )
        return self.tree_gen.generate(props, constraints)

    def generate_graph(self, num_nodes=10, **kwargs):
        """Generate graph"""
        from .core.models import GraphProperties

        props = GraphProperties(num_nodes=num_nodes, **kwargs)
        return self.graph_gen.generate(props)

    def get_edge_cases(self, problem_type="array"):
        """Get edge cases for problem type"""
        return self.edge_gen.get_edge_cases(problem_type)

    # Test execution
    def test_function(self, func, test_cases, expected_outputs=None):
        """Test a function with generated test cases"""
        return self.runner.run_test_suite(func, test_cases, expected_outputs)
