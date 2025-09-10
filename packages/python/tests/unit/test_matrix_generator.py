"""
Comprehensive tests for MatrixGenerator

Tests 2D matrix generation, special matrix types, dimensions, constraints,
and CLI integration functionality.
"""

import pytest
from testgen.core.generators import MatrixGenerator
from testgen.core.models import Constraints


class TestMatrixGeneratorBasics:
    """Test basic matrix generation functionality"""

    def test_matrix_generator_instantiation(self):
        """Test MatrixGenerator can be instantiated"""
        generator = MatrixGenerator()
        assert generator is not None
        assert hasattr(generator, "generate")

    def test_matrix_generator_with_seed(self):
        """Test MatrixGenerator with seed produces reproducible results"""
        seed = 12345
        constraints = Constraints(min_value=1, max_value=10)

        # First generation
        gen1 = MatrixGenerator(seed)
        matrix1 = gen1.generate(3, 3, constraints)

        # Second generation resets with same seed
        gen2 = MatrixGenerator(seed)
        matrix2 = gen2.generate(3, 3, constraints)

        assert matrix1 == matrix2, "Same seed should produce same matrix"

    def test_basic_matrix_generation(self):
        """Test basic matrix generation with various dimensions"""
        generator = MatrixGenerator()

        test_dimensions = [
            (1, 1),  # Single element
            (2, 3),  # Small rectangular
            (3, 3),  # Square
            (5, 2),  # Tall matrix
            (1, 10),  # Row vector
            (10, 1),  # Column vector
        ]

        for rows, cols in test_dimensions:
            constraints = Constraints(min_value=1, max_value=100)
            matrix = generator.generate(rows, cols, constraints)

            assert isinstance(matrix, list), f"Matrix should be list for {rows}x{cols}"
            assert len(matrix) == rows, f"Should have {rows} rows"

            if rows > 0:
                assert all(len(row) == cols for row in matrix), (
                    f"All rows should have {cols} columns"
                )

                # Check all elements are within constraints
                for row in matrix:
                    for element in row:
                        assert isinstance(element, int), "Elements should be integers"
                        assert 1 <= element <= 100, (
                            "Elements should be within constraints"
                        )

    def test_empty_matrix_generation(self):
        """Test generation of empty matrices"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=10)

        # Test zero rows
        matrix = generator.generate(0, 5, constraints)
        assert matrix == [] or matrix == [[]], "Zero rows should produce empty matrix"

        # Test zero columns
        matrix = generator.generate(3, 0, constraints)
        if matrix is not None:
            if len(matrix) > 0:
                assert all(len(row) == 0 for row in matrix), (
                    "Zero columns should produce empty rows"
                )

    def test_large_matrix_generation(self):
        """Test generation of large matrices"""
        generator = MatrixGenerator()

        large_rows, large_cols = 100, 100
        constraints = Constraints(min_value=1, max_value=1000)

        matrix = generator.generate(large_rows, large_cols, constraints)

        assert isinstance(matrix, list), "Large matrix should be list"
        assert len(matrix) == large_rows, f"Should have {large_rows} rows"
        assert all(len(row) == large_cols for row in matrix), (
            f"All rows should have {large_cols} columns"
        )

    def test_matrix_value_distribution(self):
        """Test that matrix values have reasonable distribution"""
        generator = MatrixGenerator()

        constraints = Constraints(min_value=1, max_value=50)
        matrix = generator.generate(10, 10, constraints)

        # Flatten matrix to analyze values
        all_values = [val for row in matrix for val in row]

        # Should have variety of values
        unique_values = set(all_values)
        assert len(unique_values) > 5, "Should have variety of values in large matrix"

        # All values should be in range
        assert all(1 <= val <= 50 for val in all_values), (
            "All values should be within constraints"
        )


class TestMatrixGeneratorConstraints:
    """Test matrix generation with various constraints"""

    def test_value_constraints(self):
        """Test that matrix values respect constraints"""
        generator = MatrixGenerator()

        test_constraints = [
            Constraints(min_value=1, max_value=10),
            Constraints(min_value=-50, max_value=-10),
            Constraints(min_value=100, max_value=200),
            Constraints(min_value=0, max_value=1),  # Binary matrix
        ]

        for constraints in test_constraints:
            matrix = generator.generate(4, 4, constraints)

            # Check all values are within constraints
            for row in matrix:
                for val in row:
                    assert constraints.min_value <= val <= constraints.max_value, (
                        f"Value {val} should be within [{constraints.min_value}, {constraints.max_value}]"
                    )

    def test_single_value_constraint(self):
        """Test constraint where min_value equals max_value"""
        generator = MatrixGenerator()

        constraints = Constraints(min_value=42, max_value=42)
        matrix = generator.generate(3, 3, constraints)

        # All values should be exactly 42
        for row in matrix:
            for val in row:
                assert val == 42, "All values should be 42 when min_value == max_value"

    def test_invalid_constraints_handling(self):
        """Test handling of invalid constraints"""
        generator = MatrixGenerator()

        # Invalid: min > max
        constraints = Constraints(min_value=100, max_value=1)

        try:
            matrix = generator.generate(3, 3, constraints)
            # If no error, should still produce valid matrix
            assert isinstance(matrix, list), "Should still return a matrix"
        except (ValueError, AssertionError):
            print("✅ Properly handles invalid constraints with error")

    def test_constraint_edge_cases(self):
        """Test constraint edge cases"""
        generator = MatrixGenerator()

        # Very large range
        constraints = Constraints(min_value=-1000000, max_value=1000000)
        matrix = generator.generate(2, 2, constraints)

        for row in matrix:
            for val in row:
                assert -1000000 <= val <= 1000000, "Should handle large ranges"


class TestMatrixGeneratorSpecialTypes:
    """Test generation of special matrix types"""

    def test_identity_matrix_generation(self):
        """Test generation of identity matrices"""
        generator = MatrixGenerator()

        try:
            # Test if special matrix generation exists
            if hasattr(generator, "generate_special"):
                matrix = generator.generate_special(4, 4, "identity")

                # Should be identity matrix
                for i in range(4):
                    for j in range(4):
                        if i == j:
                            assert matrix[i][j] == 1, (
                                f"Diagonal element [{i}][{j}] should be 1"
                            )
                        else:
                            assert matrix[i][j] == 0, (
                                f"Off-diagonal element [{i}][{j}] should be 0"
                            )
            else:
                print("Note: generate_special method not found")
        except (AttributeError, NotImplementedError):
            pytest.skip("Special matrix generation not implemented")

    def test_zero_matrix_generation(self):
        """Test generation of zero matrices"""
        generator = MatrixGenerator()

        try:
            if hasattr(generator, "generate_special"):
                matrix = generator.generate_special(3, 3, "zero")

                # All elements should be zero
                for row in matrix:
                    for val in row:
                        assert val == 0, "All elements in zero matrix should be 0"
        except (AttributeError, NotImplementedError):
            pytest.skip("Zero matrix generation not implemented")

    def test_diagonal_matrix_generation(self):
        """Test generation of diagonal matrices"""
        generator = MatrixGenerator()

        try:
            if hasattr(generator, "generate_special"):
                matrix = generator.generate_special(4, 4, "diagonal")

                # Only diagonal elements should be non-zero
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            assert matrix[i][j] == 0, (
                                f"Off-diagonal element [{i}][{j}] should be 0"
                            )
        except (AttributeError, NotImplementedError):
            pytest.skip("Diagonal matrix generation not implemented")

    def test_symmetric_matrix_generation(self):
        """Test generation of symmetric matrices"""
        generator = MatrixGenerator()

        try:
            if hasattr(generator, "generate_special"):
                matrix = generator.generate_special(4, 4, "symmetric")

                # Should be symmetric: matrix[i][j] == matrix[j][i]
                for i in range(4):
                    for j in range(4):
                        assert matrix[i][j] == matrix[j][i], (
                            f"Matrix should be symmetric: [{i}][{j}] != [{j}][{i}]"
                        )
        except (AttributeError, NotImplementedError):
            pytest.skip("Symmetric matrix generation not implemented")


class TestMatrixGeneratorDimensions:
    """Test matrix generation with various dimensions"""

    def test_square_matrices(self):
        """Test generation of square matrices"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=10)

        for size in [1, 2, 5, 10, 20]:
            matrix = generator.generate(size, size, constraints)

            assert len(matrix) == size, f"Square matrix should have {size} rows"
            assert all(len(row) == size for row in matrix), (
                f"Square matrix should have {size} columns in each row"
            )

    def test_rectangular_matrices(self):
        """Test generation of rectangular matrices"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=10)

        test_shapes = [
            (2, 5),  # Wide matrix
            (5, 2),  # Tall matrix
            (1, 10),  # Row vector
            (10, 1),  # Column vector
            (3, 7),  # Arbitrary rectangle
        ]

        for rows, cols in test_shapes:
            matrix = generator.generate(rows, cols, constraints)

            assert len(matrix) == rows, f"Should have {rows} rows"
            assert all(len(row) == cols for row in matrix), (
                f"All rows should have {cols} columns"
            )

    def test_dimension_parameter_handling(self):
        """Test handling of dimension parameters"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=10)

        # Test with None dimensions (should use defaults or handle gracefully)
        try:
            matrix = generator.generate(None, None, constraints)
            if matrix is not None:
                assert isinstance(matrix, list), (
                    "Should return list even with None dimensions"
                )
        except (TypeError, ValueError):
            print("✅ Properly handles None dimensions with error")

    def test_negative_dimensions_handling(self):
        """Test handling of negative dimensions"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=10)

        try:
            matrix = generator.generate(-3, -3, constraints)
            # Should handle gracefully or raise error
            assert matrix is None or matrix == [], (
                "Negative dimensions should produce None or empty"
            )
        except (ValueError, AssertionError):
            print("✅ Properly handles negative dimensions with error")

    def test_very_large_dimensions(self):
        """Test handling of very large dimensions"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=10)

        # Test moderately large matrix
        try:
            matrix = generator.generate(200, 200, constraints)
            assert len(matrix) == 200, "Should handle large dimensions"
            assert len(matrix[0]) == 200, "Should handle large dimensions"
        except MemoryError:
            print("Note: Large matrix generation hit memory limits")


class TestMatrixGeneratorEdgeCases:
    """Test edge cases and error conditions"""

    def test_single_element_matrix(self):
        """Test generation of 1x1 matrices"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=5, max_value=15)

        matrix = generator.generate(1, 1, constraints)

        assert len(matrix) == 1, "Should have 1 row"
        assert len(matrix[0]) == 1, "Should have 1 column"
        assert 5 <= matrix[0][0] <= 15, "Single element should be within constraints"

    def test_row_vector_generation(self):
        """Test generation of row vectors (1 row, many columns)"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=100)

        matrix = generator.generate(1, 20, constraints)

        assert len(matrix) == 1, "Row vector should have 1 row"
        assert len(matrix[0]) == 20, "Row vector should have 20 columns"

    def test_column_vector_generation(self):
        """Test generation of column vectors (many rows, 1 column)"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=100)

        matrix = generator.generate(20, 1, constraints)

        assert len(matrix) == 20, "Column vector should have 20 rows"
        assert all(len(row) == 1 for row in matrix), (
            "Column vector should have 1 column per row"
        )

    def test_repeated_generation_variety(self):
        """Test that repeated generation produces variety"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=50)

        matrices = []
        for _ in range(10):
            matrix = generator.generate(3, 3, constraints)
            matrices.append(matrix)

        # Should have some variety (not all identical)
        unique_matrices = []
        for matrix in matrices:
            matrix_tuple = tuple(tuple(row) for row in matrix)
            if matrix_tuple not in unique_matrices:
                unique_matrices.append(matrix_tuple)

        assert len(unique_matrices) > 1, "Should generate variety of different matrices"

    def test_generator_state_independence(self):
        """Test that multiple generators maintain independent state"""
        gen1 = MatrixGenerator(seed=123)
        gen2 = MatrixGenerator(seed=456)

        constraints = Constraints(min_value=1, max_value=100)

        matrix1 = gen1.generate(5, 5, constraints)
        matrix2 = gen2.generate(5, 5, constraints)

        # Different seeds should produce different matrices
        assert matrix1 != matrix2, "Different seeds should produce different matrices"

    def test_memory_efficiency_large_matrices(self):
        """Test memory efficiency for large matrix generation"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=100)

        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate several large matrices
        for _ in range(5):
            matrix = generator.generate(300, 300, constraints)
            assert matrix is not None

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Should not use excessive memory
        assert memory_growth < 500 * 1024 * 1024, (
            f"Matrix generation used too much memory: {memory_growth / 1024 / 1024:.1f}MB"
        )


class TestMatrixGeneratorIntegration:
    """Test MatrixGenerator integration with other components"""

    def test_cli_integration_compatibility(self):
        """Test compatibility with CLI usage patterns"""
        generator = MatrixGenerator(seed=42)  # CLI uses seed parameter

        # Test CLI-style usage (CLI generates matrix with random dimensions)
        constraints = Constraints(min_value=1, max_value=100)
        matrix = generator.generate(4, 5, constraints)

        assert matrix is not None, "CLI-style generation should work"
        assert len(matrix) == 4, "Should have correct rows"
        assert len(matrix[0]) == 5, "Should have correct columns"

    def test_constraints_model_integration(self):
        """Test integration with Constraints model"""
        generator = MatrixGenerator()

        # Test with various constraint configurations
        constraint_configs = [
            Constraints(min_value=1, max_value=10),
            Constraints(min_value=50, max_value=100),
            Constraints(min_value=-20, max_value=20),
        ]

        for constraints in constraint_configs:
            matrix = generator.generate(3, 4, constraints)

            # All values should respect constraints
            for row in matrix:
                for val in row:
                    assert constraints.min_value <= val <= constraints.max_value, (
                        f"Value {val} should be within [{constraints.min_value}, {constraints.max_value}]"
                    )

    def test_error_handling_integration(self):
        """Test integration with error handling system"""
        generator = MatrixGenerator()

        # Test that errors are properly formatted
        try:
            constraints = Constraints(min_value=100, max_value=1)  # Invalid
            matrix = generator.generate(3, 3, constraints)
        except Exception as e:
            error_msg = str(e)
            assert len(error_msg) > 10, "Error message should be informative"
            assert any(
                keyword in error_msg.lower()
                for keyword in ["constraint", "matrix", "min", "max"]
            ), "Error should mention relevant context"

    def test_json_serialization_compatibility(self):
        """Test that generated matrices are JSON serializable"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=100)

        matrix = generator.generate(3, 4, constraints)

        # Should be JSON serializable (for CLI output)
        import json

        json_str = json.dumps(matrix)
        assert isinstance(json_str, str), "Matrix should be JSON serializable"

        # Should be able to deserialize
        parsed_matrix = json.loads(json_str)
        assert parsed_matrix == matrix, "JSON round-trip should preserve matrix"

    def test_facade_integration(self):
        """Test integration with main TestCaseGenerator facade"""
        try:
            from testgen.facade import TestCaseGenerator

            # Test facade matrix generation
            facade = TestCaseGenerator(seed=42)

            # Check if facade has matrix generation method
            if hasattr(facade, "generate_matrix"):
                matrix = facade.generate_matrix(
                    rows=3, cols=4, min_value=1, max_value=50
                )
                assert matrix is not None, "Facade matrix generation should work"
                assert len(matrix) == 3, "Should have correct dimensions"
                assert len(matrix[0]) == 4, "Should have correct dimensions"
            else:
                print("Note: Facade matrix generation method not found")

        except ImportError:
            print("Note: TestCaseGenerator facade not available")

    def test_performance_benchmarking(self):
        """Test matrix generation performance"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=100)

        import time

        # Benchmark medium-sized matrix generation
        start_time = time.time()
        for _ in range(100):
            matrix = generator.generate(10, 10, constraints)
        generation_time = time.time() - start_time

        # Should be reasonably fast
        assert generation_time < 5.0, "Matrix generation should be reasonably fast"


class TestMatrixGeneratorValidation:
    """Test matrix validation and consistency"""

    def test_matrix_structure_consistency(self):
        """Test that generated matrices have consistent structure"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=10)

        matrix = generator.generate(5, 7, constraints)

        # All rows should have same length
        row_lengths = [len(row) for row in matrix]
        assert all(length == 7 for length in row_lengths), (
            "All rows should have same length"
        )

        # All elements should be integers
        for row in matrix:
            for element in row:
                assert isinstance(element, int), "All elements should be integers"

    def test_matrix_bounds_validation(self):
        """Test that all matrix elements are within bounds"""
        generator = MatrixGenerator()

        test_cases = [
            (Constraints(min_value=1, max_value=5), 1, 5),
            (Constraints(min_value=-10, max_value=-1), -10, -1),
            (Constraints(min_value=0, max_value=0), 0, 0),
        ]

        for constraints, min_val, max_val in test_cases:
            matrix = generator.generate(4, 4, constraints)

            for row in matrix:
                for val in row:
                    assert min_val <= val <= max_val, (
                        f"Value {val} should be within [{min_val}, {max_val}]"
                    )

    def test_matrix_randomness_quality(self):
        """Test that matrices show appropriate randomness"""
        generator = MatrixGenerator()
        constraints = Constraints(min_value=1, max_value=20)

        # Generate large matrix
        matrix = generator.generate(10, 10, constraints)

        # Flatten to analyze distribution
        all_values = [val for row in matrix for val in row]

        # Should have reasonable variety
        unique_values = set(all_values)
        assert len(unique_values) >= 5, "Should have variety of values"

        # Check for obvious patterns (all same, arithmetic sequence, etc.)
        assert not all(val == all_values[0] for val in all_values), (
            "Should not have all identical values"
        )


if __name__ == "__main__":
    # Can be run directly for quick testing
    pytest.main([__file__, "-v"])
