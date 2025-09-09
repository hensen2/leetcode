"""
Command-line interface for test case generator
Separated from core logic for better modularity
"""

import argparse
import json
import random
import sys
from datetime import datetime
from typing import Any, List, Optional

from ..core.config import Config
from ..core.generators import (
    GraphGenerator,
    IntegerGenerator,
    LinkedListGenerator,
    MatrixGenerator,
    StringGenerator,
    TreeGenerator,
)
from ..core.models import Constraints, GraphProperties, TreeProperties
from ..core.serializers import LinkedListSerializer, TreeSerializer
from ..error_handling.handlers import ErrorReporter
from ..execution.runner import EnhancedTestRunner as TestRunner
from ..patterns.edge_cases import EdgeCaseGenerator


class CLI:
    """Command-line interface for test case generation"""

    def __init__(self):
        self.edge_gen = EdgeCaseGenerator()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser"""
        parser = argparse.ArgumentParser(
            description="Generate test cases for DSA problems",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples(),
        )

        # Main arguments
        parser.add_argument(
            "type",
            choices=[
                "array",
                "string",
                "matrix",
                "tree",
                "graph",
                "linked_list",
                "edge_cases",
            ],
            help="Type of test data to generate",
        )

        parser.add_argument(
            "-n",
            "--num",
            type=int,
            default=Config.CLI_DEFAULT_NUM_TESTS,
            help=f"Number of test cases to generate (default: {Config.CLI_DEFAULT_NUM_TESTS})",
        )

        parser.add_argument(
            "-e", "--edge-cases", action="store_true", help="Include edge cases"
        )

        parser.add_argument(
            "-o", "--output", type=str, help="Output file for test cases (JSON format)"
        )

        parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

        # Size constraints
        parser.add_argument(
            "--min-size",
            type=int,
            default=Config.CLI_DEFAULT_MIN_SIZE,
            help=f"Minimum size for arrays/strings (default: {Config.CLI_DEFAULT_MIN_SIZE})",
        )

        parser.add_argument(
            "--max-size",
            type=int,
            default=Config.CLI_DEFAULT_MAX_SIZE,
            help=f"Maximum size for arrays/strings (default: {Config.CLI_DEFAULT_MAX_SIZE})",
        )

        # Value constraints
        parser.add_argument(
            "--min-value",
            type=int,
            default=Config.CLI_DEFAULT_MIN_VALUE,
            help=f"Minimum value for integers (default: {Config.CLI_DEFAULT_MIN_VALUE})",
        )

        parser.add_argument(
            "--max-value",
            type=int,
            default=Config.CLI_DEFAULT_MAX_VALUE,
            help=f"Maximum value for integers (default: {Config.CLI_DEFAULT_MAX_VALUE})",
        )

        # Type-specific options
        parser.add_argument(
            "--sorted", action="store_true", help="Generate sorted arrays"
        )

        parser.add_argument(
            "--unique", action="store_true", help="Generate arrays with unique elements"
        )

        parser.add_argument(
            "--balanced", action="store_true", help="Generate balanced trees"
        )

        parser.add_argument(
            "--bst", action="store_true", help="Generate binary search trees"
        )

        parser.add_argument(
            "--connected", action="store_true", help="Generate connected graphs"
        )

        parser.add_argument(
            "--directed", action="store_true", help="Generate directed graphs"
        )

        parser.add_argument(
            "--weighted", action="store_true", help="Generate weighted graphs"
        )

        parser.add_argument(
            "--palindrome", action="store_true", help="Generate palindrome strings"
        )

        parser.add_argument(
            "--cycle", action="store_true", help="Generate linked lists with cycles"
        )

        return parser

    def _get_examples(self) -> str:
        """Get example usage strings"""
        return """
Examples:
  %(prog)s array -n 10 -e                    # Generate 10 array test cases with edge cases
  %(prog)s tree -n 5 --balanced              # Generate 5 balanced tree test cases
  %(prog)s graph -n 3 --connected --weighted # Generate 3 connected weighted graphs
  %(prog)s string --palindrome -n 10         # Generate 10 palindrome strings
  %(prog)s edge_cases array                  # Get all edge cases for arrays
  %(prog)s linked_list --cycle -n 5          # Generate 5 linked lists with cycles
        """

    def run(self, args: List[str] = None) -> None:
        """
        Run the CLI

        Args:
            args: Command-line arguments (if None, uses sys.argv)
        """
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        # Set random seed if provided
        if parsed_args.seed is not None:
            random.seed(parsed_args.seed)

        # Generate test cases based on type
        if parsed_args.type == "edge_cases":
            test_cases = self._generate_edge_cases_only(parsed_args)
        else:
            test_cases = self._generate_test_cases(parsed_args)

        # Output results
        self._output_results(test_cases, parsed_args)

    def _generate_edge_cases_only(self, args: argparse.Namespace) -> List[Any]:
        """Generate only edge cases for the specified type"""
        # Get the subtype if specified (e.g., "edge_cases array")
        edge_case_methods = {
            "array": self.edge_gen.get_array_edge_cases,
            "string": self.edge_gen.get_string_edge_cases,
            "matrix": self.edge_gen.get_matrix_edge_cases,
            "tree": self.edge_gen.get_tree_edge_cases,
            "linked_list": self.edge_gen.get_linked_list_edge_cases,
            "graph": self.edge_gen.get_graph_edge_cases,
            "number": self.edge_gen.get_number_edge_cases,
            "boolean": self.edge_gen.get_boolean_edge_cases,
        }

        # Default to array if no specific type
        method = edge_case_methods.get("array")
        return method()

    def _generate_test_cases(self, args: argparse.Namespace) -> List[Any]:
        """Generate test cases based on arguments"""
        test_cases = []

        # Create constraints
        constraints = Constraints(
            min_value=args.min_value,
            max_value=args.max_value,
            min_length=args.min_size,
            max_length=args.max_size,
            is_sorted=args.sorted,
            is_unique=args.unique,
        )

        # Add edge cases if requested
        if args.edge_cases:
            edge_cases = self._get_edge_cases_for_type(args.type)
            test_cases.extend(edge_cases)

        # Generate random test cases
        for _ in range(args.num):
            test_case = self._generate_single_test_case(args, constraints)
            test_cases.append(test_case)

        return test_cases

    def _get_edge_cases_for_type(self, data_type: str) -> List[Any]:
        """Get edge cases for specific data type"""
        edge_case_map = {
            "array": self.edge_gen.get_array_edge_cases(),
            "string": self.edge_gen.get_string_edge_cases(),
            "matrix": self.edge_gen.get_matrix_edge_cases(),
            "tree": self.edge_gen.get_tree_edge_cases(),
            "linked_list": self.edge_gen.get_linked_list_edge_cases(),
            "graph": self.edge_gen.get_graph_edge_cases(),
        }
        return edge_case_map.get(data_type, [])

    def _generate_single_test_case(
        self, args: argparse.Namespace, constraints: Constraints
    ) -> Any:
        """Generate a single test case"""
        if args.type == "array":
            gen = IntegerGenerator(args.seed)
            size = random.randint(constraints.min_length, constraints.max_length)
            return gen.generate_array(size, constraints)

        elif args.type == "string":
            gen = StringGenerator(args.seed)
            if args.palindrome:
                length = random.randint(constraints.min_length, constraints.max_length)
                return gen.generate_palindrome(length)
            else:
                return gen.generate(None, constraints)

        elif args.type == "matrix":
            gen = MatrixGenerator(args.seed)
            rows = random.randint(1, min(20, constraints.max_length))
            cols = random.randint(1, min(20, constraints.max_length))
            return gen.generate(rows, cols, constraints)

        elif args.type == "tree":
            gen = TreeGenerator(args.seed)
            size = random.randint(1, 50)
            props = TreeProperties(
                size=size,
                balanced=args.balanced,
                bst=args.bst,
                min_val=args.min_value,
                max_val=args.max_value,
            )
            tree = gen.generate(props, constraints)
            # Convert to serializable format
            return TreeSerializer.to_array(tree)

        elif args.type == "graph":
            gen = GraphGenerator(args.seed)
            nodes = random.randint(2, 50)
            props = GraphProperties(
                num_nodes=nodes,
                connected=args.connected,
                directed=args.directed,
                weighted=args.weighted,
            )
            return gen.generate(props)

        elif args.type == "linked_list":
            gen = LinkedListGenerator(args.seed)
            size = random.randint(1, 50)
            linked_list = gen.generate(size, constraints, has_cycle=args.cycle)
            # Convert to serializable format
            return LinkedListSerializer.to_array(linked_list)

        return None

    def _output_results(self, test_cases: List[Any], args: argparse.Namespace) -> None:
        """Output test cases to file or console"""
        if args.output:
            # Save to file
            with open(args.output, "w") as f:
                json.dump(test_cases, f, indent=2, default=str)
            print(f"✅ Generated {len(test_cases)} test cases saved to {args.output}")
        else:
            # Print to console
            print(f"Generated {len(test_cases)} test cases:\n")

            # Show first few test cases
            display_limit = min(5, len(test_cases))
            for i, test_case in enumerate(test_cases[:display_limit]):
                print(f"Test {i + 1}:")
                self._print_test_case(test_case, args.type)

            if len(test_cases) > display_limit:
                print(f"\n... and {len(test_cases) - display_limit} more test cases")

    def _print_test_case(self, test_case: Any, data_type: str) -> None:
        """Print a single test case with appropriate formatting"""
        if (
            isinstance(test_case, list)
            and len(test_case) > Config.CLI_MAX_DISPLAY_LENGTH
        ):
            print(
                f"  {test_case[: Config.CLI_MAX_DISPLAY_LENGTH]}... (length: {len(test_case)})"
            )
        elif isinstance(test_case, dict):
            print(f"  Type: {data_type}")
            if "num_nodes" in test_case:
                print(
                    f"  Nodes: {test_case['num_nodes']}, Edges: {len(test_case.get('edges', []))}"
                )
            else:
                # Print dict with truncation if needed
                print(f"  {str(test_case)[:200]}...")
        elif (
            isinstance(test_case, str)
            and len(test_case) > Config.CLI_MAX_DISPLAY_LENGTH * 5
        ):
            print(
                f"  '{test_case[: Config.CLI_MAX_DISPLAY_LENGTH * 5]}...' (length: {len(test_case)})"
            )
        else:
            print(f"  {test_case}")


class EnhancedCLI:
    """Enhanced CLI with sophisticated error reporting and user guidance"""

    def __init__(self):
        self.edge_gen = EdgeCaseGenerator()
        self.runner = TestRunner()
        self.error_reporter = ErrorReporter(self.runner.error_handler)

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser with enhanced options"""
        parser = argparse.ArgumentParser(
            description="🧪 Enhanced Test Case Generator for DSA Problems",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_enhanced_examples(),
        )

        # Main arguments
        parser.add_argument(
            "type",
            choices=[
                "array",
                "string",
                "matrix",
                "tree",
                "graph",
                "linked_list",
                "edge_cases",
                "validate",
                "benchmark",
            ],
            help="Type of operation to perform",
        )

        parser.add_argument(
            "-n",
            "--num",
            type=int,
            default=Config.CLI_DEFAULT_NUM_TESTS,
            help=f"Number of test cases to generate (default: {Config.CLI_DEFAULT_NUM_TESTS})",
        )

        parser.add_argument(
            "-e",
            "--edge-cases",
            action="store_true",
            help="Include edge cases in generation",
        )

        parser.add_argument(
            "-o", "--output", type=str, help="Output file for test cases (JSON format)"
        )

        parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

        # Enhanced error reporting options
        parser.add_argument(
            "--error-report",
            choices=["text", "json", "html"],
            help="Generate detailed error report in specified format",
        )

        parser.add_argument(
            "--verbose",
            "-v",
            action="count",
            default=0,
            help="Increase verbosity (use -v, -vv, or -vvv)",
        )

        parser.add_argument(
            "--validate-function",
            type=str,
            help="Python function to validate generated test cases (format: module.function)",
        )

        parser.add_argument(
            "--benchmark",
            action="store_true",
            help="Run performance benchmarks on generated test cases",
        )

        # Size constraints
        parser.add_argument(
            "--min-size",
            type=int,
            default=Config.CLI_DEFAULT_MIN_SIZE,
            help=f"Minimum size for arrays/strings (default: {Config.CLI_DEFAULT_MIN_SIZE})",
        )

        parser.add_argument(
            "--max-size",
            type=int,
            default=Config.CLI_DEFAULT_MAX_SIZE,
            help=f"Maximum size for arrays/strings (default: {Config.CLI_DEFAULT_MAX_SIZE})",
        )

        # Value constraints
        parser.add_argument(
            "--min-value",
            type=int,
            default=Config.CLI_DEFAULT_MIN_VALUE,
            help=f"Minimum value for integers (default: {Config.CLI_DEFAULT_MIN_VALUE})",
        )

        parser.add_argument(
            "--max-value",
            type=int,
            default=Config.CLI_DEFAULT_MAX_VALUE,
            help=f"Maximum value for integers (default: {Config.CLI_DEFAULT_MAX_VALUE})",
        )

        # Type-specific options (keeping existing ones)
        parser.add_argument(
            "--sorted", action="store_true", help="Generate sorted arrays"
        )
        parser.add_argument(
            "--unique", action="store_true", help="Generate arrays with unique elements"
        )
        parser.add_argument(
            "--balanced", action="store_true", help="Generate balanced trees"
        )
        parser.add_argument(
            "--bst", action="store_true", help="Generate binary search trees"
        )
        parser.add_argument(
            "--connected", action="store_true", help="Generate connected graphs"
        )
        parser.add_argument(
            "--directed", action="store_true", help="Generate directed graphs"
        )
        parser.add_argument(
            "--weighted", action="store_true", help="Generate weighted graphs"
        )
        parser.add_argument(
            "--palindrome", action="store_true", help="Generate palindrome strings"
        )
        parser.add_argument(
            "--cycle", action="store_true", help="Generate linked lists with cycles"
        )

        return parser

    def run(self, args: List[str] = None) -> None:
        """Enhanced CLI runner with rich error reporting"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        # Configure verbosity
        self._configure_logging(parsed_args.verbose)

        # Set random seed if provided
        if parsed_args.seed is not None:
            random.seed(parsed_args.seed)
            print(f"🎲 Using random seed: {parsed_args.seed}")

        print(f"🚀 Starting {parsed_args.type} generation...")
        print(f"📊 Generating {parsed_args.num} test cases")

        try:
            # Execute the requested operation
            if parsed_args.type == "validate":
                self._run_validation(parsed_args)
            elif parsed_args.type == "benchmark":
                self._run_benchmark(parsed_args)
            else:
                self._run_generation(parsed_args)

        except Exception as e:
            self._handle_cli_error(e, parsed_args)

        # Generate error report if requested
        if parsed_args.error_report:
            self._generate_error_report(parsed_args.error_report, parsed_args)

    def _run_generation(self, args: argparse.Namespace) -> None:
        """Run test case generation with enhanced error handling"""
        start_time = datetime.now()

        try:
            # Generate test cases based on type
            if args.type == "edge_cases":
                test_cases = self._generate_edge_cases_only(args)
            else:
                test_cases = self._generate_test_cases(args)

            generation_time = (datetime.now() - start_time).total_seconds()

            # Output results with enhanced reporting
            self._output_enhanced_results(test_cases, args, generation_time)

        except Exception as e:
            self._handle_generation_error(e, args)

    def _run_validation(self, args: argparse.Namespace) -> None:
        """Run validation on existing test cases or generated ones"""
        print("🔍 Running test case validation...")

        if not args.validate_function:
            print("❌ Error: --validate-function required for validation mode")
            return

        # Load validation function
        validator_func = self._load_function(args.validate_function)
        if not validator_func:
            return

        # Generate test cases to validate
        test_cases = self._generate_test_cases(args)

        print(f"🧪 Validating {len(test_cases)} test cases...")

        # Run validation with enhanced error reporting
        results = self.runner.run_test_suite(
            validator_func, test_cases, progress_callback=self._show_progress
        )

        # Show enhanced validation results
        self._show_validation_results(results, args)

    def _run_benchmark(self, args: argparse.Namespace) -> None:
        """Run performance benchmarks"""
        print("⚡ Running performance benchmarks...")

        if not args.validate_function:
            print("❌ Error: --validate-function required for benchmark mode")
            return

        # Load function to benchmark
        benchmark_func = self._load_function(args.validate_function)
        if not benchmark_func:
            return

        # Generate test cases of varying sizes for benchmarking
        benchmark_sizes = [10, 50, 100, 500, 1000]

        print("📈 Generating benchmark test cases...")

        for size in benchmark_sizes:
            print(f"🔧 Benchmarking size {size}...")

            # Generate test cases for this size
            test_cases = self._generate_size_specific_cases(size, args)

            # Run benchmark
            results = self.runner.run_test_suite(
                benchmark_func,
                test_cases[: min(10, len(test_cases))],  # Limit for performance
                progress_callback=lambda c, t: None,  # Suppress progress for benchmarks
            )

            avg_time = results.total_time / results.total if results.total > 0 else 0
            print(f"   📊 Size {size}: {avg_time * 1000:.2f}ms average")

    def _generate_test_cases(self, args: argparse.Namespace) -> List[Any]:
        """Generate test cases with enhanced error handling"""
        test_cases = []

        # Create constraints with validation
        try:
            constraints = self._create_constraints(args)
        except ValueError as e:
            print(f"❌ Invalid constraints: {e}")
            print(
                "💡 Suggestion: Check that min_value <= max_value and min_size <= max_size"
            )
            return []

        # Add edge cases if requested
        if args.edge_cases:
            try:
                edge_cases = self._get_edge_cases_for_type(args.type)
                test_cases.extend(edge_cases)
                print(f"✅ Added {len(edge_cases)} edge cases")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate edge cases: {e}")

        # Generate random test cases with progress tracking
        print(f"🎯 Generating {args.num} random test cases...")

        successful_generations = 0
        failed_generations = 0

        for i in range(args.num):
            try:
                test_case = self._generate_single_test_case(args, constraints)
                if test_case is not None:
                    test_cases.append(test_case)
                    successful_generations += 1
                else:
                    failed_generations += 1

                # Show progress
                if (i + 1) % max(1, args.num // 10) == 0:
                    print(f"📊 Progress: {i + 1}/{args.num} cases generated")

            except Exception as e:
                failed_generations += 1
                if args.verbose > 0:
                    print(f"⚠️  Failed to generate case {i + 1}: {e}")

        # Report generation statistics
        if failed_generations > 0:
            print(
                f"⚠️  Generation issues: {failed_generations} failures out of {args.num} attempts"
            )
            print(f"✅ Successfully generated: {successful_generations} test cases")

            if failed_generations > args.num * 0.1:  # More than 10% failures
                print("💡 Suggestions:")
                print("   - Try relaxing constraints (larger value/size ranges)")
                print("   - Check for conflicting options (e.g., unique + small range)")
                print("   - Use --verbose for more details")

        return test_cases

    def _output_enhanced_results(
        self, test_cases: List[Any], args: argparse.Namespace, generation_time: float
    ) -> None:
        """Output results with enhanced formatting and statistics"""

        print(f"\n🎉 Generation completed in {generation_time:.2f}s")
        print(f"📋 Generated {len(test_cases)} test cases")

        # Analyze test case characteristics
        self._analyze_test_cases(test_cases, args)

        if args.output:
            # Save to file with metadata
            self._save_enhanced_output(test_cases, args, generation_time)
        else:
            # Display to console with rich formatting
            self._display_console_output(test_cases, args)

    def _analyze_test_cases(
        self, test_cases: List[Any], args: argparse.Namespace
    ) -> None:
        """Analyze and report test case characteristics"""
        if not test_cases:
            return

        print("\n📊 TEST CASE ANALYSIS:")

        # Type-specific analysis
        if args.type == "array":
            self._analyze_arrays(test_cases)
        elif args.type == "string":
            self._analyze_strings(test_cases)
        elif args.type == "matrix":
            self._analyze_matrices(test_cases)
        elif args.type == "tree":
            self._analyze_trees(test_cases)
        elif args.type == "graph":
            self._analyze_graphs(test_cases)

    def _analyze_arrays(self, arrays: List[List[int]]) -> None:
        """Analyze array characteristics"""
        if not arrays:
            return

        sizes = [len(arr) for arr in arrays if isinstance(arr, list)]
        if sizes:
            print(
                f"   📏 Size range: {min(sizes)} - {max(sizes)} (avg: {sum(sizes) / len(sizes):.1f})"
            )

            # Check for sorted arrays
            sorted_count = sum(
                1 for arr in arrays if isinstance(arr, list) and arr == sorted(arr)
            )
            if sorted_count > 0:
                print(
                    f"   🔄 Sorted arrays: {sorted_count}/{len(arrays)} ({sorted_count / len(arrays) * 100:.1f}%)"
                )

            # Check for unique elements
            unique_count = sum(
                1
                for arr in arrays
                if isinstance(arr, list) and len(arr) == len(set(arr))
            )
            if unique_count > 0:
                print(
                    f"   🔢 Unique elements: {unique_count}/{len(arrays)} ({unique_count / len(arrays) * 100:.1f}%)"
                )

    def _analyze_strings(self, strings: List[str]) -> None:
        """Analyze string characteristics"""
        if not strings:
            return

        lengths = [len(s) for s in strings if isinstance(s, str)]
        if lengths:
            print(
                f"   📏 Length range: {min(lengths)} - {max(lengths)} (avg: {sum(lengths) / len(lengths):.1f})"
            )

            # Check for palindromes
            palindrome_count = sum(
                1 for s in strings if isinstance(s, str) and s == s[::-1]
            )
            if palindrome_count > 0:
                print(
                    f"   🔄 Palindromes: {palindrome_count}/{len(strings)} ({palindrome_count / len(strings) * 100:.1f}%)"
                )

    def _analyze_matrices(self, matrices: List[List[List[int]]]) -> None:
        """Analyze matrix characteristics"""
        if not matrices:
            return

        shapes = [
            (len(m), len(m[0]) if m else 0) for m in matrices if isinstance(m, list)
        ]
        if shapes:
            rows = [r for r, c in shapes]
            cols = [c for r, c in shapes]
            print(
                f"   📐 Dimensions: {min(rows)}x{min(cols)} - {max(rows)}x{max(cols)}"
            )

    def _analyze_trees(self, trees: List[Any]) -> None:
        """Analyze tree characteristics"""
        non_empty = [t for t in trees if t is not None]
        if non_empty:
            print(f"   🌳 Non-empty trees: {len(non_empty)}/{len(trees)}")

    def _analyze_graphs(self, graphs: List[dict]) -> None:
        """Analyze graph characteristics"""
        if not graphs:
            return

        nodes = [g.get("num_nodes", 0) for g in graphs if isinstance(g, dict)]
        edges = [len(g.get("edges", [])) for g in graphs if isinstance(g, dict)]

        if nodes and edges:
            print(
                f"   🕸️  Nodes: {min(nodes)} - {max(nodes)} (avg: {sum(nodes) / len(nodes):.1f})"
            )
            print(
                f"   🔗 Edges: {min(edges)} - {max(edges)} (avg: {sum(edges) / len(edges):.1f})"
            )

    def _save_enhanced_output(
        self, test_cases: List[Any], args: argparse.Namespace, generation_time: float
    ) -> None:
        """Save test cases with enhanced metadata"""
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generation_time_seconds": generation_time,
                "generator_version": "enhanced_v2",
                "total_cases": len(test_cases),
                "parameters": {
                    "type": args.type,
                    "num": args.num,
                    "seed": args.seed,
                    "edge_cases_included": args.edge_cases,
                    "constraints": {
                        "min_size": args.min_size,
                        "max_size": args.max_size,
                        "min_value": args.min_value,
                        "max_value": args.max_value,
                        "sorted": args.sorted,
                        "unique": args.unique,
                    },
                },
            },
            "test_cases": test_cases,
            "error_summary": self.runner.error_handler.get_error_summary()
            if self.runner.error_handler.error_records
            else None,
        }

        try:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"✅ Enhanced output saved to {args.output}")
            print("📁 File includes metadata, parameters, and error summary")
        except Exception as e:
            print(f"❌ Failed to save output: {e}")
            print("💡 Suggestion: Check file permissions and disk space")

    def _display_console_output(
        self, test_cases: List[Any], args: argparse.Namespace
    ) -> None:
        """Display test cases to console with enhanced formatting"""
        print("\n📋 GENERATED TEST CASES:")

        display_limit = min(5, len(test_cases))
        for i, test_case in enumerate(test_cases[:display_limit]):
            print(f"\n🧪 Test Case {i + 1}:")
            self._format_test_case(test_case, args.type, args.verbose)

        if len(test_cases) > display_limit:
            print(f"\n📄 ... and {len(test_cases) - display_limit} more test cases")
            print("💡 Use --output to save all cases to a file")

    def _format_test_case(self, test_case: Any, data_type: str, verbosity: int) -> None:
        """Format individual test case for display"""
        if (
            isinstance(test_case, list)
            and len(test_case) > Config.CLI_MAX_DISPLAY_LENGTH
        ):
            preview = test_case[: Config.CLI_MAX_DISPLAY_LENGTH]
            print(f"   📊 {preview}... (length: {len(test_case)})")
            if verbosity > 1:
                print(
                    f"   📈 Statistics: min={min(test_case)}, max={max(test_case)}, unique={len(set(test_case))}"
                )
        elif isinstance(test_case, dict):
            print(f"   🔧 Type: {data_type}")
            if "num_nodes" in test_case:
                print(
                    f"   📊 Nodes: {test_case['num_nodes']}, Edges: {len(test_case.get('edges', []))}"
                )
                if verbosity > 0:
                    print(f"   🔗 Directed: {test_case.get('directed', False)}")
                    print(f"   ⚖️  Weighted: {test_case.get('weighted', False)}")
        elif (
            isinstance(test_case, str)
            and len(test_case) > Config.CLI_MAX_DISPLAY_LENGTH * 5
        ):
            preview = test_case[: Config.CLI_MAX_DISPLAY_LENGTH * 5]
            print(f"   📝 '{preview}...' (length: {len(test_case)})")
        else:
            print(f"   💾 {test_case}")

    def _show_validation_results(self, results, args: argparse.Namespace) -> None:
        """Show enhanced validation results"""
        self.runner.print_enhanced_summary(results)

        # Additional validation-specific reporting
        if results.passed == results.total:
            print("🎉 All test cases passed validation!")
        else:
            failure_rate = (results.failed + results.errors) / results.total * 100
            print(f"⚠️  Validation issues: {failure_rate:.1f}% failure rate")

            if failure_rate > 20:  # High failure rate
                print("💡 Suggestions:")
                print("   - Review your validation function logic")
                print("   - Check constraint compatibility")
                print("   - Use --verbose for detailed error information")

    def _show_progress(self, current: int, total: int) -> None:
        """Show progress during long operations"""
        if current % max(1, total // 20) == 0:  # Update every 5%
            percentage = (current / total) * 100
            print(f"📊 Progress: {current}/{total} ({percentage:.1f}%)")

    def _handle_cli_error(self, error: Exception, args: argparse.Namespace) -> None:
        """Handle CLI-level errors with helpful suggestions"""
        print(f"\n❌ CLI Error: {error}")

        # Provide specific suggestions based on error type
        if "FileNotFoundError" in str(type(error)):
            print("💡 Suggestions:")
            print("   - Check that the specified file path exists")
            print("   - Verify file permissions")
        elif "ValueError" in str(type(error)):
            print("💡 Suggestions:")
            print("   - Check parameter values and ranges")
            print("   - Ensure min <= max for size and value constraints")
        elif "ImportError" in str(type(error)):
            print("💡 Suggestions:")
            print("   - Check that the validation function module exists")
            print("   - Verify the function name is correct")

        if args.verbose > 0:
            import traceback

            print("\n🔍 Full traceback:")
            traceback.print_exc()

    def _handle_generation_error(
        self, error: Exception, args: argparse.Namespace
    ) -> None:
        """Handle generation-specific errors"""
        print(f"\n❌ Generation Error: {error}")

        print("💡 Troubleshooting suggestions:")
        print("   - Try reducing the number of test cases (-n)")
        print("   - Relax constraints (larger ranges)")
        print("   - Remove conflicting options (e.g., --unique with small ranges)")
        print("   - Use --verbose for detailed error information")

    def _generate_error_report(self, format: str, args: argparse.Namespace) -> None:
        """Generate and save detailed error report"""
        if not self.runner.error_handler.error_records:
            print("📋 No errors to report - all operations completed successfully!")
            return

        print(f"\n📄 Generating {format.upper()} error report...")

        try:
            report = self.error_reporter.generate_report(format)

            # Save report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_report_{timestamp}.{format}"

            with open(filename, "w") as f:
                f.write(report)

            print(f"✅ Error report saved to {filename}")

            # Show summary
            summary = self.runner.error_handler.get_error_summary()
            print(f"📊 Report contains {summary['total_errors']} errors")

        except Exception as e:
            print(f"❌ Failed to generate error report: {e}")

    def _create_constraints(self, args: argparse.Namespace) -> Constraints:
        """Create and validate constraints from CLI arguments"""
        constraints = Constraints(
            min_value=args.min_value,
            max_value=args.max_value,
            min_length=args.min_size,
            max_length=args.max_size,
            is_sorted=args.sorted,
            is_unique=args.unique,
        )

        # Validate constraints
        constraints.validate()
        return constraints

    def _configure_logging(self, verbosity: int) -> None:
        """Configure logging based on verbosity level"""
        import logging

        if verbosity == 0:
            level = logging.WARNING
        elif verbosity == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG

        logging.getLogger().setLevel(level)

    def _load_function(self, function_path: str) -> Optional[callable]:
        """Load a function from module.function path"""
        try:
            module_name, function_name = function_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[function_name])
            return getattr(module, function_name)
        except Exception as e:
            print(f"❌ Failed to load function {function_path}: {e}")
            print("💡 Use format: module.function (e.g., my_solutions.two_sum)")
            return None

    def _generate_size_specific_cases(
        self, size: int, args: argparse.Namespace
    ) -> List[Any]:
        """Generate test cases for a specific size (for benchmarking)"""
        # Override size constraints temporarily
        original_min = args.min_size
        original_max = args.max_size

        args.min_size = size
        args.max_size = size

        try:
            cases = self._generate_test_cases(args)
        finally:
            # Restore original constraints
            args.min_size = original_min
            args.max_size = original_max

        return cases

    def _get_enhanced_examples(self) -> str:
        """Get enhanced example usage strings"""
        return """
🧪 Enhanced Examples:

Basic Generation:
  %(prog)s array -n 10 -e                    # Generate arrays with edge cases
  %(prog)s tree -n 5 --balanced --verbose    # Generate balanced trees with details
  
Validation & Testing:
  %(prog)s validate --validate-function my_solution.two_sum -n 100
  %(prog)s benchmark --validate-function my_solution.binary_search
  
Error Reporting:
  %(prog)s array -n 1000 --error-report json --verbose
  
Advanced Options:
  %(prog)s array --unique --min-size 100 --max-size 1000 -o large_arrays.json
  %(prog)s string --palindrome -n 20 --seed 42 --verbose
        """

    # Keep existing methods for compatibility
    def _generate_edge_cases_only(self, args: argparse.Namespace) -> List[Any]:
        """Generate only edge cases - keeping original functionality"""
        edge_case_methods = {
            "array": self.edge_gen.get_array_edge_cases,
            "string": self.edge_gen.get_string_edge_cases,
            "matrix": self.edge_gen.get_matrix_edge_cases,
            "tree": self.edge_gen.get_tree_edge_cases,
            "linked_list": self.edge_gen.get_linked_list_edge_cases,
            "graph": self.edge_gen.get_graph_edge_cases,
            "number": self.edge_gen.get_number_edge_cases,
            "boolean": self.edge_gen.get_boolean_edge_cases,
        }
        method = edge_case_methods.get("array")
        return method()

    def _get_edge_cases_for_type(self, data_type: str) -> List[Any]:
        """Get edge cases for specific data type - keeping original functionality"""
        edge_case_map = {
            "array": self.edge_gen.get_array_edge_cases(),
            "string": self.edge_gen.get_string_edge_cases(),
            "matrix": self.edge_gen.get_matrix_edge_cases(),
            "tree": self.edge_gen.get_tree_edge_cases(),
            "linked_list": self.edge_gen.get_linked_list_edge_cases(),
            "graph": self.edge_gen.get_graph_edge_cases(),
        }
        return edge_case_map.get(data_type, [])

    def _generate_single_test_case(
        self, args: argparse.Namespace, constraints: Constraints
    ) -> Any:
        """Generate single test case - keeping original functionality but with enhanced error handling"""
        try:
            if args.type == "array":
                gen = IntegerGenerator(args.seed)
                size = random.randint(constraints.min_length, constraints.max_length)
                return gen.generate_array(size, constraints)

            elif args.type == "string":
                gen = StringGenerator(args.seed)
                if args.palindrome:
                    length = random.randint(
                        constraints.min_length, constraints.max_length
                    )
                    return gen.generate_palindrome(length)
                else:
                    return gen.generate(None, constraints)

            elif args.type == "matrix":
                gen = MatrixGenerator(args.seed)
                rows = random.randint(1, min(20, constraints.max_length))
                cols = random.randint(1, min(20, constraints.max_length))
                return gen.generate(rows, cols, constraints)

            elif args.type == "tree":
                gen = TreeGenerator(args.seed)
                size = random.randint(1, 50)
                props = TreeProperties(
                    size=size,
                    balanced=args.balanced,
                    bst=args.bst,
                    min_val=args.min_value,
                    max_val=args.max_value,
                )
                tree = gen.generate(props, constraints)
                return TreeSerializer.to_array(tree)

            elif args.type == "graph":
                gen = GraphGenerator(args.seed)
                nodes = random.randint(2, 50)
                props = GraphProperties(
                    num_nodes=nodes,
                    connected=args.connected,
                    directed=args.directed,
                    weighted=args.weighted,
                )
                return gen.generate(props)

            elif args.type == "linked_list":
                gen = LinkedListGenerator(args.seed)
                size = random.randint(1, 50)
                linked_list = gen.generate(size, constraints, has_cycle=args.cycle)
                return LinkedListSerializer.to_array(linked_list)

        except Exception as e:
            if args.verbose > 1:
                print(f"⚠️  Generation failed for {args.type}: {e}")
            return None


def main(args=None):
    """Enhanced main entry point"""
    cli = EnhancedCLI()

    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        # No arguments provided, show usage
        print("🧪 Enhanced Test Case Generator for DSA Problems")
        print("=" * 50)
        print("\n🚀 Quick Start:")
        print(
            "  python -m testgen array -n 10 -e           # Generate arrays with edge cases"
        )
        print("  python -m testgen string --palindrome -n 5 # Generate palindromes")
        print("  python -m testgen tree --balanced -n 3     # Generate balanced trees")
        print("\n🔧 Available Types: array, string, matrix, tree, graph, linked_list")
        print("🆘 For help: python -m testgen -h")
        print("\n💡 Enhanced Features:")
        print("  ✅ Memory-efficient generation with large ranges")
        print("  ✅ Rich error handling with context and suggestions")
        print("  ✅ Progress indicators and detailed analysis")
        print("  ✅ Multiple output formats (JSON, text)")
        return

    try:
        # Run the enhanced CLI with arguments
        cli.run(args)
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        print("🔄 Use --output to save partial results next time")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("🐛 Please report this issue with the following details:")
        print(f"   Command: python -m testgen {' '.join(args)}")
        print(f"   Error: {type(e).__name__}: {e}")

        # Check if verbose mode for full traceback
        if "-v" in args or "--verbose" in args:
            import traceback

            print("\n🔍 Full traceback:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
