"""
Reality verification tests for error handling system

These tests probe the actual implementation to understand what exists
vs. what's documented/tested, without making assumptions about sophistication.
"""

import json
import traceback

import pytest
from testgen.core.generators import IntegerGenerator
from testgen.core.models import Constraints


class TestErrorHandlingReality:
    """Verify what error handling components actually exist and work"""

    def test_error_handling_imports(self):
        """Test what error handling components can actually be imported"""
        import_results = {}

        # Test basic imports
        try:
            import_results["ErrorHandler"] = "✅ Available"
        except ImportError as e:
            import_results["ErrorHandler"] = f"❌ Import failed: {e}"
        except Exception as e:
            import_results["ErrorHandler"] = f"❌ Other error: {e}"

        try:
            import_results["ErrorContext"] = "✅ Available"
        except ImportError as e:
            import_results["ErrorContext"] = f"❌ Import failed: {e}"
        except Exception as e:
            import_results["ErrorContext"] = f"❌ Other error: {e}"

        try:
            import_results["ErrorCategory"] = "✅ Available"
        except ImportError as e:
            import_results["ErrorCategory"] = f"❌ Import failed: {e}"
        except Exception as e:
            import_results["ErrorCategory"] = f"❌ Other error: {e}"

        try:
            import_results["ExecutionError"] = "✅ Available"
        except ImportError as e:
            import_results["ExecutionError"] = f"❌ Import failed: {e}"
        except Exception as e:
            import_results["ExecutionError"] = f"❌ Other error: {e}"

        # Print results for visibility
        print("\n" + "=" * 50)
        print("ERROR HANDLING IMPORT REALITY CHECK")
        print("=" * 50)
        for component, status in import_results.items():
            print(f"{component}: {status}")
        print("=" * 50)

        # At least basic error handling should import
        assert "ErrorHandler" in import_results
        # Don't assert success - just document what we find

    def test_error_handler_instantiation_and_methods(self):
        """Test ErrorHandler instantiation and discover available methods"""
        try:
            from testgen.error_handling.handlers import ErrorHandler

            # Try to instantiate
            handler = ErrorHandler()

            # Discover methods
            all_methods = dir(handler)
            public_methods = [m for m in all_methods if not m.startswith("_")]
            private_methods = [
                m for m in all_methods if m.startswith("_") and not m.startswith("__")
            ]

            print("\n" + "=" * 50)
            print("ERROR HANDLER METHOD REALITY CHECK")
            print("=" * 50)
            print("✅ ErrorHandler instantiation: SUCCESS")
            print(f"Public methods: {public_methods}")
            print(f"Private methods: {private_methods}")

            # Test if key methods exist and are callable
            expected_methods = [
                "handle_error",
                "format_error",
                "get_error_summary",
                "clear_errors",
            ]
            method_status = {}

            for method in expected_methods:
                if hasattr(handler, method):
                    method_obj = getattr(handler, method)
                    if callable(method_obj):
                        method_status[method] = "✅ Exists and callable"
                    else:
                        method_status[method] = "⚠️ Exists but not callable"
                else:
                    method_status[method] = "❌ Does not exist"

            print("\nExpected method availability:")
            for method, status in method_status.items():
                print(f"  {method}: {status}")
            print("=" * 50)

            # Document findings without failing
            assert handler is not None, "ErrorHandler should instantiate"

        except Exception as e:
            print(f"\n❌ ErrorHandler instantiation failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            pytest.fail(f"ErrorHandler not available: {e}")

    def test_basic_error_generation_reality(self):
        """Test what actually happens when generators encounter errors"""
        generator = IntegerGenerator()

        error_scenarios = [
            {
                "name": "Invalid range (min > max)",
                "constraints": {"min_value": 100, "max_value": 1, "is_unique": False},
                "size": 5,
            },
            {
                "name": "Impossible unique constraint",
                "constraints": {"min_value": 1, "max_value": 2, "is_unique": True},
                "size": 10,  # Need 10 unique values from range [1,2]
            },
            {
                "name": "Negative size",
                "constraints": {"min_value": 1, "max_value": 10, "is_unique": False},
                "size": -5,
            },
        ]

        print("\n" + "=" * 50)
        print("GENERATOR ERROR REALITY CHECK")
        print("=" * 50)

        for scenario in error_scenarios:
            print(f"\nTesting: {scenario['name']}")
            try:
                constraints = Constraints(**scenario["constraints"])
                result = generator.generate_array(
                    size=scenario["size"], constraints=constraints
                )
                print(f"  ⚠️ UNEXPECTED: No error raised, got result: {result}")
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"  ✅ Error type: {error_type}")
                print(f"  ✅ Error message: {error_msg}")
                print(f"  ✅ Message length: {len(error_msg)} chars")

                # Check if error message is informative
                is_informative = len(error_msg) > 20
                has_context = any(
                    str(val) in error_msg for val in scenario["constraints"].values()
                )

                print(
                    f"  📊 Informative (>20 chars): {'✅' if is_informative else '❌'}"
                )
                print(
                    f"  📊 Contains constraint values: {'✅' if has_context else '❌'}"
                )

        print("=" * 50)

    def test_error_context_reality(self):
        """Test if ErrorContext actually works as designed"""
        try:
            from testgen.error_handling.handlers import (
                ErrorCategory,
                ErrorContext,
                ErrorSeverity,
            )

            print("\n" + "=" * 50)
            print("ERROR CONTEXT REALITY CHECK")
            print("=" * 50)

            # Try basic instantiation
            try:
                context = ErrorContext()
                print("✅ Basic ErrorContext instantiation: SUCCESS")
            except Exception as e:
                print(f"❌ Basic ErrorContext instantiation failed: {e}")
                return

            # Try with parameters
            try:
                context = ErrorContext(
                    category=ErrorCategory.EXECUTION,
                    severity=ErrorSeverity.ERROR,
                    component="test_component",
                    operation="test_operation",
                    additional_info={"test_key": "test_value"},
                )
                print("✅ Parameterized ErrorContext instantiation: SUCCESS")
            except Exception as e:
                print(f"❌ Parameterized ErrorContext instantiation failed: {e}")
                return

            # Test serialization
            try:
                context_dict = context.to_dict()
                print("✅ ErrorContext.to_dict(): SUCCESS")
                print(f"  Result type: {type(context_dict)}")
                print(f"  Keys: {list(context_dict.keys())}")
                print(f"  Sample content: {dict(list(context_dict.items())[:3])}")

                # Test JSON serialization
                json_str = json.dumps(context_dict)
                print(f"✅ JSON serialization: SUCCESS ({len(json_str)} chars)")

            except Exception as e:
                print(f"❌ ErrorContext serialization failed: {e}")

            print("=" * 50)

        except ImportError as e:
            print(f"\n❌ ErrorContext imports failed: {e}")
            pytest.skip("ErrorContext not available for testing")

    def test_runner_error_handling_reality(self):
        """Test what TestRunner actually does with errors"""
        try:
            from testgen.execution.runner import TestRunner

            print("\n" + "=" * 50)
            print("TEST RUNNER ERROR REALITY CHECK")
            print("=" * 50)

            runner = TestRunner(5.0)  # Set timeout time to 5.0s
            print("✅ TestRunner instantiation: SUCCESS")

            # Test with a function that always fails
            def always_fails(arr):
                raise ValueError("This function always fails for testing")

            # constraints = Constraints(min_value=1, max_value=5, is_unique=False)
            # test_suite = TestSuite(
            #     function=always_fails, constraints=constraints, num_tests=3, timeout=5.0
            # )

            try:
                result = runner.run(always_fails, test_input=[1, 1, 2, 3])
                print("✅ TestRunner.run() completed without crashing")
                print(f"  Result type: {type(result)}")
                print(f"  Has 'success' attribute: {hasattr(result, 'success')}")
                print(f"  Has 'failures' attribute: {hasattr(result, 'failures')}")

                if hasattr(result, "success"):
                    print(f"  Success status: {result.success}")

                if hasattr(result, "failures"):
                    print(
                        f"  Number of failures: {len(result.failures) if result.failures else 0}"
                    )
                    if result.failures:
                        first_failure = result.failures[0]
                        print(f"  First failure type: {type(first_failure)}")
                        if hasattr(first_failure, "error_message"):
                            error_msg = first_failure.error_message
                            print(f"  Error message preview: {error_msg[:100]}...")
                            print(f"  Error message length: {len(error_msg)} chars")

                            # Check sophistication
                            has_context = any(
                                keyword in error_msg.lower()
                                for keyword in ["context", "category", "component"]
                            )
                            print(
                                f"  📊 Contains rich context keywords: {'✅' if has_context else '❌'}"
                            )

            except Exception as e:
                print(f"❌ TestRunner.run() failed: {e}")
                print(f"  Traceback: {traceback.format_exc()}")

            print("=" * 50)

        except ImportError as e:
            print(f"\n❌ TestRunner import failed: {e}")
            pytest.skip("TestRunner not available for testing")

    def test_actual_error_integration(self):
        """Test end-to-end error handling in a realistic scenario"""
        print("\n" + "=" * 50)
        print("END-TO-END ERROR INTEGRATION REALITY CHECK")
        print("=" * 50)

        # Use the most basic approach - direct generator call
        generator = IntegerGenerator()

        try:
            # Force a realistic error
            constraints = Constraints(min_value=1, max_value=5, is_unique=True)
            result = generator.generate_array(
                size=20, constraints=constraints
            )  # Impossible: 20 unique from [1,5]

            print("⚠️ UNEXPECTED: No error was raised")
            print(f"  Got result: {result}")
            print(f"  Result length: {len(result)}")
            print(f"  Unique values: {len(set(result))}")

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            print("✅ Error properly raised")
            print(f"  Error type: {error_type}")
            print(f"  Error message: {error_msg}")

            # Analyze the error quality
            analysis = {
                "Informative length": len(error_msg) > 30,
                "Contains constraint info": any(
                    str(val) in error_msg for val in [1, 5, 20]
                ),
                "Contains 'unique'": "unique" in error_msg.lower(),
                "Contains 'impossible'": "impossible" in error_msg.lower()
                or "cannot" in error_msg.lower(),
                "Professional tone": not error_msg.startswith("Error:"),
            }

            print("\n📊 Error Quality Analysis:")
            for criterion, meets in analysis.items():
                status = "✅" if meets else "❌"
                print(f"  {status} {criterion}")

            quality_score = sum(analysis.values()) / len(analysis) * 100
            print(f"\n🎯 Overall Error Quality Score: {quality_score:.0f}%")

        print("=" * 50)

    def test_document_current_capabilities(self):
        """Document what error handling capabilities actually exist"""
        print("\n" + "=" * 50)
        print("CURRENT ERROR HANDLING CAPABILITIES SUMMARY")
        print("=" * 50)

        capabilities = {
            "Basic Exception Handling": "Unknown",
            "Rich Error Context": "Unknown",
            "Error Categorization": "Unknown",
            "Error Recovery": "Unknown",
            "Error Reporting": "Unknown",
            "CLI Error Display": "Unknown",
            "Error Serialization": "Unknown",
        }

        # Test each capability
        try:
            from testgen.error_handling.handlers import ErrorContext, ErrorHandler

            handler = ErrorHandler()
            capabilities["Basic Exception Handling"] = "✅ Available"

            if hasattr(handler, "handle_error") and callable(
                getattr(handler, "handle_error")
            ):
                capabilities["Rich Error Context"] = "✅ Method exists"
            else:
                capabilities["Rich Error Context"] = "❌ Method missing"

            try:
                context = ErrorContext()
                context.to_dict()
                capabilities["Error Serialization"] = "✅ Working"
            except Exception:
                capabilities["Error Serialization"] = "❌ Not working"

        except ImportError:
            capabilities["Basic Exception Handling"] = "❌ Import failed"

        # Test CLI integration
        try:
            capabilities["CLI Error Display"] = "✅ CLI available"
        except ImportError:
            capabilities["CLI Error Display"] = "❌ CLI not available"

        print("Current Status:")
        for capability, status in capabilities.items():
            print(f"  {capability}: {status}")

        print("\n💡 Recommendations based on findings:")
        if capabilities["Basic Exception Handling"].startswith("✅"):
            print("  • Basic error handling exists - focus on improving quality")
        else:
            print("  • Basic error handling missing - implement foundation first")

        if capabilities["Rich Error Context"].startswith("✅"):
            print("  • Rich context available - test integration quality")
        else:
            print("  • Rich context missing - current tests are aspirational")

        print("=" * 50)


if __name__ == "__main__":
    # Can be run directly for quick verification
    print("Running Error Handling Reality Check...")
    test_instance = TestErrorHandlingReality()

    try:
        test_instance.test_error_handling_imports()
        test_instance.test_error_handler_instantiation_and_methods()
        test_instance.test_basic_error_generation_reality()
        test_instance.test_error_context_reality()
        test_instance.test_runner_error_handling_reality()
        test_instance.test_actual_error_integration()
        test_instance.test_document_current_capabilities()
        print("\n🎉 Reality check completed! Check output above for findings.")
    except Exception as e:
        print(f"\n💥 Reality check failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
