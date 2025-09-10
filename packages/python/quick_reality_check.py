"""
Quick reality check script for error handling

Run this to immediately understand what error handling actually exists
vs. what's documented/tested.

Usage:
    cd packages/python/
    uv run python quick_reality_check.py
"""

import sys
import traceback
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path.cwd()))


def check_imports():
    """Check what error handling components actually import"""
    print("🔍 CHECKING IMPORTS...")
    print("-" * 40)

    components = [
        ("ErrorHandler", "testgen.error_handling.handlers", "ErrorHandler"),
        ("ErrorContext", "testgen.error_handling.handlers", "ErrorContext"),
        ("ErrorCategory", "testgen.error_handling.handlers", "ErrorCategory"),
        ("ExecutionError", "testgen.error_handling.handlers", "ExecutionError"),
        ("IntegerGenerator", "testgen.core.generators", "IntegerGenerator"),
        ("Constraints", "testgen.core.models", "Constraints"),
        ("TestRunner", "testgen.execution.runner", "TestRunner"),
    ]

    results = {}
    for name, module, class_name in components:
        try:
            exec(f"from {module} import {class_name}")
            results[name] = "✅ Available"
        except ImportError as e:
            results[name] = f"❌ Import failed: {str(e)[:50]}..."
        except Exception as e:
            results[name] = f"⚠️ Other error: {str(e)[:50]}..."

    for name, status in results.items():
        print(f"  {name}: {status}")

    return results


def check_error_handler():
    """Check ErrorHandler instantiation and methods"""
    print("\n🔧 CHECKING ERROR HANDLER...")
    print("-" * 40)

    try:
        from testgen.error_handling.handlers import ErrorHandler

        handler = ErrorHandler()
        print("✅ ErrorHandler instantiation: SUCCESS")

        methods = [m for m in dir(handler) if not m.startswith("_")]
        print(f"📋 Available methods: {methods}")

        # Test key methods
        key_methods = [
            "handle_error",
            "format_error",
            "get_error_summary",
            "clear_errors",
        ]
        for method in key_methods:
            if hasattr(handler, method):
                callable_status = (
                    "callable" if callable(getattr(handler, method)) else "not callable"
                )
                print(f"  ✅ {method}: exists ({callable_status})")
            else:
                print(f"  ❌ {method}: missing")

        return True
    except Exception as e:
        print(f"❌ ErrorHandler check failed: {e}")
        return False


def test_basic_errors():
    """Test basic error generation"""
    print("\n⚠️ TESTING BASIC ERROR GENERATION...")
    print("-" * 40)

    try:
        from testgen.core.generators import IntegerGenerator
        from testgen.core.models import Constraints

        generator = IntegerGenerator()

        # Test invalid constraints
        print("Testing invalid constraints (min > max)...")
        try:
            constraints = Constraints(min_value=100, max_value=1, is_unique=False)
            result = generator.generate_array(size=5, constraints=constraints)
            print(f"⚠️ No error raised! Got: {result}")
        except Exception as e:
            print(f"✅ Error raised: {type(e).__name__}: {e}")
            print(f"   Message length: {len(str(e))} chars")
            print(f"   Contains constraint values: {'100' in str(e) and '1' in str(e)}")

        # Test impossible unique constraint
        print("\nTesting impossible unique constraint...")
        try:
            constraints = Constraints(min_value=1, max_value=2, is_unique=True)
            result = generator.generate_array(
                size=10, constraints=constraints
            )  # Need 10 unique from [1,2]
            print(f"⚠️ No error raised! Got: {result}")
        except Exception as e:
            print(f"✅ Error raised: {type(e).__name__}: {e}")

        return True
    except ImportError as e:
        print(f"❌ Could not import generators: {e}")
        return False


def test_error_context():
    """Test ErrorContext functionality"""
    print("\n📝 TESTING ERROR CONTEXT...")
    print("-" * 40)

    try:
        from testgen.error_handling.handlers import (
            ErrorCategory,
            ErrorContext,
            ErrorSeverity,
        )

        # Basic instantiation
        context = ErrorContext()
        print("✅ Basic ErrorContext instantiation: SUCCESS")

        # Parameterized instantiation
        context = ErrorContext(
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.ERROR,
            component="test_component",
        )
        print("✅ Parameterized ErrorContext instantiation: SUCCESS")

        # Test serialization
        context_dict = context.to_dict()
        print(f"✅ Serialization: SUCCESS (keys: {list(context_dict.keys())})")

        return True
    except Exception as e:
        print(f"❌ ErrorContext test failed: {e}")
        return False


def test_runner_integration():
    """Test TestRunner error integration"""
    print("\n🏃 TESTING RUNNER ERROR INTEGRATION...")
    print("-" * 40)

    try:
        from testgen.execution.runner import TestRunner

        runner = TestRunner()
        print("✅ TestRunner instantiation: SUCCESS")

        # Test with failing function
        def always_fails(arr):
            raise ValueError("Test failure for verification")

        # constraints = Constraints(min_value=1, max_value=5, is_unique=False)
        # test_suite = TestSuite(
        #     function=always_fails, constraints=constraints, num_tests=2, timeout=5.0
        # )

        result = runner.run(always_fails, test_input=[1, 1, 2, 3])

        print("✅ TestRunner.run() completed without crashing")
        print(f"   Result type: {type(result)}")
        print(f"   Success: {getattr(result, 'success', 'unknown')}")
        print(f"   Failures: {len(getattr(result, 'failures', []))}")

        # Check error message quality
        if hasattr(result, "failures") and result.failures:
            failure = result.failures[0]
            if hasattr(failure, "error_message"):
                msg = failure.error_message
                print(f"   Error message length: {len(msg)} chars")
                print(f"   Contains 'Test failure': {'Test failure' in msg}")
                print(f"   Message preview: {msg[:100]}...")

        return True
    except Exception as e:
        print(f"❌ Runner integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run complete reality check"""
    print("🎯 ERROR HANDLING REALITY CHECK")
    print("=" * 50)
    print("This will test what error handling actually exists")
    print("vs. what's documented or assumed in tests.")
    print("=" * 50)

    results = {
        "imports": check_imports(),
        "error_handler": check_error_handler(),
        "basic_errors": test_basic_errors(),
        "error_context": test_error_context(),
        "runner_integration": test_runner_integration(),
    }

    print("\n📊 SUMMARY")
    print("=" * 50)

    # Count successes
    import_success = sum(
        1 for status in results["imports"].values() if status.startswith("✅")
    )
    total_imports = len(results["imports"])

    print(f"Imports: {import_success}/{total_imports} successful")
    print(f"ErrorHandler: {'✅ Working' if results['error_handler'] else '❌ Issues'}")
    print(f"Basic Errors: {'✅ Working' if results['basic_errors'] else '❌ Issues'}")
    print(f"Error Context: {'✅ Working' if results['error_context'] else '❌ Issues'}")
    print(
        f"Runner Integration: {'✅ Working' if results['runner_integration'] else '❌ Issues'}"
    )

    # Overall assessment
    working_components = sum(
        [
            results["error_handler"],
            results["basic_errors"],
            results["error_context"],
            results["runner_integration"],
        ]
    )

    print(f"\n🎯 Overall Status: {working_components}/4 components working")

    if working_components >= 3:
        print("💚 GOOD: Most error handling is working!")
        print("   Focus on improving test accuracy and edge cases.")
    elif working_components >= 2:
        print("🟡 PARTIAL: Some error handling working.")
        print("   Focus on fixing broken components before improving tests.")
    else:
        print("🔴 ISSUES: Major error handling problems.")
        print("   Focus on basic implementation before sophisticated testing.")

    print("\n💡 Next steps:")
    if not results["error_handler"]:
        print("   1. Fix ErrorHandler instantiation and basic methods")
    if not results["basic_errors"]:
        print("   1. Fix basic error generation in generators")
    if not results["error_context"]:
        print("   1. Fix ErrorContext serialization")
    if not results["runner_integration"]:
        print("   1. Fix TestRunner error handling integration")

    print("   2. Update tests to match actual implementation")
    print("   3. Use working components to improve non-working ones")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️ Reality check interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Reality check crashed: {e}")
        print("This might indicate serious import or setup issues.")
        traceback.print_exc()
