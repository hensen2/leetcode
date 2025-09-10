# Test 1: Import error handling components
try:
    from testgen.error_handling import ErrorHandler
    handler = ErrorHandler()
    print("✅ ErrorHandler imported")
except Exception as e:
    print(f"❌ ErrorHandler import failed: {e}")

# Test 2: Generate an error and see handling
try:
    from testgen.core.generators import IntegerGenerator
    from testgen.core.models import Constraints
    
    bad_constraints = Constraints(min_value=100, max_value=1)  # Invalid
    gen = IntegerGenerator()
    result = gen.generate_array(5, bad_constraints)
except Exception as e:
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
    # Check if error message contains rich context
    error_str = str(e)
    has_context = any(word in error_str.lower() for word in ['constraint', 'context', 'category'])
    print(f"Rich context present: {has_context}")
    
    # Check if error has additional attributes (rich error handling)
    attrs = [attr for attr in dir(e) if not attr.startswith('_')]
    print(f"Error attributes: {attrs}")
