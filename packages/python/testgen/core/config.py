"""
Configuration module for test case generator
Centralizes all magic numbers and default values
"""


class Config:
    """Central configuration for test case generator"""

    # Timing constraints
    DEFAULT_TIMEOUT = 1.0
    MAX_TIMEOUT = 10.0

    # Size constraints
    DEFAULT_MIN_SIZE = 0
    DEFAULT_MAX_SIZE = 10000
    DEFAULT_MATRIX_MAX_SIZE = 100
    DEFAULT_TREE_MAX_SIZE = 50
    DEFAULT_GRAPH_MAX_SIZE = 50

    # Value constraints
    DEFAULT_MIN_VALUE = -(10**9)
    DEFAULT_MAX_VALUE = 10**9
    DEFAULT_WEIGHT_MIN = 1
    DEFAULT_WEIGHT_MAX = 100

    # Array generation
    DEFAULT_ARRAY_MIN_LENGTH = 0
    DEFAULT_ARRAY_MAX_LENGTH = 10**4

    # String generation
    DEFAULT_STRING_MIN_LENGTH = 0
    DEFAULT_STRING_MAX_LENGTH = 10**4
    DEFAULT_CHARSET = "abcdefghijklmnopqrstuvwxyz"

    # Performance analysis
    PERF_DEFAULT_SIZES = [10, 50, 100, 500, 1000]
    PERF_DEFAULT_ITERATIONS = 3
    PERF_COMPLEXITY_TOLERANCE = 0.3  # 30% tolerance for noise

    # Graph generation
    GRAPH_MAX_ATTEMPTS_MULTIPLIER = 10  # max_attempts = num_edges * multiplier

    # Test runner
    RUNNER_MAX_RESULTS_TO_DISPLAY = 5

    # CLI defaults
    CLI_DEFAULT_NUM_TESTS = 10
    CLI_DEFAULT_MIN_SIZE = 0
    CLI_DEFAULT_MAX_SIZE = 100
    CLI_DEFAULT_MIN_VALUE = -1000
    CLI_DEFAULT_MAX_VALUE = 1000
    CLI_MAX_DISPLAY_LENGTH = 10

    # Problem-specific defaults
    TWO_SUM_DEFAULT_TESTS = 10
    TWO_SUM_MIN_SIZE = 2
    TWO_SUM_MAX_SIZE = 100
    TWO_SUM_SOLUTION_PROBABILITY = 0.8

    SLIDING_WINDOW_DEFAULT_TESTS = 10
    SLIDING_WINDOW_MIN_SIZE = 1
    SLIDING_WINDOW_MAX_SIZE = 1000

    BINARY_SEARCH_DEFAULT_TESTS = 10
    BINARY_SEARCH_MIN_SIZE = 1
    BINARY_SEARCH_MAX_SIZE = 1000
    BINARY_SEARCH_TARGET_EXISTS_PROBABILITY = 0.7

    PARENTHESES_DEFAULT_TESTS = 10
    PARENTHESES_MIN_LENGTH = 2
    PARENTHESES_MAX_LENGTH = 100
    PARENTHESES_VALID_PROBABILITY = 0.5

    # Supported bracket types
    BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
    ALL_BRACKETS = "()[]{}"
