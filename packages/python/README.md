# LeetCode Test Generator

A functional test case generator for LeetCode problems with professional CLI interface and efficient core generation capabilities.

## 🚀 **Quick Start**

```bash
# Install dependencies
cd packages/python/
uv sync
uv pip install -e .

# Generate test cases
uv run python -m testgen array -n 10 --unique --output results.json
uv run python -m testgen tree -n 5 --balanced
uv run python -m testgen graph -n 3 --connected
```

## ✨ **Key Features**

- **Professional CLI** with progress indicators and rich output formatting
- **Multiple Data Types** including arrays, strings, trees, graphs, matrices, linked lists
- **Memory Efficient** generation from large constraint ranges
- **Advanced Constraints** supporting unique values, sorting, balanced trees, connected graphs
- **JSON Export** with comprehensive metadata and analysis
- **Validation System** for testing against custom functions

## 📊 **Example Output**

```bash
$ uv run python -m testgen array -n 3 --unique
🚀 Starting array generation...
📊 Generating 3 test cases
📊 TEST CASE ANALYSIS:
   📏 Size range: 14 - 40 (avg: 25.3)
📋 GENERATED TEST CASES:
🧪 Test Case 1: [7, 5, 6, 5, 3, 10, 4, 10, 2, 5]... (length: 22)
```

## 📦 **Installation Requirements**

- Python 3.13+
- UV package manager
- Project structure: `packages/python/` contains the main package

## 📖 **Documentation**

- **[Current Status](CURRENT_STATUS.md)** - Verified features and capabilities
- **CLI Help** - `uv run python -m testgen --help` for complete command reference
- **Development History** - See `docs/history/` for development progress

## 🎯 **Usage Scenarios**

### **Basic Generation**

```bash
# Simple arrays
uv run python -m testgen array -n 5

# Constrained generation
uv run python -m testgen array --min-value 1 --max-value 100 --unique --sorted
```

### **Advanced Data Structures**

```bash
# Balanced binary trees
uv run python -m testgen tree -n 10 --balanced --bst

# Connected graphs
uv run python -m testgen graph -n 5 --connected --weighted
```

### **Validation and Testing**

```bash
# Test against your solution
uv run python -m testgen validate --validate-function my_solution.two_sum

# Generate with error reporting
uv run python -m testgen array -n 100 --error-report json
```

## 🔧 **Architecture**

```
testgen/
├── core/           # Data models, generators, constraints
├── cli/            # Command-line interface
├── execution/      # Test runners and analysis
├── error_handling/ # Error handling system
├── plugins/        # Plugin system
└── patterns/       # LeetCode-specific patterns
```

## ⚡ **Performance**

- **Memory Efficient**: Tested with 1000 unique values from 1M range
- **Fast Generation**: Sub-second performance for moderate test cases
- **Scalable Constraints**: Handles large min/max value ranges effectively

## 📄 **Current Status**

**Working Well**: CLI interface, core generation, memory efficiency, JSON output  
**Basic**: Error handling (simple exceptions)  
**Untested**: Full plugin system functionality

For detailed status information, see [CURRENT_STATUS.md](CURRENT_STATUS.md).

## 🤝 **Contributing**

This is a functional tool suitable for LeetCode practice and test case generation. For current capabilities and limitations, refer to the current status documentation.
