#!/usr/bin/env python3
"""
Test runner for FastPPI.
Runs all tests and reports results.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_pandas_operations import PandasOperationTester
from tests.test_string_operations import StringOperationTester


def main():
    """Run all tests."""
    print("FastPPI Test Suite")
    print("=" * 80)
    print()
    
    all_passed = True
    
    # Run pandas tests
    print("=" * 80)
    print("PANDAS OPERATIONS TESTS")
    print("=" * 80)
    print()
    pandas_tester = PandasOperationTester()
    pandas_success = pandas_tester.run_all_tests()
    all_passed = all_passed and pandas_success
    
    print()
    print()
    
    # Run string tests
    print("=" * 80)
    print("STRING OPERATIONS TESTS")
    print("=" * 80)
    print()
    string_tester = StringOperationTester()
    string_success = string_tester.run_all_tests()
    all_passed = all_passed and string_success
    
    print()
    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

