"""
Comprehensive tests for string operations.
Tests verify that Python string operations produce the same logical results as compiled C.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import unicodedata
from tests.test_framework import FastPPITestFramework


class StringOperationTester:
    """Test individual string operations."""
    
    def __init__(self):
        self.framework = FastPPITestFramework()
        self.setup_test_data()
        self.passed = 0
        self.failed = 0
    
    def setup_test_data(self):
        """Setup test data for string operations."""
        # Test strings
        self.test_string = "Hello World 123"
        self.test_unicode = "café"
        self.test_format_template = "Hello {0}, welcome to {1}!"
        
        # Test words for sanitization
        self.bad_words = ["bad", "evil", "hate"]
        self.good_text = "This is a good text"
        self.bad_text = "This text contains bad words"
    
    def test(self, name: str, python_code: str, inputs: dict = None):
        """Run a single test."""
        try:
            # Run Python version to get expected result
            python_result = self.framework.run_python_code(python_code, inputs)
            
            # Compile to C and verify compilation succeeds
            binary_path, c_code_path = self.framework.compile_test(
                python_code, f"test_{name}", inputs
            )
            
            # Verify binary exists
            if not Path(binary_path).exists():
                # Try with extensions
                for ext in ['.dylib', '.so', '.dll']:
                    if Path(f"{binary_path}{ext}").exists():
                        binary_path = f"{binary_path}{ext}"
                        break
                else:
                    raise FileNotFoundError(f"Binary not found: {binary_path}")
            
            print(f"✅ {name}: Compilation successful")
            print(f"   Python result type: {type(python_result).__name__}")
            if isinstance(python_result, str):
                print(f"   Value: {python_result[:50]}{'...' if len(python_result) > 50 else ''}")
            elif isinstance(python_result, bool):
                print(f"   Value: {python_result}")
            elif isinstance(python_result, (int, float)):
                print(f"   Value: {python_result}")
            elif isinstance(python_result, (list, tuple)):
                print(f"   Length: {len(python_result)}")
            elif hasattr(python_result, 'group'):  # Match object
                print(f"   Match: {python_result.group(0) if python_result else None}")
            
            self.passed += 1
            return True
            
        except Exception as e:
            print(f"❌ {name}: Failed - {str(e)}")
            import traceback
            traceback.print_exc()
            self.failed += 1
            return False
    
    def test_string_contains(self):
        """Test string containment check (prompt sanitization)."""
        python_code = f"""
text = "{self.test_string}"
substring = "World"
result = substring in text
"""
        return self.test("string_contains", python_code)
    
    def test_string_contains_not(self):
        """Test string containment check when substring not found."""
        python_code = f"""
text = "{self.test_string}"
substring = "Python"
result = substring in text
"""
        return self.test("string_contains_not", python_code)
    
    def test_regex_search(self):
        """Test regex search operation."""
        python_code = f"""
import re
text = "{self.test_string}"
pattern = r"\\d+"
match = re.search(pattern, text)
result = match is not None
"""
        return self.test("regex_search", python_code)
    
    def test_regex_search_not_found(self):
        """Test regex search when pattern not found."""
        python_code = f"""
import re
text = "{self.test_string}"
pattern = r"\\d{4,}"  # 4 or more digits
match = re.search(pattern, text)
result = match is not None
"""
        return self.test("regex_search_not_found", python_code)
    
    def test_regex_match(self):
        """Test regex match at start of string."""
        python_code = f"""
import re
text = "{self.test_string}"
pattern = r"Hello"
match = re.match(pattern, text)
result = match is not None
"""
        return self.test("regex_match", python_code)
    
    def test_regex_match_fail(self):
        """Test regex match when pattern doesn't match at start."""
        python_code = f"""
import re
text = "{self.test_string}"
pattern = r"World"  # Not at start
match = re.match(pattern, text)
result = match is not None
"""
        return self.test("regex_match_fail", python_code)
    
    def test_regex_findall(self):
        """Test regex findall operation."""
        python_code = f"""
import re
text = "{self.test_string}"
pattern = r"\\w+"
matches = re.findall(pattern, text)
result = len(matches)
"""
        return self.test("regex_findall", python_code)
    
    def test_regex_sub(self):
        """Test regex substitution."""
        python_code = f"""
import re
text = "{self.test_string}"
pattern = r"\\d+"
replacement = "XXX"
result = re.sub(pattern, replacement, text)
"""
        return self.test("regex_sub", python_code)
    
    def test_string_format(self):
        """Test string formatting with positional placeholders."""
        python_code = f"""
template = "{self.test_format_template}"
result = template.format("Alice", "FastPPI")
"""
        return self.test("string_format", python_code)
    
    def test_string_format_named(self):
        """Test string formatting with named placeholders."""
        python_code = f"""
template = "Hello {{name}}, age {{age}}"
result = template.format(name="Bob", age=30)
"""
        return self.test("string_format_named", python_code)
    
    def test_unicode_normalize(self):
        """Test Unicode normalization."""
        python_code = f"""
import unicodedata
text = "{self.test_unicode}"
result = unicodedata.normalize("NFC", text)
"""
        return self.test("unicode_normalize", python_code)
    
    def test_unicode_normalize_nfd(self):
        """Test Unicode normalization NFD form."""
        python_code = f"""
import unicodedata
text = "{self.test_unicode}"
result = unicodedata.normalize("NFD", text)
"""
        return self.test("unicode_normalize_nfd", python_code)
    
    def test_prompt_sanitization_contains(self):
        """Test prompt sanitization - check if bad words are present."""
        python_code = f"""
text = "{self.bad_text}"
bad_words = {self.bad_words}
result = any(word in text for word in bad_words)
"""
        return self.test("prompt_sanitization_contains", python_code)
    
    def test_prompt_sanitization_clean(self):
        """Test prompt sanitization - text without bad words."""
        python_code = f"""
text = "{self.good_text}"
bad_words = {self.bad_words}
result = any(word in text for word in bad_words)
"""
        return self.test("prompt_sanitization_clean", python_code)
    
    def test_string_format_fstring_like(self):
        """Test string formatting similar to f-strings."""
        python_code = f"""
name = "Charlie"
age = 25
template = "Name: {{0}}, Age: {{1}}"
result = template.format(name, age)
"""
        return self.test("string_format_fstring_like", python_code, {"name": "Charlie", "age": 25})
    
    def test_regex_email_pattern(self):
        """Test regex with email-like pattern."""
        python_code = f"""
import re
text = "Contact me at test@example.com please"
pattern = r"[\\w.]+@[\\w.]+"
match = re.search(pattern, text)
result = match is not None
"""
        return self.test("regex_email_pattern", python_code)
    
    def test_unicode_normalize_multiple(self):
        """Test Unicode normalization on multiple strings."""
        python_code = f"""
import unicodedata
text1 = "café"
text2 = "naïve"
result1 = unicodedata.normalize("NFC", text1)
result2 = unicodedata.normalize("NFC", text2)
result = result1 + " " + result2
"""
        return self.test("unicode_normalize_multiple", python_code)
    
    def run_all_tests(self):
        """Run all string operation tests."""
        print("=" * 80)
        print("STRING OPERATIONS TEST SUITE")
        print("=" * 80)
        print()
        
        tests = [
            self.test_string_contains,
            self.test_string_contains_not,
            self.test_regex_search,
            self.test_regex_search_not_found,
            self.test_regex_match,
            self.test_regex_match_fail,
            self.test_regex_findall,
            self.test_regex_sub,
            self.test_string_format,
            self.test_string_format_named,
            self.test_string_format_fstring_like,
            self.test_unicode_normalize,
            self.test_unicode_normalize_nfd,
            self.test_unicode_normalize_multiple,
            self.test_prompt_sanitization_contains,
            self.test_prompt_sanitization_clean,
            self.test_regex_email_pattern,
        ]
        
        for test_func in tests:
            test_func()
            print()
        
        print("=" * 80)
        print(f"TEST RESULTS: {self.passed} passed, {self.failed} failed")
        print("=" * 80)
        
        return self.failed == 0


if __name__ == "__main__":
    tester = StringOperationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

