/**
 * C implementation of string operations for FastPPI.
 * Supports regex, unicode normalization, string formatting, and sanitization.
 */

#ifndef STRING_C_H
#define STRING_C_H

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// String Data Structure
// ============================================================================

/**
 * Simple string structure for C operations.
 * For more complex operations, we may need to integrate with Series.
 */
typedef struct {
    char* data;           // String data (null-terminated)
    int64_t length;       // String length (excluding null terminator)
    bool is_allocated;    // Whether data was allocated (needs free)
} FastString;

/**
 * Create a new FastString from a C string.
 * If copy is true, allocates new memory. Otherwise, uses the provided pointer.
 */
FastString* fast_string_create(const char* str, bool copy);

/**
 * Free a FastString.
 */
void fast_string_free(FastString* fs);

// ============================================================================
// Prompt Sanitization (Word Checking)
// ============================================================================

/**
 * Check if any word from a list is contained in a string.
 * words: null-terminated array of word strings
 * num_words: number of words in the array
 * Returns: true if any word is found, false otherwise
 */
bool string_contains_any_word(const char* text, const char** words, int num_words);

/**
 * Check if a specific word/substring is in a string.
 * Returns: true if found, false otherwise
 */
bool string_contains(const char* text, const char* substring);

/**
 * Check if all words from a list are contained in a string.
 * Returns: true if all words are found, false otherwise
 */
bool string_contains_all_words(const char* text, const char** words, int num_words);

// ============================================================================
// String Formatting
// ============================================================================

/**
 * Format a string using format specifiers.
 * template: format string (e.g., "Hello {0}" or "Hello {name}")
 * args: array of argument strings
 * num_args: number of arguments
 * Returns: newly allocated formatted string (caller must free)
 */
char* string_format(const char* template, const char** args, int num_args);

/**
 * Format a string with named arguments.
 * template: format string (e.g., "Hello {name}")
 * names: array of argument names
 * values: array of argument values
 * num_args: number of arguments
 * Returns: newly allocated formatted string (caller must free)
 */
char* string_format_named(const char* template, const char** names, const char** values, int num_args);

// ============================================================================
// Regular Expression Operations
// ============================================================================

/**
 * Search for a pattern in a string.
 * pattern: regex pattern (simplified - basic patterns only)
 * text: text to search
 * Returns: true if pattern matches, false otherwise
 * Note: For full regex support, consider linking with a regex library like PCRE
 */
bool regex_search(const char* pattern, const char* text);

/**
 * Match a pattern at the beginning of a string.
 * Returns: true if pattern matches at start, false otherwise
 */
bool regex_match(const char* pattern, const char* text);

/**
 * Find all matches of a pattern in a string.
 * pattern: regex pattern
 * text: text to search
 * matches: output array of matched strings (caller must free each element and array)
 * num_matches: output number of matches found
 * Returns: 0 on success, -1 on error
 */
int regex_findall(const char* pattern, const char* text, char*** matches, int* num_matches);

/**
 * Substitute all matches of a pattern with a replacement string.
 * pattern: regex pattern
 * text: input text
 * replacement: replacement string
 * Returns: newly allocated string with replacements (caller must free)
 */
char* regex_sub(const char* pattern, const char* text, const char* replacement);

// ============================================================================
// Unicode Normalization
// ============================================================================

/**
 * Normalize a Unicode string.
 * form: normalization form ("NFC", "NFD", "NFKC", "NFKD")
 * text: input text (UTF-8 encoded)
 * Returns: newly allocated normalized string (caller must free)
 * Note: Requires Unicode library (e.g., ICU) for full support
 */
char* unicode_normalize(const char* form, const char* text);

/**
 * Check if a string is already normalized.
 * form: normalization form to check
 * text: input text
 * Returns: true if normalized, false otherwise
 */
bool unicode_is_normalized(const char* form, const char* text);

#endif // STRING_C_H

