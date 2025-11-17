/**
 * C implementation of pandas operations for FastPPI.
 * This header defines the interface for pandas DataFrame and Series operations.
 */

#ifndef PANDAS_C_H
#define PANDAS_C_H

#include "data_structures.h"
#include <stdint.h>

// ============================================================================
// DataFrame Operations
// ============================================================================

/**
 * Compute mean of DataFrame columns.
 * Returns array of means, one per column.
 */
double* pandas_df_mean(const DataFrame* df);

/**
 * Fill missing values in DataFrame.
 * strategy: "mean", "median", "most_frequent", or value
 */
DataFrame* pandas_df_fillna(const DataFrame* df, const char* strategy, double fill_value);

/**
 * Read CSV file into DataFrame.
 */
DataFrame* pandas_read_csv(const char* filename);

/**
 * Make HTTP GET request and parse JSON array into DataFrame.
 * URL: HTTP endpoint URL
 * Returns DataFrame* with data from JSON array, or NULL on error.
 */
DataFrame* pandas_http_get_json(const char* url);

/**
 * Get column from DataFrame by name.
 * Returns Series containing the column data.
 */
Series* pandas_df_getitem(const DataFrame* df, const char* column_name);

// ============================================================================
// Series Operations
// ============================================================================

/**
 * Compute mean of Series.
 * Returns mean value, or NaN if all values are null.
 */
double pandas_series_mean(const Series* series);

/**
 * Fill missing values in Series with numeric value.
 * Returns new Series with filled values.
 */
Series* pandas_series_fillna(const Series* series, double fill_value, const char* strategy);

/**
 * Fill missing values in Series with string value.
 * Returns new Series with filled values (for string dtype Series).
 */
Series* pandas_series_fillna_str(const Series* series, const char* fill_value);

/**
 * Apply function to Series elements.
 * For numeric operations: use function pointer (future enhancement)
 * For now: simplified for common operations
 */
Series* pandas_series_apply(const Series* series, const char* op_type);

/**
 * Convert Series to different dtype.
 * target_dtype: 'f' = float64, 'i' = int64, 's' = string
 */
Series* pandas_series_astype(const Series* series, char target_dtype);

/**
 * Convert Series strings to lowercase.
 * Only works for string dtype Series.
 */
Series* pandas_series_str_lower(const Series* series);

/**
 * Convert Series strings to uppercase.
 * Only works for string dtype Series.
 */
Series* pandas_series_str_upper(const Series* series);

/**
 * Strip whitespace from Series strings.
 * Only works for string dtype Series.
 */
Series* pandas_series_str_strip(const Series* series);

/**
 * Check for null values in Series.
 * Returns new boolean Series where True indicates null.
 */
Series* pandas_series_isna(const Series* series);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Extract numeric array from DataFrame column.
 */
double* dataframe_column_to_array(const DataFrame* df, const char* column_name, int64_t* out_length);

/**
 * Extract numeric array from Series.
 */
double* series_to_array(const Series* series, int64_t* out_length);

/**
 * Extract all numeric data from DataFrame as flattened array.
 * Data is stored row-major (row0_col0, row0_col1, ..., row1_col0, ...)
 * Only extracts numeric columns (dtype 'f' or 'i').
 * Returns array and sets out_length to total number of elements.
 */
double* dataframe_to_array(const DataFrame* df, int64_t* out_length);

#endif // PANDAS_C_H

