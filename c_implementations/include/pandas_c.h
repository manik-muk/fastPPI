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

/**
 * Concatenate DataFrames along axis (0 = rows, 1 = columns).
 * dfs: Array of DataFrame pointers
 * num_dfs: Number of DataFrames to concatenate
 * axis: 0 for vertical (rows), 1 for horizontal (columns)
 * Returns new DataFrame with concatenated data.
 */
DataFrame* pandas_concat(DataFrame** dfs, int num_dfs, int axis);

/**
 * Sort DataFrame by column values.
 * by: Column name to sort by
 * ascending: 1 for ascending, 0 for descending
 * Returns new sorted DataFrame.
 */
DataFrame* pandas_df_sort_values(const DataFrame* df, const char* by, int ascending);

/**
 * Group DataFrame by column(s).
 * by: Column name(s) to group by (for now, single column name)
 * Returns a GroupBy object (simplified - just returns DataFrame for now).
 * Note: Full groupby with aggregations requires more complex implementation.
 */
DataFrame* pandas_df_groupby(const DataFrame* df, const char* by);

/**
 * Convert categorical Series to dummy/indicator variables (one-hot encoding).
 * Returns DataFrame with one column per unique value, filled with 1.0/0.0.
 */
DataFrame* pandas_series_get_dummies(const Series* series);

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

/**
 * Extract day component from datetime Series.
 * Input Series must have dtype 'd' (datetime64).
 * Returns new int64 Series with day values (1-31).
 */
Series* pandas_series_dt_day(const Series* series);

/**
 * Extract month component from datetime Series.
 * Input Series must have dtype 'd' (datetime64).
 * Returns new int64 Series with month values (1-12).
 */
Series* pandas_series_dt_month(const Series* series);

/**
 * Extract year component from datetime Series.
 * Input Series must have dtype 'd' (datetime64).
 * Returns new int64 Series with year values.
 */
Series* pandas_series_dt_year(const Series* series);

/**
 * Convert Series to datetime64.
 * Input Series can be string Series (dtype 's') with date strings.
 * Returns new Series with dtype 'd' (datetime64, stored as int64 nanoseconds since epoch).
 */
Series* pandas_to_datetime(const Series* series);

/**
 * Compute rolling window mean of Series.
 * window_size: Size of the rolling window
 * Returns new Series with rolling mean values.
 */
Series* pandas_series_rolling_mean(const Series* series, int64_t window_size);

/**
 * Compute rolling window sum of Series.
 * window_size: Size of the rolling window
 * Returns new Series with rolling sum values.
 */
Series* pandas_series_rolling_sum(const Series* series, int64_t window_size);

/**
 * Compute exponential moving average (EMA) of Series.
 * span: Span parameter for EMA (if provided, alpha is calculated from span)
 * alpha: Alpha parameter for EMA (decay factor, 0 < alpha <= 1)
 * If both span and alpha are provided, span takes precedence.
 * Returns new Series with EMA values.
 */
Series* pandas_series_ewm_mean(const Series* series, double span, double alpha);

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

