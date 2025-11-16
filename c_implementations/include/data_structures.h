/**
 * Common data structures for pandas C implementations.
 */

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

// Forward declarations
typedef struct DataFrame DataFrame;
typedef struct Series Series;

/**
 * Series - represents a pandas Series
 * Similar to a 1D array with optional index
 */
struct Series {
    double* data;           // Data array
    char** string_data;     // For string series (alternative to data)
    int64_t length;         // Number of elements
    int64_t* index;         // Optional index array
    bool has_nulls;         // Whether series contains NaN/None
    int64_t* null_mask;     // Bitmask for null values (1 = null)
    char dtype;             // 'f' = float64, 'i' = int64, 's' = string, 'b' = bool
    char* name;             // Series name
};

/**
 * DataFrame - represents a pandas DataFrame
 * Collection of named Series (columns)
 */
struct DataFrame {
    Series** columns;       // Array of Series pointers
    char** column_names;    // Column names
    int64_t num_columns;    // Number of columns
    int64_t num_rows;       // Number of rows (all columns have same length)
    int64_t* index;         // Optional index array
};

// Memory management
Series* series_create(int64_t length, char dtype);
void series_free(Series* series);
DataFrame* dataframe_create(int64_t num_rows, int64_t num_columns);
void dataframe_free(DataFrame* df);
void dataframe_add_column(DataFrame* df, const char* name, Series* series);

// Utility functions
bool is_null(const Series* series, int64_t idx);
void set_null(Series* series, int64_t idx, bool is_null);

#endif // DATA_STRUCTURES_H

