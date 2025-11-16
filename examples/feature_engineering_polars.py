"""
Polars implementation of feature_engineering.py for benchmarking.
"""
import polars as pl


def preprocess_data_polars(csv_path: str):
    """
    Polars version of the preprocessing pipeline.
    """
    # Load CSV
    df = pl.read_csv(csv_path)

    # 1. Normalize the string field to lowercase
    df = df.with_columns(
        pl.col("city").cast(pl.Utf8).str.to_lowercase().alias("city")
    )

    # 2. Create a new column using a condition (high income flag)
    # Polars doesn't have apply with lambda, but we can use when/then/otherwise
    df = df.with_columns(
        pl.when((pl.col("income").is_not_null()) & (pl.col("income") > 60000))
        .then(1)
        .otherwise(0)
        .alias("high_income_flag")
    )

    # 3. Apply normalization to age (subtract mean)
    age_mean = df.select(pl.col("age").mean()).item()
    df = df.with_columns(
        pl.when(pl.col("age").is_not_null())
        .then(pl.col("age") - age_mean)
        .otherwise(pl.col("age"))
        .alias("age_centered")
    )

    # 4. Drop rows with negative or corrupted values
    df = df.filter(
        pl.col("income").is_null() | (pl.col("income") >= 0)
    )

    # 5. Fill missing categorical values
    df = df.with_columns(
        pl.col("city").fill_null("unknown").alias("city")
    )

    # Return processed data
    X = df.drop("purchase_amount")
    y = df.select("purchase_amount")

    return X, y


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "examples/example_data.csv"
    X_processed, y = preprocess_data_polars(csv_path)
    print(f"Processed {len(X_processed)} rows")
    print(f"Columns: {X_processed.columns}")

