import argparse
import pandas as pd


def preprocess_data(csv_path: str):
    # Load CSV
    df = pd.read_csv(csv_path)

    # -----------------------------
    # PANDAS-SPECIFIC PREPROCESSING
    # -----------------------------

    # 1. Normalize the string field to lowercase
    df["city"] = df["city"].astype(str).str.lower()

    # 2. Create a new column using a lambda apply
    # Example: flag rows where income is above 60000
    df["high_income_flag"] = df["income"].apply(
        lambda x: 1 if pd.notnull(x) and x > 60000 else 0
    )

    # 3. Apply a lambda to normalize age (subtract mean)
    age_mean = df["age"].mean(skipna=True)
    df["age_centered"] = df["age"].apply(
        lambda x: x - age_mean if pd.notnull(x) else x
    )

    # 4. Drop rows with negative or corrupted values (example)
    df = df[df["income"].isna() | (df["income"] >= 0)]

    # 5. Fill missing categorical values
    df["city"] = df["city"].fillna("unknown")

    # Return processed data
    X = df.drop(columns=["purchase_amount"])
    y = df["purchase_amount"]

    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that takes arguments.")
    parser.add_argument("csv_path", type=str, help="The path to the CSV file")
    args = parser.parse_args()
    X_processed, y = preprocess_data(args.csv_path)
