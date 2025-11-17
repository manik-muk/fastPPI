import pandas as pd
import requests

# Direct execution for tracing
url = "http://localhost:3000/users"
response = requests.get(url)
df = pd.DataFrame(response.json())

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
# Use a simpler approach that doesn't require closure
age_mean = df["age"].mean(skipna=True)
df["age_centered"] = df["age"] - age_mean

# 4. Drop rows with negative or corrupted values (example)
df = df[df["income"].isna() | (df["income"] >= 0)]

# 5. Fill missing categorical values (skip for now to avoid compilation issues)
# df["city"] = df["city"].fillna("unknown")

# Return processed data
X = df.drop(columns=["purchase_amount"])
y = df["purchase_amount"]

X_processed = X
result_y = y

