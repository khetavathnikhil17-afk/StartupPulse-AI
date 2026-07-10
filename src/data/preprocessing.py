import pandas as pd
from pathlib import Path

# ----------------------------
# Load Dataset
# ----------------------------
DATA_PATH = "data/raw/employee_reviews.csv.csv"

df = pd.read_csv(DATA_PATH)

print(f"Original Shape: {df.shape}")

# ----------------------------
# Drop unnecessary columns
# ----------------------------
drop_columns = [
    "Unnamed: 0",
    "link",
    "advice-to-mgmt"
]

df = df.drop(columns=drop_columns, errors="ignore")

# ----------------------------
# Fill missing summaries
# ----------------------------
df["summary"] = df["summary"].fillna("")

# ----------------------------
# Create one review column
# ----------------------------
df["review"] = (
    df["summary"].astype(str)
    + ". "
    + df["pros"].astype(str)
    + ". "
    + df["cons"].astype(str)
)

# ----------------------------
# Remove duplicate reviews
# ----------------------------
df = df.drop_duplicates(subset=["review"])

# ----------------------------
# Remove empty reviews
# ----------------------------
df = df[df["review"].str.strip() != ""]

# ----------------------------
# Keep useful columns
# ----------------------------
df = df[
    [
        "company",
        "location",
        "job-title",
        "overall-ratings",
        "work-balance-stars",
        "culture-values-stars",
        "carrer-opportunities-stars",
        "comp-benefit-stars",
        "senior-mangemnet-stars",
        "review",
    ]
]

# ----------------------------
# Save cleaned dataset
# ----------------------------
output_path = "data/processed/clean_employee_reviews.csv"

df.to_csv(output_path, index=False)

print("\nCleaning Completed Successfully!")
print(f"New Shape: {df.shape}")
print(f"Saved to: {output_path}")

print("\nFirst 5 rows:")
print(df.head())