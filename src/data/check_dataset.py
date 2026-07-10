import pandas as pd
from pathlib import Path

# Locate CSV files in data/raw
data_folder = Path("data/raw")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("❌ No CSV file found in data/raw/")
    exit()

dataset_path = csv_files[0]

print(f"Using dataset: {dataset_path}")

# Try reading with different separators
for sep in [",", ";", "\t"]:
    try:
        df = pd.read_csv(dataset_path, sep=sep)
        if len(df.columns) > 1:
            print(f"\n✅ Successfully loaded using separator: '{sep}'")
            break
    except Exception:
        continue

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())