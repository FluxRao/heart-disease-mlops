
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

# Download UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into one DataFrame
df = pd.concat([X, y], axis=1)

# Save the raw data
data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(exist_ok=True)

output_path = data_dir / "heart_disease_raw.csv"
df.to_csv(output_path, index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nSaved to:", output_path)
