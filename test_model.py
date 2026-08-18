import pandas as pd

df = pd.read_csv("models/rolling_origin_model_summary.csv")

print("COLUMNS:")
print(df.columns.tolist())

print("\nSHAPE:")
print(df.shape)

print("\nDATA:")
print(df.to_string(index=False))