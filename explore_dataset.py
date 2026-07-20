import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print(df["Exam_Score"].describe())