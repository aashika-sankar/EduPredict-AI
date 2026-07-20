import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# -------------------------
# Load Original Dataset
# -------------------------

df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

print("Dataset Shape:", df.shape)

# -------------------------
# Missing Values
# -------------------------

categorical_cols = [
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------
# Numerical Columns
# -------------------------

numeric_cols = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

# -------------------------
# Categorical Columns
# -------------------------

cat_features = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Motivation_Level",
    "Internet_Access",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Peer_Influence",
    "Learning_Disabilities",
    "Parental_Education_Level",
    "Distance_from_Home",
    "Gender"
]

for col in cat_features:
    df[col] = df[col].astype(str)

# -------------------------
# Feature Engineering
# -------------------------

df["Study_Consistency"] = (
    df["Hours_Studied"] *
    df["Sleep_Hours"]
)

df["Balance_Index"] = (
    df["Sleep_Hours"] +
    df["Physical_Activity"] +
    df["Attendance"]
)

df["Performance_Trend"] = (
    df["Previous_Scores"] +
    df["Attendance"]
)

df["Engagement"] = (
    df["Hours_Studied"] +
    df["Attendance"] +
    df["Tutoring_Sessions"]
)

df["Study_Attendance"] = (
    df["Hours_Studied"] *
    df["Attendance"]
)

df["Previous_Study"] = (
    df["Previous_Scores"] *
    df["Hours_Studied"]
)

df["Attendance_Previous"] = (
    df["Attendance"] *
    df["Previous_Scores"]
)

df["Study_Efficiency"] = (
    df["Previous_Scores"] /
    (df["Hours_Studied"] + 1)
)

# -------------------------
# Save processed dataset
# -------------------------

df.to_csv(
    "dataset/processed_student_data.csv",
    index=False
)

print("Processed dataset saved.")

# -------------------------
# Save categorical feature names
# -------------------------

joblib.dump(
    cat_features,
    "models/categorical_features.pkl"
)

print("Categorical feature list saved.")

# -------------------------
# Train/Test Split
# -------------------------

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

joblib.dump(
    (X_train, X_test, y_train, y_test),
    "models/train_test_split.pkl"
)

print("Train/Test split saved.")