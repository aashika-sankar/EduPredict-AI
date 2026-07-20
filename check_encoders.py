import joblib

encoders = joblib.load("models/encoders.pkl")

for col in encoders:
    print(col)
    print(encoders[col].classes_)
    print()