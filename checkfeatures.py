import joblib

model = joblib.load("models/best_model.pkl")

print("Model Features:")
for i, feature in enumerate(model.feature_names_):
    print(f"{i+1}. {feature}")