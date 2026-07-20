import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor,Pool

import optuna

# Load the cleaned dataset
df = pd.read_csv("dataset/processed_student_data.csv")

#features
X = df.drop(columns=["Exam_Score"])
#target
y = df["Exam_Score"]

from sklearn.preprocessing import LabelEncoder

encoders = {}

categorical_features = X.select_dtypes(include=["object", "string"]).columns

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, "models/encoders.pkl")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.20,random_state=42)

# Detect categorical columns

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

train_pool = Pool(
    X_train,
    y_train,
    cat_features=categorical_features
)

test_pool = Pool(
    X_test,
    y_test,
    cat_features=categorical_features
)

LabelEncoder()

#evaluate model function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return r2, mae, mse, rmse

# Create the model and evaluating it

lr_model = LinearRegression()
r2, mae, mse, rmse = evaluate_model(lr_model, X_train, X_test, y_train, y_test)

dt_model = DecisionTreeRegressor(random_state=42)
dt_r2, dt_mae, dt_mse, dt_rmse = evaluate_model(dt_model, X_train, X_test, y_train, y_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_r2, rf_mae, rf_mse, rf_rmse = evaluate_model(rf_model, X_train, X_test, y_train, y_test)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_r2, gb_mae, gb_mse, gb_rmse = evaluate_model(gb_model, X_train, X_test, y_train, y_test)
# -----------------------------
# XGBoost Regressor
# -----------------------------
print("\n🔍 Tuning XGBoost...")

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5]
}

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

xgb_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)

xgb_r2, xgb_mae, xgb_mse, xgb_rmse = evaluate_model(
    xgb_model,
    X_train,
    X_test,
    y_train,
    y_test
)

print("\n🔍 Optimizing CatBoost with Optuna...")

def objective(trial):

    params = {

        "iterations": trial.suggest_int("iterations", 500, 2500),

        "depth": trial.suggest_int("depth", 4, 10),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.15,
            log=True
        ),

        "l2_leaf_reg": trial.suggest_int(
            "l2_leaf_reg",
            1,
            20
        ),

        "random_strength": trial.suggest_float(
            "random_strength",
            0,
            10
        ),

        "bagging_temperature": trial.suggest_float(
            "bagging_temperature",
            0,
            10
        ),

        "border_count": trial.suggest_int(
            "border_count",
            64,
            255
        ),

        "loss_function": "RMSE",

        "eval_metric": "R2",

        "random_seed": 42,

        "verbose": 0
    }

    model = CatBoostRegressor(**params)

    model.fit(

        train_pool,

        eval_set=test_pool,

        early_stopping_rounds=100,

        use_best_model=True

    )

    predictions = model.predict(test_pool)

    return r2_score(y_test, predictions)


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=150)

print("\nBest Parameters:")
print(study.best_params)

cat_model = CatBoostRegressor(
    **study.best_params,
    loss_function="RMSE",
    eval_metric="R2",
    random_seed=42,
    verbose=0
)

cat_model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=100,
    use_best_model=True
)

predictions = cat_model.predict(test_pool)

cat_r2 = r2_score(y_test, predictions)
cat_mae = mean_absolute_error(y_test, predictions)
cat_mse = mean_squared_error(y_test, predictions)
cat_rmse = np.sqrt(cat_mse)

# Maximum prediction on the training data
train_predictions = cat_model.predict(X_train)

print("\nMaximum predicted score on training data:")
print(train_predictions.max())

print("\nTop 10 predicted scores:")
print(sorted(train_predictions, reverse=True)[:10])

print("\nMaximum actual score:")
print(y_train.max())

print("\n🔍 Training Stacking Regressor...")

stack_model = StackingRegressor(

    estimators=[
        ("cat", cat_model),
        ("xgb", xgb_model),
        ("rf", rf_model),
        ("gb", gb_model)
    ],

    final_estimator=Ridge(),

    passthrough=True,

    n_jobs=-1
)

stack_r2, stack_mae, stack_mse, stack_rmse = evaluate_model(
    stack_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# Store model results
results = {}
results["Linear Regression"] = {
    "R² Score": r2,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse
}
results["Decision Tree"] = {
    "R² Score": dt_r2,
    "MAE": dt_mae,
    "MSE": dt_mse,
    "RMSE": dt_rmse
}

results["Random Forest"] = {
    "R² Score": rf_r2,
    "MAE": rf_mae,
    "MSE": rf_mse,
    "RMSE": rf_rmse
}

results["Gradient Boosting"] = {
    "R² Score": gb_r2,
    "MAE": gb_mae,
    "MSE": gb_mse,
    "RMSE": gb_rmse
}

results["XGBoost"] = {
    "R² Score": xgb_r2,
    "MAE": xgb_mae,
    "MSE": xgb_mse,
    "RMSE": xgb_rmse
}

results["CatBoost"] = {
    "R² Score": cat_r2,
    "MAE": cat_mae,
    "MSE": cat_mse,
    "RMSE": cat_rmse
}

results["Stacking"] = {
    "R² Score": stack_r2,
    "MAE": stack_mae,
    "MSE": stack_mse,
    "RMSE": stack_rmse
}

print("\nModel Comparison")
print("------------------------------")

for model, metrics in results.items():
    print(f"\n{model}")
    print(f"R² Score : {metrics['R² Score']:.4f}")
    print(f"MAE      : {metrics['MAE']:.4f}")
    print(f"MSE      : {metrics['MSE']:.4f}")
    print(f"RMSE     : {metrics['RMSE']:.4f}")
    
print("\nTraining Features:")
for i, col in enumerate(X.columns, 1):
    print(f"{i}. {col}")
        
# Save the best model
joblib.dump(cat_model, "models/best_model.pkl")

print("\n✅ Best model saved successfully!")

# Save feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": cat_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("✅ Feature importance saved successfully!")