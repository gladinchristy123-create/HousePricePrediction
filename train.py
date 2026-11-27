
import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load dataset
print("Loading California Housing Dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df["MedHouseVal"] *= 100000
df.to_csv("data/california_housing.csv", index=False)

# EDA
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe())

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["MedHouseVal"], bins=40, kde=True)
plt.title("House Price Distribution")
plt.xlabel("Median House Value ($)")
plt.tight_layout()
plt.savefig("results/price_distribution.png")
plt.close()

# Prepare data
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# MLflow experiment tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("California_House_Price_Prediction")

# Train and evaluate
results = []

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Set displayed run user
        mlflow.set_tag("mlflow.user", "Ancy")

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("model_name", name)
        mlflow.log_param("dataset", "California Housing")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        # Save model
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, f"models/{filename}")

        # Save model artifact in MLflow
        mlflow.log_artifact(f"models/{filename}")

        print(f"MAE: ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R2: {r2:.4f}")

# Save comparison
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nAll models trained and saved successfully!")
print("MLflow experiment tracking completed!")
