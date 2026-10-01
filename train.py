import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_NAME = "house-price-predictor"
DATA_PATH = r"C:\Users\user\Downloads\Telegram Desktop\Mlops_house_prediction_clean_v1.csv"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mlops-house-prediction")

df = pd.read_csv(DATA_PATH)
print(f"Loaded shape: {df.shape}")

FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]
X = df[FEATURES]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    n_estimators = 150
    max_depth = 8

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("data_source", DATA_PATH)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model",
        registered_model_name=MODEL_NAME,
        serialization_format="cloudpickle",
    )

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

client = MlflowClient()
latest = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
client.set_registered_model_alias(MODEL_NAME, "champion", latest)
print(f"Alias 'champion' -> version {latest}")