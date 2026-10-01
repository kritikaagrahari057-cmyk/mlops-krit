
import shutil
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

shutil.rmtree("model", ignore_errors=True)
mlflow.artifacts.download_artifacts(
    artifact_uri="models:/house-price-predictor@champion",
    dst_path="model",
)
print("Model exported to ./model")