import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from src.services.inferenceService.app.core.config import Settings

settings = Settings()

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.MODEL_NAME)


def train_and_register():
    # Dummy dataset
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50]
    })
    y = [5, 10, 15, 20, 25]

    model = RandomForestRegressor()
    model.fit(X, y)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # 1. Log model + REGISTER it
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="farmerXential_model"
        )

        print("Run ID:", run_id)
        print("Model registered!")

    return run_id


def set_latest_alias():
    client = MlflowClient()

    # get latest version
    versions = client.get_latest_versions("farmerXential_model")
    latest_version = versions[0].version

    print("Latest version:", latest_version)

    # set alias "latest"
    client.set_registered_model_alias(
        name="farmerXential_model",
        alias="prod",
        version=latest_version
    )

    print("Alias 'latest' set!")


if __name__ == "__main__":
    train_and_register()
    set_latest_alias()
