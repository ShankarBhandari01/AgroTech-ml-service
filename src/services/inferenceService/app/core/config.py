from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    MODEL_NAME: str = "farmerXential_model"
    MODEL_ALIAS: str = "prod"
    USE_LOCAL_MODEL: bool = True

settings = Settings()