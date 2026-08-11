from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    MODEL_NAME: str = "farmerXential_model"
    MODEL_ALIAS: str = "prod"
    USE_LOCAL_MODEL: bool = True
    AGRONOMIC_MODEL_PATH: str = "artifacts/agronomic_risk.joblib"
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/agrotech"

    # Copernicus Data Space Ecosystem (CDSE) Sentinel Hub — real spectral indices for coldstart
    # inference (see SentinelClient). Left blank disables the real fetch and the pipeline falls back
    # to the synthetic index model. Defaults point at CDSE (free tier); commercial Sentinel Hub URLs
    # can be substituted via env.
    SENTINEL_CLIENT_ID: str = ""
    SENTINEL_CLIENT_SECRET: str = ""
    SENTINEL_TOKEN_URL: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    SENTINEL_STATS_URL: str = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

settings = Settings()
