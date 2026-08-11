from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    AGRONOMIC_MODEL_PATH: str = "artifacts/agronomic_risk.joblib"
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/agrotech"

    # Copernicus Data Space Ecosystem (CDSE) Sentinel Hub — the canopy signal (see SentinelClient).
    # Left blank, the satellite path is disabled: no Crop Health Index, no learned model
    # contribution, and the physics-derived hazards carry the assessment on their own. Nothing is
    # synthesised to fill the gap. Defaults point at CDSE (free tier).
    SENTINEL_CLIENT_ID: str = ""
    SENTINEL_CLIENT_SECRET: str = ""
    SENTINEL_TOKEN_URL: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    SENTINEL_STATS_URL: str = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

settings = Settings()
