from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    AGRONOMIC_MODEL_PATH: str = "artifacts/agronomic_risk.joblib"

    # Where the vegetation hazard term comes from: "persistence" or "model".
    #
    # Defaults to persistence because it measurably ranks better. Leave-one-cluster-out against the
    # continuous target, the trained classifier scores Spearman 0.191 and carrying the field's own
    # peer anomaly forward scores 0.351 — better in 5 of 6 clusters. Set to "model" to A/B it; that
    # is the only setting that loads the joblib at all.
    VEGETATION_HAZARD_SOURCE: str = "persistence"
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
