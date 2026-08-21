from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    AGRONOMIC_MODEL_PATH: str = "artifacts/agronomic_risk.joblib"

    # Where the vegetation hazard term comes from: "model" or "persistence".
    #
    # Both map a peer-standardised NDVI anomaly through `risk.vegetation_hazard_from_anomaly`; they
    # differ only in where the anomaly comes from — predicted 30 days out, or the field's own
    # observed value carried forward.
    #
    # Defaults to the model since 2026-08-21. The previous default was persistence because the
    # *classifier* ranked worse than it: Spearman 0.196 against 0.351. Regressing `forward_z`
    # instead lifts the model to 0.342 — level with persistence, improving in all 30 of 5 seeds x 6
    # blocked folds — and P@25 to 0.741 against persistence's 0.51. The case for this default rests
    # on P@25, not on rho, where the two are level.
    VEGETATION_HAZARD_SOURCE: str = "model"
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
