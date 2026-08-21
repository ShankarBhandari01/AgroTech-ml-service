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
    # *classifier* ranked worse than it; regressing `forward_z` instead is what changed the answer.
    #
    # Against BOTH baselines, not just the easy one — from `artifacts/metrics.json`, seed 42,
    # leave-one-cluster-out over six clusters (mean of the folds):
    #
    #                      P@25     rho
    #     regressor        0.760   0.377
    #     site climatology 0.680   0.363
    #     persistence      0.467   0.368
    #     classifier arm   0.647   0.235
    #
    # The case rests entirely on P@25: ahead of persistence in 6 of 6 folds and of climatology in
    # 4 of 6. On rho the three are within 0.014 of each other and the result is not meaningful.
    #
    # Forward in time it is weaker, and this is the standing caveat on the default: over the three
    # forward-chaining folds the regressor takes P@25 0.493 against persistence's 0.533 and
    # climatology's 0.587 — it loses to both — while rho is 0.476 against 0.428 and 0.475, i.e.
    # level with climatology. Climatology is the opponent to beat, not persistence.
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
