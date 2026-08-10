import pandas as pd
from argotech.features.schema import FEATURES

class FeatureStore:

    def __init__(self):
        self.feature_list = list(FEATURES.keys())

    def build_features(self, payload: dict) -> pd.DataFrame:
        # normalize naming
        if "yield_value" in payload:
            payload["yield"] = payload.pop("yield_value")

        # enforce schema + order
        row = [payload[f] for f in self.feature_list]

        return pd.DataFrame([row], columns=self.feature_list)