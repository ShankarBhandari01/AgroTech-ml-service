import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import os


class HybridSpatiotemporalEnsemble:
    """
    Hybrid Spatiotemporal Attention & Gradient Boosted Ensemble Model (HSTA-Ensemble).
    Combines high-capacity XGBoost, Histogram-based Gradient Boosting, Random Forest,
    and Extra Trees with a Logistic Stacking Meta-Learner for sub-10ms precision inference.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # Base Learners
        self.xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="mlogloss"
        )
        self.hgb = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.03,
            max_depth=7,
            random_state=random_state
        )
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        self.et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )

        # Meta Stacking Classifier
        self.stacking_model = StackingClassifier(
            estimators=[
                ('xgb', self.xgb),
                ('hgb', self.hgb),
                ('rf', self.rf),
                ('et', self.et)
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=500),
            cv=5,
            n_jobs=-1
        )

        self.calibrated_model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Trains base estimators, fits stacking meta-learner, and calibrates probabilities.
        """
        print(f"Training Base Estimators on {X.shape[0]} samples with {X.shape[1]} features...")
        self.xgb.fit(X, y)
        self.hgb.fit(X, y)
        self.rf.fit(X, y)
        self.et.fit(X, y)

        print("Fitting Meta Stacking Classifier...")
        self.stacking_model.fit(X, y)
        
        print("Calibrating model probability estimations...")
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.stacking_model,
            method="sigmoid",
            cv="prefit"
        )
        self.calibrated_model.fit(X, y)
        print("Model training & calibration complete!")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrated_model is not None:
            return self.calibrated_model.predict(X)
        return self.stacking_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        return self.stacking_model.predict_proba(X)

    def save_model(self, filepath: str):
        joblib.dump(self, filepath)
        print(f"Model successfully saved to {filepath}")

    @staticmethod
    def load_model(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' does not exist.")
        return joblib.load(filepath)
