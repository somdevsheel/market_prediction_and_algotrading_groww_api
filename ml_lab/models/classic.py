"""Classic ML models (scikit-learn / XGBoost / LightGBM). Small, regularised configs fixed in advance."""
from __future__ import annotations

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

HAR = ["rv_d", "rv_w", "rv_m"]


def classifiers(seed: int = 0) -> dict:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "random_forest": RandomForestClassifier(400, max_depth=4, min_samples_leaf=40, random_state=seed, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                                 random_state=seed, eval_metric="logloss", n_jobs=4),
        "lightgbm": LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=7, learning_rate=0.03, min_child_samples=40,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1),
    }


def regressors(seed: int = 0) -> dict:
    return {
        "har_baseline": ("har", LinearRegression()),            # uses only HAR features (standard benchmark)
        "ridge": ("all", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        "random_forest": ("all", RandomForestRegressor(400, max_depth=6, min_samples_leaf=20, random_state=seed, n_jobs=-1)),
        "xgboost": ("all", XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                                        colsample_bytree=0.7, random_state=seed, n_jobs=4)),
        "lightgbm": ("all", LGBMRegressor(n_estimators=400, max_depth=4, num_leaves=15, learning_rate=0.03,
                                          min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                          random_state=seed, verbose=-1)),
    }
