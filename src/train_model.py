"""
train_model.py (v2 - fixed)
============================
End-to-end training on the cleaned, leakage-free dataset produced by
data_cleaning.py.

Fixes applied vs. the original project's train_model.py:
  1. LOCALITY (389-category, high-cardinality) is encoded with sklearn's
     TargetEncoder(cv=5, target_type="continuous"), which internally
     cross-fits during fit_transform (each row's encoded value comes from
     a model that never saw that row's own target) - no leakage, no manual
     KFold bookkeeping needed.
  2. Model selection uses RandomizedSearchCV with 5-fold CV, not a single
     train/test split - the reported test metrics come from a held-out set
     the search never touched.
  3. Three model families are compared (Ridge / RandomForest / Hist
     GradientBoosting) instead of hand-picked hyperparameters for one.
  4. Metrics: R2, MAE, RMSE, MedAE (robust to outliers) and a floor-safe
     MAPE - computed on data that's already been scoped+winsorized in
     data_cleaning.py, so it no longer explodes.
  5. Everything needed for inference (fitted pipeline, feature lists,
     metrics, scope/assumptions) is saved in one bundle.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBRegressor

CLEAN_PATH = "data/clean_dataset.csv"
TARGET = "TARGET_PRICE_SQFT"

HIGH_CARD_CAT = ["LOCALITY"]
LOW_CARD_CAT = ["PROP_TYPE_CLEAN", "FURNISH_LABEL", "FACING_LABEL", "AGE_LABEL", "AREA_BUCKET", "BALCONY_BUCKET"]
NUMERIC = [
    "BEDROOM_NUM_CLEAN", "TOTAL_FLOOR", "FEATURE_COUNT", "AMENITY_COUNT",
    "LANDMARK_COUNT", "FLOOR_RATIO", "BALCONY_NUM", "AREA_SQFT",
    "LATITUDE", "LONGITUDE", "LISTING_AGE_DAYS", "IS_READY_TO_MOVE",
    "IS_UNDER_CONSTR", "IS_RERA", "IS_RESALE", "IS_NEW_BOOKING",
    "IS_VERIFIED", "HAS_SOCIETY", "IS_TOP_FLOOR", "IS_GROUND_FLOOR",
    "FLOOR_RATIO_WAS_MISSING", "BALCONY_NUM_WAS_MISSING",
    "BEDROOM_NUM_CLEAN_WAS_MISSING", "TOTAL_FLOOR_WAS_MISSING",
]


def robust_mape(y_true, y_pred, floor):
    """MAPE with a documented, non-arbitrary floor on the denominator instead
    of clip(1) - floor is the 1st percentile of the training target, so we
    never divide by a near-zero number that isn't representative of the
    actual price scale being modeled."""
    denom = np.clip(y_true, floor, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def build_preprocessor(scale_numeric: bool):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC),
        ("low_card", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), LOW_CARD_CAT),
        ("high_card", TargetEncoder(target_type="continuous", cv=5, random_state=42), HIGH_CARD_CAT),
    ])


def main():
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    y = np.log1p(df[TARGET])  # log-space target: price/sqft is right-skewed
    X = df[NUMERIC + LOW_CARD_CAT + HIGH_CARD_CAT]

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y, df[TARGET], test_size=0.2, random_state=42
    )

    candidates = {
        "ridge": {
            "pipeline": Pipeline([("pre", build_preprocessor(scale_numeric=True)), ("model", Ridge())]),
            "param_dist": {"model__alpha": [0.1, 1, 3, 10, 30, 100, 300]},
            "n_iter": 7,
        },
        "random_forest": {
            "pipeline": Pipeline([("pre", build_preprocessor(scale_numeric=False)), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
            "param_dist": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 8, 12, 16, 24],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
            "n_iter": 15,
        },
        "hist_gb": {
            "pipeline": Pipeline([("pre", build_preprocessor(scale_numeric=False)), ("model", HistGradientBoostingRegressor(random_state=42))]),
            "param_dist": {
                "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
                "model__max_depth": [3, 5, 7, None],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.1, 1.0],
                "model__min_samples_leaf": [10, 20, 40],
            },
            "n_iter": 20,
        },
        "xgboost": {
            "pipeline": Pipeline([("pre", build_preprocessor(scale_numeric=False)), ("model", XGBRegressor(random_state=42, n_jobs=4, tree_method="hist"))]),
            "param_dist": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [3, 4, 5, 6, 8],
                "model__learning_rate": [0.02, 0.05, 0.08, 0.1],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__reg_lambda": [0.5, 1, 2, 5],
                "model__min_child_weight": [1, 3, 5],
            },
            "n_iter": 20,
        },
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_name, best_score, best_search = None, -np.inf, None

    for name, cfg in candidates.items():
        t0 = time.time()
        search = RandomizedSearchCV(
            cfg["pipeline"], cfg["param_dist"], n_iter=cfg["n_iter"],
            cv=cv, scoring="r2", random_state=42, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0
        results[name] = {
            "cv_best_r2": search.best_score_,
            "best_params": search.best_params_,
            "seconds": round(elapsed, 1),
        }
        print(f"[{name}] CV R2={search.best_score_:.4f}  ({elapsed:.1f}s)  params={search.best_params_}")
        if search.best_score_ > best_score:
            best_name, best_score, best_search = name, search.best_score_, search

    print(f"\nSelected model: {best_name} (CV R2={best_score:.4f})")
    best_pipeline = best_search.best_estimator_

    # ---- final held-out evaluation (test set was never touched by CV search) ----
    pred_log = best_pipeline.predict(X_test)
    pred = np.expm1(pred_log)

    floor = float(y_train_raw.quantile(0.01))
    pct_err = np.abs(pred - y_test_raw.values) / y_test_raw.values * 100
    within_band = {f"within_{t}pct": float((pct_err <= t).mean() * 100) for t in [5, 10, 15, 20, 30, 50]}

    # per-segment (property type) breakdown - the aggregate number alone hides
    # that this model is much stronger on Apartments than on Villas
    segment_metrics = {}
    for ptype, mask in X_test.groupby("PROP_TYPE_CLEAN").groups.items():
        yt, yp = y_test_raw.loc[mask], pd.Series(pred, index=X_test.index).loc[mask]
        segment_metrics[ptype] = {
            "n": int(len(mask)),
            "r2": float(r2_score(yt, yp)) if len(mask) > 1 else None,
            "mae": float(mean_absolute_error(yt, yp)),
            "medae": float(median_absolute_error(yt, yp)),
        }

    metrics = {
        "model_selected": best_name,
        "cv_r2_all_candidates": {k: v["cv_best_r2"] for k, v in results.items()},
        "test_r2": r2_score(y_test_raw, pred),
        "test_mae": mean_absolute_error(y_test_raw, pred),
        "test_rmse": mean_squared_error(y_test_raw, pred) ** 0.5,
        "test_medae": median_absolute_error(y_test_raw, pred),
        "test_mape_sklearn": float(mean_absolute_percentage_error(y_test_raw, pred) * 100),
        "test_mape_robust_floored": robust_mape(y_test_raw.values, pred, floor),
        "mape_floor_used_inr_per_sqft": floor,
        "accuracy_bands": within_band,
        "segment_metrics": segment_metrics,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(json.dumps(metrics, indent=2))

    with open("reports/improved_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("reports/model_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump({
        "pipeline": best_pipeline,
        "target_is_log1p": True,
        "target_col": TARGET,
        "numeric_cols": NUMERIC,
        "low_card_cat_cols": LOW_CARD_CAT,
        "high_card_cat_cols": HIGH_CARD_CAT,
        "metrics": metrics,
        "scope": {
            "transact_label": "Sale",
            "prop_types": ["Apartment", "Villa", "Builder_Floor", "Farmhouse"],
            "target_winsorized_to_pct": [1, 99],
        },
    }, "models/model_v2.pkl")

    return best_pipeline, X_test, y_test_raw, pred


if __name__ == "__main__":
    main()
