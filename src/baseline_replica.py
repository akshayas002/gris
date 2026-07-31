"""
baseline_replica.py
====================
Faithfully reproduces the ORIGINAL project's modeling methodology so we have
an honest, apples-to-apples "before" number to compare the fixed pipeline
against. This intentionally re-creates the bugs identified in review:

  1. No scoping: Sale + Rent + Unknown transaction types, and Land parcels,
     are all trained on together even though they represent different
     price scales / mechanisms.
  2. Locality target-encoding is fit on the FULL dataframe (train+test)
     BEFORE the train/test split -> classic target leakage.
  3. No outlier trimming on the target.
  4. Same GradientBoosting hyperparameters as the shipped model.pkl
     (learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.8).
  5. Same (buggy) MAPE formula: denominator clipped only at 1, so near-zero
     targets blow the metric up.

Do not copy this file's methodology - it exists purely as the "before" arm
of the before/after comparison in reports/BEFORE_AFTER.md.
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RAW_PATH = "data/raw_engineered.csv"

NUM_COLS = [
    "BEDROOM_NUM_CLEAN", "TOTAL_FLOOR", "FEATURE_COUNT", "AMENITY_COUNT",
    "LANDMARK_COUNT", "FLOOR_RATIO", "BALCONY_NUM", "AREA_SQFT",
    "LATITUDE", "LONGITUDE", "LISTING_AGE_DAYS", "IS_READY_TO_MOVE",
    "IS_UNDER_CONSTR", "IS_RERA", "IS_RESALE", "IS_NEW_BOOKING",
    "IS_VERIFIED", "HAS_SOCIETY", "IS_TOP_FLOOR", "IS_GROUND_FLOOR",
]
CAT_COLS = ["PROP_TYPE_CLEAN", "FURNISH_LABEL", "FACING_LABEL", "AGE_LABEL", "AREA_BUCKET"]


def buggy_mape(y_true, y_pred):
    # Reproduces the original clip(1) denominator bug on purpose.
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    # --- bug #2: target-encode locality on the FULL dataframe before split ---
    locality_means = df.groupby("LOCALITY")["TARGET_PRICE_SQFT"].mean()
    df["LOCALITY_ENC"] = df["LOCALITY"].map(locality_means)
    NUM_COLS_WITH_LEAK = NUM_COLS + ["LOCALITY_ENC"]

    y = df["TARGET_PRICE_SQFT"]
    X = df[NUM_COLS_WITH_LEAK + CAT_COLS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS_WITH_LEAK),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_COLS),
    ])

    model = GradientBoostingRegressor(
        learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.8, random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])

    y_train_log = np.log1p(y_train.clip(lower=1))
    pipe.fit(X_train, y_train_log)

    pred_log = pipe.predict(X_test)
    pred = np.expm1(pred_log)

    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "mape_buggy": buggy_mape(y_test.values, pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "note": "Reproduces original bugs on purpose: mixed Sale/Rent/Unknown/Land, "
                "pre-split target encoding leakage, no outlier trimming, buggy MAPE.",
    }
    print(json.dumps(metrics, indent=2))
    with open("reports/baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
