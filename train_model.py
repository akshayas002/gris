"""
Real Estate Price Prediction — Train Model
Dataset: Hyderabad Kaggle dataset
Target: PRICE
"""

import json
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)


# ─── 1. Load ──────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─── 2. Fix target column ─────────────────────────────────────────────────────
def fix_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    PRICE is the target. Fill nulls using median of MIN_PRICE / MAX_PRICE.
    Drop rows where all three are null (can't recover).
    """
    def to_numeric(col):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return np.nan

    df["PRICE"]     = to_numeric("PRICE")
    df["MIN_PRICE"] = to_numeric("MIN_PRICE")
    df["MAX_PRICE"]  = to_numeric("MAX_PRICE")

    # Fill using midpoint of min/max when PRICE is null
    mid = (df["MIN_PRICE"] + df["MAX_PRICE"]) / 2
    df["PRICE"] = df["PRICE"].fillna(mid)

    before = len(df)
    df = df.dropna(subset=["PRICE"])
    print(f"Target: dropped {before - len(df)} rows with no recoverable price. {len(df):,} remain.")

    # Log-transform target (right-skewed real estate prices behave better in log space)
    df["PRICE_LOG"] = np.log1p(df["PRICE"])
    return df


# ─── 3. Feature engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    # Average area (avoids choice between min/max)
    df["AVG_AREA_SQFT"] = (
        pd.to_numeric(df.get("MIN_AREA_SQFT"), errors="coerce").fillna(0) +
        pd.to_numeric(df.get("MAX_AREA_SQFT"), errors="coerce").fillna(0)
    ) / 2
    df["AVG_AREA_SQFT"] = df["AVG_AREA_SQFT"].replace(0, np.nan)

    # Amenity count from the AMENITIES string
    if "AMENITIES" in df.columns:
        df["AMENITY_COUNT"] = (
            df["AMENITIES"]
            .fillna("")
            .apply(lambda x: len(str(x).split(",")) if x else 0)
        )

    # Extract lat/long from MAP_DETAILS if it's JSON
    if "MAP_DETAILS" in df.columns:
        def extract_coords(val):
            try:
                d = json.loads(str(val))
                return float(d.get("latitude", np.nan)), float(d.get("longitude", np.nan))
            except Exception:
                # Try regex fallback: "lat":17.xxx,"lng":78.xxx
                lat = re.search(r'"lat(?:itude)?"\s*:\s*([\d.]+)', str(val))
                lng = re.search(r'"l(?:ng|ong(?:itude)?)\s*"\s*:\s*([\d.]+)', str(val))
                return (
                    float(lat.group(1)) if lat else np.nan,
                    float(lng.group(1)) if lng else np.nan,
                )
        coords = df["MAP_DETAILS"].apply(extract_coords)
        df["LAT"] = coords.apply(lambda x: x[0])
        df["LNG"] = coords.apply(lambda x: x[1])
        print(f"Extracted lat/long for {df['LAT'].notna().sum():,} rows")

    # Floor ratio: which floor out of total (proxy for view / desirability)
    floor_num   = pd.to_numeric(df.get("FLOOR_NUM"),   errors="coerce")
    total_floor = pd.to_numeric(df.get("TOTAL_FLOOR"), errors="coerce")
    df["FLOOR_RATIO"] = floor_num / total_floor.replace(0, np.nan)

    # Property age bucket
    age = pd.to_numeric(df.get("AGE"), errors="coerce")
    df["AGE_BUCKET"] = pd.cut(
        age,
        bins=[-1, 0, 3, 7, 15, 999],
        labels=["new", "0-3yr", "3-7yr", "7-15yr", "15yr+"]
    ).astype(str)

    # Binary: is verified?
    if "VERIFIED" in df.columns:
        df["IS_VERIFIED"] = df["VERIFIED"].apply(
            lambda x: 1 if str(x).strip().lower() in ("yes", "1", "true", "verified") else 0
        )

    return df


# ─── 4. Target encoding for high-cardinality location ─────────────────────────
def target_encode(df: pd.DataFrame, col: str, target: str = "PRICE_LOG") -> pd.DataFrame:
    """
    Replace locality string with mean log price for that locality.
    Uses leave-one-out style (only training mean should be computed and stored).
    Returns df with new column {col}_ENC.
    """
    df[col] = df[col].fillna("unknown").str.strip().str.lower()
    mean_map = df.groupby(col)[target].mean()
    global_mean = df[target].mean()
    df[f"{col}_ENC"] = df[col].map(mean_map).fillna(global_mean)
    print(f"Target-encoded '{col}': {df[col].nunique()} unique values → single numeric column")
    return df, mean_map


# ─── 5. Select & clean feature matrix ─────────────────────────────────────────
NUMERIC_COLS = [
    "BEDROOM_NUM",
    "AVG_AREA_SQFT",
    "AGE",
    "TOTAL_FLOOR",
    "FLOOR_NUM",
    "FLOOR_RATIO",
    "BALCONY_NUM",
    "TOTAL_LANDMARK_COUNT",
    "REGISTERED_DAYS",
    "AMENITY_COUNT",
    "IS_VERIFIED",
    "location_ENC",
    "AREA_ENC",
]

CATEGORICAL_COLS = [
    "PROPERTY_TYPE",
    "FURNISH",
    "FACING",
    "TRANSACT_TYPE",
    "RES_COM",
    "AGE_BUCKET",
]

# !! DATA LEAKAGE: NEVER include these in features
LEAKAGE_COLS = ["PRICE_SQFT", "PRICE_PER_UNIT_AREA", "MIN_PRICE", "MAX_PRICE", "PRICE"]


def build_feature_matrix(df: pd.DataFrame):
    # Only keep columns that actually exist in df
    num_cols = [c for c in NUMERIC_COLS     if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Convert numerics
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill categoricals
    for col in cat_cols:
        df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()

    all_feature_cols = num_cols + cat_cols
    X = df[all_feature_cols].copy()
    y = df["PRICE_LOG"].copy()

    print(f"\nFeature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Numeric    : {num_cols}")
    print(f"  Categorical: {cat_cols}")
    print(f"  Missing values:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
    return X, y, num_cols, cat_cols


# ─── 6. Preprocessor pipeline ─────────────────────────────────────────────────
def build_preprocessor(num_cols, cat_cols):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])


# ─── 7. Train & evaluate ──────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)   # reverse log1p
    y_true     = np.expm1(y_test)

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true.clip(1))) * 100

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"  R²   : {r2:.4f}  (higher = better, 1.0 = perfect)")
    print(f"  MAE  : ₹{mae:>14,.0f}")
    print(f"  RMSE : ₹{rmse:>14,.0f}")
    print(f"  MAPE : {mape:.1f}%  (mean absolute % error)")
    print(f"{'─'*40}")

    return {"name": name, "r2": r2, "mae": mae, "rmse": rmse, "mape": mape, "model": model}


def plot_actual_vs_predicted(y_test, y_pred_log, name: str, save_path: Path):
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{name} — evaluation", fontsize=13)

    # Actual vs predicted scatter
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color="#1D9E75")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("Actual price (₹)")
    axes[0].set_ylabel("Predicted price (₹)")
    axes[0].set_title("Actual vs predicted")

    # Residuals
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, color="#7F77DD", edgecolor="none", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", lw=1)
    axes[1].set_xlabel("Residual (actual − predicted) in ₹")
    axes[1].set_title("Residual distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {save_path}")


def plot_feature_importance(model_pipeline, feature_names: list, save_path: Path):
    try:
        rf = model_pipeline.named_steps["model"]
        importances = rf.feature_importances_
        idx = np.argsort(importances)[-15:]  # top 15

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([feature_names[i] for i in idx], importances[idx], color="#378ADD")
        ax.set_xlabel("Feature importance")
        ax.set_title("Top 15 features — Random Forest")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved importance plot → {save_path}")
    except Exception as e:
        print(f"  Could not plot feature importances: {e}")


# ─── 8. Prediction function ───────────────────────────────────────────────────
def predict_price(
    model_pipeline,
    location_enc_map: dict,
    area_enc_map: dict,
    global_mean_log: float,
    features: dict,
) -> dict:
    """
    Predict price for a single listing.

    features dict example:
    {
        "BEDROOM_NUM": 3,
        "AVG_AREA_SQFT": 1400,
        "AGE": 2,
        "TOTAL_FLOOR": 10,
        "FLOOR_NUM": 4,
        "BALCONY_NUM": 2,
        "TOTAL_LANDMARK_COUNT": 5,
        "REGISTERED_DAYS": 30,
        "AMENITY_COUNT": 8,
        "IS_VERIFIED": 1,
        "location": "gachibowli",
        "AREA": "gachibowli",
        "PROPERTY_TYPE": "flat",
        "FURNISH": "semi-furnished",
        "FACING": "east",
        "TRANSACT_TYPE": "sell",
        "RES_COM": "residential",
    }
    """
    row = features.copy()

    # Target encode location
    loc = str(row.pop("location", "unknown")).strip().lower()
    row["location_ENC"] = location_enc_map.get(loc, global_mean_log)

    area = str(row.pop("AREA", "unknown")).strip().lower()
    row["AREA_ENC"] = area_enc_map.get(area, global_mean_log)

    # FLOOR_RATIO
    fn = float(row.get("FLOOR_NUM", 0) or 0)
    tf = float(row.get("TOTAL_FLOOR", 1) or 1)
    row["FLOOR_RATIO"] = fn / tf if tf > 0 else 0

    # Age bucket
    age = float(row.get("AGE", 0) or 0)
    if age <= 0:   row["AGE_BUCKET"] = "new"
    elif age <= 3: row["AGE_BUCKET"] = "0-3yr"
    elif age <= 7: row["AGE_BUCKET"] = "3-7yr"
    elif age <= 15: row["AGE_BUCKET"] = "7-15yr"
    else:           row["AGE_BUCKET"] = "15yr+"

    df_input = pd.DataFrame([row])

    log_price = model_pipeline.predict(df_input)[0]
    price = np.expm1(log_price)

    # Rough confidence range: ±15% (replace with proper PI from quantile RF later)
    return {
        "predicted_price": round(price),
        "low_estimate":    round(price * 0.85),
        "high_estimate":   round(price * 1.15),
        "in_lakhs":        round(price / 1e5, 2),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(csv_path: str = "data/listings_latest.csv"):
    print("=" * 50)
    print("Real Estate Price Prediction — Training")
    print("=" * 50)

    # Load
    df = load_data(csv_path)

    # Fix target
    df = fix_target(df)

    # Feature engineering
    df = engineer_features(df)

    # Target encode high-cardinality location columns
    df, location_enc_map = target_encode(df, "location")
    df, area_enc_map     = target_encode(df, "AREA")
    global_mean_log      = df["PRICE_LOG"].mean()

    # Build feature matrix
    X, y, num_cols, cat_cols = build_feature_matrix(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    preprocessor = build_preprocessor(num_cols, cat_cols)
    all_feature_names = num_cols + cat_cols

    results = []

    # ── Model 1: Ridge Regression (better than plain LR for this data) ──
    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",        Ridge(alpha=10.0)),
    ])
    ridge_pipeline.fit(X_train, y_train)
    res = evaluate_model(ridge_pipeline, X_test, y_test, "Ridge Regression (baseline)")
    results.append(res)
    plot_actual_vs_predicted(
        y_test, ridge_pipeline.predict(X_test),
        "Ridge Regression", PLOT_DIR / "ridge_eval.png"
    )

    # ── Model 2: Random Forest ──
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    print("\nTraining Random Forest (this takes ~30s)...")
    rf_pipeline.fit(X_train, y_train)
    res_rf = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
    results.append(res_rf)
    plot_actual_vs_predicted(
        y_test, rf_pipeline.predict(X_test),
        "Random Forest", PLOT_DIR / "rf_eval.png"
    )
    plot_feature_importance(rf_pipeline, all_feature_names, PLOT_DIR / "feature_importance.png")

    # ── Model 3: Gradient Boosting (usually best for tabular data) ──
    gb_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )),
    ])
    print("\nTraining Gradient Boosting (this takes ~60s)...")
    gb_pipeline.fit(X_train, y_train)
    res_gb = evaluate_model(gb_pipeline, X_test, y_test, "Gradient Boosting")
    results.append(res_gb)
    plot_actual_vs_predicted(
        y_test, gb_pipeline.predict(X_test),
        "Gradient Boosting", PLOT_DIR / "gb_eval.png"
    )

    # ── Pick best model by R² ──
    best = max(results, key=lambda r: r["r2"])
    print(f"\nBest model: {best['name']}  (R² = {best['r2']:.4f})")

    # ── Save ──
    model_artifacts = {
        "pipeline":         best["model"],
        "location_enc_map": dict(location_enc_map),
        "area_enc_map":     dict(area_enc_map),
        "global_mean_log":  global_mean_log,
        "feature_names":    all_feature_names,
        "num_cols":         num_cols,
        "cat_cols":         cat_cols,
        "metrics": {
            "r2": best["r2"], "mae": best["mae"],
            "rmse": best["rmse"], "mape": best["mape"],
        },
    }
    out_path = OUTPUT_DIR / "model.pkl"
    joblib.dump(model_artifacts, out_path)
    print(f"\nSaved model artifacts → {out_path}")

    # ── Quick sanity check ──
    print("\nSanity check — predicting a sample 3BHK in Gachibowli:")
    sample = {
        "BEDROOM_NUM":          3,
        "AVG_AREA_SQFT":        1400,
        "AGE":                  2,
        "TOTAL_FLOOR":          12,
        "FLOOR_NUM":            5,
        "BALCONY_NUM":          2,
        "TOTAL_LANDMARK_COUNT": 6,
        "REGISTERED_DAYS":      45,
        "AMENITY_COUNT":        8,
        "IS_VERIFIED":          1,
        "location":             "gachibowli",
        "AREA":                 "gachibowli",
        "PROPERTY_TYPE":        "flat",
        "FURNISH":              "semi-furnished",
        "FACING":               "east",
        "TRANSACT_TYPE":        "sell",
        "RES_COM":              "residential",
    }
    pred = predict_price(
        best["model"],
        model_artifacts["location_enc_map"],
        model_artifacts["area_enc_map"],
        global_mean_log,
        sample,
    )
    print(f"  Predicted: ₹{pred['predicted_price']:,}  ({pred['in_lakhs']} Lakhs)")
    print(f"  Range:     ₹{pred['low_estimate']:,} – ₹{pred['high_estimate']:,}")
    print("\nDone.")


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/listings_latest.csv"
    main(csv)
