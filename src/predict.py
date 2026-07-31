"""
predict.py
==========
Loads models/model_v2.pkl and predicts price/sqft (and total price, given
area) for a new listing. Also reports the model's documented scope, so
callers know when a prediction is out-of-scope (e.g. land, rentals) rather
than silently returning a number the model was never trained to give.

Usage:
    python predict.py --area_sqft 1400 --bedrooms 3 --prop_type Apartment \
        --locality Gachibowli --furnish Semi_Furnished --facing East \
        --age_label New --area_bucket Medium
"""
import argparse

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/model_v2.pkl"

# Reasonable defaults for fields not exposed on the CLI, so a partial
# description still gets a prediction.
DEFAULTS = {
    "TOTAL_FLOOR": 10, "FEATURE_COUNT": 8, "AMENITY_COUNT": 6, "LANDMARK_COUNT": 3,
    "FLOOR_RATIO": 0.5, "BALCONY_NUM": 2, "LATITUDE": 17.44, "LONGITUDE": 78.38,
    "LISTING_AGE_DAYS": 30, "IS_READY_TO_MOVE": 1, "IS_UNDER_CONSTR": 0,
    "IS_RERA": 1, "IS_RESALE": 0, "IS_NEW_BOOKING": 1, "IS_VERIFIED": 1,
    "HAS_SOCIETY": 1, "IS_TOP_FLOOR": 0, "IS_GROUND_FLOOR": 0,
    "FLOOR_RATIO_WAS_MISSING": 0, "BALCONY_NUM_WAS_MISSING": 0,
    "BEDROOM_NUM_CLEAN_WAS_MISSING": 0, "TOTAL_FLOOR_WAS_MISSING": 0,
    "BALCONY_BUCKET": "Medium",
}


def predict(area_sqft, bedrooms, prop_type, locality, furnish, facing, age_label, area_bucket, overrides=None):
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]

    row = dict(DEFAULTS)
    row.update({
        "AREA_SQFT": area_sqft,
        "BEDROOM_NUM_CLEAN": bedrooms,
        "PROP_TYPE_CLEAN": prop_type,
        "LOCALITY": locality,
        "FURNISH_LABEL": furnish,
        "FACING_LABEL": facing,
        "AGE_LABEL": age_label,
        "AREA_BUCKET": area_bucket,
    })
    if overrides:
        row.update(overrides)

    cols = bundle["numeric_cols"] + bundle["low_card_cat_cols"] + bundle["high_card_cat_cols"]
    X = pd.DataFrame([row])[cols]

    pred_log = pipe.predict(X)[0]
    price_per_sqft = float(np.expm1(pred_log))
    total_price = price_per_sqft * area_sqft

    in_scope = prop_type in bundle["scope"]["prop_types"]
    return {
        "predicted_price_per_sqft_inr": round(price_per_sqft, 0),
        "predicted_total_price_inr": round(total_price, 0),
        "in_scope": in_scope,
        "scope_note": None if in_scope else (
            f"Model was trained only on {bundle['scope']['prop_types']} (Sale only). "
            f"'{prop_type}' is outside that scope - treat this prediction with caution."
        ),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--area_sqft", type=float, required=True)
    ap.add_argument("--bedrooms", type=float, required=True)
    ap.add_argument("--prop_type", default="Apartment")
    ap.add_argument("--locality", default="Gachibowli")
    ap.add_argument("--furnish", default="Semi_Furnished")
    ap.add_argument("--facing", default="East")
    ap.add_argument("--age_label", default="New")
    ap.add_argument("--area_bucket", default="Medium")
    args = ap.parse_args()

    result = predict(
        args.area_sqft, args.bedrooms, args.prop_type, args.locality,
        args.furnish, args.facing, args.age_label, args.area_bucket,
    )
    print(result)
