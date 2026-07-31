"""
model_bridge.py
================
Translates the existing frontend's PredictRequest vocabulary (property_type
in flat/villa/house, furnish in furnished/semi-furnished/unfurnished, age in
years, floor numbers) into the feature schema models/model_v2.pkl actually
expects (PROP_TYPE_CLEAN in Apartment/Villa/Builder_Floor/Farmhouse,
FURNISH_LABEL in Furnished/Semi_Furnished/Unfurnished, AGE_LABEL bucket,
FLOOR_RATIO, etc.) - so templates/index.html didn't need to change at all.

Reuses geo_viz.AREA_COORDS to turn the free-text locality the user types
into an approximate (lat, lng) - the model actually uses these as real
numeric features, so a locality-aware estimate is meaningfully better than
a single hardcoded default for every prediction.

Documented approximations (same caveats as src/prepare_app_data.py):
  - AREA_BUCKET edges are a reasonable approximation, not the original
    (unrecoverable - see eda_feature_engineering.py's merge-conflict issue).
  - "house" (Independent House) is mapped to Builder_Floor, the closest
    available category to what that UI option represents.
"""
from typing import Optional

import numpy as np
import pandas as pd

from geo_viz import _get_coord

PROP_TYPE_MAP = {"flat": "Apartment", "villa": "Villa", "house": "Builder_Floor"}
FURNISH_MAP = {"furnished": "Furnished", "semi-furnished": "Semi_Furnished", "unfurnished": "Unfurnished"}

# Approximate bucket edges (sqft) - see module docstring
AREA_BUCKET_EDGES = [(500, "Micro"), (800, "Small"), (1400, "Medium"), (2200, "Large"), (3500, "XL")]


def _area_bucket(area_sqft: float) -> str:
    for edge, label in AREA_BUCKET_EDGES:
        if area_sqft < edge:
            return label
    return "XXL"


def _age_label(age_years: float) -> str:
    if age_years <= 0:
        return "New"
    if age_years <= 5:
        return "0_5_yrs"
    if age_years <= 10:
        return "5_10_yrs"
    if age_years <= 17:
        return "Resale_Old"
    return "Resale_Very_Old"


def predict_price(bundle: dict, req) -> dict:
    """req is app.py's PredictRequest (or anything with the same attributes)."""
    pipe = bundle["pipeline"]

    prop_type = PROP_TYPE_MAP.get((req.property_type or "").lower().strip(), "Land")
    furnish = FURNISH_MAP.get((req.furnish or "").lower().strip(), "Semi_Furnished")
    age_label = _age_label(req.age)
    lat, lng = _get_coord(req.location)
    if np.isnan(lat):
        lat, lng = 17.3850, 78.4867  # Hyderabad center fallback

    total_floor = max(req.total_floors, 1)
    floor_ratio = req.floor / total_floor

    row = {
        "BEDROOM_NUM_CLEAN": req.bedrooms,
        "TOTAL_FLOOR": total_floor,
        "FEATURE_COUNT": max(req.amenity_count, 1),
        "AMENITY_COUNT": req.amenity_count,
        "LANDMARK_COUNT": req.landmarks,
        "FLOOR_RATIO": floor_ratio,
        "BALCONY_NUM": 2,
        "AREA_SQFT": req.area_sqft,
        "LATITUDE": lat,
        "LONGITUDE": lng,
        "LISTING_AGE_DAYS": 30,
        "IS_READY_TO_MOVE": 0 if req.age <= 0 else 1,
        "IS_UNDER_CONSTR": 1 if req.age <= 0 else 0,
        "IS_RERA": 1,
        "IS_RESALE": 1 if req.age > 0 else 0,
        "IS_NEW_BOOKING": 1 if req.age <= 0 else 0,
        "IS_VERIFIED": req.verified,
        "HAS_SOCIETY": 1,
        "IS_TOP_FLOOR": 1 if req.floor >= total_floor else 0,
        "IS_GROUND_FLOOR": 1 if req.floor <= 0 else 0,
        "FLOOR_RATIO_WAS_MISSING": 0,
        "BALCONY_NUM_WAS_MISSING": 0,
        "BEDROOM_NUM_CLEAN_WAS_MISSING": 0,
        "TOTAL_FLOOR_WAS_MISSING": 0,
        "PROP_TYPE_CLEAN": prop_type,
        "FURNISH_LABEL": furnish,
        "FACING_LABEL": "Unknown",
        "AGE_LABEL": age_label,
        "AREA_BUCKET": _area_bucket(req.area_sqft),
        "BALCONY_BUCKET": "Medium",
        "LOCALITY": req.location.lower().strip(),
    }

    cols = bundle["numeric_cols"] + bundle["low_card_cat_cols"] + bundle["high_card_cat_cols"]
    X = pd.DataFrame([row])[cols]

    pred_log = pipe.predict(X)[0]
    price_per_sqft = float(np.expm1(pred_log))

    in_scope = prop_type in bundle["scope"]["prop_types"]
    return {
        "predicted_price_per_sqft_inr": price_per_sqft,
        "in_scope": in_scope,
        "scope_note": None if in_scope else (
            f"'{req.property_type}' maps outside the model's trained scope "
            f"({bundle['scope']['prop_types']}) - treat this estimate with caution."
        ),
    }
