"""
prepare_app_data.py
====================
Builds data/app_listings.csv - the dataset the running app (app.py,
recommender.py, geo_viz.py) actually loads.

WHY THIS IS A SEPARATE STEP FROM data_cleaning.py
---------------------------------------------------
data_cleaning.py produces data/clean_dataset.csv for MODEL TRAINING: it
deliberately drops PRICE_INR, PROP_ID/SPID, and anything derived from the
price target, because those would leak into a model that predicts price.

None of that applies here. The app needs to *display and filter by* the
real, known price of a listing (that's not leakage - leakage is about
predicting a target from features derived from itself, not about showing a
user the true price of a property they're browsing). So this script starts
from the same raw export and applies the same scope decisions (Sale-only,
residential-only) for a coherent, honest listings set, but keeps the fields
a browsing/recommendation UI actually needs: real price, an identifier,
and the location/amenity/spec fields recommender.py and geo_viz.py expect.

KNOWN, DOCUMENTED APPROXIMATIONS
----------------------------------
The engineered export doesn't preserve every field the original recommender
UI was built against. Rather than fabricate data, these are approximated
from what *is* available, and called out here explicitly:

  - AMENITIES: the engineered export only kept AMENITY_COUNT (a number), not
    the original list of amenity names. There's no honest way to reconstruct
    "which" amenities from a count, so AMENITIES is left empty for every row.
    Practical effect: the amenity-Jaccard component of recommender.py's
    scoring (30% of the final score) contributes 0 for every listing - the
    ranking still works, it's just driven by the cosine-similarity and
    location components instead. This is a real limitation of working from
    the engineered export rather than the missing raw scrape, not something
    hidden - see README.
  - AGE (years): the export only kept AGE_LABEL (a bucket: New,
    Under_Construction, 0_5_yrs, 5_10_yrs, Resale_Old, Resale_Very_Old).
    Mapped to a representative numeric midpoint per bucket for display and
    for recommender.py's numeric similarity feature.
  - FLOOR_NUM: the export only kept FLOOR_RATIO (floor / total_floor) and
    TOTAL_FLOOR, not the raw floor number. Reconstructed as
    round(FLOOR_RATIO * TOTAL_FLOOR).
  - PROPERTY_TYPE / FURNISH / FACING: relabeled to lowercase, UI-vocabulary
    values (matching templates/index.html's dropdowns: "flat", "villa",
    "house") since the frontend was built against that vocabulary. See the
    PROP_TYPE_MAP below.
"""
import json

import numpy as np
import pandas as pd

RAW_PATH = "data/raw_engineered.csv"
OUT_PATH = "data/app_listings.csv"
LOG_PATH = "reports/app_data_log.json"

RESIDENTIAL_TYPES = ["Apartment", "Villa", "Builder_Floor", "Farmhouse"]

# Maps the engineered export's PROP_TYPE_CLEAN values to the vocabulary the
# existing frontend dropdowns use (templates/index.html: flat/villa/house).
PROP_TYPE_MAP = {
    "Apartment": "flat",
    "Villa": "villa",
    "Builder_Floor": "house",
    "Farmhouse": "house",
}
FURNISH_MAP = {
    "Furnished": "furnished",
    "Semi_Furnished": "semi-furnished",
    "Unfurnished": "unfurnished",
}
FACING_MAP = {
    "East": "east", "West": "west", "North": "north", "South": "south",
    "NE": "north-east", "Unknown": "unknown",
}
AGE_LABEL_TO_YEARS = {
    "New": 0, "Under_Construction": 0, "0_5_yrs": 2.5,
    "5_10_yrs": 7.5, "Resale_Old": 12, "Resale_Very_Old": 20,
}


def main():
    entries = []
    df = pd.read_csv(RAW_PATH, low_memory=False)
    entries.append(f"Loaded raw export: {len(df)} rows")

    # de-dupe the merge-conflict-era duplicate columns (same as data_cleaning.py)
    for base, dup in [
        ("AREA_SQFT", "AREA_SQFT.1"), ("LOCALITY", "LOCALITY.1"),
        ("PROP_TYPE_CLEAN", "PROP_TYPE_CLEAN.1"), ("LISTING_AGE_DAYS", "LISTING_AGE_DAYS.1"),
        ("LOCALITY_MEDIAN_PRICE", "LOCALITY_MEDIAN_PRICE.1"),
    ]:
        if dup in df.columns:
            df = df.drop(columns=[dup])

    # same scope as the model: Sale only, residential types only
    df = df[df["TRANSACT_LABEL"] == "Sale"]
    df = df[df["PROP_TYPE_CLEAN"].isin(RESIDENTIAL_TYPES)]
    entries.append(f"Scoped to Sale + residential types: {len(df)} rows")

    lo, hi = df["TARGET_PRICE_SQFT"].quantile([0.01, 0.99])
    df = df[df["TARGET_PRICE_SQFT"].between(lo, hi)]
    entries.append(f"Winsorized target outliers: {len(df)} rows remain")

    out = pd.DataFrame(index=df.index)
    out["PROP_ID"] = df["PROP_ID"].astype(str)

    loc = df["LOCALITY"].astype(str).str.strip()
    out["location"] = loc.where(loc.str.len() > 1, "unknown").str.lower()
    out["AREA"] = out["location"]

    out["PRICE"] = df["PRICE_INR"]  # real recorded price - legitimate to display/filter by
    out["BEDROOM_NUM"] = df["BEDROOM_NUM_CLEAN"]
    out["MIN_AREA_SQFT"] = df["AREA_SQFT"]
    out["MAX_AREA_SQFT"] = df["AREA_SQFT"]
    out["AMENITIES"] = ""  # documented limitation - see module docstring
    out["AGE"] = df["AGE_LABEL"].map(AGE_LABEL_TO_YEARS).fillna(10)
    out["FLOOR_NUM"] = (df["FLOOR_RATIO"].fillna(0.5) * df["TOTAL_FLOOR"].fillna(1)).round()
    out["TOTAL_FLOOR"] = df["TOTAL_FLOOR"]
    out["BALCONY_NUM"] = df["BALCONY_NUM"]
    out["TOTAL_LANDMARK_COUNT"] = df["LANDMARK_COUNT"]
    out["REGISTERED_DAYS"] = df["LISTING_AGE_DAYS"]
    out["PROPERTY_TYPE"] = df["PROP_TYPE_CLEAN"].map(PROP_TYPE_MAP).fillna("flat")
    out["FURNISH"] = df["FURNISH_LABEL"].map(FURNISH_MAP).fillna("unfurnished")
    out["FACING"] = df["FACING_LABEL"].map(FACING_MAP).fillna("unknown")
    out["VERIFIED"] = df["IS_VERIFIED"].fillna(0).astype(int)
    out["LAT"] = df["LATITUDE"]
    out["LNG"] = df["LONGITUDE"]

    # sanity-clip clearly bad coordinates to the Hyderabad metro area, same
    # bound used during the model-side investigation of this same issue
    bad = ~(out["LAT"].between(16.8, 18.0) & out["LNG"].between(77.9, 78.9))
    out.loc[bad, ["LAT", "LNG"]] = np.nan
    entries.append(f"Nulled out-of-bounds coordinates: {int(bad.sum())} rows")

    out = out.reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    entries.append(f"Wrote {OUT_PATH}: {len(out)} rows, {len(out.columns)} cols")

    with open(LOG_PATH, "w") as f:
        json.dump({"steps": entries}, f, indent=2)
    for e in entries:
        print(e)


if __name__ == "__main__":
    main()
