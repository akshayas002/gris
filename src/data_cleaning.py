"""
data_cleaning.py
=================
Rebuilds a clean, leakage-free modeling dataset from the project's engineered
export (data/hyderabad_engineered.csv).

WHY THIS FILE EXISTS
---------------------
The original eda_feature_engineering.py had an unresolved git merge conflict
(<<<<<<< / ======= / >>>>>>>) that made the script un-runnable, and the CSV
it had already produced showed the scars of that conflict: duplicate columns
(AREA_SQFT vs AREA_SQFT.1, LOCALITY vs LOCALITY.1, etc.) and several
target-leaking columns kept as "features"
(LOCALITY_MEDIAN_PRICE, PRICE_PER_BED, LOG_PRICE_SQFT).

This script starts from that same export but:
  1. Removes exact-duplicate columns.
  2. Drops every column that is mathematically derived from the target.
  3. Scopes the dataset to the segment that can be modeled honestly
     (see SCOPING DECISIONS below).
  4. Winsorizes remaining data-entry-error outliers in the target.
  5. Writes out a clean CSV + a JSON log of every row/column dropped and why,
     so the whole cleaning process is auditable.

SCOPING DECISIONS (documented, not hidden)
-------------------------------------------
Investigating TARGET_PRICE_SQFT by TRANSACT_LABEL showed the original target
mixed three incompatible price bases in one column:
    Sale     -> median price/sqft ~10,000 INR   (correct scale for a sale)
    Rent     -> median price/sqft ~6,300 INR    (this is monthly rent/sqft,
                                                  not a sale price - different
                                                  economic quantity entirely)
    Unknown  -> median price/sqft ~28 INR       (almost certainly unlabeled
                                                  rentals; ~4300x smaller than
                                                  the Sale median)
Training one regressor on all three at once means the model is being asked to
predict three different things with one number - that alone can produce the
kind of exploded MAPE the original run showed. We restrict this model to
TRANSACT_LABEL == 'Sale'.

Within Sale, PROP_TYPE_CLEAN == 'Land' has both a different price mechanism
(no structure, no floors/furnishing/amenities) and a ~4x higher median
price/sqft than built residential property. Mixing land into a "residential
price/sqft" model contaminates both the feature relationships (amenity count,
furnishing etc. are meaningless for land) and the target scale. We scope the
model to built residential types: Apartment, Villa, Builder_Floor, Farmhouse.
(Serviced_Apt / Studio / Other are dropped for a different reason: <5 rows
each after filtering - too few to model or evaluate honestly.)
"""
import json
import numpy as np
import pandas as pd

RAW_PATH = "data/raw_engineered.csv"
OUT_CSV = "data/clean_dataset.csv"
LOG_PATH = "reports/cleaning_log.json"

RESIDENTIAL_TYPES = ["Apartment", "Villa", "Builder_Floor", "Farmhouse"]

# Columns that are mathematically derived from the target (TARGET_PRICE_SQFT)
# and therefore cannot be used as model inputs without leaking the answer.
LEAKAGE_COLS = [
    "LOG_PRICE_SQFT",        # = log1p(TARGET_PRICE_SQFT), literally the label
    "LOCALITY_MEDIAN_PRICE", # locality-level aggregate of the target itself
    "PRICE_PER_BED",         # = PRICE_INR / bedrooms, and PRICE_INR IS the target*area
    "PRICE_INR",             # total price = target_price_sqft * area -> leakage
]

# Identifier / no-signal columns
ID_COLS = ["SPID", "PROP_ID"]


def log(entries, msg):
    print(msg)
    entries.append(msg)


def main():
    entries = []
    df = pd.read_csv(RAW_PATH, low_memory=False)
    log(entries, f"Loaded raw export: {df.shape[0]} rows, {df.shape[1]} cols")

    # ---- 1. de-duplicate columns produced by the unresolved merge conflict ----
    dup_pairs = [
        ("AREA_SQFT", "AREA_SQFT.1"),
        ("LOCALITY", "LOCALITY.1"),
        ("PROP_TYPE_CLEAN", "PROP_TYPE_CLEAN.1"),
        ("LISTING_AGE_DAYS", "LISTING_AGE_DAYS.1"),
        ("LOCALITY_MEDIAN_PRICE", "LOCALITY_MEDIAN_PRICE.1"),
    ]
    for base, dup in dup_pairs:
        if dup in df.columns:
            identical = df[base].equals(df[dup]) or (df[base] == df[dup]).mean() > 0.999
            log(entries, f"Dropping duplicate column '{dup}' (identical to '{base}': {identical})")
            df = df.drop(columns=[dup])

    # ---- 2. drop target-leaking columns ----
    for c in LEAKAGE_COLS:
        if c in df.columns and c != "PRICE_INR":  # keep PRICE_INR temporarily for sanity checks below
            pass
    leak_present = [c for c in LEAKAGE_COLS if c in df.columns]
    log(entries, f"Dropping leakage columns: {leak_present}")
    df = df.drop(columns=[c for c in leak_present if c != "PRICE_INR"])
    # PRICE_INR dropped too (kept out of LEAKAGE_COLS loop above only for the sanity note)
    if "PRICE_INR" in df.columns:
        df = df.drop(columns=["PRICE_INR"])

    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])

    # ---- 3. scope: Sale only ----
    before = len(df)
    df = df[df["TRANSACT_LABEL"] == "Sale"].copy()
    log(entries, f"Scoped to TRANSACT_LABEL == 'Sale': {before} -> {len(df)} rows")
    df = df.drop(columns=["TRANSACT_LABEL"])  # now constant, no signal left

    # ---- 4. scope: residential built property types only ----
    before = len(df)
    df = df[df["PROP_TYPE_CLEAN"].isin(RESIDENTIAL_TYPES)].copy()
    log(entries, f"Scoped to residential types {RESIDENTIAL_TYPES}: {before} -> {len(df)} rows")

    # ---- 5. winsorize remaining data-entry-error outliers in the target ----
    lo, hi = df["TARGET_PRICE_SQFT"].quantile([0.01, 0.99])
    before = len(df)
    df = df[(df["TARGET_PRICE_SQFT"] >= lo) & (df["TARGET_PRICE_SQFT"] <= hi)].copy()
    log(entries, f"Trimmed TARGET_PRICE_SQFT to [1st,99th] percentile = [{lo:.0f}, {hi:.0f}]: {before} -> {len(df)} rows")

    # ---- 6. basic missing-value handling (documented, not silent) ----
    # FLOOR_RATIO / BALCONY_NUM / BALCONY_BUCKET / BEDROOM_NUM_CLEAN have real
    # missingness (not encoded as 0) - impute with median/mode and add
    # "was_missing" flags so the model can learn if missingness itself is informative.
    for col in ["FLOOR_RATIO", "BALCONY_NUM", "BEDROOM_NUM_CLEAN", "TOTAL_FLOOR"]:
        missing_frac = df[col].isnull().mean()
        if missing_frac > 0:
            df[f"{col}_WAS_MISSING"] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df[col].median())
            log(entries, f"Imputed '{col}' median (missing frac={missing_frac:.2%}), added '{col}_WAS_MISSING' flag")

    if df["BALCONY_BUCKET"].isnull().any():
        df["BALCONY_BUCKET"] = df["BALCONY_BUCKET"].fillna("Unknown")

    log(entries, f"Final clean dataset: {df.shape[0]} rows, {df.shape[1]} cols")

    df.to_csv(OUT_CSV, index=False)
    with open(LOG_PATH, "w") as f:
        json.dump({"steps": entries, "final_shape": list(df.shape)}, f, indent=2)


if __name__ == "__main__":
    main()
