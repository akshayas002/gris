"""
Property Recommendation Engine
Approach: Rule-based hard filter → cosine similarity re-rank → amenity boost
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class UserQuery:
    """What the user types into the dashboard search form."""
    location:      str
    budget_min:    float
    budget_max:    float
    bedrooms:      Optional[int]  = None
    property_type: Optional[str]  = None
    furnish:       Optional[str]  = None
    amenities:     List[str]      = field(default_factory=list)
    top_n:         int            = 10


SIM_NUMERIC = [
    "BEDROOM_NUM", "AVG_AREA_SQFT", "PRICE_NORM",
    "AMENITY_COUNT", "TOTAL_LANDMARK_COUNT", "FLOOR_NUM", "AGE",
]
SIM_CATEGORICAL = ["PROPERTY_TYPE_ENC", "FURNISH_ENC", "FACING_ENC"]


def _safe_str(series: pd.Series) -> pd.Series:
    """Cast any series to str before using .str accessor."""
    return series.fillna("unknown").astype(str).str.strip()


def _safe_median(series, fallback: float = 0.0) -> float:
    """Return median, fallback if all NaN."""
    if series is None:
        return fallback
    val = pd.to_numeric(series, errors="coerce").median()
    return float(val) if pd.notna(val) else fallback


class PropertyRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df, self.scaler, self.matrix = self._build_similarity_matrix(df)
        print(f"Recommender ready: {len(self.df):,} listings indexed")

    def _build_similarity_matrix(self, df: pd.DataFrame):
        df = df.copy()

        # Normalise location columns
        for col in ["location", "AREA", "SECONDARY_AREA"]:
            if col in df.columns:
                df[col] = _safe_str(df[col]).str.lower()

        # Coerce numeric columns
        for col in ["BEDROOM_NUM", "AGE", "FLOOR_NUM", "TOTAL_FLOOR",
                    "BALCONY_NUM", "TOTAL_LANDMARK_COUNT", "REGISTERED_DAYS"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Price 0-1 normalisation
        price = pd.to_numeric(df.get("PRICE"), errors="coerce").fillna(0)
        df["PRICE_NORM"] = (price - price.min()) / (price.max() - price.min() + 1)

        # Average area
        mn = pd.to_numeric(df.get("MIN_AREA_SQFT"), errors="coerce").fillna(0)
        mx = pd.to_numeric(df.get("MAX_AREA_SQFT"), errors="coerce").fillna(0)
        df["AVG_AREA_SQFT"] = ((mn + mx) / 2).replace(0, np.nan)

        # Amenity count + set for Jaccard
        amenity_raw = df.get("AMENITIES", pd.Series([""] * len(df), index=df.index))
        df["AMENITY_COUNT"] = amenity_raw.fillna("").apply(
            lambda x: len([a for a in str(x).split(",") if a.strip()])
        )
        df["AMENITY_SET"] = amenity_raw.fillna("").apply(
            lambda x: {a.strip().lower() for a in str(x).split(",") if a.strip()}
        )

        # Encode categoricals — always cast to str first
        for col, enc_col in [
            ("PROPERTY_TYPE", "PROPERTY_TYPE_ENC"),
            ("FURNISH",       "FURNISH_ENC"),
            ("FACING",        "FACING_ENC"),
        ]:
            if col in df.columns:
                df[enc_col] = pd.factorize(_safe_str(df[col]).str.lower())[0]
            else:
                df[enc_col] = 0

        available = [c for c in SIM_NUMERIC + SIM_CATEGORICAL if c in df.columns]
        mat = df[available].copy()
        for c in available:
            mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0)

        scaler = StandardScaler()
        matrix = scaler.fit_transform(mat)
        return df, scaler, matrix

    def recommend(self, query: UserQuery) -> pd.DataFrame:
        df = self.df.copy()

        # ── Stage 1: Hard filters ─────────────────────────────────────────────
        price       = pd.to_numeric(df.get("PRICE"), errors="coerce")
        budget_mask = (price >= query.budget_min * 0.9) & (price <= query.budget_max * 1.1)
        df          = df[budget_mask]
        if len(df) == 0:
            return pd.DataFrame()

        # Location match with fallback
        loc_q    = query.location.lower().strip()
        loc_mask = df["location"].str.contains(loc_q, na=False, regex=False)
        if "AREA" in df.columns:
            loc_mask = loc_mask | df["AREA"].str.contains(loc_q, na=False, regex=False)
        loc_df = df[loc_mask] if loc_mask.sum() >= 5 else df

        # Bedrooms ±1
        if query.bedrooms is not None:
            beds     = pd.to_numeric(loc_df.get("BEDROOM_NUM"), errors="coerce")
            filtered = loc_df[beds.between(query.bedrooms - 1, query.bedrooms + 1).fillna(False)]
            loc_df   = filtered if len(filtered) >= 3 else loc_df

        # Property type
        if query.property_type:
            col      = _safe_str(loc_df.get("PROPERTY_TYPE", pd.Series(dtype=object))).str.lower()
            filtered = loc_df[col.str.contains(query.property_type.lower(), regex=False, na=False)]
            loc_df   = filtered if len(filtered) >= 3 else loc_df

        # Furnish
        if query.furnish:
            col      = _safe_str(loc_df.get("FURNISH", pd.Series(dtype=object))).str.lower()
            filtered = loc_df[col.str.contains(query.furnish.lower(), regex=False, na=False)]
            loc_df   = filtered if len(filtered) >= 3 else loc_df

        if len(loc_df) == 0:
            return pd.DataFrame()

        # ── Stage 2: Query vector ─────────────────────────────────────────────
        global_price = pd.to_numeric(self.df.get("PRICE"), errors="coerce")
        price_mid    = (query.budget_min + query.budget_max) / 2
        price_norm   = (price_mid - global_price.min()) / (global_price.max() - global_price.min() + 1)

        query_row = {
            "BEDROOM_NUM":          float(query.bedrooms) if query.bedrooms else _safe_median(loc_df.get("BEDROOM_NUM")),
            "AVG_AREA_SQFT":        _safe_median(loc_df.get("AVG_AREA_SQFT")),
            "PRICE_NORM":           float(price_norm),
            "AMENITY_COUNT":        float(len(query.amenities)),
            "TOTAL_LANDMARK_COUNT": _safe_median(loc_df.get("TOTAL_LANDMARK_COUNT")),
            "FLOOR_NUM":            _safe_median(loc_df.get("FLOOR_NUM")),
            "AGE":                  _safe_median(loc_df.get("AGE")),
            "PROPERTY_TYPE_ENC":    self._encode_cat("PROPERTY_TYPE", query.property_type),
            "FURNISH_ENC":          self._encode_cat("FURNISH", query.furnish),
            "FACING_ENC":           0,
        }

        available        = [c for c in SIM_NUMERIC + SIM_CATEGORICAL if c in self.df.columns]
        q_vec            = np.array([[query_row.get(c, 0) for c in available]], dtype=float)
        q_vec_scaled     = self.scaler.transform(q_vec)
        candidate_pos    = [self.df.index.get_loc(i) for i in loc_df.index]
        candidate_vecs   = self.matrix[candidate_pos]

        # ── Stage 3: Cosine similarity ────────────────────────────────────────
        cos_scores = cosine_similarity(q_vec_scaled, candidate_vecs)[0]

        # ── Stage 4: Amenity Jaccard ──────────────────────────────────────────
        query_amenities = {a.lower().strip() for a in query.amenities}

        def jaccard(listing_set):
            if not query_amenities or not listing_set:
                return 0.0
            inter = len(query_amenities & listing_set)
            union = len(query_amenities | listing_set)
            return inter / union if union else 0.0

        jaccard_scores = loc_df["AMENITY_SET"].apply(jaccard).values
        loc_exact      = loc_df["location"].str.contains(loc_q, na=False, regex=False).astype(float).values

        # ── Final score: 50% cosine + 30% amenity + 20% location ─────────────
        final_scores = 0.50 * cos_scores + 0.30 * jaccard_scores + 0.20 * loc_exact

        loc_df = loc_df.copy()
        loc_df["similarity_score"] = final_scores
        loc_df["amenity_match"]    = jaccard_scores
        loc_df["location_exact"]   = loc_exact.astype(bool)

        def match_reason(row):
            parts = []
            if row["location_exact"]:
                parts.append(f"in {loc_q.title()}")
            if row["amenity_match"] > 0.3:
                parts.append("amenity match")
            score = row["similarity_score"]
            if score > 0.8:   parts.append("very similar")
            elif score > 0.5: parts.append("similar")
            return ", ".join(parts) if parts else "budget match"

        loc_df["match_reason"] = loc_df.apply(match_reason, axis=1)

        result   = loc_df.sort_values("similarity_score", ascending=False).head(query.top_n)
        out_cols = [
            "PROP_ID", "location", "AREA", "PRICE", "BEDROOM_NUM",
            "AVG_AREA_SQFT", "PROPERTY_TYPE", "FURNISH", "FACING",
            "AGE", "TOTAL_FLOOR", "FLOOR_NUM", "AMENITIES",
            "TOTAL_LANDMARK_COUNT", "VERIFIED",
            "similarity_score", "amenity_match", "match_reason",
        ]
        out_cols = [c for c in out_cols if c in result.columns]
        return result[out_cols].reset_index(drop=True)

    def similar_to(self, prop_id: str, top_n: int = 6) -> pd.DataFrame:
        mask = self.df["PROP_ID"].astype(str) == str(prop_id)
        if not mask.any():
            return pd.DataFrame()
        loc_idx  = self.df.index.get_loc(self.df[mask].index[0])
        scores   = cosine_similarity(self.matrix[loc_idx].reshape(1, -1), self.matrix)[0]
        result   = self.df.copy()
        result["_sim"] = scores
        result   = result[~mask].sort_values("_sim", ascending=False).head(top_n)
        out_cols = ["PROP_ID", "location", "PRICE", "BEDROOM_NUM",
                    "AVG_AREA_SQFT", "PROPERTY_TYPE", "FURNISH", "_sim"]
        out_cols = [c for c in out_cols if c in result.columns]
        return result[out_cols].rename(columns={"_sim": "similarity_score"}).reset_index(drop=True)

    def _encode_cat(self, col: str, value: Optional[str]) -> int:
        if value is None or col not in self.df.columns:
            return 0
        cats = _safe_str(self.df[col]).str.lower().unique().tolist()
        v    = value.lower().strip()
        return cats.index(v) if v in cats else 0

    def top_localities(self, top_n: int = 15) -> pd.DataFrame:
        if "location" not in self.df.columns:
            return pd.DataFrame()
        price = pd.to_numeric(self.df.get("PRICE"), errors="coerce")
        return (
            self.df.assign(PRICE_NUM=price)
            .groupby("location")
            .agg(listing_count=("PROP_ID", "count"),
                 median_price =("PRICE_NUM", "median"),
                 avg_bedrooms =("BEDROOM_NUM", "mean"))
            .sort_values("listing_count", ascending=False)
            .head(top_n)
            .reset_index()
        )