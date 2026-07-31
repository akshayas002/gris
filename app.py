"""
FastAPI Backend — Property Recommendation System
Endpoints:
  GET  /                    → web dashboard
  GET  /health              → status check
  POST /recommend           → property recommendations
  GET  /similar/{prop_id}   → listings similar to a given property
  GET  /map/heatmap         → price heatmap (Folium HTML)
  GET  /map/clusters        → KMeans cluster map (Folium HTML)
  GET  /insights/areas      → area price data for charts
  GET  /insights/localities → top localities table
  POST /predict             → ML price prediction
  GET  /listings            → paginated listing browse

NOTE ON DATA/MODEL SOURCE (v2 update)
---------------------------------------
The original data/listings_latest.csv (raw scrape) this app was built
against was never available in the delivered project. This version runs on
data/app_listings.csv - built by src/prepare_app_data.py from the project's
engineered export - and models/model_v2.pkl, a leak-fixed model trained on
this same data (see reports/BEFORE_AFTER.md). Scope: Sale-only, residential
property types only (Apartment/Villa/Builder_Floor/Farmhouse) - see
src/prepare_app_data.py's docstring for exactly what that means and what's
approximated (amenity list, floor number, age-in-years).
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from recommender import PropertyRecommender, UserQuery
from geo_viz import (
    build_price_heatmap, build_cluster_map,
    build_listing_map, area_price_summary,
)
from model_bridge import predict_price

# ── Update these paths if your files are elsewhere ────────────────────────────
DATA_PATH  = "data/app_listings.csv"
MODEL_PATH = "models/model_v2.pkl"
# ─────────────────────────────────────────────────────────────────────────────

MAP_DIR = Path("static/maps")
MAP_DIR.mkdir(parents=True, exist_ok=True)

state: dict = {}


# ─── JSON-safe record cleaner ─────────────────────────────────────────────────
def clean_record(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if isinstance(v, set):
            continue                        # AMENITY_SET not needed in API response
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ─── App startup ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data and models...")

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: python src/prepare_app_data.py first."
        )
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # app_listings.csv is already clean (location is plain text, PRICE is
    # numeric) - just derive the one convenience column downstream code expects
    mn = pd.to_numeric(df.get("MIN_AREA_SQFT"), errors="coerce").fillna(0)
    mx = pd.to_numeric(df.get("MAX_AREA_SQFT"), errors="coerce").fillna(0)
    df["AVG_AREA_SQFT"] = ((mn + mx) / 2).replace(0, np.nan)

    state["df"]          = df
    state["recommender"] = PropertyRecommender(df)

    # Always regenerate maps so location names are clean
    for cached in (MAP_DIR / "price_heatmap.html", MAP_DIR / "cluster_map.html"):
        if cached.exists():
            cached.unlink()
    build_price_heatmap(df)
    build_cluster_map(df)

    # ML model
    if Path(MODEL_PATH).exists():
        state["model"] = joblib.load(MODEL_PATH)
        print("ML model loaded (model_v2.pkl)")
    else:
        state["model"] = None
        print("No model_v2.pkl found — /predict returns 501. Run src/train_model.py first.")

    print(f"Ready. {len(df):,} listings loaded.")
    yield
    state.clear()


app = FastAPI(title="Property Recommendation System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Schemas ──────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    location:      str
    budget_min:    float
    budget_max:    float
    bedrooms:      Optional[int]   = None
    property_type: Optional[str]   = None
    furnish:       Optional[str]   = None
    amenities:     List[str]       = []
    top_n:         int             = 10


class PredictRequest(BaseModel):
    bedrooms:      int
    area_sqft:     float
    location:      str
    property_type: str = "flat"
    furnish:       str = "semi-furnished"
    age:           int = 0
    floor:         int = 1
    total_floors:  int = 5
    amenity_count: int = 5
    landmarks:     int = 3
    verified:      int = 1


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the web dashboard."""
    path = Path("templates/index.html")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return HTMLResponse(
        "<h2>Dashboard not found.</h2><p>Place <code>index.html</code> in the <code>templates/</code> folder.</p>",
        status_code=404,
    )


@app.get("/health")
def health():
    return {"status": "ok", "listings": len(state.get("df", []))}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    query = UserQuery(
        location=req.location, budget_min=req.budget_min,
        budget_max=req.budget_max, bedrooms=req.bedrooms,
        property_type=req.property_type, furnish=req.furnish,
        amenities=req.amenities, top_n=req.top_n,
    )
    results = state["recommender"].recommend(query)
    if results.empty:
        return {"results": [], "count": 0,
                "message": "No listings matched. Try widening the budget or changing location."}

    build_listing_map(state["df"], results, "recommendations_map.html")
    return {
        "results": [clean_record(r) for r in results.to_dict(orient="records")],
        "count":   len(results),
        "map_url": "/static/maps/recommendations_map.html",
    }


@app.get("/similar/{prop_id}")
def similar(prop_id: str, top_n: int = Query(6, ge=1, le=20)):
    results = state["recommender"].similar_to(prop_id, top_n=top_n)
    if results.empty:
        raise HTTPException(404, f"Property '{prop_id}' not found")
    return {"results": [clean_record(r) for r in results.to_dict(orient="records")],
            "count": len(results)}


@app.get("/map/heatmap", response_class=HTMLResponse)
def heatmap():
    path = MAP_DIR / "price_heatmap.html"
    if not path.exists():
        build_price_heatmap(state["df"])
    return path.read_text(encoding="utf-8")


@app.get("/map/clusters", response_class=HTMLResponse)
def clusters():
    path = MAP_DIR / "cluster_map.html"
    if not path.exists():
        build_cluster_map(state["df"])
    return path.read_text(encoding="utf-8")


@app.get("/insights/areas")
def areas(top_n: int = Query(15, ge=5, le=30)):
    return {"data": area_price_summary(state["df"], top_n=top_n)}


@app.get("/insights/localities")
def localities(top_n: int = Query(15, ge=1, le=50)):
    data = state["recommender"].top_localities(top_n=top_n)
    return {"data": [clean_record(r) for r in data.to_dict(orient="records")]}


@app.post("/predict")
def predict(req: PredictRequest):
    if state["model"] is None:
        raise HTTPException(501, "Model not loaded. Run src/train_model.py first.")

    try:
        result = predict_price(state["model"], req)
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    price = result["predicted_price_per_sqft_inr"] * req.area_sqft
    response = {
        "predicted_price": round(price),
        "low_estimate":    round(price * 0.85),
        "high_estimate":   round(price * 1.15),
        "in_lakhs":        round(price / 1e5, 2),
        "location":        req.location,
        "bedrooms":        req.bedrooms,
        "area_sqft":       req.area_sqft,
    }
    if not result["in_scope"]:
        response["scope_note"] = result["scope_note"]
    return response


@app.get("/listings")
def listings(
    page:          int            = Query(1,    ge=1),
    page_size:     int            = Query(20,   ge=5, le=100),
    location:      Optional[str]  = Query(None),
    min_price:     Optional[float]= Query(None),
    max_price:     Optional[float]= Query(None),
    bedrooms:      Optional[int]  = Query(None),
    property_type: Optional[str]  = Query(None),
):
    df              = state["df"].copy()
    df["PRICE_NUM"] = pd.to_numeric(df.get("PRICE"), errors="coerce")

    if location:
        df = df[df["location"].str.contains(location.lower(), na=False, regex=False)]
    if min_price is not None:
        df = df[df["PRICE_NUM"] >= min_price]
    if max_price is not None:
        df = df[df["PRICE_NUM"] <= max_price]
    if bedrooms is not None:
        beds = pd.to_numeric(df.get("BEDROOM_NUM"), errors="coerce")
        df   = df[beds == bedrooms]
    if property_type:
        pt_col = df.get("PROPERTY_TYPE", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
        df     = df[pt_col.str.contains(property_type.lower(), regex=False, na=False)]

    total   = len(df)
    start   = (page - 1) * page_size
    page_df = df.iloc[start: start + page_size]

    out_cols = ["PROP_ID", "location", "PRICE_NUM", "BEDROOM_NUM",
                "AVG_AREA_SQFT", "PROPERTY_TYPE", "FURNISH", "VERIFIED"]
    out_cols = [c for c in out_cols if c in page_df.columns]
    records  = [clean_record(r) for r in
                page_df[out_cols].rename(columns={"PRICE_NUM": "PRICE"}).to_dict(orient="records")]

    return {
        "data":      records,
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     max(1, (total + page_size - 1) // page_size),
    }
