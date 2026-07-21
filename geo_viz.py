"""
Geo Visualization Engine
Outputs: price heatmap, listing cluster map, listing-specific map, area summary
All maps saved as self-contained HTML files.
"""

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from sklearn.cluster import KMeans
from pathlib import Path

MAP_DIR = Path("static/maps")
MAP_DIR.mkdir(parents=True, exist_ok=True)

HYD_LAT, HYD_LNG = 17.3850, 78.4867

# Known Hyderabad locality centroids — used when lat/lng is missing from dataset
AREA_COORDS = {
    "gachibowli":        (17.4401, 78.3489),
    "hitech city":       (17.4435, 78.3772),
    "madhapur":          (17.4484, 78.3908),
    "kondapur":          (17.4600, 78.3600),
    "kukatpally":        (17.4849, 78.3996),
    "banjara hills":     (17.4100, 78.4400),
    "jubilee hills":     (17.4314, 78.4070),
    "miyapur":           (17.4965, 78.3579),
    "manikonda":         (17.4018, 78.3862),
    "nallagandla":       (17.4620, 78.3230),
    "uppal":             (17.4060, 78.5590),
    "lb nagar":          (17.3469, 78.5524),
    "dilsukhnagar":      (17.3688, 78.5246),
    "secunderabad":      (17.4399, 78.4983),
    "begumpet":          (17.4418, 78.4636),
    "ameerpet":          (17.4376, 78.4482),
    "kompally":          (17.5455, 78.4862),
    "bachupally":        (17.5231, 78.4127),
    "nizampet":          (17.5086, 78.3908),
    "pragathi nagar":    (17.5100, 78.4000),
    "financial district":(17.4156, 78.3419),
    "narsingi":          (17.3900, 78.3580),
    "kokapet":           (17.4094, 78.3322),
    "tellapur":          (17.4620, 78.2990),
    "attapur":           (17.3670, 78.4190),
    "mehdipatnam":       (17.3925, 78.4354),
    "tolichowki":        (17.4040, 78.4140),
    "shamshabad":        (17.2430, 78.4294),
    "adibatla":          (17.3008, 78.5679),
    "maheshwaram":       (17.2690, 78.4470),
    "shankarpally":      (17.4570, 78.2280),
    "patancheru":        (17.5310, 78.2630),
    "sultanpur":         (17.4570, 78.2820),
    "mokila":            (17.4630, 78.2650),
    "chevella":          (17.3070, 78.1490),
    "gandipet":          (17.3870, 78.3360),
    "puppalaguda":       (17.4005, 78.3665),
    "nanakramguda":      (17.4244, 78.3536),
    "raidurgam":         (17.4279, 78.3731),
    "serilingampally":   (17.4820, 78.3308),
    "hafeezpet":         (17.4847, 78.3627),
    "aminpur":           (17.5221, 78.3622),
    "boduppal":          (17.4298, 78.5674),
    "peerzadiguda":      (17.4420, 78.5613),
    "pocharam":          (17.4818, 78.5654),
    "ghatkesar":         (17.4450, 78.6842),
    "alwal":             (17.5030, 78.5158),
    "malkajgiri":        (17.4560, 78.5297),
    "sainikpuri":        (17.4817, 78.5528),
    "kapra":             (17.4713, 78.5625),
    "ecil":              (17.4691, 78.5569),
    "hayathnagar":       (17.3360, 78.5980),
    "vanasthalipuram":   (17.3388, 78.5480),
    "saroornagar":       (17.3431, 78.5367),
    "ramanthapur":       (17.4023, 78.5568),
    "tarnaka":           (17.4296, 78.5320),
    "amberpet":          (17.4169, 78.5215),
    "narayanguda":       (17.3950, 78.4856),
    "himayathnagar":     (17.4027, 78.4807),
    "somajiguda":        (17.4243, 78.4570),
    "punjagutta":        (17.4328, 78.4498),
    "sr nagar":          (17.4523, 78.4297),
    "erragadda":         (17.4564, 78.4286),
    "yousufguda":        (17.4388, 78.4260),
    "banjara hills road no 12": (17.4180, 78.4360),
}


def _get_coord(location_str: str) -> tuple:
    """Return (lat, lng) for a locality name string, or (nan, nan)."""
    loc = str(location_str).lower().strip()
    # Exact match first
    if loc in AREA_COORDS:
        lat, lng = AREA_COORDS[loc]
        return (lat + np.random.uniform(-0.004, 0.004),
                lng + np.random.uniform(-0.004, 0.004))
    # Partial match
    for key, (lat, lng) in AREA_COORDS.items():
        if key in loc or loc in key:
            return (lat + np.random.uniform(-0.004, 0.004),
                    lng + np.random.uniform(-0.004, 0.004))
    return (np.nan, np.nan)


def _extract_lat_lng(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign LAT/LNG to every row.
    Priority: existing LAT/LNG cols → MAP_DETAILS JSON → locality name fallback.
    By the time data reaches here, df['location'] is already a plain string
    (parsed by app.py's parse_location function).
    """
    df = df.copy()

    # If LAT/LNG already exist and are mostly populated, keep them
    if "LAT" in df.columns and df["LAT"].notna().sum() > len(df) * 0.5:
        return df

    # Try MAP_DETAILS JSON
    if "MAP_DETAILS" in df.columns:
        import json, re as _re
        def _parse_map(val):
            try:
                d = json.loads(str(val))
                return float(d.get("latitude", np.nan)), float(d.get("longitude", np.nan))
            except Exception:
                lat = _re.search(r'"lat(?:itude)?"\\s*:\\s*([\\d.]+)', str(val))
                lng = _re.search(r'"l(?:ng|ong(?:itude)?)?"\\s*:\\s*([\\d.]+)', str(val))
                return (float(lat.group(1)) if lat else np.nan,
                        float(lng.group(1)) if lng else np.nan)
        coords    = df["MAP_DETAILS"].apply(_parse_map)
        df["LAT"] = coords.apply(lambda x: x[0])
        df["LNG"] = coords.apply(lambda x: x[1])

    # Fallback: locality name → known centroid (with jitter)
    if "location" in df.columns:
        missing = df["LAT"].isna() if "LAT" in df.columns else pd.Series(True, index=df.index)
        if missing.any():
            fallback = df.loc[missing, "location"].apply(_get_coord)
            df.loc[missing, "LAT"] = fallback.apply(lambda x: x[0])
            df.loc[missing, "LNG"] = fallback.apply(lambda x: x[1])

    # Final: still-missing rows get Hyderabad center with large jitter
    if "LAT" in df.columns:
        still_missing = df["LAT"].isna()
        df.loc[still_missing, "LAT"] = HYD_LAT + np.random.uniform(-0.05, 0.05, still_missing.sum())
        df.loc[still_missing, "LNG"] = HYD_LNG + np.random.uniform(-0.05, 0.05, still_missing.sum())

    return df


def build_price_heatmap(df: pd.DataFrame, output: str = "price_heatmap.html") -> str:
    df          = _extract_lat_lng(df.copy())
    df["PRICE_NUM"] = pd.to_numeric(df.get("PRICE"), errors="coerce")
    valid       = df.dropna(subset=["LAT", "LNG", "PRICE_NUM"])

    if len(valid) == 0:
        print("No geo data available for heatmap")
        return ""

    p_min, p_max = valid["PRICE_NUM"].min(), valid["PRICE_NUM"].max()
    valid        = valid.copy()
    valid["WEIGHT"] = (valid["PRICE_NUM"] - p_min) / (p_max - p_min + 1)

    m = folium.Map(location=[HYD_LAT, HYD_LNG], zoom_start=11, tiles="CartoDB positron")

    HeatMap(
        valid[["LAT", "LNG", "WEIGHT"]].values.tolist(),
        min_opacity=0.3,
        max_opacity=0.85,
        radius=22,
        blur=25,
        gradient={0.2: "#1D9E75", 0.5: "#EF9F27", 0.8: "#D85A30", 1.0: "#E24B4A"},
    ).add_to(m)

    # Locality median price labels
    if "location" in valid.columns:
        summary = (
            valid.groupby("location")
            .agg(median_price=("PRICE_NUM", "median"),
                 lat=("LAT", "mean"), lng=("LNG", "mean"))
            .reset_index()
        )
        for _, row in summary.iterrows():
            p_lakh = row["median_price"] / 1e5
            folium.Marker(
                location=[row["lat"], row["lng"]],
                icon=folium.DivIcon(
                    html=(f'<div style="font-family:\'Inter\', sans-serif; font-size:10px; font-weight:600; '
                          f'background:rgba(255,255,255,0.95); padding:3px 8px; border-radius:12px; '
                          f'box-shadow: 0 1px 3px rgba(0,0,0,0.15); border: 1px solid #e2e0d8; '
                          f'white-space:nowrap; color:#0a5940; text-align:center;">₹{p_lakh:.0f}L</div>'),
                    icon_size=(70, 22),
                ),
                tooltip=f"{row['location'].title()} — median ₹{p_lakh:.1f}L",
            ).add_to(m)

    out_path = MAP_DIR / output
    m.save(str(out_path))
    print(f"Saved price heatmap -> {out_path}  ({len(valid):,} points)")
    return str(out_path)


def build_cluster_map(df: pd.DataFrame, n_clusters: int = 10,
                      output: str = "cluster_map.html") -> str:
    df    = _extract_lat_lng(df.copy())
    valid = df.dropna(subset=["LAT", "LNG"]).copy()

    if len(valid) < n_clusters:
        n_clusters = max(2, len(valid) // 50)

    coords       = valid[["LAT", "LNG"]].values
    km           = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    valid["CLUSTER"] = km.fit_predict(coords)

    COLORS = ["#0A5940", "#7F77DD", "#D85A30", "#378ADD",
              "#EF9F27", "#D4537E", "#17B2C3", "#A259D9",
              "#639922", "#E24B4A"]

    m  = folium.Map(location=[HYD_LAT, HYD_LNG], zoom_start=11, tiles="CartoDB positron")
    mc = MarkerCluster(
        options={"maxClusterRadius": 40, "disableClusteringAtZoom": 14}
    ).add_to(m)

    valid["PRICE_NUM"] = pd.to_numeric(valid.get("PRICE"), errors="coerce")

    for _, row in valid.iterrows():
        c      = int(row["CLUSTER"]) % len(COLORS)
        p_lakh = row["PRICE_NUM"] / 1e5 if pd.notna(row.get("PRICE_NUM")) else 0
        beds   = int(row["BEDROOM_NUM"]) if pd.notna(row.get("BEDROOM_NUM")) else "?"
        loc    = str(row.get("location", "")).title()
        prop_type = str(row.get('PROPERTY_TYPE','')).upper()
        furnish = str(row.get('FURNISH','')).capitalize()
        
        popup_html = (
            f'<div style="font-family:\'Inter\', sans-serif; font-size:12px; color:#1a1a18; min-width:180px; padding:4px;">'
            f'  <div style="font-weight:600; font-size:13px; margin-bottom:4px; color:{COLORS[c]};">{loc}</div>'
            f'  <div style="display:flex; align-items:baseline; gap:6px; margin-bottom:6px;">'
            f'    <span style="font-family:\'DM Mono\', monospace; font-size:15px; font-weight:700; color:#0a5940;">₹{p_lakh:.1f}L</span>'
            f'    <span style="color:#6b6a64; font-size:11px;">({beds} BHK)</span>'
            f'  </div>'
            f'  <div style="display:flex; gap:4px; flex-wrap:wrap;">'
            f'    <span style="font-size:9px; font-weight:600; background:#e8f4f0; color:#0a5940; padding:2px 5px; border-radius:4px; border:1px solid #c6e5d9;">{prop_type}</span>'
            f'    <span style="font-size:9px; font-weight:600; background:#f0ede6; color:#6b6a64; padding:2px 5px; border-radius:4px; border:1px solid #e2e0d8;">{furnish}</span>'
            f'  </div>'
            f'</div>'
        )
        
        folium.CircleMarker(
            location=[row["LAT"], row["LNG"]],
            radius=5, color=COLORS[c], fill=True,
            fill_color=COLORS[c], fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"₹{p_lakh:.1f}L — {loc}",
        ).add_to(mc)

    # Zone labels at centroids
    for cid in range(n_clusters):
        cluster_rows = valid[valid["CLUSTER"] == cid]
        center       = km.cluster_centers_[cid]
        med_price    = cluster_rows["PRICE_NUM"].median() / 1e5 if cluster_rows["PRICE_NUM"].notna().any() else 0
        folium.Marker(
            location=center.tolist(),
            icon=folium.DivIcon(
                html=(f'<div style="background:{COLORS[cid % len(COLORS)]}; color:#fff; '
                      f'font-family:\'Inter\', sans-serif; font-size:10px; font-weight:600; '
                      f'padding:4px 10px; border-radius:999px; box-shadow: 0 2px 5px rgba(0,0,0,0.15); '
                      f'border: 1px solid rgba(255,255,255,0.3); text-align: center; '
                      f'white-space:nowrap;">Zone {cid+1} ({len(cluster_rows)})</div>'),
                icon_size=(150, 26),
            ),
            tooltip=f"Zone {cid+1}: {len(cluster_rows)} listings · median ₹{med_price:.0f}L",
        ).add_to(m)

    out_path = MAP_DIR / output
    m.save(str(out_path))
    print(f"Saved cluster map -> {out_path}  ({len(valid):,} listings, {n_clusters} zones)")
    return str(out_path)


def build_listing_map(df: pd.DataFrame, listings_df: pd.DataFrame,
                      output: str = "listings_map.html") -> str:
    """Plot a set of recommendation results on a focused map."""
    listings_df = _extract_lat_lng(listings_df.copy())
    valid       = listings_df.dropna(subset=["LAT", "LNG"])
    if len(valid) == 0:
        return ""

    center_lat = valid["LAT"].mean()
    center_lng = valid["LNG"].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles="CartoDB positron")

    for _, row in valid.iterrows():
        price  = pd.to_numeric(row.get("PRICE"), errors="coerce")
        p_lakh = price / 1e5 if pd.notna(price) else 0
        beds   = int(row["BEDROOM_NUM"]) if pd.notna(row.get("BEDROOM_NUM")) else "?"
        score  = float(row.get("similarity_score", 0))
        loc    = str(row.get("location", "")).title()
        reason = str(row.get("match_reason", ""))
        color  = "#0A5940" if score > 0.7 else "#EF9F27" if score > 0.4 else "#888780"

        popup_html = (
            f'<div style="font-family:\'Inter\', sans-serif; font-size:12px; color:#1a1a18; min-width:190px; padding:4px;">'
            f'  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
            f'    <span style="font-family:\'DM Mono\', monospace; font-size:15px; font-weight:700; color:#0a5940;">₹{p_lakh:.1f}L</span>'
            f'    <span style="font-size:10px; font-weight:600; color:{color}; background:{color}15; padding:2px 6px; border-radius:999px;">{int(score*100)}% Match</span>'
            f'  </div>'
            f'  <div style="font-weight:500; font-size:11px; color:#6b6a64; margin-bottom:4px;">{beds} BHK · {loc}</div>'
            f'  <div style="font-size:10px; color:#6b6a64; background:#f0ede6; padding:4px 6px; border-radius:4px; border:1px solid #e2e0d8; font-style:italic;">'
            f'    {reason}'
            f'  </div>'
            f'</div>'
        )

        folium.CircleMarker(
            location=[row["LAT"], row["LNG"]],
            radius=9, color=color, fill=True,
            fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"₹{p_lakh:.1f}L · {beds}BHK · {loc}",
        ).add_to(m)

    out_path = MAP_DIR / output
    m.save(str(out_path))
    return str(out_path)


def area_price_summary(df: pd.DataFrame, top_n: int = 15) -> list:
    """Returns area comparison data for the Insights bar charts."""
    df          = df.copy()
    df["PRICE_NUM"] = pd.to_numeric(df.get("PRICE"), errors="coerce")
    loc_col     = "location" if "location" in df.columns else "AREA"
    if loc_col not in df.columns:
        return []

    agg_dict = {
        "median_price":  ("PRICE_NUM", "median"),
        "listing_count": ("PRICE_NUM", "count"),
    }
    if "AVG_AREA_SQFT" in df.columns:
        agg_dict["avg_area"] = ("AVG_AREA_SQFT", "mean")

    summary = (
        df.dropna(subset=["PRICE_NUM"])
        .groupby(loc_col)
        .agg(**agg_dict)
        .sort_values("listing_count", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={loc_col: "area"})
    )
    summary["median_price_lakh"] = (summary["median_price"] / 1e5).round(1)

    # Replace NaN with None for JSON safety
    return [
        {k: (None if isinstance(v, float) and np.isnan(v) else v)
         for k, v in row.items()}
        for row in summary.to_dict(orient="records")
    ]
