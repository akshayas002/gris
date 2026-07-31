# Hyderabad Property Finder — Complete Project (frontend + backend + fixed ML model)

Your original frontend (`templates/index.html`) and backend logic
(`recommender.py`, `geo_viz.py`) are here **unchanged** — the dashboard,
recommendation engine, and maps all look and behave exactly as you built
them. What's new is that they're now wired to real, available data and the
leak-fixed model, instead of a raw file that was never in the delivered
project and a model with a broken evaluation metric.

## What was actually broken, and what's fixed here

| Problem | Where | Fix |
|---|---|---|
| `data/listings_latest.csv` (the raw scrape) was never included, so `app.py` couldn't start | `app.py` | New `src/prepare_app_data.py` builds `data/app_listings.csv` from the engineered export that *did* survive, scoped to Sale + residential listings |
| `models/model.pkl`'s `/predict` used a leaked, broken model (MAPE 1.9M% in the original run) | `train_model.py` | Replaced with `models/model_v2.pkl` — leak-free `TargetEncoder`, 4-model CV search (Ridge/RF/HistGB/XGBoost), documented scope |
| `/predict`'s feature schema didn't match any available model | `app.py` | New `model_bridge.py` translates the frontend's existing vocabulary (flat/villa/house, furnished/semi-furnished) into the new model's schema — **`templates/index.html` needed zero changes** |
| Unrecognized property type (e.g. "plot") silently defaulted to a safe category instead of being flagged | `model_bridge.py` | Now explicitly maps to an out-of-scope sentinel and returns a `scope_note` in the response |
| Leftover duplicate frontend draft (`static/index.html`) | cleanup | Removed — `templates/index.html` is the only, current version |

## Run it

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Build the app's data (one-time; only needed if data/app_listings.csv isn't already present)
python src/prepare_app_data.py

# 2. Start the server
uvicorn app:app --reload --port 8000
```
Open **http://localhost:8000** — dashboard, recommendations, maps, and the
price predictor all work against real data and the fixed model.

`models/model_v2.pkl` is already trained and included, so you don't need to
re-run `src/train_model.py` unless you want to retrain it yourself (takes a
few minutes — see `reports/BEFORE_AFTER.md` for what that pipeline does).

## What's honestly still limited (carried over from the underlying data, not hidden)

- **~80% of listings have no recorded locality name** (`": "` in the source
  data — a real bug in the original scraper/feature-engineering step, not
  something introduced here). These show up as `location: "unknown"`
  throughout the app. Filtering/searching by a specific locality only works
  reliably for the ~20% of listings that do have one.
- **No amenity list survived feature engineering**, only a count. The
  recommender's amenity-match scoring (30% of its ranking formula) is
  effectively neutral for every listing as a result — ranking is still
  driven by the cosine-similarity and location components.
- **Villa price predictions are meaningfully less reliable than Apartment
  predictions** (R²=0.32 vs 0.79 on held-out data) — the available features
  don't capture what actually drives luxury pricing. Worth stating in the UI
  if this becomes a real product decision.
- **Age (years) and floor number are approximated** from buckets/ratios in
  the source data (see `src/prepare_app_data.py` and `model_bridge.py`
  docstrings) since the exact original values weren't preserved upstream.

## Project layout

```
app.py                    FastAPI backend (updated: new data/model paths, new /predict)
model_bridge.py           NEW — translates frontend vocab → new model's feature schema
recommender.py            unchanged
geo_viz.py                unchanged
templates/index.html      unchanged
static/                   maps/ regenerate on startup

data/
  raw_engineered.csv       the project's original engineered export (input)
  clean_dataset.csv        leak-free, model-training dataset (src/data_cleaning.py output)
  app_listings.csv         app-facing dataset (src/prepare_app_data.py output)

models/model_v2.pkl        the fixed, retrained model (see reports/BEFORE_AFTER.md)

src/
  data_cleaning.py         builds clean_dataset.csv for model training
  prepare_app_data.py      builds app_listings.csv for the running app
  train_model.py           4-model CV search, saves model_v2.pkl
  baseline_replica.py      reproduces the original bugs, for honest before/after comparison
  predict.py               standalone CLI predictor (independent of the web app)

reports/                   BEFORE_AFTER.md + all metrics from the model rebuild
plots/                     feature importance, residuals, before/after chart
```

## If you want to talk about this build in an interview

Beyond the model story (see `reports/BEFORE_AFTER.md`), this integration
step itself is worth mentioning: taking an existing frontend/backend that
depended on a file that didn't exist and a model that couldn't be called
correctly, and getting it running end-to-end without changing the UI code
at all — by building an explicit adapter layer (`model_bridge.py`,
`prepare_app_data.py`) rather than rewriting the frontend to match the
model. That's a realistic "someone else's half-finished project" scenario.
