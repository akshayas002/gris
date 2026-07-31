# Before / After — full writeup

## 1. What "before" actually reproduces

The original `train_model.py` shipped a `GradientBoostingRegressor
(learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.8)` trained on
`data/listings_latest.csv`, which was not included in the delivered project —
so its exact reported numbers (R²=0.626, MAE=₹46.8 lakh **total price**,
MAPE=1,949,677%) can't be re-run bit-for-bit here.

What *can* be reproduced faithfully is the **methodology**: the same three
bugs (mixed transaction types/property types, pre-split target encoding,
no outlier trimming), applied to the same real project data
(`hyderabad_engineered.csv`) that survives in this delivery. That's what
`src/baseline_replica.py` does — same hyperparameters, same encoding order,
same evaluation formula (including the `clip(1)` MAPE bug), just predicting
**price/sqft** (this repo's actual target column) rather than total price, so
the "before" and "after" numbers in this writeup are on the same units.

Result: R²=0.465, MAE=₹8,322/sqft, MAPE=61.6%. Not as extreme as the
1.9-million-percent figure from the original run (that dataset had additional
data-entry errors this one doesn't happen to reproduce), but the same root
causes are visibly present: an inflated-then-undermined R², and a MAPE that's
still nearly meaningless because Rent and near-zero "Unknown" listings are
still in the mix dragging denominators toward zero.

## 2. What changed, mechanically

**Scope.** `TRANSACT_LABEL` grouping showed three incompatible price bases in
one column (Sale median ≈ ₹10,000/sqft, Rent median ≈ ₹6,300/sqft — actually
monthly rent per sqft, not a sale price — and "Unknown" median ≈ ₹28/sqft,
almost certainly unlabeled rentals). `PROP_TYPE_CLEAN` showed Land at ~4x the
median price/sqft of built residential property, with no shared feature
mechanism (no floors, no furnishing, no amenities). The fixed pipeline scopes
to `TRANSACT_LABEL == 'Sale'` and `PROP_TYPE_CLEAN in {Apartment, Villa,
Builder_Floor, Farmhouse}` — a defensible, single, coherent modeling target
instead of four different things averaged together.

**Leakage.** `LOCALITY_MEDIAN_PRICE`, `PRICE_PER_BED`, `LOG_PRICE_SQFT`, and
`PRICE_INR` are dropped entirely — each is a direct function of the label.
Locality (389 categories, too many for one-hot) is encoded with sklearn's
`TargetEncoder(cv=5)`, which cross-fits internally during `fit_transform`
(each training row's encoded value comes from folds that excluded that row),
then applies the train-fitted global mapping to the test set — no
train/test boundary crossing at any point.

**Outliers.** Even within the Sale+residential scope, `TARGET_PRICE_SQFT` had
a min of ₹37/sqft and a max of ₹366,666/sqft — implausible data-entry errors
at both ends. Winsorized to the [1st, 99th] percentile ([₹3,126, ₹217,813]),
dropping 94 rows (2.0%) rather than an arbitrary hand-picked cutoff.

**Model selection.** Ridge, RandomForest, and HistGradientBoosting were each
tuned with `RandomizedSearchCV` under 5-fold CV; the winner (RandomForest,
CV R²=0.697) was picked by cross-validated score, not a single lucky split.
The reported test metrics come from a 20% held-out set the search never saw.

**Metrics.** MAPE is still reported with a floor (at the 1st-percentile
training value, ₹3,500/sqft) so it stays interpretable rather than
recreating the `clip(1)` blowup — though after the scope+winsorize fixes
above, the *raw* sklearn MAPE (28.96%) and the floored version (28.93%)
now agree almost exactly, which is itself evidence the underlying data
problem (not just the formula) was the real fix.

## 3. Honest result: aggregate R² barely moves, and here's why that's fine

| | Before | After |
|---|---|---|
| R² (raw ₹/sqft) | 0.465 | 0.461 |
| MAE (₹/sqft) | 8,322 | 8,211 |
| MAPE | 61.6% | 28.9% |

Two effects cancel out in the raw-R² row: removing leakage *should* lower R²
(the model can no longer partially "cheat" via locality means computed on
overlapping data), while removing genuinely un-modelable noise (land, rent,
near-zero data errors) *should* raise it. Net: roughly flat. MAPE is the
metric that moves clearly and substantively, because it's the one most
directly wrecked by near-zero-price rows in the original scope.

Reporting a flat top-line R² honestly, with the mechanism explained, is more
credible than a report that only shows numbers going up.

## 4. Where the model is actually strong (and where it isn't)

|Segment|Test n|R²|MAE (₹/sqft)|MedAE (₹/sqft)|
|---|---|---|---|---|
|Apartment|631|0.796|913|560|
|Villa|256|0.246|25,747|10,107|
|Builder_Floor|9|-0.003|17,817|2,769|
|Farmhouse|6|0.791|13,032|10,298|

Apartments are ~70% of the scoped market and the model explains ~80% of
price/sqft variance there, with a median error of ~8% of price — a genuinely
useful working model for the bulk of the market. Villas are a small
(~28% of test rows), high-variance luxury segment where the available
features (no interior finish grade, no plot-shape/corner premium, no
builder-brand signal) plainly don't capture what drives price — this is a
data/feature-availability limitation, not something more hyperparameter
tuning would fix. Builder_Floor/Farmhouse have too few rows (9 and 6) to
draw any conclusion at all; reported for transparency, not as evidence of
anything.

## 5. Adding XGBoost as a fourth candidate

XGBoost was added to the model search (RandomizedSearchCV, same 5-fold CV,
20 iterations — matching the search budget already used for HistGB) and won
on CV R² (0.703 vs Random Forest's 0.697), so it's now the model
`train_model.py` selects and `models/model_v2.pkl` contains.

This is a genuine improvement on every aggregate metric (CV R², test R²,
MAE, RMSE) — but not a uniform one. Breaking the held-out test set down by
segment shows why:

| Segment | Random Forest R² | XGBoost R² |
|---|---|---|
| Apartment (631 rows) | 0.796 | 0.794 |
| Villa (256 rows) | 0.246 | 0.319 |

XGBoost's aggregate gain comes almost entirely from the Villa segment. On
Apartments — 70% of the scoped market — the two models are statistically
indistinguishable, and Random Forest actually has a slightly lower median
error overall (₹882/sqft vs ₹967/sqft) and puts more predictions within 5%
and 10% of the true price (30.6%/47.7% vs 25.5%/44.6%).

**Why this happened:** XGBoost's boosting objective and stronger
regularization options let it fit the long right tail (expensive villas)
more precisely without a matching loss on the bulk of the data — the
opposite failure mode of a model that overfits the tail *at the expense of*
the bulk. Whether that trade is worth it depends on what the model is for:
a business that cares about pricing high-end listings accurately should
prefer XGBoost; one that mostly transacts in the apartment segment might
reasonably prefer Random Forest's slightly tighter typical-case error
despite the lower aggregate R².

The pipeline selects by CV R² because that's the standard, defensible
criterion when no business-specific loss function has been specified — but
the segment breakdown is saved specifically so that choice can be revisited
with more context, rather than treating "the model with the higher number"
as automatically correct.

## 6. What would be next if this were a real production model

- Collect features that likely explain the Villa segment specifically
  (plot size/shape, interior finish tier, builder reputation).
- If Villas matter for the business, consider a separate model per
  property-type segment rather than one shared regressor.
- Revisit whether "Unknown"-labeled listings can be recovered as Rent with a
  rule-based fix rather than dropped outright, to grow the Rent-specific
  dataset for a separate rental-price model.
