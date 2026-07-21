"""
Hyderabad Real Estate Dataset
EDA + Feature Engineering Pipeline
====================================
Target: Predict property price (PRICE_SQFT / MIN_PRICE)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json, re, ast, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. Load Data
# ─────────────────────────────────────────────
<<<<<<< HEAD
df = pd.read_csv('/home/claude/hyderabad.csv')
=======
df = pd.read_csv('./data/listings_latest.csv')
>>>>>>> 31795b2 (updated the frontend)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 1. HELPER FUNCTIONS
# ─────────────────────────────────────────────

def parse_price(x):
    """Convert '69.25 L', '2.91 Cr', '69.3 - 83.7 L' → float (₹)"""
    try:
        if pd.isna(x) or 'Request' in str(x): return np.nan
        x = str(x).replace(',','').strip()
        # Handle range → take midpoint
        if '-' in x:
            parts = re.findall(r'[\d.]+', x)
            if len(parts) >= 2:
                v = (float(parts[0]) + float(parts[1])) / 2
            else:
                v = float(parts[0])
            unit = 'Cr' if 'Cr' in x else 'L'
        else:
            parts = re.findall(r'[\d.]+', x)
            v = float(parts[0]) if parts else np.nan
            unit = 'Cr' if 'Cr' in x else 'L'
        if unit == 'Cr': return v * 1e7
        elif unit == 'L': return v * 1e5
    except: return np.nan

def parse_area(x):
    """Convert '1215 sq.ft.' or '1155-1395 sq.ft.' → float (average sqft)"""
    try:
        if pd.isna(x): return np.nan
        x = str(x).replace('sq.ft.','').strip()
        if '-' in x:
            parts = re.findall(r'[\d.]+', x)
            if len(parts) >= 2:
                return (float(parts[0]) + float(parts[1])) / 2
        parts = re.findall(r'[\d.]+', x)
        return float(parts[0]) if parts else np.nan
    except: return np.nan

def extract_locality(loc):
    """Extract locality name from location JSON string"""
    try:
        if pd.isna(loc): return None
        d = json.loads(loc.replace("'", '"'))
        return d.get('LOCALITY_NAME', None)
    except:
        try:
            m = re.search(r"LOCALITY_NAME.*?'([^']+)'", str(loc))
            return m.group(1) if m else None
        except: return None

def extract_lat_lon(map_details):
    """Extract latitude and longitude from MAP_DETAILS JSON string"""
    try:
        if pd.isna(map_details): return np.nan, np.nan
        d = json.loads(map_details.replace("'", '"'))
        return float(d.get('LATITUDE', np.nan)), float(d.get('LONGITUDE', np.nan))
    except:
        return np.nan, np.nan

def count_features(feat_str):
    """Count number of features/amenities from comma-separated string"""
    try:
        if pd.isna(feat_str) or feat_str in ['N', '', 'nan']: return 0
        return len(str(feat_str).split(','))
    except: return 0

def has_tag(tag_str, keyword):
    """Check if a tag keyword exists in the tag string"""
    try:
        if pd.isna(tag_str): return 0
        return int(keyword.upper() in str(tag_str).upper())
    except: return 0

# ─────────────────────────────────────────────
# 2. PARSE RAW COLUMNS
# ─────────────────────────────────────────────
df['PRICE_INR']   = df['PRICE'].apply(parse_price)
df['AREA_SQFT']   = df['AREA'].apply(parse_area)
df['LOCALITY']    = df['location'].apply(extract_locality)

lat_lon = df['MAP_DETAILS'].apply(extract_lat_lon)
df['LATITUDE']    = lat_lon.apply(lambda x: x[0])
df['LONGITUDE']   = lat_lon.apply(lambda x: x[1])

# ─────────────────────────────────────────────
# 3. TARGET VARIABLE
# ─────────────────────────────────────────────
# Use PRICE_SQFT as primary target (price per sq ft), fallback to computed
df['TARGET_PRICE_SQFT'] = np.where(
    (df['PRICE_SQFT'] > 0) & (df['PRICE_SQFT'] < 1e7),
    df['PRICE_SQFT'],
    np.where(
        (df['AREA_SQFT'] > 0) & (~df['PRICE_INR'].isna()),
        df['PRICE_INR'] / df['AREA_SQFT'],
        np.nan
    )
)
# Remove extreme outliers in target (>99.5th percentile)
q995 = df['TARGET_PRICE_SQFT'].quantile(0.995)
df = df[df['TARGET_PRICE_SQFT'].notna() & (df['TARGET_PRICE_SQFT'] > 0) & (df['TARGET_PRICE_SQFT'] <= q995)].copy()
df['LOG_PRICE_SQFT'] = np.log1p(df['TARGET_PRICE_SQFT'])

print(f"After cleaning target: {df.shape}")
print(f"TARGET_PRICE_SQFT range: ₹{df['TARGET_PRICE_SQFT'].min():.0f} – ₹{df['TARGET_PRICE_SQFT'].max():.0f}")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# --- 4a. Bedroom clipping (outliers > 10 are likely errors) ---
df['BEDROOM_NUM_CLEAN'] = df['BEDROOM_NUM'].clip(upper=10)

# --- 4b. Furnishing label ---
furnish_map = {0: 'Unfurnished', 1: 'Semi_Furnished', 2: 'Semi_Furnished', 4: 'Furnished'}
df['FURNISH_LABEL'] = df['FURNISH'].map(furnish_map).fillna('Unknown')

# --- 4c. Facing label ---
facing_map = {0: 'Unknown', 1: 'North', 2: 'South', 3: 'East', 4: 'West',
              5: 'NE', 6: 'NW', 7: 'SE', 8: 'SW'}
df['FACING_LABEL'] = df['FACING'].map(facing_map).fillna('Unknown')

# --- 4d. Property Age label ---
age_map = {0: 'New', 1: 'Under_Construction', 2: '0_5_yrs', 3: '5_10_yrs', 5: 'Resale_Old', 6: 'Resale_Very_Old'}
df['AGE_LABEL'] = df['AGE'].map(age_map).fillna('Unknown')

# --- 4e. Transaction type ---
df['TRANSACT_LABEL'] = df['TRANSACT_TYPE'].map({1.0: 'Sale', 2.0: 'Rent'}).fillna('Unknown')

# --- 4f. Property type simplified ---
ptype_map = {
    'Residential Apartment': 'Apartment',
    'Residential Land': 'Land',
    'Independent House/Villa': 'Villa',
    'Independent/Builder Floor': 'Builder_Floor',
    'Serviced Apartments': 'Serviced_Apt',
    'Studio Apartment': 'Studio',
    'Farm House': 'Farmhouse',
    'Other': 'Other'
}
df['PROP_TYPE_CLEAN'] = df['PROPERTY_TYPE'].map(ptype_map).fillna('Other')

# --- 4g. Feature count (number of amenity features listed) ---
df['FEATURE_COUNT'] = df['FEATURES'].apply(count_features)

# --- 4h. Amenities count ---
def count_amenities(x):
    try:
        if pd.isna(x) or str(x).strip() == 'nan': return 0
        return len(str(x).split(','))
    except: return 0
df['AMENITY_COUNT'] = df['AMENITIES'].apply(count_amenities)

# --- 4i. Landmark count (already numeric, but fill NaN) ---
df['LANDMARK_COUNT'] = df['TOTAL_LANDMARK_COUNT'].fillna(0)

# --- 4j. Tag-based binary features ---
df['IS_READY_TO_MOVE']    = df['SECONDARY_TAGS'].apply(lambda x: has_tag(x, 'READY TO MOVE'))
df['IS_UNDER_CONSTR']     = df['SECONDARY_TAGS'].apply(lambda x: has_tag(x, 'UNDER CONSTRUCTION'))
df['IS_RERA']             = df['SECONDARY_TAGS'].apply(lambda x: has_tag(x, 'RERA'))
df['IS_RESALE']           = df['SECONDARY_TAGS'].apply(lambda x: has_tag(x, 'RESALE'))
df['IS_NEW_BOOKING']      = df['SECONDARY_TAGS'].apply(lambda x: has_tag(x, 'NEW BOOKING'))

# --- 4k. Verified listing ---
df['IS_VERIFIED'] = (df['VERIFIED'] == 'Y').astype(int)

# --- 4l. Has society/building name ---
df['HAS_SOCIETY'] = df['SOCIETY_NAME'].notna().astype(int)

# --- 4m. Floor ratio (floor_num / total_floor) ---
df['FLOOR_NUM_NUM']   = pd.to_numeric(df['FLOOR_NUM'], errors='coerce')
df['TOTAL_FLOOR_NUM'] = pd.to_numeric(df['TOTAL_FLOOR'], errors='coerce')

df['FLOOR_RATIO'] = np.where(
    (df['FLOOR_NUM_NUM'].notna()) & (df['TOTAL_FLOOR_NUM'].notna()) & (df['TOTAL_FLOOR_NUM'] > 0),
    df['FLOOR_NUM_NUM'] / df['TOTAL_FLOOR_NUM'],
    np.nan
)

# --- 4n. Is top floor ---
df['IS_TOP_FLOOR'] = np.where(
    (df['FLOOR_NUM_NUM'].notna()) & (df['TOTAL_FLOOR_NUM'].notna()),
    (df['FLOOR_NUM_NUM'] == df['TOTAL_FLOOR_NUM']).astype(int),
    0
)

# --- 4o. Is ground floor ---
df['IS_GROUND_FLOOR'] = np.where(
    df['FLOOR_NUM_NUM'].notna(),
    (df['FLOOR_NUM_NUM'] == 0).astype(int),
    0
)

# --- 4p. Balcony bucket ---
df['BALCONY_BUCKET'] = pd.cut(
    df['BALCONY_NUM'].fillna(0),
    bins=[-1, 0, 1, 2, 100],
    labels=['None', '1', '2', '3+']
)

# --- 4q. Locality-level median price (target encoding proxy) ---
locality_price = df.groupby('LOCALITY')['TARGET_PRICE_SQFT'].median().rename('LOCALITY_MEDIAN_PRICE')
df = df.join(locality_price, on='LOCALITY')

# --- 4r. Listing age (days since registration) ---
def parse_days_ago(x):
    try:
        if pd.isna(x): return np.nan
        x = str(x).lower()
        if 'today' in x or 'hours' in x: return 0
        m = re.search(r'(\d+)\s*(day|week|month|year)', x)
        if not m: return np.nan
        n, unit = int(m.group(1)), m.group(2)
        return {'day':1,'week':7,'month':30,'year':365}[unit] * n
    except: return np.nan

df['LISTING_AGE_DAYS'] = df['REGISTERED_DAYS'].apply(parse_days_ago)

# --- 4s. Price per bedroom (only for apartments) ---
df['PRICE_PER_BED'] = np.where(
    (df['BEDROOM_NUM_CLEAN'] > 0) & df['PRICE_INR'].notna(),
    df['PRICE_INR'] / df['BEDROOM_NUM_CLEAN'],
    np.nan
)

# --- 4t. Area bucket ---
df['AREA_BUCKET'] = pd.cut(
    df['AREA_SQFT'],
    bins=[0, 800, 1200, 1800, 2500, 5000, np.inf],
    labels=['Micro', 'Small', 'Medium', 'Large', 'XL', 'XXL']
)

# ─────────────────────────────────────────────
# 5. FINAL ENGINEERED FEATURE LIST
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    'BEDROOM_NUM_CLEAN', 'TOTAL_FLOOR', 'FEATURE_COUNT', 'AMENITY_COUNT',
    'LANDMARK_COUNT', 'FLOOR_RATIO', 'BALCONY_NUM', 'AREA_SQFT',
    'LATITUDE', 'LONGITUDE', 'LOCALITY_MEDIAN_PRICE', 'LISTING_AGE_DAYS',
    'IS_READY_TO_MOVE', 'IS_UNDER_CONSTR', 'IS_RERA', 'IS_RESALE',
    'IS_NEW_BOOKING', 'IS_VERIFIED', 'HAS_SOCIETY', 'IS_TOP_FLOOR',
    'IS_GROUND_FLOOR'
]
CATEGORICAL_FEATURES = [
    'PROP_TYPE_CLEAN', 'FURNISH_LABEL', 'FACING_LABEL', 'AGE_LABEL',
    'TRANSACT_LABEL', 'AREA_BUCKET', 'BALCONY_BUCKET', 'LOCALITY'
]
TARGET = 'TARGET_PRICE_SQFT'

print("\n=== Engineered Feature Summary ===")
print(f"Numeric features : {len(NUMERIC_FEATURES)}")
print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
print(f"Target           : {TARGET}")

# ─────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────
palette = sns.color_palette("muted")
sns.set_style("whitegrid")

# ── Figure 1: EDA Overview ──────────────────────────────────
fig = plt.figure(figsize=(22, 24))
fig.suptitle("Hyderabad Real Estate — EDA Overview", fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Target distribution
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df['TARGET_PRICE_SQFT'], bins=60, kde=True, color=palette[0], ax=ax1)
ax1.set_title("Price/SqFt Distribution")
ax1.set_xlabel("₹ per sq.ft.")

# 2. Log-price distribution
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(df['LOG_PRICE_SQFT'], bins=60, kde=True, color=palette[1], ax=ax2)
ax2.set_title("Log(Price/SqFt) Distribution")
ax2.set_xlabel("log(₹/sq.ft.)")

# 3. Price by property type
ax3 = fig.add_subplot(gs[0, 2])
order3 = df.groupby('PROP_TYPE_CLEAN')['TARGET_PRICE_SQFT'].median().sort_values(ascending=False).index
sns.boxplot(data=df, x='PROP_TYPE_CLEAN', y='TARGET_PRICE_SQFT', order=order3,
            palette='muted', ax=ax3, showfliers=False)
ax3.set_title("Price/SqFt by Property Type")
ax3.set_xlabel("")
ax3.tick_params(axis='x', rotation=30)

# 4. Price by bedrooms
ax4 = fig.add_subplot(gs[1, 0])
bed_df = df[df['BEDROOM_NUM_CLEAN'].between(1,6)]
sns.boxplot(data=bed_df, x='BEDROOM_NUM_CLEAN', y='TARGET_PRICE_SQFT',
            palette='muted', ax=ax4, showfliers=False)
ax4.set_title("Price/SqFt by Bedrooms")
ax4.set_xlabel("Number of Bedrooms")

# 5. Price by furnishing
ax5 = fig.add_subplot(gs[1, 1])
sns.boxplot(data=df, x='FURNISH_LABEL', y='TARGET_PRICE_SQFT',
            palette='muted', ax=ax5, showfliers=False)
ax5.set_title("Price/SqFt by Furnishing")
ax5.set_xlabel("")

# 6. Price by property age
ax6 = fig.add_subplot(gs[1, 2])
age_order = ['New','Under_Construction','0_5_yrs','5_10_yrs','Resale_Old','Resale_Very_Old']
age_order = [a for a in age_order if a in df['AGE_LABEL'].unique()]
sns.boxplot(data=df, x='AGE_LABEL', y='TARGET_PRICE_SQFT', order=age_order,
            palette='muted', ax=ax6, showfliers=False)
ax6.set_title("Price/SqFt by Property Age")
ax6.set_xlabel("")
ax6.tick_params(axis='x', rotation=30)

# 7. Property type distribution
ax7 = fig.add_subplot(gs[2, 0])
vc = df['PROP_TYPE_CLEAN'].value_counts()
ax7.pie(vc.values, labels=vc.index, autopct='%1.1f%%', colors=palette[:len(vc)], startangle=90)
ax7.set_title("Property Type Distribution")

# 8. Bedroom distribution
ax8 = fig.add_subplot(gs[2, 1])
bed_vc = df['BEDROOM_NUM_CLEAN'].value_counts().sort_index().head(8)
ax8.bar(bed_vc.index.astype(str), bed_vc.values, color=palette[2])
ax8.set_title("Bedroom Count Distribution")
ax8.set_xlabel("# Bedrooms")
ax8.set_ylabel("Count")

# 9. Furnishing distribution
ax9 = fig.add_subplot(gs[2, 2])
f_vc = df['FURNISH_LABEL'].value_counts()
ax9.bar(f_vc.index, f_vc.values, color=palette[3])
ax9.set_title("Furnishing Distribution")
ax9.set_ylabel("Count")

# 10. Missing values heatmap
ax10 = fig.add_subplot(gs[3, 0:2])
miss_cols = df[NUMERIC_FEATURES].isnull().sum()
miss_cols = miss_cols[miss_cols > 0].sort_values(ascending=False)
ax10.barh(miss_cols.index, miss_cols.values / len(df) * 100, color=palette[4])
ax10.set_title("Missing Value % (Numeric Features)")
ax10.set_xlabel("% Missing")
ax10.axvline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
ax10.legend()

# 11. Correlation with target
ax11 = fig.add_subplot(gs[3, 2])
corr_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
corr = df[corr_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
colors = ['#d62728' if v < 0 else '#2ca02c' for v in corr.values]
ax11.barh(corr.index, corr.values, color=colors)
ax11.set_title("Feature Correlation with Target")
ax11.set_xlabel("Pearson r")
ax11.axvline(0, color='black', linewidth=0.8)

<<<<<<< HEAD
plt.savefig('/mnt/user-data/outputs/eda_overview.png', dpi=150, bbox_inches='tight')
=======
plt.savefig('./data/eda_overview.png', dpi=150, bbox_inches='tight')
>>>>>>> 31795b2 (updated the frontend)
plt.close()
print("Saved: eda_overview.png")

# ── Figure 2: Feature Engineering Insights ─────────────────
fig2, axes = plt.subplots(3, 3, figsize=(22, 18))
fig2.suptitle("Hyderabad Real Estate — Feature Engineering Insights", fontsize=18, fontweight='bold')

# 1. Price by RERA status
ax = axes[0][0]
sns.boxplot(data=df, x='IS_RERA', y='TARGET_PRICE_SQFT', palette=['#d62728','#2ca02c'], ax=ax, showfliers=False)
ax.set_title("Price: RERA vs Non-RERA")
ax.set_xticklabels(['Non-RERA','RERA'])

# 2. Price by Ready to Move
ax = axes[0][1]
sns.boxplot(data=df, x='IS_READY_TO_MOVE', y='TARGET_PRICE_SQFT', palette=['#ff7f0e','#1f77b4'], ax=ax, showfliers=False)
ax.set_title("Price: Ready-to-Move vs Not")
ax.set_xticklabels(['Not RTM','Ready to Move'])

# 3. Price by Facing
ax = axes[0][2]
face_order = df.groupby('FACING_LABEL')['TARGET_PRICE_SQFT'].median().sort_values(ascending=False).index
sns.barplot(data=df, x='FACING_LABEL', y='TARGET_PRICE_SQFT', order=face_order,
            palette='muted', ax=ax, errorbar=None)
ax.set_title("Median Price by Facing Direction")
ax.tick_params(axis='x', rotation=30)

# 4. Area vs Price scatter
ax = axes[1][0]
sample = df[df['AREA_SQFT'] < 10000].sample(min(1500, len(df)), random_state=42)
ax.scatter(sample['AREA_SQFT'], sample['TARGET_PRICE_SQFT'], alpha=0.3, s=10, color=palette[0])
ax.set_title("Area vs Price/SqFt")
ax.set_xlabel("Area (sq.ft.)")
ax.set_ylabel("₹/sq.ft.")

# 5. Feature count vs price
ax = axes[1][1]
feat_grp = df.groupby('FEATURE_COUNT')['TARGET_PRICE_SQFT'].median().reset_index()
ax.plot(feat_grp['FEATURE_COUNT'], feat_grp['TARGET_PRICE_SQFT'], marker='o', color=palette[1])
ax.set_title("Feature Count vs Median Price/SqFt")
ax.set_xlabel("# Features Listed")
ax.set_ylabel("Median ₹/sq.ft.")

# 6. Locality median price (top 20)
ax = axes[1][2]
loc_med = (df[df['LOCALITY'] != ':']
           .groupby('LOCALITY')['TARGET_PRICE_SQFT'].median()
           .sort_values(ascending=False).head(20))
ax.barh(loc_med.index[::-1], loc_med.values[::-1], color=palette[2])
ax.set_title("Top 20 Localities by Median Price/SqFt")
ax.set_xlabel("Median ₹/sq.ft.")

# 7. Floor ratio vs price
ax = axes[2][0]
fl_df = df[df['FLOOR_RATIO'].notna()]
ax.scatter(fl_df['FLOOR_RATIO'], fl_df['TARGET_PRICE_SQFT'], alpha=0.3, s=10, color=palette[3])
ax.set_title("Floor Ratio vs Price/SqFt")
ax.set_xlabel("Floor Ratio (floor/total)")
ax.set_ylabel("₹/sq.ft.")

# 8. Landmark count vs price
ax = axes[2][1]
lm_grp = df.groupby(pd.cut(df['LANDMARK_COUNT'], bins=[0,5,10,20,35,50], include_lowest=True))['TARGET_PRICE_SQFT'].median()
ax.bar([str(i) for i in lm_grp.index], lm_grp.values, color=palette[4])
ax.set_title("Nearby Landmarks vs Price/SqFt")
ax.set_xlabel("Landmark Count Bucket")
ax.set_ylabel("Median ₹/sq.ft.")
ax.tick_params(axis='x', rotation=20)

# 9. Price by area bucket
ax = axes[2][2]
ab_order = ['Micro','Small','Medium','Large','XL','XXL']
ab_order_present = [x for x in ab_order if x in df['AREA_BUCKET'].cat.categories]
sns.boxplot(data=df, x='AREA_BUCKET', y='TARGET_PRICE_SQFT', order=ab_order_present,
            palette='muted', ax=ax, showfliers=False)
ax.set_title("Price/SqFt by Area Bucket")
ax.set_xlabel("Area Bucket")

plt.tight_layout()
<<<<<<< HEAD
plt.savefig('/mnt/user-data/outputs/feature_engineering_insights.png', dpi=150, bbox_inches='tight')
=======
plt.savefig('./data/feature_engineering_insights.png', dpi=150, bbox_inches='tight')
>>>>>>> 31795b2 (updated the frontend)
plt.close()
print("Saved: feature_engineering_insights.png")

# ── Figure 3: Geo Map ────────────────────────────────────────
geo_df = df[df['LATITUDE'].notna() & df['LONGITUDE'].notna()
            & df['LATITUDE'].between(17.2, 17.8)
            & df['LONGITUDE'].between(78.2, 78.7)].copy()

fig3, ax = plt.subplots(figsize=(14, 10))
sc = ax.scatter(geo_df['LONGITUDE'], geo_df['LATITUDE'],
                c=geo_df['TARGET_PRICE_SQFT'], cmap='YlOrRd',
                s=20, alpha=0.6, vmin=geo_df['TARGET_PRICE_SQFT'].quantile(0.05),
                vmax=geo_df['TARGET_PRICE_SQFT'].quantile(0.95))
plt.colorbar(sc, ax=ax, label='₹/sq.ft.')
ax.set_title("Hyderabad — Property Price Heatmap (by Location)", fontsize=15, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
<<<<<<< HEAD
plt.savefig('/mnt/user-data/outputs/geo_price_map.png', dpi=150, bbox_inches='tight')
=======
plt.savefig('./data/geo_price_map.png', dpi=150, bbox_inches='tight')
>>>>>>> 31795b2 (updated the frontend)
plt.close()
print("Saved: geo_price_map.png")

# ─────────────────────────────────────────────
# 7. SAVE ENGINEERED DATASET
# ─────────────────────────────────────────────
KEEP_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
    'SPID', 'PROP_ID', 'TARGET_PRICE_SQFT', 'LOG_PRICE_SQFT',
    'PRICE_INR', 'AREA_SQFT', 'LOCALITY', 'PROP_TYPE_CLEAN',
    'PRICE_PER_BED', 'LISTING_AGE_DAYS', 'LOCALITY_MEDIAN_PRICE'
]
KEEP_COLS = [c for c in KEEP_COLS if c in df.columns]
df_out = df[KEEP_COLS].copy()
<<<<<<< HEAD
df_out.to_csv('/mnt/user-data/outputs/hyderabad_engineered.csv', index=False)
=======
df_out.to_csv('./data/hyderabad_engineered.csv', index=False)
>>>>>>> 31795b2 (updated the frontend)
print(f"\nEngineered dataset saved: {df_out.shape}")
print("\nFinal feature columns:")
for c in KEEP_COLS:
    dtype = str(df_out[c].dtypes)
    miss  = df_out[c].isna().sum()
    print(f"  {c:<35} dtype={dtype:<12} missing={miss}")
