import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import re

DATA_DIR = Path("data")
MODEL_DIR = Path("backend/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = DATA_DIR / r"C:\Users\Shawn\Downloads\geocoded_by_geoapify-9_11_2025, 3_26_07 am.csv"
OUT_FILE = DATA_DIR / "nyc_master_features.csv"

REQUIRED_USER_FEATURES = [
    "avg_rent_usd", "crime_rate_per_1000", "avg_aqi", "green_cover_pct",
    "avg_commute_time_min", "walkability_score", "internet_speed_mbps", "road_quality_index",
    "num_hospitals", "num_schools", "avg_income_growth_rate_pct", "avg_business_density_per_100_residents"
]

print(" Loading dataset:", MASTER_FILE)
df = pd.read_csv(MASTER_FILE)
print(" Loaded:", df.shape, "rows")

df.columns = df.columns.str.strip()

rename_map = {
    "original_locality": "locality",
    "original_borough": "borough",
    "original_neighborhood": "neighborhood",
    "original_avg_rent_usd": "avg_rent_usd",
    "original_crime_rate_per_1000": "crime_rate_per_1000",
    "original_avg_aqi": "avg_aqi",
    "original_green_cover_pct": "green_cover_pct",
    "original_avg_commute_time_min": "avg_commute_time_min",
    "original_walkability_score": "walkability_score",
    "original_internet_speed_mbps": "internet_speed_mbps",
    "original_road_quality_index": "road_quality_index",
    "original_num_hospitals": "num_hospitals",
    "original_num_schools": "num_schools",
    "original_avg_income_growth_rate_pct": "avg_income_growth_rate_pct",
    "original_avg_business_density_per_100_residents": "avg_business_density_per_100_residents"
}
df = df.rename(columns=rename_map)

# ----------- KEY SECTION: Create _orig columns and KEEP them SEPARATE ----------- #
df["avg_rent_usd_orig"] = df["avg_rent_usd"]
df["crime_rate_per_1000_orig"] = df["crime_rate_per_1000"]
df["avg_aqi_orig"] = df["avg_aqi"]
df["avg_commute_time_min_orig"] = df["avg_commute_time_min"]
# ------------------------------------------------------------------------------- #

df = df.drop_duplicates(subset=["locality"]).reset_index(drop=True)

geo_cols = ["locality", "borough", "neighborhood", "lat", "lon"]

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

ord_map = {
    "Low": 0, "Medium": 1, "High": 2, "Very High": 3, "Very Low": 0,
    "Unsafe": 0, "Moderate": 1, "Safe": 2,
    "Poor": 0, "Average": 1, "Good": 2, "Excellent": 3,
    "Few": 1, "Many": 2, "Very Few": 0,
    "Yes": 1, "No": 0
}

ordinal_cols = []
for col in df.columns:
    if df[col].dtype == "object" and set(df[col].unique()).intersection(ord_map.keys()):
        ordinal_cols.append(col)

print(" Ordinal columns found:", ordinal_cols)

for col in ordinal_cols:
    df[col] = df[col].map(ord_map).fillna(0)

nominal_cols = [
    c for c in df.columns
    if df[c].dtype == "object" and c not in geo_cols and c not in ordinal_cols
]

print(" Nominal columns (one-hot):", nominal_cols)
df_onehot = pd.get_dummies(df[nominal_cols], prefix=nominal_cols, drop_first=True)

# Here, define the numeric_cols to EXCLUDE any *_orig columns!
numeric_cols = [
    col for col in df.select_dtypes(include=[np.number]).columns if not col.endswith("_orig")
]

# ------- CONSTRUCT THE FINAL DATAFRAME (Geo, *_orig, Normalized, Onehot) -------
df_final = pd.concat(
    [
        df[geo_cols + [
            "avg_rent_usd_orig", "crime_rate_per_1000_orig", "avg_aqi_orig", "avg_commute_time_min_orig"
        ]].reset_index(drop=True),
        df[numeric_cols].reset_index(drop=True),  # Only processed features, not *_orig
        df_onehot.reset_index(drop=True)
    ],
    axis=1
)
# -------------------------------------------------------------------------------

df_final = df_final.loc[:, ~df_final.columns.duplicated()]
df_final = df_final.fillna(0)

df_final.columns = [re.sub(r'[^\w\d_]', '_', col) for col in df_final.columns]

print("\n--- USER SCORING COLUMN CHECKS ---")
for feat in REQUIRED_USER_FEATURES:
    if feat not in df_final.columns:
        print(f"WARNING: Column '{feat}' is missing after feature engineering!")
    else:
        vals = df_final[feat]
        if np.all(vals == vals.iloc[0]):
            print(f"WARNING: Column '{feat}' is constant (value={vals.iloc[0]}) and won't impact user score.")
        else:
            print(f"✅ Column '{feat}': min={vals.min()}, max={vals.max()}, std={vals.std()}")

# Scale all numerics except geo, *orig, and one-hot
scaler = StandardScaler()
numeric_features = [col for col in df_final.columns
                    if col not in geo_cols
                    and not col.endswith("_orig")
                    and col not in df_onehot.columns]

df_final[numeric_features] = scaler.fit_transform(df_final[numeric_features])

frontend_cols = [
    "avg_rent_usd_orig",
    "crime_rate_per_1000_orig",
    "avg_aqi_orig",
    "avg_commute_time_min_orig",
    "lat",
    "lon"
]

print("\n--- FRONTEND DATA PRESENCE CHECKS ---")
for col in frontend_cols:
    if col not in df_final.columns and col not in df.columns:
        print(f"WARNING: Frontend column '{col}' is missing from dataset!")
    else:
        col_data = df_final[col] if col in df_final.columns else df[col]
        if col_data.isnull().all():
            print(f"WARNING: Frontend column '{col}' contains only null values!")
        else:
            print(f"✅ Frontend column '{col}' present with {col_data.notnull().sum()} non-null values.")

df_final.to_csv(OUT_FILE, index=False)
joblib.dump(scaler, MODEL_DIR / "numeric_scaler.pkl")
joblib.dump(numeric_features, MODEL_DIR / "feature_cols.pkl")

print(f"\n Saved preprocessed data → {OUT_FILE}")
print(f" Total features: {len(numeric_features)}")
