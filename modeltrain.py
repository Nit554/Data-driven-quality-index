import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import re


DATA_DIR = Path("data")
MODEL_DIR = Path("backend/model")
DOCS_DIR = Path("docs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = Path(r"C:\Users\Shawn\Downloads\rqitest\data\nyc_master_features.csv")
OUT_MODEL = MODEL_DIR / "model.pkl"

print("🔹 Loading features from:", FEATURE_FILE)
df = pd.read_csv(FEATURE_FILE)
print(" Loaded:", df.shape, "rows")

geo_cols = ["locality", "borough", "neighborhood", "lat", "lon"]  # <-- ADD lat/lon here

# Confirm presence and non-null of geo_cols including lat/lon
print("\n--- GEO COLUMNS PRESENCE CHECK ---")
for col in geo_cols:
    if col not in df.columns:
        print(f"WARNING: Geographic column '{col}' is missing!")
    else:
        non_null_count = df[col].notnull().sum()
        print(f"✅ Column '{col}' present with {non_null_count} non-null values.")

# Copy original columns (_orig) but exclude them from model features if they exist
orig_cols = ["avg_rent_usd", "crime_rate_per_1000", "avg_aqi", "avg_commute_time_min"]
print("\n--- ORIGINAL FEATURE COLUMNS PRESENCE CHECK ---")
for col in orig_cols:
    if col in df.columns:
        df[f"{col}_orig"] = df[col]
        non_null_count = df[col].notnull().sum()
        print(f"✅ Original column '{col}' copied as '{col}_orig' with {non_null_count} non-null values.")
    else:
        print(f"WARNING: Original column '{col}' missing from data.")

# Clean column names to remove special chars for compatibility
df.columns = [re.sub(r'[^\w\d_]', '_', col) for col in df.columns]

# Define training feature columns excluding geo and original (_orig) columns
feature_cols = [col for col in df.columns if col not in geo_cols and not col.endswith('_orig')]
feature_cols = list(dict.fromkeys(feature_cols))  # Remove duplicates

X = df[feature_cols]

positive_features = [
    "green_cover_pct", "walkability_score", "internet_speed_mbps",
    "num_hospitals", "num_schools", "road_quality_index"
]
negative_features = [
    "crime_rate_per_1000", "avg_rent_usd", "avg_aqi", "avg_commute_time_min"
]

positive_features = [f for f in positive_features if f in X.columns]
negative_features = [f for f in negative_features if f in X.columns]

if len(positive_features) == 0 or len(negative_features) == 0:
    y = np.random.uniform(40, 80, len(X))
    print("\nWARNING: Positive or Negative features missing, using random target values.")
else:
    pos_score = X[positive_features].sum(axis=1)
    neg_score = X[negative_features].sum(axis=1)
    raw = pos_score - neg_score
    y = 100 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

df["Livability_Score"] = y
print("\nTarget variable 'Livability_Score' created.")
print(f"Target score stats -- min: {y.min():.2f}, max: {y.max():.2f}, mean: {y.mean():.2f}, std: {y.std():.2f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nRandomForest Model Trained — RMSE: {rmse:.3f}, R²: {r2:.3f}")

try:
    import lightgbm as lgb
    lgbm = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1  # suppress output
    )
    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="rmse")
    y_pred_lgbm = lgbm.predict(X_test)
    rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    print(f"LightGBM Model — RMSE: {rmse_lgbm:.3f}, R²: {r2_lgbm:.3f}")
    model = lgbm if rmse_lgbm < rmse else rf
except ImportError:
    print("LightGBM not installed — using RandomForest only.")
    model = rf

# Double check alignment
print(f"\nFeature importances length: {len(model.feature_importances_)}")
print(f"Feature_cols length: {len(feature_cols)}")

joblib.dump(model, OUT_MODEL)
joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

print("\nModel saved at:", OUT_MODEL)
print("Feature columns saved to:", MODEL_DIR / "feature_cols.pkl")

if hasattr(model, "feature_importances_"):
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[:15], y=importance.index[:15], palette="viridis")
    plt.title("Top 15 Important Features (Livability Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "feature_importance.png")
    print("Feature importance plot saved → docs/feature_importance.png")
