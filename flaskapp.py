from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Robust CORS

DATA_PATH = Path("data/nyc_master_features.csv")
MODEL_PATH = Path("backend/model/model.pkl")
FEATURES_PATH = Path("backend/model/feature_cols.pkl")

# Load data and model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)
feature_cols = [col for col in feature_cols if col in df.columns and not col.endswith('_orig')]

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

feature_groups = {
    "affordability": ["avg_rent_usd"],
    "safety": ["crime_rate_per_1000"],
    "environment": ["avg_aqi", "green_cover_pct"],
    "connectivity": ["avg_commute_time_min", "walkability_score"],
    "infrastructure": ["internet_speed_mbps", "road_quality_index"],
    "healthcare": ["num_hospitals"],
    "education": ["num_schools"],
    "economy": ["avg_income_growth_rate_pct", "avg_business_density_per_100_residents"]
}

@app.route("/recommend_custom", methods=["POST"])
def recommend_custom():
    data = request.get_json()
    weights_raw = data.get("weights", {})
    weights = {k.lower(): float(v) for k, v in weights_raw.items() if float(v) > 0}
    valid_keys = set(feature_groups.keys())
    weights = {k: v for k, v in weights.items() if k in valid_keys}
    total_w = sum(weights.values()) or 1
    weights = {k: v / total_w for k, v in weights.items()}

    budget = data.get("budget", None)
    max_commute = data.get("max_commute_min", None)
    df_filtered = df.copy()

    # Filtering on original rent and commute columns, if available
    if budget and "avg_rent_usd_orig" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["avg_rent_usd_orig"] <= budget]
    if max_commute and "avg_commute_time_min_orig" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["avg_commute_time_min_orig"] <= max_commute]

    used_feature_cols = [col for col in feature_cols if col in df_filtered.columns]
    X = df_filtered[used_feature_cols]
    base_score = model.predict(X)
    df_filtered["ml_score"] = normalize(pd.Series(base_score)) * 100

    user_score = np.zeros(len(df_filtered), dtype=float)
    for pref, w in weights.items():
        for col in feature_groups[pref]:
            if col in df_filtered.columns:
                series = df_filtered[col].fillna(df_filtered[col].median())
                if any(k in col for k in ["rate", "rent", "time", "aqi"]):
                    series = 1 - normalize(series)
                else:
                    series = normalize(series)
                user_score += w * series

    df_filtered["user_score"] = normalize(pd.Series(user_score)) * 100
    alpha = 0.7
    df_filtered["final_score"] = alpha * df_filtered["ml_score"] + (1 - alpha) * df_filtered["user_score"]

    if "lat" not in df_filtered.columns:
        df_filtered["lat"] = np.random.uniform(40.6, 40.9, len(df_filtered))
    if "lon" not in df_filtered.columns:
        df_filtered["lon"] = np.random.uniform(-74.1, -73.8, len(df_filtered))

    columns_for_output = [
        "locality", "borough", "final_score", "ml_score", "user_score",
        "avg_rent_usd", "avg_rent_usd_orig",
        "avg_aqi", "avg_aqi_orig",
        "crime_rate_per_1000", "crime_rate_per_1000_orig",
        "num_hospitals", "num_schools", "lat", "lon"
    ]

    output_cols = [c for c in columns_for_output if c in df_filtered.columns]
    top = df_filtered.sort_values("final_score", ascending=False).head(10)
    result = top[output_cols].to_dict(orient="records")

    return jsonify({"recommendations": result})

@app.route("/compare_localities", methods=["POST"])
def compare_localities():
    data = request.get_json()
    locs = data.get("localities", [])
    subset = df[df["locality"].isin(locs)]
    return jsonify(subset.to_dict(orient="records"))

@app.route("/explain", methods=["GET"])
def explain():
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        top = importance.sort_values(ascending=False).head(10)
        return jsonify(top.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_dict(orient="records"))
    return jsonify({"error": "Model does not support feature importance"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
