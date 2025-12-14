# scripts/01_merge_datasets.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUT = DATA_DIR / "nyc_merged.csv"

files = [
    r"C:\Users\Shawn\Downloads\nyc_housing_dataset_200.csv",
    r"C:\Users\Shawn\Downloads\nyc_crime_dataset_aligned.csv",
    r"C:\Users\Shawn\Downloads\nyc_environment_dataset_aligned.csv",
    r"C:\Users\Shawn\Downloads\nyc_transport_dataset_aligned.csv",
    r"C:\Users\Shawn\Downloads\nyc_infrastructure_dataset_aligned.csv",
    r"C:\Users\Shawn\Downloads\nyc_social_amenities_dataset_aligned.csv",
    r"C:\Users\Shawn\Downloads\nyc_demographics_dataset_aligned.csv",
    r"C:\Users\Shawn\Downloads\nyc_accessibility_inclusivity_dataset_aligned.csv",
    r"C:\Users\Shawn\Downloads\nyc_economic_opportunities_dataset_aligned.csv"
]

# load housing 
base = pd.read_csv(DATA_DIR / files[0])
base = base.drop_duplicates(subset=["locality"]).set_index("locality")

for fname in files[1:]:
    df = pd.read_csv(DATA_DIR / fname).drop_duplicates(subset=["locality"]).set_index("locality")
   
    base = base.join(df, how="left", rsuffix=f"_{fname.split('.')[0]}")

NYCmerged = base.reset_index()

print("Rows inNYCmerged :",NYCmerged.shape[0])

DATA_DIR.mkdir(exist_ok=True)

NYCmerged.to_csv(OUT, index=False)
print("Saved:", OUT)