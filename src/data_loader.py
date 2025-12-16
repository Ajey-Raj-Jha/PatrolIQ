import pandas as pd
import os

# ----------------------------
# Paths
# ----------------------------
RAW_DATA_PATH = "data/raw/Crimes_-_2001_to_Present_20251213.csv"
PROCESSED_DIR = "data/processed"
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "crimes_500k_clean.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ----------------------------
# Configuration
# ----------------------------
SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

COLUMNS_REQUIRED = [
    "ID",
    "Case Number",
    "Date",
    "Primary Type",
    "Description",
    "Location Description",
    "Arrest",
    "Domestic",
    "Beat",
    "District",
    "Ward",
    "Community Area",
    "Latitude",
    "Longitude"
]

# ----------------------------
# Load data
# ----------------------------
print("Loading raw dataset...")
df = pd.read_csv(RAW_DATA_PATH, usecols=COLUMNS_REQUIRED)

print(f"Initial shape: {df.shape}")

# ----------------------------
# Basic cleaning
# ----------------------------
print("Dropping rows with missing Date / Latitude / Longitude...")
df = df.dropna(subset=["Date", "Latitude", "Longitude"])

print(f"Shape after dropping missing values: {df.shape}")

# Convert Date to datetime
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%m/%d/%Y %I:%M:%S %p",
    errors="coerce"
)

df = df.dropna(subset=["Date"])

# ----------------------------
# Sampling
# ----------------------------
print(f"Sampling {SAMPLE_SIZE} records...")
df_sampled = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

print(f"Sampled shape: {df_sampled.shape}")

# ----------------------------
# Save processed data
# ----------------------------
df_sampled.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned dataset saved to: {OUTPUT_PATH}")
print("STEP 1 COMPLETE")
