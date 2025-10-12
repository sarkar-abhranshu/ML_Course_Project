import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import *
import joblib
import os


def load_and_clean_data():
    # Load and perform intitial cleaning
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        FIRE_YEAR,
        DISCOVERY_DOY,
        STAT_CAUSE_CODE,
        STAT_CAUSE_DESCR,
        FIRE_SIZE,
        FIRE_SIZE_CLASS,
        LATITUDE,
        LONGITUDE,
        STATE,
        COUNTY,
        OWNER_DESCR
    FROM Fires
    WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL AND FIRE_SIZE IS NOT NULL AND STAT_CAUSE_CODE IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Remove outliers
    df = df[(df["LATITUDE"] >= 24) & (df["LATITUDE"] <= 50)]
    df = df[(df["LONGITUDE"] >= -125) & (df["LONGITUDE"] <= -65)]
    df = df[df["FIRE_SIZE"] >= 0]

    print(f"Loaded and cleaned: {len(df)} records")
    return df


def create_spatial_features(df):
    # Create spatial features
    print("Creating spatial features...")

    # Create spatial grid cells
    df["lat_grid"] = (df["LATITUDE"] // GRID_SIZE) * GRID_SIZE
    df["lon_grid"] = (df["LONGITUDE"] // GRID_SIZE) * GRID_SIZE
    df["grid_cell"] = df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)

    # Historical fire density in the grid cell
    grid_fire_counts = df.groupby("grid_cell").size()
    df["grid_fire_density"] = df["grid_cell"].map(grid_fire_counts)

    # Distance from coast (simplified)
    coastal_lon_west = -125
    coastal_lon_east = -75
    df["dist_to_west_coast"] = abs(df["LONGITUDE"] - coastal_lon_west)
    df["dist_to_east_coast"] = abs(df["LONGITUDE"] - coastal_lon_east)
    df["min_coast_distance"] = df[["dist_to_west_coast", "dist_to_east_coast"]].min(
        axis=1
    )

    return df


def create_temporal_features(df):
    # Create temporal features
    print("Creating temporal features...")

    # Cyclical encoding of day of year
    df["doy_sin"] = np.sin(2 * np.pi * df["DISCOVERY_DOY"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["DISCOVERY_DOY"] / 365)

    # Season
    df["season"] = np.select(
        condlist=[
            (df["DISCOVERY_DOY"] < 80) | (df["DISCOVERY_DOY"] >= 356),  # Winter
            (df["DISCOVERY_DOY"] >= 80) & (df["DISCOVERY_DOY"] < 172),  # Spring
            (df["DISCOVERY_DOY"] >= 172) & (df["DISCOVERY_DOY"] < 264),  # Summer
            (df["DISCOVERY_DOY"] >= 264) & (df["DISCOVERY_DOY"] < 356),  # Fall
        ],
        choicelist=["Winter", "Spring", "Summer", "Fall"],
        default="Unknown",
    )

    # Fire season flag (typically may-october in most regions)
    df["is_fire_season"] = (
        (df["DISCOVERY_DOY"] >= 120) & (df["DISCOVERY_DOY"] <= 300)
    ).astype(int)

    # Historical fires in the same time window
    df = df.sort_values(["grid_cell", "FIRE_YEAR", "DISCOVERY_DOY"])
    df["fires_in_region_recent"] = df.groupby(["grid_cell", "FIRE_YEAR"]).cumcount()

    return df


def create_target_variable(df):
    # Create target variable for prediction
    print("Creating target variable...")

    # Binary classification: Will there be a significant fire?
    df["fire_occurred"] = (df["FIRE_SIZE"] >= FIRE_SIZE_THRESHOLD).astype(int)

    # Multi-class: Fire size category
    df["fire_risk_level"] = pd.cut(
        df["FIRE_SIZE"],
        bins=[-0.1, 0.25, 10, 100, np.inf],
        labels=["None/Small", "Medium", "Large", "Very Large"],
    )

    return df


def aggregate_to_grid_level(df):
    # Aggregate to grid-level predictions, Each row represents a grid cell in a specific year and time period, Target: likelihood of fire occurence
    print("Aggregating to grid level...")

    # monthly time bins
    df["month"] = ((df["DISCOVERY_DOY"] - 1) // 30) + 1

    # Group by grid cell, year and month
    grid_features = (
        df.groupby(["grid_cell", "lat_grid", "lon_grid", "FIRE_YEAR", "month"])
        .agg(
            {
                "fire_occurred": "max",  # Did any fire occur?
                "FIRE_SIZE": ["sum", "max", "mean", "count"],
                "grid_fire_density": "first",
                "min_coast_distance": "first",
                "is_fire_season": "first",
                "STATE": lambda x: x.mode()[0] if len(x) > 0 else None,
            }
        )
        .reset_index()
    )

    # Flatten column names
    grid_features.columns = [
        "_".join(col).strip("_") for col in grid_features.columns.values
    ]

    # Rename for clarity
    grid_features.rename(
        columns={
            "fire_occurred_max": "target_fire_occurred",
            "FIRE_SIZE_sum": "total_fire_size",
            "FIRE_SIZE_max": "max_fire_size",
            "FIRE_SIZE_mean": "mean_fire_size",
            "FIRE_SIZE_count": "num_fires",
            "STATE_<lambda>": "state",
            "grid_fire_density_first": "grid_fire_density",
            "min_coast_distance_first": "min_coast_distance",
            "is_fire_season_first": "is_fire_season",
        },
        inplace=True,
    )

    return grid_features


def prepare_features_and_target(df):
    # Prepare final feature matrix and target
    print("Preparing features and target...")

    # Select features for modeling
    feature_cols = [
        "lat_grid",
        "lon_grid",
        "FIRE_YEAR",
        "month",
        "grid_fire_density",
        "min_coast_distance",
        "is_fire_season",
    ]

    # Handling categorical variables
    if "state" in df.columns:
        # Encode state
        le = LabelEncoder()
        df["state_encoded"] = le.fit_transform(df["state"].fillna("Unknown"))
        feature_cols.append("state_encoded")

        # Save encoder
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(le, f"{MODEL_DIR}state_encoder.pkl")

    X = df[feature_cols].copy()
    y = df["target_fire_occurred"].copy()

    # Handling any remaining missing values
    X = X.fillna(X.mean())

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Positive class ratio: {y.mean():.3f}")

    return X, y, feature_cols


def split_and_scale_data(X, y):
    # Splitting data and scaling features
    print("Splitting and scaling data...")

    # Split: train, validation, test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, f"{MODEL_DIR}scaler.pkl")

    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    return (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)


def main():
    print("Starting Data Preprocessing...")
    print("=" * 50)

    # Load and clean
    df = load_and_clean_data()

    # Feature engineering
    df = create_spatial_features(df)
    df = create_temporal_features(df)
    df = create_target_variable(df)

    # Aggregate to grid level, best for regional prediction
    df_grid = aggregate_to_grid_level(df)
    X, y, feature_cols = prepare_features_and_target(df_grid)

    # Split and scale
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale_data(X, y)

    # Save processed data
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_cols": feature_cols,
        },
        f"{MODEL_DIR}processed_data.pkl",
    )

    print(f"\nProcessed data saved to {MODEL_DIR}")
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
