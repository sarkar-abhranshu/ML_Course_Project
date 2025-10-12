import sqlite3
import pandas as pd
import numpy as np
import joblib
from config import *


def create_historical_grid_density():
    # creating historical fire density for each grid cell from database and saving it for use in predictions
    print("Loading historical fire data...")
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        LATITUDE,
        LONGITUDE,
        FIRE_SIZE
    FROM Fires
    WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL AND LATITUDE >= 24 AND LATITUDE <= 50 AND LONGITUDE >= -125 AND LONGITUDE <= -65
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df)} historical fires")

    # Create grid cells
    df["lat_grid"] = (df["LATITUDE"] // GRID_SIZE) * GRID_SIZE
    df["lon_grid"] = (df["LONGITUDE"] // GRID_SIZE) * GRID_SIZE
    df["grid_cell"] = df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)

    # calculating fire density per grid cell
    grid_density = (
        df.groupby("grid_cell")
        .agg({"FIRE_SIZE": ["count", "sum", "mean", "max"]})
        .reset_index()
    )

    # flattening column
    grid_density.columns = [
        "grid_cell",
        "fire_count",
        "total_fire_size",
        "mean_fire_size",
        "max_fire_size",
    ]

    # storing lat/lon for easy lookup
    grid_coords = (
        df.groupby("grid_cell")
        .agg({"lat_grid": "first", "lon_grid": "first"})
        .reset_index()
    )

    grid_density = grid_density.merge(grid_coords, on="grid_cell")

    # saving for use in predicitons
    joblib.dump(grid_density, f"{MODEL_DIR}historical_grid_density.pkl")

    print(f"\nGrid statistics:")
    print(f"Total grid cells: {len(grid_density)}")
    print(
        f"Fire count per grid - Mean: {grid_density['fire_count'].mean():.2f}, "
        f"Median: {grid_density['fire_count'].median():.2f}, "
        f"Max: {grid_density['fire_count'].max():.0f}"
    )

    return grid_density


if __name__ == "__main__":
    grid_density = create_historical_grid_density()
    print("\nSample grid densities:")
    print(grid_density.head(10))
    print(f"\nSaved to {MODEL_DIR}historical_grid_density.pkl")
