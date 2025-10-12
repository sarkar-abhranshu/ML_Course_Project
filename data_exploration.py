import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import DB_PATH, PLOTS_DIR
import os


def load_data():
    # Load data from the SQLite database
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        FIRE_YEAR,
        DISCOVERY_DATE,
        DISCOVERY_DOY,
        STAT_CAUSE_CODE,
        STAT_CAUSE_DESCR,
        FIRE_SIZE,
        FIRE_SIZE_CLASS,
        LATITUDE,
        LONGITUDE,
        STATE,
        COUNTY,
        OWNER_DESCR,
        CONT_DATE,
        CONT_DOY
    FROM Fires
    WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL AND FIRE_SIZE IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"No. of records loaded: {len(df)}")
    return df


def explore_temporal_patterns(df):
    # Analyze temporal patterns in wildfire occurences
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Fires per year
    plt.figure(figsize=(12, 6))
    fires_per_year = df["FIRE_YEAR"].value_counts().sort_index()
    plt.bar(fires_per_year.index, fires_per_year.values)
    plt.xlabel("Year")
    plt.ylabel("Number of Fires")
    plt.title("Wildfire Occurences by Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}fires_by_year.png")
    plt.close()

    # Fires by day of year (seasonality)
    plt.figure(figsize=(12, 6))
    fires_per_doy = df["DISCOVERY_DOY"].value_counts().sort_index()
    plt.plot(fires_per_doy.index, fires_per_doy.values)
    plt.xlabel("Day of Year")
    plt.ylabel("Number of Fires")
    plt.title("Wildfire Seasonality")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}fires_seasonality.png")
    plt.close()


def explore_spatial_patterns(df):
    # Analyze spatial distribution of wildfires

    # Fires by state
    plt.figure(figsize=(14, 8))
    top_states = df["STATE"].value_counts().head(15)
    plt.barh(range(len(top_states)), top_states.values)
    plt.yticks(range(len(top_states)), top_states.index)
    plt.xlabel("Number of Fires")
    plt.title("Top 15 States by Wildfire Occurences")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}fires_by_state.png")
    plt.close()

    # Spatial heatmap
    plt.figure(figsize=(14, 8))
    plt.hexbin(df["LONGITUDE"], df["LATITUDE"], gridsize=100, cmap="YlOrRd", mincnt=1)
    plt.colorbar(label="Fire Count")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Wildfire Density Map (USA)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}spatial_density.png")
    plt.close()

    print("\nSpatial Stats:")
    print(f"States with fires: {df['STATE'].nunique()}")
    print(f"Top 5 states:\n{top_states.head()}")


def explore_fire_causes(df):
    # Analyze fire causes

    plt.figure(figsize=(12, 6))
    causes = df["STAT_CAUSE_DESCR"].value_counts()
    plt.barh(range(len(causes)), causes.values)
    plt.yticks(range(len(causes)), causes.index)
    plt.xlabel("Number of Fires")
    plt.title("Wildfire Causes")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}fire_causes.png")
    plt.close()

    print("\nFire Causes:")
    print(causes)


def explore_fire_sizes(df):
    # Analyze fire size distribution

    # Fire size distribution (in log scale for normalized values)
    plt.figure(figsize=(12, 6))
    fire_sizes = df[df["FIRE_SIZE"] > 0]["FIRE_SIZE"]
    plt.hist(np.log10(fire_sizes), bins=100, edgecolor="black")
    plt.xlabel("Log10(Fire Size in Acres)")
    plt.ylabel("Frequency")
    plt.title("Fire Size Distribution")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}fire_size_distribution.png")
    plt.close()

    # Fire size classes
    plt.figure(figsize=(10, 6))
    size_classes = df["FIRE_SIZE_CLASS"].value_counts().sort_index()
    plt.bar(size_classes.index, size_classes.values)
    plt.xlabel("Fire Size Class")
    plt.ylabel("Count")
    plt.title("Fire Size Classes")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}fire_size_classes.png")
    plt.close()

    print("\nFire Size Statistics")
    print(f"Mean Size: {df['FIRE_SIZE'].mean():.2f} acres")
    print(f"Median size: {df['FIRE_SIZE'].median():.2f} acres")
    print(f"Max size: {df['FIRE_SIZE'].max():.2f} acres")
    print(f"\nSize class distribution:\n{size_classes}")


def main():
    print("Starting Data Exploration...")
    print("=" * 50)

    # Load data
    df = load_data()

    # Basic info
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())

    # Explore different aspects
    explore_temporal_patterns(df)
    explore_spatial_patterns(df)
    explore_fire_causes(df)
    explore_fire_sizes(df)

    print(f"\nPlots saved to {PLOTS_DIR}")
    print("Exploration complete!")


if __name__ == "__main__":
    main()
