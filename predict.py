import numpy as np
import pandas as pd
import joblib
from config import *

# Global cache for historical grid density data
_historical_grid_density = None


def load_best_model():
    # load best performing model
    model_name = "random_forest"

    model = joblib.load(f"{MODEL_DIR}{model_name}_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}scaler.pkl")

    try:
        state_encoder = joblib.load(f"{MODEL_DIR}state_encoder.pkl")
    except:
        state_encoder = None

    feature_cols = joblib.load(f"{MODEL_DIR}processed_data.pkl")["feature_cols"]

    return model, scaler, state_encoder, feature_cols


def load_historical_grid_density():
    # Loading historical grid density data (cached)
    global _historical_grid_density

    if _historical_grid_density is None:
        try:
            _historical_grid_density = joblib.load(
                f"{MODEL_DIR}historical_grid_density.pkl"
            )
            print(
                f"Loaded historical data for {len(_historical_grid_density)} grid cells"
            )
        except FileNotFoundError:
            print(
                "Warning: Historical grid density not found. Run load_historical_data.py first!"
            )
            print("Using default values for now.")
            _historical_grid_density = pd.DataFrame()

    return _historical_grid_density


def get_grid_fire_density(latitude, longitude):
    # gets historical fire density for a specific location

    # creating grid cell ID
    lat_grid = (latitude // GRID_SIZE) * GRID_SIZE
    lon_grid = (longitude // GRID_SIZE) * GRID_SIZE
    grid_cell = f"{lat_grid}_{lon_grid}"

    # loading historical data
    hist_data = load_historical_grid_density()

    if len(hist_data) == 0:
        return 50.0  # falling back to default

    # look up grid cell
    cell_data = hist_data[hist_data["grid_cell"] == grid_cell]

    if len(cell_data) > 0:
        return float(cell_data["fire_count"].values[0])
    else:
        # grid cell has no historical fires, using median of all cells
        median_density = hist_data["fire_count"].median()
        return float(median_density)


def predict_fire_likelihood(latitude, longitude, year, month, state=None):
    """
    Predict wildfire likelihood for specific location and time

    Parameters:
    -----------
    latitude : float
        Latitude of the location
    longitude : float
        Longitude of the location
    year : int
        Year for prediction
    month : int
        Month (1-12)
    state : str, optional
        State abbreviation

    Returns:
    --------
    dict : Prediction results
    """
    # load model and preprocessors
    model, scaler, state_encoder, feature_cols = load_best_model()

    # create grid cell
    lat_grid = (latitude // GRID_SIZE) * GRID_SIZE
    lon_grid = (longitude // GRID_SIZE) * GRID_SIZE

    # estimate grid fire density
    grid_fire_density = get_grid_fire_density(latitude, longitude)

    # calculate distance to coast
    coastal_lon_west = -125
    coastal_lon_east = -75
    dist_to_west_coast = abs(longitude - coastal_lon_west)
    dist_to_east_coast = abs(longitude - coastal_lon_east)
    min_coast_distance = min(dist_to_west_coast, dist_to_east_coast)

    # determine if fire season
    # assuming fire season is typically may-october (months 5-10)
    is_fire_season = 1 if 5 <= month <= 10 else 0

    # create feature vector
    features = {
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "FIRE_YEAR": year,
        "month": month,
        "grid_fire_density": grid_fire_density,
        "min_coast_distance": min_coast_distance,
        "is_fire_season": is_fire_season,
    }

    if state and state_encoder and "state_encoded" in feature_cols:
        try:
            features["state_encoded"] = state_encoder.transform([state])[0]
        except:
            features["state_encoded"] = -1  # unknown state

    # create dataframe with correct column order
    X = pd.DataFrame([features])[feature_cols]

    # scale features
    X_scaled = scaler.transform(X)

    # predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0, 1]

    # create result
    result = {
        "location": {"latitude": latitude, "longitude": longitude, "state": state},
        "time": {"year": year, "month": month, "is_fire_season": bool(is_fire_season)},
        "prediction": {
            "fire_likely": bool(prediction),
            "probability": float(probability),
            "risk_level": get_risk_level(probability),
        },
    }

    return result


def get_risk_level(probability):
    # convert probability to risk level
    if probability < 0.2:
        return "Low"
    elif probability < 0.4:
        return "Moderate"
    elif probability < 0.8:
        return "High"
    elif probability < 0.8:
        return "Very High"
    else:
        return "Extreme"


def predict_region_grid(lat_min, lat_max, lon_min, lon_max, year, month, state=None):
    """
    Predict wildfire likelihood across a grid of locations

    Parameters:
    -----------
    lat_min, lat_max : float
        Latitude range
    lon_min, lon_max : float
        Longitude range
    year : int
        Year for prediction
    month : int
        Month (1-12)
    state : str, optional
        State abbreviation

    Returns:
    --------
    pd.DataFrame : Predictions for all grid cells
    """
    # create grid
    lats = np.arange(lat_min, lat_max, GRID_SIZE)
    lons = np.arange(lon_min, lon_max, GRID_SIZE)

    results = []

    print(f"Predicting for {len(lats) * len(lons)} grid cells...")

    for lat in lats:
        for lon in lons:
            result = predict_fire_likelihood(lat, lon, year, month, state)
            results.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "probability": result["prediction"]["probability"],
                    "risk_level": result["prediction"]["risk_level"],
                }
            )

    return pd.DataFrame(results)


def main():
    # example usage
    print("Wildfire Prediction System")
    print("=" * 50)

    # example 1, single location prediction
    print("\nExample 1: Single Location Prediction")
    print("-" * 50)

    result = predict_fire_likelihood(
        latitude=40.0, longitude=-120.0, year=2024, month=7, state="CA"
    )

    print(f"Location: {result['location']}")
    print(f"Time: {result['time']}")
    print(f"Fire Likely: {result['prediction']['fire_likely']}")
    print(f"Probability: {result['prediction']['probability']:.2f}%")
    print(f"Risk Level: {result['prediction']['risk_level']}")

    # example 2, region grid prediction
    print("\n\nExample 2: Region Grid Prediction")
    print("-" * 50)

    region_results = predict_region_grid(
        lat_min=39.0,
        lat_max=41.0,
        lon_min=-121.0,
        lon_max=-119.0,
        year=2024,
        month=8,
        state="CA",
    )

    print(region_results.head(10))
    print(f"\nTotal cells predicted: {len(region_results)}")
    print(
        f"High risk cells: {len(region_results[region_results['risk_level'].isin(['High', 'Very High', 'Extreme'])])}"
    )

    # saving results
    region_results.to_csv(f"{RESULTS_DIR}region_predictions.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR}region_predictions.csv")


if __name__ == "__main__":
    main()
