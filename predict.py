import numpy as np
import pandas as pd
import joblib
from config import *

# Global cache for historical grid density data
_historical_grid_density = None

RISK_THRESHOLDS = {
    "Low": 0.2,
    "Moderate": 0.4,
    "High": 0.8,
    "Very High": 0.95,
}


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
    """
    Gets historical fire density for a specific location
    
    FIX #3: Added comprehensive error handling for missing historical data
    """
    try:
        # Create grid cell ID
        lat_grid = (latitude // GRID_SIZE) * GRID_SIZE
        lon_grid = (longitude // GRID_SIZE) * GRID_SIZE
        grid_cell = f"{lat_grid}_{lon_grid}"

        # Load historical data
        hist_data = load_historical_grid_density()

        # CHECK 1: Data exists
        if hist_data is None or len(hist_data) == 0:
            print(f"Warning: No historical data available. Using default density of 50.")
            return 50.0

        # CHECK 2: Grid cell exists
        cell_data = hist_data[hist_data["grid_cell"] == grid_cell]
        
        if len(cell_data) > 0:
            return float(cell_data["fire_count"].values[0])
        else:
            # CHECK 3: Use median as fallback
            if "fire_count" in hist_data.columns and len(hist_data) > 0:
                median_density = hist_data["fire_count"].median()
                if pd.isna(median_density):
                    return 50.0
                return float(median_density)
            else:
                return 50.0
                
    except Exception as e:
        # CATCH-ALL: Any unexpected error
        print(f"Error getting grid fire density for ({latitude}, {longitude}): {e}")
        print(f"Falling back to default density of 50.")
        return 50.0


def validate_input(latitude, longitude, year, month):
    """
    Validate input parameters for prediction

    Parameters:
    -----------
    latitude : float
        Must be between 24 and 50 (continental US)
    longitude : float
        Must be between -125 and -65 (continental US)
    year : int
        Year for prediction
    month : int
        Month must be between 1 and 12

    Returns:
    --------
    bool : True if all inputs are valid

    Raises:
    -------
    ValueError : If any input is out of valid range
    """
    if not (24 <= latitude <= 50):
        raise ValueError(f"Latitude {latitude} out of range [24, 50]")
    if not (-125 <= longitude <= -65):
        raise ValueError(f"Longitude {longitude} out of range [-125, -65]")
    if not (1 <= month <= 12):
        raise ValueError(f"Month {month} out of range [1, 12]")
    if not isinstance(year, (int, np.integer)) or year < 1900 or year > 2100:
        raise ValueError(f"Year {year} out of valid range")

    return True


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

    Raises:
    -------
    ValueError : If inputs are out of valid ranges
    """
    # validate inputs
    validate_input(latitude, longitude, year, month)

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

    # adding engineered features that match preprocessing
    if "doy_sin" in feature_cols or "doy_cos" in feature_cols:
        # approximating day of year based on month
        approx_doy = month * 30.44 - 15
        if approx_doy > 365:
            approx_doy -= 365

        if "doy_sin" in feature_cols:
            features["doy_sin"] = np.sin(2 * np.pi * approx_doy / 365)
        if "doy_cos" in feature_cols:
            features["doy_cos"] = np.cos(2 * np.pi * approx_doy / 365)

        # Add season encoding if present
        if "season_encoded" in feature_cols:
            season_map = {
                1: 0,
                2: 1,
                3: 2,
                4: 2,
                5: 2,
                6: 2,
                7: 2,
                8: 2,
                9: 1,
                10: 1,
                11: 1,
                12: 0,
            }
            features["season_encoded"] = season_map.get(month, -1)

        # Add fires_in_region_recent if present (default to 0 for single predictions)
        if "fires_in_region_recent" in feature_cols:
            features["fires_in_region_recent"] = 0

    if state and state_encoder and "state_encoded" in feature_cols:
        try:
            features["state_encoded"] = state_encoder.transform([state])[0]
        except ValueError:
            print(
                f"Warning: State '{state}' not found in training data. Using unknown state."
            )
            # using the encoded value for "Unknown" if available
            try:
                features["state_encoded"] = state_encoder.transform(["Unknown"])[0]
            except:
                # using mean of all encoded values
                features["state_encoded"] = state_encoder.transform(
                    state_encoder.categories_[0]
                )[0]

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
    """
    Convert probability to risk level category

    Parameters:
    -----------
    probability : float
        Fire occurrence probability (0-1)

    Returns:
    --------
    str : Risk level category
    """
    if probability < RISK_THRESHOLDS["Low"]:
        return "Low"
    elif probability < RISK_THRESHOLDS["Moderate"]:
        return "Moderate"
    elif probability < RISK_THRESHOLDS["High"]:
        return "High"
    elif probability < RISK_THRESHOLDS["Very High"]:
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
    # validating inputs
    validate_input(lat_min, lon_min, year, month)
    validate_input(lat_max, lon_max, year, month)

    if lat_max <= lat_min:
        raise ValueError(
            f"lat_max ({lat_max}) must be greater than lat_min ({lat_min})"
        )
    if lon_max <= lon_min:
        raise ValueError(
            f"lon_max ({lon_max}) must be greater than lon_min ({lon_min})"
        )

    # create grid
    lats = np.arange(lat_min, lat_max, GRID_SIZE)
    lons = np.arange(lon_min, lon_max, GRID_SIZE)

    results = []

    print(f"Predicting for {len(lats) * len(lons)} grid cells...")

    for lat in lats:
        for lon in lons:
            try:
                result = predict_fire_likelihood(lat, lon, year, month, state)
                results.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "probability": result["prediction"]["probability"],
                        "risk_level": result["prediction"]["risk_level"],
                    }
                )
            except Exception as e:
                print(f"Error predicting for ({lat}, {lon}): {e}")
                continue

    return pd.DataFrame(results)


def main():
    # example usage
    print("Wildfire Prediction System")
    print("=" * 50)

    # example 1, single location prediction
    print("\nExample 1: Single Location Prediction")
    print("-" * 50)

    try:
        result = predict_fire_likelihood(
            latitude=40.0, longitude=-120.0, year=2024, month=7, state="CA"
        )

        print(f"Location: {result['location']}")
        print(f"Time: {result['time']}")
        print(f"Fire Likely: {result['prediction']['fire_likely']}")
        print(f"Probability: {result['prediction']['probability']:.2f}%")
        print(f"Risk Level: {result['prediction']['risk_level']}")
    except Exception as e:
        print(f"Error in single prediction: {e}")

    # example 2, region grid prediction
    print("\n\nExample 2: Region Grid Prediction")
    print("-" * 50)

    try:
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

        # counting cells by risk level
        risk_counts = region_results["risk_level"].value_counts()
        print(f"\nRisk level distribution:")
        print(risk_counts)

        high_risk = len(
            region_results[
                region_results["risk_level"].isin(["High", "Very High", "Extreme"])
            ]
        )
        print(f"\nHigh/Very High/Extreme risk cells: {high_risk}")

        # saving results
        region_results.to_csv(f"{RESULTS_DIR}region_predictions.csv", index=False)
        print(f"\nResults saved to {RESULTS_DIR}region_predictions.csv")
    except Exception as e:
        print(f"Error in region prediction: {e}")


if __name__ == "__main__":
    main()
