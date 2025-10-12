# config settings for the project

# Database
DB_PATH = "FPA_FOD_20170508.sqlite"

# Data params
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# Feature engineering
GRID_SIZE = 0.1  # spatial grid degrees (approx 11km)
TIME_WINDOW_DAYS = 90  # for temporal features

# Model params
MODELS_TO_TRAIN = ["logistic", "random_forest", "xgboost"]

# Thresholds
FIRE_SIZE_THRESHOLD = 0.25  # acres - fires above this are 'significant'
LARGE_FIRE_THRESHOLD = 100.0  # acres - large fires

# Output paths
MODEL_DIR = "models/"
PLOTS_DIR = "plots/"
RESULTS_DIR = "results/"
