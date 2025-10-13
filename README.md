# Wildfire Risk Prediction System

A machine learning project that predicts wildfire risk across the United States using historical fire data, meteorological features, and spatial-temporal patterns. The system classifies fire risk levels and predicts the likelihood of significant wildfires in specific regions.

## Project Overview

This project implements multiple machine learning models (Logistic Regression, Random Forest, and XGBoost) to predict wildfire occurrence and severity based on:
- Historical fire patterns and spatial density
- Geographic features (latitude, longitude, elevation)
- Temporal patterns (season, month, day of year)
- Regional characteristics

The models are trained on the FPA-FOD wildfire database and use techniques like SMOTE for handling class imbalance.

## Features

- **Data Preprocessing**: Automated data cleaning, feature engineering, and grid-based spatial aggregation
- **Multiple Models**: Compare Logistic Regression, Random Forest, and XGBoost classifiers
- **Class Imbalance Handling**: SMOTE implementation for better prediction of rare events
- **Visualization**: Comprehensive plots for data exploration and model evaluation
- **Risk Assessment**: Multi-level risk classification (Low, Moderate, High, Very High)
- **Regional Predictions**: Batch prediction capabilities for multiple geographic locations

## Project Structure

```
ML_Course_Project/
├── config.py                  # Configuration parameters and settings
├── load_historical_data.py    # Load and process historical fire data
├── data_exploration.py        # Exploratory data analysis and visualization
├── data_preprocessing.py      # Feature engineering and data preparation
├── model_training.py          # Train and evaluate ML models
├── model_evaluation.py        # Model comparison and performance metrics
├── predict.py                 # Make predictions on new data
├── test.py                    # Test predictions with sample data
├── requirements.txt           # Python dependencies
├── models/                    # Saved models and preprocessed data
├── plots/                     # Generated visualizations
└── results/                   # Model outputs and predictions
```

## Prerequisites

- Python 3.8 or higher
- SQLite database: `FPA_FOD_20170508.sqlite` (Fire Program Analysis Fire-Occurrence Database)
- Virtual environment (recommended)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sarkar-abhranshu/ML_Course_Project.git
cd ML_Course_Project
```

### 2. Create and Activate Virtual Environment

**Windows:**
```cmd
python -m venv wildfire
.\wildfire\Scripts\activate
```

### 3. Install Dependencies

```cmd
pip install -r requirements.txt
```

The main dependencies include:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- imbalanced-learn
- joblib

### 4. Download the Database

Download the FPA-FOD database (`FPA_FOD_20170508.sqlite`) and place it in the project root directory. This database can be obtained from:
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires)

## Running the Project

### Complete Pipeline

Run the following scripts in order to execute the full pipeline:

#### Step 1: Load Historical Data
```cmd
python load_historical_data.py
```
This creates spatial grid cells and calculates historical fire density metrics.

#### Step 2: Explore the Data (Optional)
```cmd
python data_exploration.py
```
Generates visualizations for understanding the dataset.

#### Step 3: Preprocess Data
```cmd
python data_preprocessing.py
```
Performs feature engineering, creates train/validation/test splits, and saves processed data.

#### Step 4: Train Models
```cmd
python model_training.py
```
Trains multiple ML models with class balancing and saves the best performing models.

#### Step 5: Evaluate Models
```cmd
python model_evaluation.py
```
Compares model performance and generates evaluation metrics.

#### Step 6: Make Predictions
```python
python predict.py
```

## Configuration

Edit `config.py` to modify project parameters:

- `GRID_SIZE`: Spatial grid resolution (default: 0.1 degrees ≈ 11km)
- `FIRE_SIZE_THRESHOLD`: Minimum fire size for classification (default: 1.0 acres)
- `TEST_SIZE`: Test set proportion (default: 0.2)
- `MODELS_TO_TRAIN`: List of models to train

## Output Files

### Models Directory (`models/`)
- `*_model.pkl`: Trained model files
- `scaler.pkl`: Feature scaler
- `processed_data.pkl`: Preprocessed dataset
- `historical_grid_density.pkl`: Historical fire patterns

### Plots Directory (`plots/`)
- Data distribution visualizations
- Feature importance plots
- Model performance comparisons
- Confusion matrices

### Results Directory (`results/`)
- `model_comparison.csv`: Performance metrics for all models
- `region_predictions.csv`: Predictions for multiple locations

## Model Performance

The system trains and compares three models:
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting classifier (typically best performing)

Models are evaluated using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Class-specific performance metrics

## Key Parameters

- **Grid Size**: 0.1 degrees (approximately 11km squares)
- **Time Window**: 90 days for temporal features
- **Fire Size Threshold**: 1.0 acres (significant fires)
- **Large Fire Threshold**: 100.0 acres
- **Class Balancing**: SMOTE oversampling

## Troubleshooting

### Database Not Found
Ensure `FPA_FOD_20170508.sqlite` is in the project root directory.

### Missing Dependencies
```cmd
pip install --upgrade -r requirements.txt
```

### Module Not Found Errors
Ensure the virtual environment is activated and you're in the project directory.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is created for educational purposes as part of an ML course.

## Contact

**Author**: Abhranshu Sarkar, Akhilesh M  
**Repository**: [github.com/sarkar-abhranshu/ML_Course_Project](https://github.com/sarkar-abhranshu/ML_Course_Project)

---

**Note**: This project uses historical data for educational purposes. For real-world wildfire prediction, consult official sources like NOAA, USFS, or local fire management agencies.
