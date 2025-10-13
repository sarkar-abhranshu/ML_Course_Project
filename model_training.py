import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os
from config import *


def load_processed_data():
    # load preprocessed data
    data = joblib.load(f"{MODEL_DIR}processed_data.pkl")
    return (
        data["X_train"],
        data["X_val"],
        data["X_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"],
        data["feature_cols"],
    )


def handle_class_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels

    Returns:
    --------
    tuple : (X_resampled, y_resampled) - balanced training data
    """
    print("\nHandling class imbalance using SMOTE...")
    print(f"Original distribution: {np.bincount(y_train)}")

    sampler = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print(f"Resampled distribution: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train logistic regression model

    Parameters:
    -----------
    X_train : array-like
        Training features (should be resampled)
    y_train : array-like
        Training labels (should be resampled)
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels

    Returns:
    --------
    model : Trained LogisticRegression model
    """
    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # validation performance
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    print("Validation Performance:")
    print(classification_report(y_val, y_val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

    return model


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train random forest model

    Parameters:
    -----------
    X_train : array-like
        Training features (should be resampled)
    y_train : array-like
        Training labels (should be resampled)
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels

    Returns:
    --------
    model : Trained RandomForestClassifier model
    """
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # validation performance
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    print("Validation Performance:")
    print(classification_report(y_val, y_val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

    # important features
    print("\nTop 10 Important Features:")
    feature_importance = pd.DataFrame(
        {
            "feature": joblib.load(f"{MODEL_DIR}processed_data.pkl")["feature_cols"],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print(feature_importance.head(10))

    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model

    Parameters:
    -----------
    X_train : array-like
        Training features (should be resampled and balanced)
    y_train : array-like
        Training labels (should be resampled and balanced)
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels

    Returns:
    --------
    model : Trained XGBClassifier model
    """
    print("\nTraining XGBoost...")

    # calculating scale_pos_weight from RESAMPLED (balanced) data
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)

    if num_pos > 0:
        scale_pos_weight = num_neg / num_pos
    else:
        scale_pos_weight = 1.0

    print(f"Class distribution in training data:")
    print(f"  - Negative samples: {num_neg}")
    print(f"  - Positive samples: {num_pos}")
    print(f"  - Class ratio (neg/pos): {scale_pos_weight:.4f}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # validation performance
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    print("Validation Performance:")
    print(classification_report(y_val, y_val_pred))
    print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

    return model


def main():
    print("Starting Model Training...")
    print("=" * 50)

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = load_processed_data()

    print(f"\nDataset Information:")
    print(f"Features being used: {feature_cols}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

    # models to train
    models = {}

    if "logistic" in MODELS_TO_TRAIN:
        models["logistic"] = train_logistic_regression(
            X_train_resampled, y_train_resampled, X_val, y_val
        )

    if "random_forest" in MODELS_TO_TRAIN:
        models["random_forest"] = train_random_forest(
            X_train_resampled, y_train_resampled, X_val, y_val
        )

    if "xgboost" in MODELS_TO_TRAIN:
        models["xgboost"] = train_xgboost(
            X_train_resampled, y_train_resampled, X_val, y_val
        )

    # save all models
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"{MODEL_DIR}{name}_model.pkl")
        print(f"\nSaved {name} model")

    print("\nAll models trained and saved!")


if __name__ == "__main__":
    main()
