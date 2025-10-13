import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import joblib
import os
from config import *


def load_model_and_data(model_name):
    # loading trained model and test data
    model = joblib.load(f"{MODEL_DIR}{model_name}_model.pkl")
    data = joblib.load(f"{MODEL_DIR}processed_data.pkl")
    return model, data


def evaluate_model(model, X_test, y_test, model_name):
    # comprehensive model evaluation
    print(f"\n{'=' * 50}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'=' * 50}")

    # predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # roc-auc
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # average precision
    avg_precision = average_precision_score(y_test, y_proba)
    print(f"Average Precision Score: {avg_precision:.4f}")

    return {
        "predictions": y_pred,
        "probabilities": y_proba,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
    }


def plot_confusion_matrix(cm, model_name):
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Fire", "Fire"],
        yticklabels=["No Fire", "Fire"],
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}confusion_matrix_{model_name}.png")
    plt.close()


def plot_roc_curves(results_dict, y_test):
    # plot roc curves for all models
    plt.figure(figsize=(10, 8))

    for model_name, results in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, results["probabilities"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}roc_curves_comparison.png")
    plt.close()


def plot_precision_recall_curves(results_dict, y_test):
    # plot precision-recall curves for all models
    plt.figure(figsize=(10, 8))

    for model_name, results in results_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, results["probabilities"])
        plt.plot(
            recall,
            precision,
            label=f"{model_name} (AP = {results['average_precision']:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Model Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}precision_recall_comparison.png")
    plt.close()


def create_results_summary(results_dict):
    # create summary table for all models
    summary = pd.DataFrame(
        {
            model_name: {
                "ROC-AUC": results["roc_auc"],
                "Average Precision": results["average_precision"],
                "True Negatives": results["confusion_matrix"][0, 0],
                "False Positives": results["confusion_matrix"][0, 1],
                "False Negatives": results["confusion_matrix"][1, 0],
                "True Positives": results["confusion_matrix"][1, 1],
            }
            for model_name, results in results_dict.items()
        }
    ).T

    # calculate additional metrics
    summary["accuracy"] = (
        (summary["True Positives"] + summary["True Negatives"])
        / (summary["True Positives"]
        + summary["True Negatives"]
        + summary["False Positives"]
        + summary["False Negatives"])
    )

    summary["precision"] = summary["True Positives"] / (
        summary["True Positives"] + summary["False Positives"]
    )

    summary["recall"] = summary["True Positives"] / (
        summary["True Positives"] + summary["False Negatives"]
    )

    summary["f1_score"] = (
        2
        * summary["precision"]
        * summary["recall"]
        / (summary["precision"] + summary["recall"])
    )

    return summary


def main():
    print("Startng Model Evaluation...")
    print("=" * 50)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load test data
    data = joblib.load(f"{MODEL_DIR}processed_data.pkl")
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Test set size: {len(y_test)}")
    print(f"Positive class ratio: {y_test.mean():.3f}")

    # evaluate all trained models
    results_dict = {}
    for model_name in MODELS_TO_TRAIN:
        try:
            model, _ = load_model_and_data(model_name)
            results = evaluate_model(model, X_test, y_test, model_name)
            results_dict[model_name] = results

            # plot confusion matrix
            plot_confusion_matrix(results["confusion_matrix"], model_name)

        except FileNotFoundError:
            print(f"\nModel {model_name} not found. Skipping...")

    # comparative plots
    if len(results_dict) > 0:
        plot_roc_curves(results_dict, y_test)
        plot_precision_recall_curves(results_dict, y_test)

        # summary table
        summary = create_results_summary(results_dict)
        print("\n" + "=" * 50)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 50)
        print(summary.round(4))

        # save summary
        summary.to_csv(f"{RESULTS_DIR}model_comparison.csv")

        # identify best model
        best_model = summary["ROC-AUC"].idxmax()
        print(f"\nBest model (by ROC-AUC): {best_model}")
        print(f"ROC-AUC: {summary.loc[best_model, 'ROC-AUC']:.4f}")

    print(f"\nEvaluation complete! Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
