from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import numpy as np
import pandas as pd


#summarize model metric in string format and saves sample_submission_{modelname_roc_mse_r^2}.csv
def regression_metrics(model_name, y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    """Print model performance metrics in a clean format."""
    print(f"\n📊 Results for {model_name}:")

    print(f"Mean Absolute Error (MAE): {mae}")
    # Mean Squared Error
    print(f"Mean Squared Error (MSE): {mse}")

    # Root Mean Squared Error
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # R² Score (coefficient of determination)
    print(f"R² Score: {r2}")

def classification_metric(model_name, y_pred, y_test):
    from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
    )  
        # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Try computing ROC AUC (works only if probabilities available)
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = None
    except Exception:
        roc_auc = None

    # Print metrics
    print("\n📊 SVC Model Performance")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC  : {roc_auc:.4f}")
    else:
        print("ROC AUC  : N/A (probabilities not available)")
    print("-" * 40)



def regression_graph(y_pred, y_test):
    plt.figure(figsize=(7, 5))
    # Scatter actual vs predicted
    plt.scatter(y_test, y_pred, color='steelblue', alpha=0.6, label="Predicted vs Actual")
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect Fit")

    # plt.title(f"{type} — Predicted vs Actual")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def output_confusion_matrix(y_pred, y_test):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
