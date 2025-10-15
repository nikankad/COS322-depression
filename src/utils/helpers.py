from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, roc_curve, roc_auc_score

#students_dfsummarize model metric in string format and saves sample_submission_{modelname_roc_mse_r^2}.csv
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
    f1_score, roc_auc_score 
    )  
        # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)


    # Print metrics
    print("\n📊 SVC Model Performance")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
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

def generate_classification_charts(y_pred, y_test, y_pred_prob):
        # Confusion Matrix
        cnf_matrix = sk_confusion_matrix(y_test, y_pred)
        class_names = [0, 1]  # name of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion Matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()
