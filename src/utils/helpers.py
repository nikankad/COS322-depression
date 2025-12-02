# from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    log_loss, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def prepare_xy(df: pd.DataFrame):
        # """ Prepare X, y from df: drop NA, select numeric cols, handle id if present."""

        numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])

        X = numeric_df.drop(columns=["depression", "id"])
        y = numeric_df["depression"]

        return X, y


def preprocessing(df):
    # Preprocessing
    # rename columns
    df.rename(
        columns={
            "Working Professional or Student": "Working Student",
            "Have you ever had suicidal thoughts ?": "Suicidal Thoughts",
            "Family History of Mental Illness": "Family Mental Illness",
        },
        inplace=True,
    )

    # Convert all column names to snake_case
    df.columns = (
        df.columns.str.strip()  # remove leading/trailing spaces
        .str.replace(" ", "_")  # replace spaces with underscores
        .str.replace("[^A-Za-z0-9_]+", "", regex=True)  # remove special characters
        .str.lower()  # convert to lowercase
    )
    # Convert Yes/No columns to binary (1/0)
    df = df.map(lambda x: 1 if x == "Yes" else 0 if x == "No" else x)
    # Make working_student binary: 1 if working, 0 if student
    df["working_student"] = df["working_student"].map(
        {"Working Professional": 1, "Student": 0}
    )
    # Convert gender to binary: Male = 1, Female = 0
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    # remove sleep_duration occurences that have less than 10 appearences
    # Replace sleep_duration occurrences that have less than 10 appearances with the most common value
    sleep_duration_mode = df["sleep_duration"].mode()[0]
    df["sleep_duration"] = df["sleep_duration"].where(
        df["sleep_duration"].isin(
            df["sleep_duration"]
            .value_counts()[df["sleep_duration"].value_counts() >= 10]
            .index
        ),
        sleep_duration_mode,
    )

    # Replace dietary_habits occurrences that have less than 10 appearances with the most common value
    dietary_habits_mode = df["dietary_habits"].mode()[0]
    df["dietary_habits"] = df["dietary_habits"].where(
        df["dietary_habits"].isin(
            df["dietary_habits"]
            .value_counts()[df["dietary_habits"].value_counts() >= 10]
            .index
        ),
        dietary_habits_mode,
    )
    df["dietary_habits"] = df["dietary_habits"].map(
        {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
    )

    # if profession is student then make profession "Student"
    df.loc[df["working_student"] == 0, "profession"] = "Student"
    # If profession is still NaN, set to "Unemployed"
    df.loc[df["profession"].isna(), "profession"] = "Unemployed"

    # gdp_df = pd.read_csv(
    #     "/Users/nikan/Desktop/School/Sems/Spring 2025/COS 322/COS322-depression/data/ExtraData/CityGDP.csv"
    # )

    # # Normalize city names for reliable merge
    # df["city"] = df["city"].str.strip().str.lower()
    # gdp_df["city"] = gdp_df["city"].str.strip().str.lower()

    # # Merge GDP info
    # df = df.merge(gdp_df[["city", "gdp", "ppp"]], on="city", how="left")

    sleep = {
        "More than 8 hours": 9,
        "Less than 5 hours": 4,
        "5-6 hours": 5.5,
        "7-8 hours": 7.5,
        "1-2 hours": 1.5,
        "6-8 hours": 7,
        "4-6 hours": 5,
        "6-7 hours": 6.5,
        "10-11 hours": 10.5,
        "8-9 hours": 8.5,
        "9-11 hours": 10,
        "2-3 hours": 2.5,
        "3-4 hours": 3.5,
        "Moderate": 6,
        "4-5 hours": 4.5,
        "9-6 hours": 7.5,
        "1-3 hours": 2,
        "1-6 hours": 4,
        "8 hours": 8,
        "10-6 hours": 8,
        "Unhealthy": 3,
        "Work_Study_Hours": 6,
        "3-6 hours": 3.5,
        "9-5": 7,
        "9-5 hours": 7,
    }
    df["sleep_duration"] = df["sleep_duration"].map(sleep)
    df.loc[:, "sleep_duration"] = df["sleep_duration"].fillna(
        df["sleep_duration"].mode()[0]
    )

    # Apply Label Encoding to sleep_duration
    label_encoder = LabelEncoder()
    # Degree
    df["degree"] = df["degree"].astype(str).fillna("Unknown")
    df["degree"] = label_encoder.fit_transform(df["degree"])

    # df["city"] = df["city"].astype(str).fillna("Unknown")
    # df["city"] = label_encoder.fit_transform(df["city"])
    # Profession
    df["profession"] = df["profession"].astype(str).fillna("Unknown")
    df["profession"] = label_encoder.fit_transform(df["profession"])

    df.fillna(df.select_dtypes(include=["number"]).median(), inplace=True)

    # # combine pressure columns
    # df["pressure"] = df["academic_pressure"] + df["work_pressure"]
    df.drop(columns=["name", "city"], inplace=True)
    # # Combine satisfaction columns
    # df["satisfaction"] = df["study_satisfaction"] + df["job_satisfaction"]
    # df.drop(columns=["study_satisfaction", "study_satisfaction"], inplace=True)

    # scale

    # Identify numeric columns safely
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [
        c
        for c in numeric_cols
        if c
        not in [
            "id",
            "depression",
            # "gender",
            # "family_mental_illness",
            # "working_student",
            # "suicidal_thoughts",
        ]
    ]

    # Convert possible string numerics
    df[cols_to_scale] = df[cols_to_scale].apply(pd.to_numeric, errors="coerce")

    # Fill missing after coercion
    df[cols_to_scale] = df[cols_to_scale].fillna(df[cols_to_scale].median())

    # Scale
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df


def generate_submission(df, modelName):
    """
    Generates a CSV file with columns 'ids' and 'depression'
    saved to the directory in the environment variable RESULSTS_LOCATION.
    Expects y_pred to be a DataFrame or array-like with 'id' and 'y_pred' columns.
    """
    results_dir = os.getenv("SUBMISSION_LOCATION")
    if not results_dir:
        raise EnvironmentError("SUBMISSION_LOCATION not set in environment variables.")

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(
        results_dir, f"submission_{modelName}_{df['id'].iloc[0]}.csv"
    )

    # Handle both DataFrame and tuple inputs
    if isinstance(df, pd.DataFrame):
        df = df.rename(columns={"id": "id", "y_pred": "depression"})
    else:
        raise TypeError("y_pred must be a DataFrame with 'id' and 'y_pred' columns.")

    df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")


def report_metrics(model, threshold, X_test, y_test):
    """
    Comprehensive metrics report for binary classification.
    """
    # Get predictions
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    
    # ============= CLASSIFICATION REPORT =============
    print("="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    # ============= ADDITIONAL METRICS =============
    print("\n" + "="*60)
    print("ADDITIONAL METRICS")
    print("="*60)
    
    # Confusion Matrix values
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Advanced metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)  # Matthews Correlation Coefficient
    kappa = cohen_kappa_score(y_test, y_pred)  # Cohen's Kappa
    
    # Probability-based metrics
    logloss = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    
    # ROC and PR AUC
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    avg_precision = average_precision_score(y_test, probs)
    
    # Print metrics
    print(f"Threshold:                {threshold:.3f}")
    print(f"\nConfusion Matrix Components:")
    print(f"  True Positives (TP):    {tp}")
    print(f"  True Negatives (TN):    {tn}")
    print(f"  False Positives (FP):   {fp}")
    print(f"  False Negatives (FN):   {fn}")
    
    print(f"\nBasic Metrics:")
    print(f"  Accuracy:               {accuracy:.4f}")
    print(f"  Precision (PPV):        {precision:.4f}")
    print(f"  Recall (Sensitivity):   {recall:.4f}")
    print(f"  Specificity (TNR):      {specificity:.4f}")
    print(f"  F1-Score:               {f1:.4f}")
    
    print(f"\nAdvanced Metrics:")
    print(f"  Balanced Accuracy:      {balanced_acc:.4f}")
    print(f"  NPV (Neg Pred Value):   {npv:.4f}")
    print(f"  MCC (Matthews Corr):    {mcc:.4f}")
    print(f"  Cohen's Kappa:          {kappa:.4f}")
    
    print(f"\nProbability Metrics:")
    print(f"  ROC AUC:                {roc_auc:.4f}")
    print(f"  PR AUC (Avg Precision): {avg_precision:.4f}")
    print(f"  Log Loss:               {logloss:.4f}")
    print(f"  Brier Score:            {brier:.4f}")
    
    # Additional ratios
    print(f"\nDiagnostic Ratios:")
    ppv = precision  # Positive Predictive Value (same as precision)
    lr_plus = recall / (1 - specificity) if specificity < 1 else float('inf')
    lr_minus = (1 - recall) / specificity if specificity > 0 else float('inf')
    dor = lr_plus / lr_minus if lr_minus > 0 else float('inf')  # Diagnostic Odds Ratio
    
    print(f"  LR+ (Likelihood Ratio+): {lr_plus:.4f}")
    print(f"  LR- (Likelihood Ratio-): {lr_minus:.4f}")
    print(f"  DOR (Diagnostic Odds):   {dor:.4f}")
    
    # ============= VISUALIZATIONS =============
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. ROC Curve
    ax[0, 0].plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})', linewidth=2)
    ax[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax[0, 0].set_title("ROC Curve", fontsize=12, fontweight='bold')
    ax[0, 0].set_xlabel("False Positive Rate")
    ax[0, 0].set_ylabel("True Positive Rate")
    ax[0, 0].legend()
    ax[0, 0].grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0, 1],
                cbar_kws={'label': 'Count'})
    ax[0, 1].set_title(f"Confusion Matrix (threshold={threshold:.3f})", 
                       fontsize=12, fontweight='bold')
    ax[0, 1].set_xlabel("Predicted Label")
    ax[0, 1].set_ylabel("True Label")
    
    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
    ax[1, 0].plot(recall_vals, precision_vals, 
                  label=f'PR (AP={avg_precision:.3f})', linewidth=2)
    ax[1, 0].set_title("Precision-Recall Curve", fontsize=12, fontweight='bold')
    ax[1, 0].set_xlabel("Recall")
    ax[1, 0].set_ylabel("Precision")
    ax[1, 0].legend()
    ax[1, 0].grid(True, alpha=0.3)
    
    # 4. Probability Distribution
    ax[1, 1].hist(probs[y_test == 0], bins=30, alpha=0.6, label='Class 0 (Negative)', 
                  color='blue', edgecolor='black')
    ax[1, 1].hist(probs[y_test == 1], bins=30, alpha=0.6, label='Class 1 (Positive)', 
                  color='red', edgecolor='black')
    ax[1, 1].axvline(threshold, color='green', linestyle='--', linewidth=2, 
                     label=f'Threshold={threshold:.3f}')
    ax[1, 1].set_title("Predicted Probability Distribution", 
                       fontsize=12, fontweight='bold')
    ax[1, 1].set_xlabel("Predicted Probability")
    ax[1, 1].set_ylabel("Frequency")
    ax[1, 1].legend()
    ax[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ============= THRESHOLD ANALYSIS =============
    print("\n" + "="*60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*60)
    
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
    print("-" * 60)
    for thresh in thresholds_to_test:
        y_pred_temp = (probs >= thresh).astype(int)
        prec = precision_score(y_test, y_pred_temp, zero_division=0)
        rec = recall_score(y_test, y_pred_temp, zero_division=0)
        f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
        acc = (y_test == y_pred_temp).mean()
        print(f"{thresh:<12.2f} {prec:<12.4f} {rec:<12.4f} {f1_temp:<12.4f} {acc:<12.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'mcc': mcc,
        'kappa': kappa,
        'roc_auc': roc_auc,
        'pr_auc': avg_precision,
        'log_loss': logloss,
        'brier_score': brier,
        'confusion_matrix': cm
    }


def find_best_threshold(model, X_test, y_test):
        probs = model.predict_proba(X_test)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)

        best_idx = np.argmax(f1_scores)

        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]

        return best_threshold, best_f1, best_precision, best_recall

import time
from sklearn.model_selection import train_test_split

from utils.helpers import prepare_xy, report_metrics, find_best_threshold


def compare_models(df, models_dict, nn_model_path=None, save_path='model_comparison.csv', 
                   show_detailed_reports=False, optimize_thresholds=False):
    """
    Compare multiple models and save results to CSV.
    
    Args:
        df: DataFrame with data
        models_dict: Dictionary of {'Model Name': model_instance}
        nn_model_path: Path to saved Neural Network model (if applicable)
        save_path: Path to save CSV results
        show_detailed_reports: If True, show full report_metrics for each model
        optimize_thresholds: If True, find and use optimal threshold for each model
    
    Returns:
        DataFrame with comparison results
    """
    # Prepare data
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = []
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_name}...")
        print(f"{'='*60}")
        
        try:
            # Handle Neural Network differently (load) vs others (train)
            if model_name == 'NeuralNetwork' or 'Neural' in model_name:
                # Load pre-trained neural network
                if nn_model_path is None:
                    nn_model_path = os.environ.get("WEIGHT_FILE_LOCATION")
                
                if nn_model_path is None:
                    print(f"✗ No model path provided for {model_name}")
                    continue
                
                print(f"Loading from: {nn_model_path}")
                model.load_our_model(nn_model_path, X_test, y_test)
                training_time = 'N/A (loaded)'
                
            else:
                # Train other models
                print(f"Training {model_name}...")
                start_train = time.time()
                model.train(df)
                training_time = round(time.time() - start_train, 2)
                print(f"Training completed in {training_time}s")
            
            # Create wrapper for consistency with report_metrics
            class ModelWrapper:
                def __init__(self, model, is_neural_net):
                    self.model = model
                    self.is_neural_net = is_neural_net
                
                def predict_proba(self, X):
                    if self.is_neural_net:
                        # Neural network returns 1D array of P(class=1)
                        probs_class1 = self.model.predict_proba(X)
                        probs_class0 = 1 - probs_class1
                        return np.column_stack([probs_class0, probs_class1])
                    else:
                        # Other models use sklearn's fitted model
                        return self.model.model.predict_proba(X)
            
            is_nn = (model_name == 'NeuralNetwork' or 'Neural' in model_name)
            wrapped_model = ModelWrapper(model, is_nn)
            
            # Optimize threshold if requested
            original_threshold = model.threshold
            if optimize_thresholds:
                best_thresh, best_f1, best_prec, best_rec = find_best_threshold(
                    wrapped_model, X_test, y_test
                )
                print(f"Optimal threshold found: {best_thresh:.3f} (F1={best_f1:.4f})")
                print(f"  Original threshold: {original_threshold:.3f}")
                model.threshold = best_thresh
            
            # Measure inference time
            start_time = time.time()
            probs = wrapped_model.predict_proba(X_test)[:, 1]
            inference_time = time.time() - start_time
            
            # Use report_metrics to get all metrics
            print(f"\nGenerating metrics for {model_name}...")
            if show_detailed_reports:
                # Show full detailed report
                metrics_dict = report_metrics(wrapped_model, model.threshold, X_test, y_test)
            else:
                # Suppress output but still get metrics
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    metrics_dict = report_metrics(wrapped_model, model.threshold, X_test, y_test)
                finally:
                    sys.stdout = old_stdout
            
            # Add model name and timing info to metrics
            metrics_dict['Model'] = model_name
            metrics_dict['Training_Time_Sec'] = training_time
            metrics_dict['Inference_Time_Sec'] = round(inference_time, 4)
            metrics_dict['Threshold'] = model.threshold
            metrics_dict['Original_Threshold'] = original_threshold if optimize_thresholds else original_threshold
            
            # Extract confusion matrix values
            cm = metrics_dict['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            metrics_dict['True_Positives'] = int(tp)
            metrics_dict['True_Negatives'] = int(tn)
            metrics_dict['False_Positives'] = int(fp)
            metrics_dict['False_Negatives'] = int(fn)
            
            # Calculate NPV (not in report_metrics)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics_dict['NPV'] = npv
            
            # Rename keys to match expected format
            metrics_dict['Accuracy'] = metrics_dict.pop('accuracy')
            metrics_dict['Precision'] = metrics_dict.pop('precision')
            metrics_dict['Recall'] = metrics_dict.pop('recall')
            metrics_dict['Specificity'] = metrics_dict.pop('specificity')
            metrics_dict['F1_Score'] = metrics_dict.pop('f1')
            metrics_dict['Balanced_Accuracy'] = metrics_dict.pop('balanced_accuracy')
            metrics_dict['MCC'] = metrics_dict.pop('mcc')
            metrics_dict['Cohen_Kappa'] = metrics_dict.pop('kappa')
            metrics_dict['ROC_AUC'] = metrics_dict.pop('roc_auc')
            metrics_dict['PR_AUC'] = metrics_dict.pop('pr_auc')
            metrics_dict['Log_Loss'] = metrics_dict.pop('log_loss')
            metrics_dict['Brier_Score'] = metrics_dict.pop('brier_score')
            
            # Remove confusion_matrix from dict (already extracted values)
            del metrics_dict['confusion_matrix']
            
            results.append(metrics_dict)
            
            print(f"✓ {model_name} completed successfully")
            print(f"  ROC AUC: {metrics_dict['ROC_AUC']:.4f}")
            print(f"  Accuracy: {metrics_dict['Accuracy']:.4f}")
            print(f"  F1 Score: {metrics_dict['F1_Score']:.4f}")
            print(f"  MCC: {metrics_dict['MCC']:.4f}")
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print("\n⚠ No models were successfully evaluated!")
        return None
    
    # Reorder columns for better readability
    column_order = [
        'Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'NPV',
        'F1_Score', 'Balanced_Accuracy', 'ROC_AUC', 'PR_AUC', 
        'MCC', 'Cohen_Kappa', 'Log_Loss', 'Brier_Score',
        'True_Positives', 'True_Negatives', 'False_Positives', 'False_Negatives',
        'Threshold', 'Original_Threshold', 'Training_Time_Sec', 'Inference_Time_Sec'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df_results.columns]
    df_results = df_results[column_order]
    
    # Sort by ROC_AUC
    df_results = df_results.sort_values('ROC_AUC', ascending=False)
    
    # Save to CSV
    df_results.to_csv(save_path, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Results saved to {save_path}")
    print(f"{'='*70}")
    
    # Display summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    summary_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'MCC']
    print(df_results[summary_cols].to_string(index=False))
    
    print("\n" + "="*70)
    print("ADVANCED METRICS")
    print("="*70)
    advanced_cols = ['Model', 'Balanced_Accuracy', 'Specificity', 'PR_AUC', 'Log_Loss', 'Brier_Score']
    print(df_results[advanced_cols].to_string(index=False))
    
    print("\n" + "="*70)
    print("TRAINING & INFERENCE TIMES")
    print("="*70)
    print(df_results[['Model', 'Training_Time_Sec', 'Inference_Time_Sec']].to_string(index=False))
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX SUMMARY")
    print("="*70)
    cm_cols = ['Model', 'True_Positives', 'True_Negatives', 'False_Positives', 'False_Negatives']
    print(df_results[cm_cols].to_string(index=False))
    
    # Highlight best model
    best_model = df_results.iloc[0]
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL (by ROC AUC): {best_model['Model']}")
    print("="*70)
    print(f"  ROC AUC:           {best_model['ROC_AUC']:.4f}")
    print(f"  Accuracy:          {best_model['Accuracy']:.4f}")
    print(f"  F1 Score:          {best_model['F1_Score']:.4f}")
    print(f"  MCC:               {best_model['MCC']:.4f}")
    print(f"  Precision:         {best_model['Precision']:.4f}")
    print(f"  Recall:            {best_model['Recall']:.4f}")
    print(f"  Balanced Accuracy: {best_model['Balanced_Accuracy']:.4f}")
    print("="*70)
    
    return df_results