import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, log_loss,
    brier_score_loss, confusion_matrix
)
from utils.helpers import prepare_xy


def compare_models(df, models_dict, model_paths_dict=None, save_path='model_comparison.csv'):
    """
    Compare multiple models and save results to CSV.
    
    Args:
        df: DataFrame with data
        models_dict: Dictionary of {'Model Name': model_instance}
        model_paths_dict: Dictionary of {'Model Name': 'path/to/model/file'}
                         If None, assumes models are already trained
        save_path: Path to save CSV results
    
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
        print(f"Evaluating {model_name}...")
        print(f"{'='*60}")
        
        try:
            # Load model if path provided
            if model_paths_dict and model_name in model_paths_dict:
                model_path = model_paths_dict[model_name]
                print(f"Loading from: {model_path}")
                
                if model_name == 'NeuralNetwork' or 'Neural' in model_name:
                    model.load_our_model(model_path, X_test, y_test)
                else:
                    model.load_model(model_path)
            
            # Measure inference time
            start_time = time.time()
            
            # Get predictions
            if model_name == 'NeuralNetwork' or 'Neural' in model_name:
                y_scores = model.predict_proba(X_test)
            else:
                y_scores_full = model.predict_proba(X_test)
                # Handle different output formats
                if y_scores_full.ndim == 2 and y_scores_full.shape[1] == 2:
                    y_scores = y_scores_full[:, 1]
                else:
                    y_scores = y_scores_full.ravel()
            
            inference_time = time.time() - start_time
            
            y_pred = (y_scores >= model.threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'Specificity': specificity,
                'F1_Score': f1_score(y_test, y_pred, zero_division=0),
                'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred),
                'ROC_AUC': roc_auc_score(y_test, y_scores),
                'PR_AUC': average_precision_score(y_test, y_scores),
                'MCC': matthews_corrcoef(y_test, y_pred),
                'Cohen_Kappa': cohen_kappa_score(y_test, y_pred),
                'Log_Loss': log_loss(y_test, y_scores),
                'Brier_Score': brier_score_loss(y_test, y_scores),
                'True_Positives': int(tp),
                'True_Negatives': int(tn),
                'False_Positives': int(fp),
                'False_Negatives': int(fn),
                'Threshold': model.threshold,
                'Inference_Time_Sec': round(inference_time, 4),
            }
            
            # Get model file size if path exists
            if model_paths_dict and model_name in model_paths_dict:
                model_path = model_paths_dict[model_name]
                if os.path.exists(model_path):
                    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    metrics['Model_Size_MB'] = round(model_size_mb, 2)
                    metrics['Model_Path'] = model_path
                else:
                    metrics['Model_Size_MB'] = 'N/A'
                    metrics['Model_Path'] = model_path
            
            results.append(metrics)
            print(f"✓ {model_name} completed successfully")
            print(f"  ROC AUC: {metrics['ROC_AUC']:.4f}")
            print(f"  Accuracy: {metrics['Accuracy']:.4f}")
            
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
    
    # Sort by ROC_AUC
    df_results = df_results.sort_values('ROC_AUC', ascending=False)
    
    # Save to CSV
    df_results.to_csv(save_path, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Results saved to {save_path}")
    print(f"{'='*60}")
    
    # Display summary
    print("\nModel Comparison Summary:")
    summary_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'MCC']
    available_cols = [col for col in summary_cols if col in df_results.columns]
    print(df_results[available_cols].to_string(index=False))
    
    return df_results