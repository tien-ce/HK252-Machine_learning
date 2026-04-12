from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    average_precision_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def evaluate_metrics(y_true, y_pred, y_prob=None, plot_cm=True, name_model= None):
    """
    Function to calculate and print classification model evaluation metrics.
    
    Parameters:
    - y_true: Array containing ground truth labels
    - y_pred: Array containing labels predicted by the model
    - y_prob: (Optional) Array containing predicted probabilities for the positive class (used for PR-AUC)
    - plot_cm: Boolean, whether to display the Confusion Matrix plot
    
    Returns:
    - dictionary containing the values of the metrics
    """
    
    # 1. Calculate basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 2. Calculate PR-AUC (Only if y_prob is provided)
    pr_auc = None
    if y_prob is not None:
        pr_auc = average_precision_score(y_true, y_prob)
        
    # 3. Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. Print evaluation report
    print("-" * 35)
    name= "MODEL"
    if name_model:
        name= name_model
    print(f"{name} EVALUATION REPORT")
    print("-" * 35)
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision        : {prec:.4f}")
    print(f"Recall           : {rec:.4f}")
    print(f"F1-score         : {f1:.4f}")
    
    if pr_auc is not None:
        print(f"PR-AUC           : {pr_auc:.4f}")
    else:
        print("PR-AUC           : Skipped (y_prob required)")
        
    print("-" * 35)
    print("Confusion Matrix (Text):")
    print(cm)
    print("-" * 35)
    
    # 5. (Optional) Plot Confusion Matrix
    if plot_cm:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title('Confusion Matrix')
        plt.show()
        
    # Return results as a dictionary for easy reuse
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-score': f1,
        'PR-AUC': pr_auc,
        'Confusion Matrix': cm
    }

def save_metrics_to_json(metrics_dict, file_path="log/metrics.json"):
    """
    Saves metrics results to a JSON file.
    
    Parameters:
    - metrics_dict: dictionary returned by evaluate_metrics
    - file_path: path to the JSON file
    """

    # Extract directory from the file path
    folder = os.path.dirname(file_path)
    
    # If a directory is specified and does not exist, create it
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Helper function to convert non-serializable data types
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if o is None:
            return None
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    
    # Write the dictionary to the JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4, default=convert)
    
    print(f"Results successfully saved to: {file_path}")
