import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from pre_processing.handle_data import split_data, fill_missing_value, convert_categorical_to_numerical, save_log
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    average_precision_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

from model.decision_tree import Decision_Tree_Model
from model.gradient_boosting import Gradient_Boosting_Model
from model.random_forest_library import Random_Forest_Model
from model.random_forest import RandomForestClassifier
from model.SVM import SVM_Model




def preprocess_data(file_path, random_state):

    # Load data
    data = pd.read_csv(file_path)
    
    # Fill missing values
    data_filled, log = fill_missing_value(data)

    # Convert categorical data to numerical data
    data_cleaned = convert_categorical_to_numerical(data_filled)

    # Split data into train, and test sets
    train_data, test_data = split_data(data_cleaned, random_state= random_state)
    save_log(log)
    
    return train_data, test_data

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

def normalize_data(X_train, X_test=None):
    """
    Normalizes data to the range [0, 1] using MinMaxScaler.
    
    Parameters:
    - X_train: The training dataset
    - X_test: The test dataset (optional)
    
    Returns:
    - If X_test is provided: (X_train_scaled, X_test_scaled)
    - If X_test is None: (X_train_scaled, scaler)
    """
    print("Normalizing data...\n")
    scaler = MinMaxScaler()
    
    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        # Transform the test data using the parameters learned from training data
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    # Return the scaled training data and the scaler object for reuse (e.g., for future inference)
    return X_train_scaled, scaler

def get_X_and_Y(dataset):
    y= dataset['stroke']
    X= dataset.drop(columns=['stroke'])
    return X, y

def apply_SMOTE(X,y, random_state):
    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X, y = smote.fit_resample(X, y)
    return X, y

# Main execution
if __name__ == "__main__":
    file_path = 'Data/healthcare-dataset-stroke-data.csv'
    random_state= 42
    train_data, test_data= preprocess_data(file_path, random_state= random_state)
    train_data, label_train= get_X_and_Y(train_data)
    test_data, label_test= get_X_and_Y(test_data)


    command = sys.argv[1]
    if command == 'Decision_Tree_Model':
        model = Decision_Tree_Model(train_data, label_train, random_state= random_state)
    elif command == 'Gradient_Boosting_Model':
        train_data, test_data= normalize_data(train_data, test_data)
        train_data, label_train= apply_SMOTE(train_data, label_train, random_state= random_state)
        model = Gradient_Boosting_Model(train_data, label_train, random_state= random_state)
    elif command == 'Random_Forest_Library_Model':
        train_data, test_data= normalize_data(train_data, test_data)
        train_data, label_train= apply_SMOTE(train_data, label_train, random_state= random_state)
        model = Random_Forest_Model(train_data, label_train, random_state= random_state)
    elif command == 'Random_Forest_Model':
        classifier = RandomForestClassifier(n_estimators = 10, feature_subset_size = 4, max_depth = 5, min_samples_split = 3, random_state= random_state)
        classifier.fit (train_data.values, label_train.values)
    else:
        train_data, test_data= normalize_data(train_data, test_data)
        train_data, label_train= apply_SMOTE(train_data, label_train, random_state= random_state)
        model = SVM_Model(train_data, label_train, random_state= random_state)


    y_pred= model.predict(test_data)
    y_prob= model.predict_proba(test_data)[:, 1]
    metrics= evaluate_metrics(label_test, y_pred, y_prob, name_model= command)
    save_metrics_to_json(metrics, f"log/{command}_results.json")