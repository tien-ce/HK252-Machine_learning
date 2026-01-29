import csv
import pandas as pd
from sklearn.model_selection import train_test_split
def split_data (main_data, train_size = 0.7, validate_size = 0.1,test_size=0.2, random_state=42):
    """
    Splits the dataset into train, validation, and test sets.
    Returns: 3 lists (train_list, val_list, test_list)
    """
    # Validate sizes
    if (train_size + validate_size + test_size) != 1.0:
        raise ValueError("Train, validation, and test sizes must sum to 1.0")
    # First split into train and temp (validate + test)
    temp_size = validate_size + test_size
    train_data, temp_data = train_test_split(main_data,
                                            train_size=train_size, 
                                            random_state=random_state)
    # Then split temp into validate and test
    validate_ratio = validate_size / temp_size
    val_data, test_data = train_test_split(temp_data, 
                                           train_size=validate_ratio, 
                                           random_state=random_state)
    return train_data, val_data, test_data

def fill_missing_value(df):
    """
    Identifies and fills missing values in the DataFrame.
    - Numerical columns: Filled with median.
    - Categorical columns: Filled with mode.
    """
    df_cleaned = df.copy()
    # Replace "Unknown" values with NaN
    df_cleaned = df_cleaned.replace("Unknown", pd.NA)
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            if df_cleaned[col].dtype in ['float64', 'int64']:
                # Fill numerical missing values with median
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
            else:
                # Fill categorical missing values with the most frequent value (mode)
                mode_val = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                
    return df_cleaned