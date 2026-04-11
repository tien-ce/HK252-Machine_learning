import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data (main_data, train_size = 0.8, test_size=0.2, random_state=42):
    """
    Splits the dataset into train, and test sets.
    Returns: 2 lists (train_list, test_list)
    """
    # Validate sizes
    if (train_size + test_size) != 1.0:
        raise ValueError("Train, and test sizes must sum to 1.0")
    train_data, test_data = train_test_split(main_data,
                                            train_size=train_size, 
                                            random_state=random_state)
    return train_data, test_data


# Fill missing values
def _fill_bmi_by_age(df):
    log = {}

    bins = [0, 10, 30, 60, 90, 200]
    labels = ['0-10', '10-30', '30-60', '60-90', '90+']
    
    age_groups = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    # Log distribution
    log['age_group_distribution'] = age_groups.value_counts().to_dict()
    
    group = df.groupby(age_groups, observed= False)
    
    # Log sample group
    if '10-30' in group.groups:
        log['sample_group_10_30_size'] = len(group.get_group('10-30'))
    else:
        log['sample_group_10_30_size'] = 0
    
    group_medians = group["bmi"].transform('median')
    
    # Log median (unique values per group)
    log['bmi_group_medians'] = df.groupby(age_groups, observed= False)['bmi'].median().to_dict()
    
    df['bmi'] = df['bmi'].fillna(group_medians)
    
    return df, log

# Fill categorical data by most frequent value
def _fill_categorical_data(df):
    log = {}
    
    df_cat = df.select_dtypes(include=['object'])
    
    for column in df_cat.columns:
        most_frequent = df_cat[column].mode()[0]
        
        log[column] = {
            "most_frequent": most_frequent,
            "num_missing_before": int(df[column].isna().sum())
        }
        
        df[column] = df[column].fillna(most_frequent)
    
    return df, log

# Export map (categorical to numerical) to JSON file
def _export_categorical_map(df_cat, file_path):

    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Create a dictionary to store the mapping for each categorical column
    cat_map = {}
    for column in df_cat.columns:
        # Assign numerical codes
        unique_labels = df_cat[column].unique()
        cat_map[column] = {label: idx for idx, label in enumerate(unique_labels)}
    # Write the mapping to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(cat_map, json_file, indent=4)
    
    print(f"categorical_map successfully saved to {file_path}")

# Conver categorical data to numerical data
def convert_categorical_to_numerical(df):
    df_cat = df.select_dtypes(include=['object'])
    
    cat_map = {}
    
    for column in df_cat.columns:
        unique_labels = df_cat[column].unique()
        mapping = {label: int(idx) for idx, label in enumerate(unique_labels)}
        
        cat_map[column] = mapping
        
        df[column] = df[column].astype('category').cat.codes
    
    _export_categorical_map(df_cat, 'log/categorical_map.json')
    
    return df


def fill_missing_value(main_data):
    """
    Fills missing values + return log
    """
    final_log = {}

    main_data, bmi_log = _fill_bmi_by_age(main_data)
    final_log['bmi_processing'] = bmi_log

    main_data, cat_log = _fill_categorical_data(main_data)
    final_log['categorical_fill'] = cat_log

    return main_data, final_log

def save_log(log_data, file_path="log/data_processing_log.json"):
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"data_processing_log.json successfully saved to {file_path}")