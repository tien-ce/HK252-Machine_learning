import csv
import json
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


# Fill missing values
def _fill_bmi_by_age(df):
    """Internal helper to fill BMI missing values based on age ranges."""
    # Define the bins and labels as requested
    bins = [0, 10, 30, 60, 90, 200]
    labels = ['0-10', '10-30', '30-60', '60-90', '90+']
    
    # Create temporary grouping
    age_groups = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    print ("Age groups:", age_groups.value_counts())
    
    # Calculate group medians and fill
    group = df.groupby(age_groups)
    print("10-30 group: \n", group.get_group('10-30'))
    # Mean BMI for each age group
    group_medians = group["bmi"].transform('median')
    print ("Group medians: \n", group_medians)    
    df['bmi'] = df['bmi'].fillna(group_medians)
    return df

# Fill categorical data by most frequent value
def _fill_categorical_data(df):
    # Select categorical columns
    df_cat = df.select_dtypes(include=['object'])
    # Get most frequent value for each categorical column
    for column in df_cat.columns:
        most_frequent = df_cat[column].mode()[0]
        print (f"Most frequent value for {column}: {most_frequent}")
        # Fill missing values with most frequent
        df[column] = df[column].fillna(most_frequent)
    return df

# Export map (categorical to numerical) to JSON file
def _export_categorical_map(df_cat, file_path):
    # Create a dictionary to store the mapping for each categorical column
    cat_map = {}
    for column in df_cat.columns:
        # Assign numerical codes
        unique_labels = df_cat[column].unique()
        cat_map[column] = {label: idx for idx, label in enumerate(unique_labels)}
    # Write the mapping to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(cat_map, json_file, indent=4)

# Conver categorical data to numerical data
def convert_categorical_to_numerical(df):
    # Convert categorical columns to numerical 
    df_cat = df.select_dtypes(include=['object'])
    _export_categorical_map(df_cat, 'Data/categorical_map.json')
    for column in df_cat.columns:
        df[column] = df[column].astype('category').cat.codes
    return df
def fill_missing_value(main_data):
    """
    Fills missing values in the dataset.
    Currently fills missing 'bmi' values based on age groups.
    Returns: DataFrame with filled missing values
    """
    # Fill missing BMI values based on age groups
    main_data = _fill_bmi_by_age(main_data)
    
    # Fill missing categorical data
    main_data = _fill_categorical_data(main_data)
    return main_data