import pandas as pd
from pre_processing.handle_data import split_data, fill_missing_value

# Define global variables
file_path = 'Data/healthcare-dataset-stroke-data.csv'
train_data, val_data, test_data = None, None, None
# Preprocess data function
def preprocess_data():
    # Load data
    data = pd.read_csv(file_path)
    
    # Fill missing values
    data_cleaned = fill_missing_value(data)
    
    # Split data into train, validation, and test sets
    global train_data, val_data, test_data
    train_data, val_data, test_data = split_data(data_cleaned)


# Export data into CSV files
def export_data():
    if train_data is not None and val_data is not None and test_data is not None:
        train_data.to_csv('Data/train_data.csv', index=False)
        val_data.to_csv('Data/val_data.csv', index=False)
        test_data.to_csv('Data/test_data.csv', index=False)
    else:
        raise ValueError("Data has not been preprocessed. Please run preprocess_data() first.")
    

# Main execution
if __name__ == "__main__":
    # Run preprocessing
    preprocess_data()
    export_data()
    print("Data preprocessing and export completed.")