from sklearn.ensemble import RandomForestClassifier

def Random_Forest_Model(X_train, y_train,
                        n_estimators=300, max_depth=None,
                        min_samples_split=2, min_samples_leaf=1,
                        random_state=42):
    """
    Function to train a Random Forest model.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of each tree
    - min_samples_split: Minimum number of samples required to split an internal node
    - min_samples_leaf: Minimum number of samples required to be at a leaf node
    - random_state: Seed for reproducibility
    
    Returns:
    - model: The trained Random Forest model
    """
    
    # 1. Initialize the model
    print(f"Initializing Random Forest with n_estimators={n_estimators}...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight='balanced',  
        n_jobs=-1        
    )
    
    # 2. Train the model
    print("Training the model...")
    model.fit(X_train, y_train)
    
    return model