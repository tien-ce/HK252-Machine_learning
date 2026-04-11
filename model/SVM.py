from sklearn.svm import SVC

def SVM_Model(X_train, y_train, kernel='rbf', C=1.0, random_state=42):
    """
    Function to train and evaluate an SVM model.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - kernel: Specifies the kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid')
    - C: Regularization parameter
    - random_state: Seed for reproducibility
    
    Returns:
    - model: The trained SVM model
    """
    
    # 1. Initialize the model
    print(f"Initializing SVM model with kernel='{kernel}', C={C}...")
    model = SVC(
        kernel=kernel, 
        C=C, 
        random_state=random_state, 
        probability=True, 
        class_weight='balanced'
    )
    
    # 2. Train the model
    print("Training the model...")
    model.fit(X_train, y_train)
    
    return model