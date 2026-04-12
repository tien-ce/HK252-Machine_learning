from sklearn.ensemble import RandomForestClassifier

class Random_Forest_Model:
    def __init__(self, n_estimators=300, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, 
                 random_state=42):
        """
        Initialize the Random Forest model with hyperparameters.
        
        Parameters:
        - n_estimators: Number of trees in the forest
        - max_depth: Maximum depth of each tree
        - min_samples_split: Minimum samples required to split an internal node
        - min_samples_leaf: Minimum samples required to be at a leaf node
        - random_state: Seed for reproducibility
        """
        # 1. Initialize the model
        print(f"Initializing Random Forest with n_estimators={n_estimators}...")
        self.__model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced',  
            n_jobs=-1        
        )

    def train(self, X_train, y_train):
        """
        Function to train the Random Forest model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        
        Returns:
        - model: The trained Random Forest model
        """
        
        # 2. Train the model
        print("Training the model...")
        self.__model.fit(X_train, y_train)
        
        return self.__model

    def getModel(self):
        """
        Returns the trained model instance.
        """
        return self.__model