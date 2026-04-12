from sklearn.ensemble import GradientBoostingClassifier

class Gradient_Boosting_Model:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                max_depth=3, random_state=42):
        # 1. Initialize the model
        print(f"Initializing Gradient Boosting with n_estimators={n_estimators}, learning_rate={learning_rate}...")
        self.__model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )    
    
    def train(self, X_train, y_train):
        """
        Function to train a Gradient Boosting model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - n_estimators: Number of boosting stages to perform
        - learning_rate: Shrinks the contribution of each tree
        - max_depth: Maximum depth of the individual regression estimators
        - random_state: Seed for reproducibility
        
        Returns:
        - model: The trained Gradient Boosting model
        """
        
        # 2. Train the model
        print("Training the model...")
        self.__model.fit(X_train, y_train)
        
        return self.__model
    
    def getModel(self):
        return self.__model