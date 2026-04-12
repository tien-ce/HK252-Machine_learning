from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class Decision_Tree_Model:
    def __init__(self, min_samples_split=2, random_state=42):
        # 1. Initialize the model setup
        print(f"Initializing Decision Tree with min_samples_split={min_samples_split}...")
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.__model = None
        self.best_params = None
        self.best_f1 = -1

    def train(self, X_train, y_train):
        """
        Function to train and tune a Decision Tree model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        
        Returns:
        - model: The trained Decision Tree model with best hyperparameters
        """
        
        print("Training and tuning the model...")
        
        # Split data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=self.random_state
        )

        # Hyperparameter tuning loops
        for criterion in ["gini", "entropy", "log_loss"]:
            for max_depth in [3, 4, 5, 6, 8, 10, 12, None]:
                for min_samples_leaf in [1, 2, 5, 10, 20]:
                    
                    # 2. Initialize temporary model with current grid parameters
                    model = DecisionTreeClassifier(
                        criterion=criterion,                 
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,  
                        min_samples_split=self.min_samples_split,
                        random_state=self.random_state,
                        class_weight='balanced'
                    )
                    
                    model.fit(X_tr, y_tr)

                    # Validate the model
                    val_pred = model.predict(X_val)
                    f1 = f1_score(y_val, val_pred, zero_division=0)
                    
                    # Update best model if improvement is found
                    if f1 > self.best_f1:
                        self.best_f1 = f1
                        self.__model = model
                        self.best_params = {
                            "criterion": criterion,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf
                        }

        print("Best params on validation:", self.best_params)
        print("Best validation F1:", self.best_f1)
        
        return self.__model

    def getModel(self):
        """
        Returns the trained Decision Tree model.
        """
        if self.__model is None:
            print("Warning: The model hasn't been trained yet.")
        return self.__model