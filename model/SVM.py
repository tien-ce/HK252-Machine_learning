from sklearn.svm import SVC

class SVM_Model:
    def __init__(self, kernel='rbf', C=1.0, random_state=42):
        """
        Initialize the SVM model with hyperparameters.
        
        Parameters:
        - kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        - C: Regularization parameter
        - random_state: Seed for reproducibility
        """
        # 1. Initialize the model
        print(f"Initializing SVM model with kernel='{kernel}', C={C}...")
        self.__model = SVC(
            kernel=kernel, 
            C=C, 
            random_state=random_state, 
            probability=True, 
            class_weight='balanced'
        )
    
    def train(self, X_train, y_train):
        """
        Function to train the SVM model.
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        
        Returns:
        - model: The trained SVM model
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