from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
def Decision_Tree_Model(X_train, y_train, 
                        min_samples_split=2, 
                        random_state=42):
    """
    Function to train a Decision Tree model.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - max_depth: The maximum depth of the tree
    - min_samples_split: The minimum number of samples required to split an internal node
    - random_state: Seed used by the random number generator for reproducibility
    
    Returns:
    - model: The trained Decision Tree model
    """
    
    # 1. Initialize the model
    print(f"Initializing Decision Tree...")
    
    best_model = None
    best_f1 = -1
    best_params = None

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=random_state
    )

    for criterion in ["gini", "entropy", "log_loss"]:
        for max_depth in [3, 4, 5, 6, 8, 10, 12, None]:
            for min_samples_leaf in [1, 2, 5, 10, 20]:
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state,
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)

                val_pred = model.predict(X_val)
                f1 = f1_score(y_val, val_pred, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_params = {
                        "criterion": criterion,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf
                    }

    print("Best params on validation:", best_params)
    print("Best validation F1:", best_f1)
    
    return best_model