import numpy as np

class RandomForestClassifier:
    def __init__(self, n_estimators = 100, feature_subset_size = 2, max_depth=None, min_samples_split = None,random_state=None):
        """
            @n_estimators: The number of trees in the random forest
            @feature_subset_size: The number of features to select for each tree
            @max_depth: The maximum depth of each tree
            @min_samples_split: The minimum number of samples required to split an internal node
            @random_state: The random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.feature_subset_size = feature_subset_size
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    
    def build_forest (self, dataset):
        """
            Function to build the random forest
                @dataset: The dataset to build the random forest with
        """
        bost_strapped_datasets = self.bootstrap_dataset (dataset, self.feature_subset_size)   # Create bootstrapped datasets
        for i in range (self.n_estimators):
            tree = DecisionTreeClassifier (max_depth = self.max_depth, min_samples_split = self.min_samples_split) # Create a decision tree
            tree.root = tree.build_tree (bost_strapped_datasets[i])
            self.trees.append (tree) # Add the tree to the forest
    
    def bootstrap_dataset (self, dataset, feature_subset_size):
        """
            Function to create bootstrapped datasets
                @dataset: The dataset to create bootstrapped datasets from
                @feature_subset_size: The number of features to select for each tree
        """
        bootstrapped_datasets = []
        n_samples, n_features = dataset.shape
        for _ in range (self.n_estimators):
            # Randomly select samples with replacement (Example: [1, 2, 3, 4, 5] -> [1, 1, 3, 4, 5])
            indices = np.random.choice (n_samples, size = n_samples, replace = True)
            bootstrapped_dataset = dataset[indices]

            # Randomly select features without replacement (Example: [0, 1, 2, 3, 4] -> [0, 2])
            feature_indices = np.random.choice (n_features - 1, size = feature_subset_size, replace = False)
            bootstrapped_dataset = bootstrapped_dataset[:, np.append(feature_indices, n_features - 1)] # Append the target variable index
            bootstrapped_datasets.append (bootstrapped_dataset)
        
        return bootstrapped_datasets

    def print_forest (self):
        """
            Function to print the random forest
        """
        for i, tree in enumerate (self.trees):
            print (f"Tree {i + 1}:")
            tree.print_tree ()
            print ("\n")
    
    def fit (self, X, y):
        """
            Function to fit the random forest to the data
                @X: The feature matrix
                @y: The target variable
        """
        dataset = np.concatenate ((X, y.reshape(-1, 1)), axis = 1) # Combine features and target variable into a single dataset
        self.build_forest (dataset)

    def predict (self, X):
        """
            Function to predict the target variable for the given feature matrix
                @X: The feature matrix
        """
        predictions = [self.make_predictions(x) for x in X] # Get predictions from each feature vector
        return predictions # Return the array of predictions

    def make_predictions (self, x):
        """
            Function to make a prediction for a single feature vector using the random forest
                @x: The feature vector to predict the target variable for
        """
        tree_predictions = [tree.make_prediction (x, tree.root) for tree in self.trees] # Get predictions from each tree in the forest
        return max (tree_predictions, key = tree_predictions.count) # Return the most common prediction among the trees as the final prediction
    
class Node():
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, information_gain = None, value = None):
        # For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.information_gain = information_gain

        # For leaf node
        self.value = value


class DecisionTreeClassifier():
    def __init__(self, min_samples_split = 2, max_depth = 2):
        self.root = None

        # Stop conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree (self, dataset, curr_depth = 0):
        """
        Recursive function to build the decision tree
            @dataset: The dataset to build the tree with
            @curr_depth: The current depth of the tree
        """
        # Split the dataset into features and target variable
        X, y = dataset[:,:-1], dataset[:,-1]
        number_samples, number_features = np.shape(X)

        # Split until stopping conditions are met
        if number_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split (dataset, number_features)
            if not best_split:
                # If there is no valid split, compute leaf node
                # print ("No valid split found. Creating a leaf node.")
                leaf_value = self.calculate_leaf_value (y)
                return Node (value = leaf_value)
            if best_split ["information_gain"] > 0: # If there is a valid split
                # Recur left
                left_subtree = self.build_tree (best_split["dataset_left"], curr_depth = curr_depth + 1)

                # Recur right
                right_subtree = self.build_tree (best_split["dataset_right"], curr_depth= curr_depth + 1)

                # Retrun decision node
                return Node (feature_index = best_split["feature_index"], threshold = best_split ["threshold"], left = left_subtree,
                            right = right_subtree, information_gain= best_split["information_gain"])
        
        # Compute leaf node
        leaf_value = self.calculate_leaf_value (y)
        # Return leaf node
        return Node (value = leaf_value)
    
    def get_best_split (self, dataset, number_features):
        """
            Function to find the best split for the dataset
                @dataset: The dataset to find the best split for
                @number_samples: The number of samples in the dataset
                @number_features: The number of features in the dataset
        """
        best_split = {}
        max_info_gain = -float("inf")

        # Loop over all the features
        for feature_index in range (number_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique (feature_values) # Get all unique values of the feature to consider as possible thresholds

            # Loop over all of present dataset values of the feature to find the best threshold
            for threshold in possible_thresholds:
                # Get current split
                dataset_left, dataset_right = self.split (dataset, feature_index, threshold)
    
                # Check if the split is valid
                if len (dataset_left) > 0 and len (dataset_right) > 0: # If both left and right splits have at least one sample
                    y, left_y, right_y = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]
                    curr_info_gain = self.information_gain (y, left_y, right_y)
                    # Update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["information_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        # print (f"Best split: {best_split}")
        return best_split

    def split (self, dataset, feature_index, threshold):
        """
            Function to split the dataset based on the feature index and threshold
                @dataset: The dataset to split
                @feature_index: The index of the feature to split on
                @threshold: The threshold valuand len (dataset_right) > 0
        """
        dataset_left = np.array ([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array ([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def information_gain (self, parent, left_child, right_child, mode = "entropy"):
        """
            Function to calculate the information gain of a split
                @parent: The target variable of the parent node
                @left_child: The target variable of the left child node
                @right_child: The target variable of the right child node
                @mode: The method to calculate the information gain (default is "entropy")
        """
        weight_left = len (left_child) / len (parent)
        weight_right = len (right_child) / len (parent)
        
        if mode == "gini":
            gain = self.gini_index (parent) - (weight_left * self.gini_index (left_child) + weight_right * self.gini_index (right_child))
        else:
            gain = self.entropy (parent) - (weight_left * self.entropy (left_child) + weight_right * self.entropy (right_child))
        return gain
    
    def entropy (self, y):
        """
            Function to calculate the entropy of a set of labels
                @y: The target variable to calculate the entropy for
        """
        class_labels = np.unique (y)
        entropy = 0

        for cls in class_labels:
            p_cls = len (y[y == cls]) / len (y) # Probability of samples belonging to class cls
            # Entropy formula: - sum (p_cls * log2 (p_cls)) for all classes with p_cls being the probability of samples belonging to class cls
            entropy += -p_cls * np.log2 (p_cls) 
        return entropy
    
    def gini_index (self, y):
        """
            Function to calculate the gini index of a set of labels
                @y: The target variable to calculate the gini index for
        """
        class_labels = np.unique (y)
        gini = 1

        for cls in class_labels:
            p_cls = len (y[y == cls]) / len (y) # Probability of samples belonging to class cls
            # Gini index formula: 1 - sum (p_cls^2) for all classes with p_cls being the probability of samples belonging to class cls
            gini -= p_cls ** 2 
        return gini
    
    def calculate_leaf_value (self, y):
        """
            Function to calculate the value of a leaf node
                @y: The target variable to calculate the leaf value for
        """
        y = list (y) # Convert the target variable to a list to use the count method
        return max (y, key = y.count) # Return the most common class label in the leaf node as the leaf value
    
    def print_tree (self, tree = None, indent = " "):
        """
            Function to print the decision tree
                @tree: The decision tree to print (default is None, which means to print the entire tree)
                @indent: The indentation to use for each level of the tree (default is a single space)
        """
        if tree is None:
            tree = self.root
        
        # If the node is a leaf node, print the value
        if tree.value is not None:
            print (tree.value)
        else:
            # Print the feature index and threshold for the decision node
            print ("X" + str (tree.feature_index) + " <= " + str (tree.threshold) + " ?")

            # Print the left subtree with increased indentation
            print (indent + "Left:")
            self.print_tree (tree.left, indent + indent)

            # Print the right subtree with increased indentation
            print (indent + "Right:")
            self.print_tree (tree.right, indent + indent)
    
    def fit (self, X, y):
        """
            Function to fit the decision tree to the training data
                @X: The feature matrix of the training data
                @y: The target variable of the training data
        """
        dataset = np.concatenate ((X, y.reshape (-1, 1)), axis = 1) # Combine the feature matrix and target variable into a single dataset
        self.root = self.build_tree (dataset) # Build the decision tree using the combined dataset
    
    def predict (self, X):
        """
            Function to predict the target variable for the given feature matrix
                @X: The feature matrix to predict the target variable for
        """
        predictions = [self.make_prediction (x, self.root) for x in X] # Make a prediction for each sample in the feature vector using the decision tree
        return predictions
    
    def make_prediction (self, x, tree):
        """
            Function to make a prediction for a single sample using the decision tree
                @x: The feature vector of the sample to predict the target variable for
                @tree: The decision tree to use for making the prediction
        """
        # If the node is a leaf node, return the value
        if tree.value is not None:
            return tree.value
        
        # Traverse the decision tree based on the feature index and threshold
        feature_value = x[tree.feature_index]
        if feature_value <= tree.threshold:
            return self.make_prediction (x, tree.left) # Traverse left subtree
        else:
            return self.make_prediction (x, tree.right) # Traverse right subtree