from .decision_tree import Decision_Tree_Model
from .gradient_boosting import Gradient_Boosting_Model
from .random_forest_library import Random_Forest_Model
from .random_forest import RandomForestClassifier
from .SVM import SVM_Model

def choose_model(command, random_state= 42):
    if command == 'Decision_Tree_Model':
        model = Decision_Tree_Model(random_state= random_state)
    elif command == 'Gradient_Boosting_Model':
        model = Gradient_Boosting_Model(random_state= random_state)
    elif command == 'Random_Forest_Library_Model':
        model = Random_Forest_Model(random_state= random_state)
    elif command == 'Random_Forest_Model':
        model = RandomForestClassifier(n_estimators = 10, feature_subset_size = 4, max_depth = 5, min_samples_split = 3, random_state= random_state)
    else:
        model = SVM_Model(random_state= random_state)

    return model