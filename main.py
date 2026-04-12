import sys
from pre_processing.handle_data import preprocess_data, get_X_and_Y, normalize_data, apply_SMOTE
from model.model import choose_model
from evaluation.evaluate import evaluate_metrics, save_metrics_to_json


# Main execution
if __name__ == "__main__":
    file_path = 'dataset/healthcare-dataset-stroke-data.csv'
    random_state= 42

    # Data processing
    train_data, test_data= preprocess_data(file_path, random_state= random_state)
    train_data, label_train= get_X_and_Y(train_data)
    test_data, label_test= get_X_and_Y(test_data)


    # Choosing model and training
    command = sys.argv[1]
    model= choose_model(command, random_state)

    if command != 'Decision_Tree_Model': # Don't apply normalize and SMOTE for decision tree model
        train_data, test_data= normalize_data(train_data, test_data)
        train_data, label_train= apply_SMOTE(train_data, label_train, random_state= random_state)

    model.train(train_data, label_train)

    if command != 'Random_Forest_Model': # Random_Forest_Model is already a model, so there's no need to use getModel().
        model= model.getModel()
    

    # Evaluation
    y_pred= model.predict(test_data)
    y_prob= model.predict_proba(test_data)[:, 1]
    metrics= evaluate_metrics(label_test, y_pred, y_prob, name_model= command)
    save_metrics_to_json(metrics, f"log/{command}_results.json")