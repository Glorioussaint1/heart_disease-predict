import sys
import os

# Set project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add module paths to sys.path
sys.path.append(os.path.join(project_root, 'data_loading'))
sys.path.append(os.path.join(project_root, 'model_training'))
sys.path.append(os.path.join(project_root, 'model_evaluation'))
sys.path.append(os.path.join(project_root, 'predict'))

# Print sys.path to verify paths
print("Current sys.path:")
for path in sys.path:
    print(path)

# Import the modules after adding their paths
try:
    from data_loading import data_loading
    from model_training import model_training
    from model_evaluation import model_evaluation
    from predict import predict
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Load and preprocess data
data = data_loading.load_data()
X_train, X_test, y_train, y_test, scaler = data_loading.preprocess_data(data)

# Train models
log_reg, rf_clf, svc_clf = model_training.train_models(X_train, y_train)

# Evaluate models
for model, name in zip([log_reg, rf_clf, svc_clf], ["Logistic Regression", "Random Forest", "SVC"]):
    accuracy, report, matrix = model_evaluation.evaluate_model(model, X_test, y_test)
    model_evaluation.print_evaluation_results(name, accuracy, report, matrix)

# Make a prediction with the best model (Random Forest in this case)
new_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prediction = predict.make_prediction(rf_clf, scaler, new_data)
print("Prediction (0 = No Heart Disease, 1 = Heart Disease):", prediction)
