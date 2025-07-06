# main.py

from src.preprocess import load_and_preprocess_data
from src.model import train_logistic_regression, train_random_forest
from src.evaluate import plot_roc_curve, plot_confusion_matrix

if __name__ == "__main__":
    # Load and preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/loan_data.csv')

    # Train models
    log_reg_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate Logistic Regression
    print("Logistic Regression Evaluation:")
    plot_roc_curve(log_reg_model, X_test, y_test)
    plot_confusion_matrix(log_reg_model, X_test, y_test)

    # Evaluate Random Forest
    print("Random Forest Evaluation:")
    plot_roc_curve(rf_model, X_test, y_test)
    plot_confusion_matrix(rf_model, X_test, y_test)
