import os
import logging

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.data_splitter import split_features_target, split_data
from src.scaler import scale_data

from src.train import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_tuned_random_forest
)
from src.evaluation import (
    evaluate_model,
    evaluate_logistic_with_threshold,
    cross_validate_model
)
from src.save_model import save_model


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_folders():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def main():
    setup_logging()
    create_folders()

    try:
        input_file = "data/credit.csv"
        processed_file = "data/Processed_Credit_Dataset.csv"

        logging.info("Program started.")

        df = load_data(input_file)

        df = preprocess_data(df)
        df.to_csv("data/Processed_Credit_Dataset.csv", index=False)

        X, y = split_features_target(df)

        X_train, X_test, y_train, y_test = split_data(X, y)

        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

        logistic_model = train_logistic_regression(X_train_scaled, y_train)
        decision_tree_model = train_decision_tree(X_train_scaled, y_train)
        random_forest_model = train_random_forest(X_train_scaled, y_train)
        tuned_random_forest_model = train_tuned_random_forest(X_train_scaled, y_train)

        logistic_acc, logistic_cm, logistic_pred = evaluate_model(
            logistic_model,
            X_test_scaled,
            y_test
        )

        logistic_threshold_acc, logistic_threshold_cm, logistic_threshold_pred = evaluate_logistic_with_threshold(
            logistic_model,
            X_test_scaled,
            y_test,
            threshold=0.7
        )

        decision_tree_acc, decision_tree_cm, decision_tree_pred = evaluate_model(
            decision_tree_model,
            X_test_scaled,
            y_test
        )

        random_forest_acc, random_forest_cm, random_forest_pred = evaluate_model(
            random_forest_model,
            X_test_scaled,
            y_test
        )

        tuned_rf_acc, tuned_rf_cm, tuned_rf_pred = evaluate_model(
            tuned_random_forest_model,
            X_test_scaled,
            y_test
        )

        logistic_scores, logistic_mean, logistic_std = cross_validate_model(
            logistic_model,
            X_train_scaled,
            y_train
        )

        random_forest_scores, random_forest_mean, random_forest_std = cross_validate_model(
            tuned_random_forest_model,
            X_train_scaled,
            y_train
        )

        print("Logistic Regression Accuracy:", logistic_acc)
        print("Logistic Regression Confusion Matrix:")
        print(logistic_cm)
        print()

        print("Logistic Regression Accuracy with 0.7 Threshold:", logistic_threshold_acc)
        print("Logistic Regression Confusion Matrix with 0.7 Threshold:")
        print(logistic_threshold_cm)
        print()

        print("Decision Tree Accuracy:", decision_tree_acc)
        print("Decision Tree Confusion Matrix:")
        print(decision_tree_cm)
        print()

        print("Random Forest Accuracy:", random_forest_acc)
        print("Random Forest Confusion Matrix:")
        print(random_forest_cm)
        print()

        print("Tuned Random Forest Accuracy:", tuned_rf_acc)
        print("Tuned Random Forest Confusion Matrix:")
        print(tuned_rf_cm)
        print()

        print("Logistic Regression Cross Validation Scores:", logistic_scores)
        print("Logistic Regression Mean Accuracy:", logistic_mean)
        print("Logistic Regression Standard Deviation:", logistic_std)
        print()

        print("Random Forest Cross Validation Scores:", random_forest_scores)
        print("Random Forest Mean Accuracy:", random_forest_mean)
        print("Random Forest Standard Deviation:", random_forest_std)
        print()

        save_model(logistic_model, "models/logistic_model.pkl")
        save_model(decision_tree_model, "models/decision_tree_model.pkl")
        save_model(random_forest_model, "models/random_forest_model.pkl")
        save_model(tuned_random_forest_model, "models/tuned_random_forest_model.pkl")
        save_model(scaler, "models/scaler.pkl")

        logging.info("Program finished successfully.")

    except Exception as e:
        logging.error("Program failed: %s", e)
        print("An error occurred:", e)


if __name__ == "__main__":
    main()