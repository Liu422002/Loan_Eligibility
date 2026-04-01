import logging
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold


def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        logging.info("Model evaluated successfully. Accuracy: %.4f", acc)

        return acc, cm, y_pred
    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        raise


def evaluate_logistic_with_threshold(model, X_test, y_test, threshold=0.7):
    try:
        probabilities = model.predict_proba(X_test)
        y_pred_threshold = (probabilities[:, 1] >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_threshold)
        cm = confusion_matrix(y_test, y_pred_threshold)

        logging.info(
            "Logistic Regression evaluated with threshold %.2f. Accuracy: %.4f",
            threshold,
            acc
        )

        return acc, cm, y_pred_threshold
    except Exception as e:
        logging.error("Error evaluating Logistic Regression with threshold: %s", e)
        raise


def cross_validate_model(model, X_train, y_train):
    try:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kfold)

        mean_score = scores.mean()
        std_score = scores.std()

        logging.info(
            "Cross-validation completed successfully. Mean: %.4f, Std: %.4f",
            mean_score,
            std_score
        )

        return scores, mean_score, std_score
    except Exception as e:
        logging.error("Error during cross-validation: %s", e)
        raise