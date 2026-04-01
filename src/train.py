import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(X_train, y_train):
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        logging.info("Logistic Regression model trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training Logistic Regression model: %s", e)
        raise


def train_decision_tree(X_train, y_train):
    try:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        logging.info("Decision Tree model trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training Decision Tree model: %s", e)
        raise


def train_random_forest(X_train, y_train):
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training Random Forest model: %s", e)
        raise


def train_tuned_random_forest(X_train, y_train):
    try:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            max_features="sqrt",
            random_state=42
        )
        model.fit(X_train, y_train)
        logging.info("Tuned Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training Tuned Random Forest model: %s", e)
        raise