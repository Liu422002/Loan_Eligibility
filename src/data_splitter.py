import logging
from sklearn.model_selection import train_test_split


def split_features_target(df):
    try:
        X = df.drop("Loan_Approved", axis=1)
        y = df["Loan_Approved"]
        logging.info("Features and target split successfully.")
        return X, y
    except Exception as e:
        logging.error("Error splitting features and target: %s", e)
        raise


def split_data(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        logging.info("Data split successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error splitting data: %s", e)
        raise