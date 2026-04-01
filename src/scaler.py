import logging
from sklearn.preprocessing import MinMaxScaler


def scale_data(X_train, X_test):
    try:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Data scaled successfully.")
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        logging.error("Error scaling data: %s", e)
        raise