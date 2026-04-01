import logging
import pandas as pd


def convert_column_types(df):
    try:
        df["Credit_History"] = df["Credit_History"].astype("object")
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")
        logging.info("Column types converted successfully.")
        return df
    except Exception as e:
        logging.error("Error converting column types: %s", e)
        raise


def fill_missing_values(df):
    try:
        df["Gender"].fillna("Male", inplace=True)
        df["Married"].fillna(df["Married"].mode()[0], inplace=True)
        df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
        df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
        df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
        df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
        df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

        logging.info("Missing values filled successfully.")
        return df
    except Exception as e:
        logging.error("Error filling missing values: %s", e)
        raise


def drop_columns(df):
    try:
        if "Loan_ID" in df.columns:
            df = df.drop("Loan_ID", axis=1)
        logging.info("Unnecessary columns dropped successfully.")
        return df
    except Exception as e:
        logging.error("Error dropping columns: %s", e)
        raise


def encode_data(df):
    try:
        df = pd.get_dummies(
            df,
            columns=[
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self_Employed",
                "Property_Area"
            ],
            dtype=int
        )

        df["Loan_Approved"] = df["Loan_Approved"].replace({"Y": 1, "N": 0})

        logging.info("Categorical variables encoded successfully.")
        return df
    except Exception as e:
        logging.error("Error encoding data: %s", e)
        raise


def preprocess_data(df):
    df = convert_column_types(df)
    df = fill_missing_values(df)
    df = drop_columns(df)
    df = encode_data(df)
    return df