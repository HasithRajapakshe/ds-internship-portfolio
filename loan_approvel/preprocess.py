"""Preprocess"""
import pandas as pd
from sklearn.impute import SimpleImputer

SELECTED_COLUMNS = [
    'income', 'credit_score', 'loan_amount', 'loan_approved'
]


def run():

    df = pd.read_csv(
        r"C:\Users\hasit\OneDrive\Desktop\loan_approvel\loan_approval.csv",
        usecols=SELECTED_COLUMNS
    )

    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df['loan_approved'] = df['loan_approved'].astype(int)

    print(df.head())
    return df
