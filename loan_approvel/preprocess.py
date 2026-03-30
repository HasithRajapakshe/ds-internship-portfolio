"""Preprocessing: imputes missing values and scales features."""

from sklearn.impute import SimpleImputer


def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')

    feature_cols = df.columns.difference(['loan_approved'])
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    print(df.head())
    return df
