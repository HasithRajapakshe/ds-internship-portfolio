import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import pandas as pd

numerical_features = ['Area', 'Bedrooms',
                      'Bathrooms', 'Floors', 'YearBuilt', 'Garage']
categorical_features = ['Location', 'Condition']


def convert_garage(df):
    # Convert garage column into 0 and 1
    df['Garage'] = df['Garage'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df


def minmax_scaling(df):
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df


def polynomial_features(df):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[numerical_features])

    # Updated to get_feature namesout
    poly_feature_names = poly.get_feature_names_out(numerical_features)

    poly_df = pd.DataFrame(
        poly_features, columns=poly_feature_names, index=df.index)

    df = pd.concat([df, poly_df], axis=1)
    df.drop(numerical_features, axis=1, inplace=True)
    return df


def ordinal_encoder(df):
    encoder = ce.OrdinalEncoder(cols=['Condition'])
    df['Condition'] = encoder.fit_transform(df['Condition'])
    return df


def one_hot_encoder(df):
    encoder = ce.OneHotEncoder(cols=['Location'])
    df = encoder.fit_transform(df)
    return df


def apply_all(df):
    df = convert_garage(df)
    df = minmax_scaling(df)
    df = polynomial_features(df)
    df = ordinal_encoder(df)
    df = one_hot_encoder(df)
    return df
