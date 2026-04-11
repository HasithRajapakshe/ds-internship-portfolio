import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


numerical_features = ['Area', 'Bedrooms',
                      'Bathrooms', 'Floors', 'YearBuilt', 'Garage']
categorical_features = ['Location', 'Condition']

# 1. Feature creation


def create_features(df):
    df['Property_Age'] = 2026 - df['YearBuilt']
    df['Floor_Density'] = df['Area'] / df['Floors']
    df['Total_rooms_per_floor'] = df['Floor_Density'] / \
        df['Bedrooms'] + df['Bathrooms']
    df['Is_Modern'] = (df['YearBuilt'] > 2000).astype(int)
    df['Modern_with_Garage'] = ((df['YearBuilt'] > 2000) &
                                (df['Location'] == 'Suburban') &
                                (df['Garage'] == 'Yes')).astype(int)
    print(df.head())
    return df


# 2. Feature transformation
def transform_features(df):
    df['Garage'] = df['Garage'].map({'Yes': 1, 'No': 0})

    condition_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    df['Condition'] = df['Condition'].map(condition_map)

    df = pd.get_dummies(df, columns=['Location'], dtype=int)
    print("transform features output")
    print(df.head())
    return df


# 3. Feature selection
def select_features(df, target_col='Price', threshold=0.01):
    correlation_matrix = df.corr(numeric_only=True)

    price_corr = correlation_matrix[target_col].abs()  # get positive values
    # extract price colum from the matrix
    # remove the price self correlation
    price_corr = price_corr.drop(labels=[target_col])

    print("\nCorrelation with Price (all features):")
    # arrange into ascending oder
    print(price_corr.sort_values(ascending=False))

    # filter by threshold value
    selected_cols = price_corr[price_corr > threshold].index.tolist()

    print(f"\nSelected features: {selected_cols}")
    return df[selected_cols + [target_col]]


# 4. Feature extraction (PCA) creating new features combining
def extract_features(df, target_col='Price'):

    y = df[target_col]
    feature_cols = [col for col in df.columns if col !=
                    target_col]  # take every colum expect price
    X = df[feature_cols]

    pca = PCA(n_components=0.95)  # get the 95% from the all data using PC1,2,3
    X_pca = pca.fit_transform(X)

    # Creates readable column names: PC1, PC2, PC3
    pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    # transform into dataframes
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca[target_col] = y.reset_index(drop=True)  # reattach the price columns

    print(df_pca.head())
    return df_pca


def apply_all(df):
    df = create_features(df)
    df = transform_features(df)
    df = select_features(df)
    df = extract_features(df)
    return df
