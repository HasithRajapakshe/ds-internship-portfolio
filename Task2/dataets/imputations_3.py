import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

df = pd.read_csv('dataets/income.csv')
print(df)

# only these two colums proccesed/check only the these colums
impute_features = ['Income', 'Age']


df_mean = df.copy()  # take the copy of the df
# calculate the arithmatic mean
mean_imputer = SimpleImputer(strategy='mean')
df_mean[impute_features] = mean_imputer.fit_transform(df_mean[impute_features])
print("Mean Imputed:\n", df_mean.head(31), "\n")

df_mode = df.copy()
# replace with mode
mode_imputer = SimpleImputer(strategy='most_frequent')
df_mode[impute_features] = mode_imputer.fit_transform(df_mode[impute_features])
print("Mode Imputed:\n", df_mode.head(31), "\n")

df_knn = df.copy()
# use the neighbors values according to the rows amount
knn_imputer = KNNImputer(n_neighbors=5)
df_knn[impute_features] = knn_imputer.fit_transform(df_knn[impute_features])
print("KNN Imputed:\n", df_knn.head(31), "\n")
