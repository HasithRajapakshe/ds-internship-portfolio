import numpy as np
from sklearn.impute import KNNImputer

data = np.array([[1, 2, np.nan], [3, 4, 5], [6, np.nan, 8]])

imputer = KNNImputer(n_neighbors=2, weights="uniform")

data_imputed = imputer.fit_transform(data)
print(data_imputed)
