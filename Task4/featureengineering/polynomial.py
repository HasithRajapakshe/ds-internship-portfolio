from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def polynomial_features(x_train, x_test):
    poly = PolynomialFeatures(degree=2)
    x_train_poly = pd.DataFrame(
        poly.fit_transform(x_train),
        columns=poly.get_feature_names_out(x_train.columns)
    )
    x_test_poly = pd.DataFrame(
        poly.transform(x_test),
        columns=poly.get_feature_names_out(x_train.columns)
    )
    return x_train_poly, x_test_poly
