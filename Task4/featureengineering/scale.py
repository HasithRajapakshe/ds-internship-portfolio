from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def scale(x_train, x_test):
    scaler = MinMaxScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test), columns=x_test.columns)
    return x_train_scaled, x_test_scaled
