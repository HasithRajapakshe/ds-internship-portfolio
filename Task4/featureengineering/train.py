from sklearn.linear_model import LinearRegression
import joblib


def model_train(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    joblib.dump(model, 'model.pkl')
    return model
