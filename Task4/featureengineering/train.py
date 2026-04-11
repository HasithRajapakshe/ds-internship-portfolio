from xgboost import XGBRegressor
import joblib


def model_train(x_train, y_train):
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(x_train, y_train)
    joblib.dump(model, 'model.pkl')
    return model
