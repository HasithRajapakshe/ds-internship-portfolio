"""Train/test split and model training."""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib


def split_data(df):
    x = df[['income', 'credit_score', 'loan_amount']]
    y = df['loan_approved'].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    joblib.dump(model, 'model1.pkl')
    return model


def train_model2(x_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, 'model2.pkl')
    return model


def train_model3(x_train, y_train):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    joblib.dump(model, 'model3.pkl')
    return model
