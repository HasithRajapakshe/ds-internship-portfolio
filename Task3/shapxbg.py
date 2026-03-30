"""this is the shap using xgboost"""
import shap
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\Wholesale-customers-data.csv")


x = df.drop(columns=["Channel"])
y = df["Channel"]-1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()
model.fit(x_train, y_train)

joblib.dump(model, 'shap_model.pkl')

print(classification_report(y_test, model.predict(x_test)))

explainer = shap.TreeExplainer(model)
shap_values = explainer(x_test)

shap.waterfall_plot(shap_values[0])
