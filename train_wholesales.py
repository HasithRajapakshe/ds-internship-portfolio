import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\Wholesale-customers-data.csv")
print(df.head())

x = df[['Region', 'Fresh', 'Milk', 'Grocery',
       'Frozen', 'Detergents_Paper', 'Delicassen']]

y = df['Channel']-1

print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = XGBClassifier()
model.fit(x_train, y_train)
xgb_pred = model.predict(x_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)


print(classification_report(y_test, xgb_pred))
print(classification_report(y_test, lr_pred))
print(classification_report(y_test, dt_pred))
print(classification_report(y_test, rf_pred))
