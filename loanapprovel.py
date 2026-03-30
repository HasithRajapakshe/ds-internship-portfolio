import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestClassifier


selected_colums = ['income', 'credit_score',
                   'loan_amount', 'years_employed', 'points', 'loan_approved']
df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\loan_approval.csv", usecols=selected_colums)

print(df.head())

x = df[['income', 'credit_score',
        'loan_amount', 'years_employed', 'points']]
y = df['loan_approved']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


# add pipeline methood
# for logisticRegression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(x_train, y_train)

lg_pred = pipe.predict(x_test)
print("for logistic regression")
print(classification_report(y_test, lg_pred))

# for randomforest

pipe2 = Pipeline([('scaler', StandardScaler),
                 ('model', RandomForestClassifier())])
pipe2.fit(x_train, y_train)
rf_pred = pipe2.predict(x_test)
print("for random forest")
print(classification_report(y_test, rf_pred))
