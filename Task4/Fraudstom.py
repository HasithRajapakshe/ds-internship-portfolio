import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\Fraud Detection Dataset.csv")

selected_columns = ['Transaction_Amount', 'Transaction_Type', 'Time_of_Transaction', 'Device_Used', 'Location',
                    'Fraudulent', 'Previous_Fraudulent_Transactions', 'Account_Age', 'Number_of_Transactions_Last_24H', 'Payment_Method']

df = df[selected_columns]


numeric_cols = ['Transaction_Amount', 'Time_of_Transaction',
                'Previous_Fraudulent_Transactions', 'Account_Age',
                'Number_of_Transactions_Last_24H']

imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

encode_col = ["Transaction_Type", "Device_Used", "Location", "Payment_Method"]

label = LabelEncoder()

for col in encode_col:
    df[col] = label.fit_transform(df[col])

print(df.head())


x = df.drop(columns=["Fraudulent"])
y = df["Fraudulent"]

print("Class counts:")
print(df["Fraudulent"].value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

joblib.dump(model, 'logistic_regression_model3.pkl')


print(classification_report(y_test, y_pred))
