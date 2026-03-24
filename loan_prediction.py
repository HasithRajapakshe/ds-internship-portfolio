import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

selected_colums = ['ID', 'loan_amount',
                   'income', 'credit_type', 'Credit_Score', 'Status']
df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\Loan_Default.csv", usecols=selected_colums)

print(df.head())

x = df[['Credit_Score', 'income', 'loan_amount', 'credit_type']]
y = df['Status']
x = pd.get_dummies(x, columns=['credit_type'])

imputer = SimpleImputer(strategy='median')
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)


scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


print(x_train.head(), '\n')
print(y_train.head(), '\n')

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

x_pred = model.predict(x_test)

joblib.dump(model, 'model_loan.pkl')
