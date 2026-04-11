import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib


SELECTED_COLUMNS = [
    'income', 'credit_score', 'loan_amount', 'loan_approved'
]

df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\loan_approval.csv", usecols=SELECTED_COLUMNS)

df['loan_approved'] = df['loan_approved'].astype(int)

print(df.head())

x = df[['credit_score', 'income', 'loan_amount']]
y = df['loan_approved']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


model = XGBClassifier(n_estimators=200, max_depth=5)
model.fit(x_train, y_train)

joblib.dump(model, 'pdp_model.pkl')

key_features = ["credit_score", "income", "loan_amount"]


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Partial Dependence Plots",
             fontsize=16, fontweight="bold")

PartialDependenceDisplay.from_estimator(
    model,
    x_train,
    features=key_features,
    feature_names=x_train.columns,
    kind="average",
    ax=axes
)

plt.tight_layout()
plt.show()
