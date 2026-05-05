import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

selected_colums = ['ID', 'loan_amount',
                   'income', 'credit_type', 'Credit_Score', 'Status']
df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\Loan_Default.csv", usecols=selected_colums)

print(df.head())

x = df[['Credit_Score', 'income', 'loan_amount', 'credit_type']]
y = df['Status']

# use one hot encoding for credit type
x = pd.get_dummies(x, columns=['credit_type'])

# use for missing data(income)
imputer = SimpleImputer(strategy='median')
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

# adjust the data into low distange range to reduce the outlier
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)


# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


print(x_train.head(), '\n')
print(y_train.head(), '\n')

# model train using logical regression
model = LogisticRegression(max_iter=1000)  # use that to get the best solution
model.fit(x_train, y_train)

x_pred = model.predict(x_test)

joblib.dump(model, 'model_loan.pkl')


# confusion matrix
# compare the actual value and the predicted values
confussion_matrix = confusion_matrix(y_test, x_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confussion_matrix, display_labels=model.classes_)  # use the status classes
disp.plot()
plt.show()
print(confussion_matrix)

# use roc
for m in [model]:
    fpr, tpr, thresholds = roc_curve(y_test, x_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
plt.show()


# use precision score and recall
precision_score = precision_score(y_test, x_pred)
recall_score = recall_score(y_test, x_pred)
print(f"Precision Score: {precision_score}")
print(f"Recall Score: {recall_score}")
