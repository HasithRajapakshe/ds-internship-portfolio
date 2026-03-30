import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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
model = RandomForestClassifier(
    n_estimators=100, max_depth=5, max_features='sqrt', max_leaf_nodes=6)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_pred, y_test))

joblib.dump(model, 'hyperpara_model.pkl')

# GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
