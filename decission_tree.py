"""
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.tree import plot_tree


df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\Iris.csv")

print(df.head())

# splite dataset
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

depths = range(1, 16)
test_accuracy = []
train_accuracy = []

for depth in depths:
    dtc = DecisionTreeClassifier(max_depth=depth)
    dtc.fit(x_train, y_train)
    test_accuracy.append(dtc.score(x_test, y_test))
    train_accuracy.append(dtc.score(x_train, y_train))


best_depth = depths[test_accuracy.index(max(test_accuracy))]


print(x_train.head(), '\n')
print(y_train.head(), '\n')

dtc = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dtc.fit(x_train, y_train)

joblib.dump(dtc, 'model_iris.pkl')

y_pred = dtc.predict(x_test)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(dtc, filled=True)
plt.show()


importances = dtc.feature_importances_
plot = plt.figure(figsize=(12, 8))
plt.barh(range(len(importances)), importances)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(depths, train_accuracy, marker='o', label="Cross-Validation Accuracy")
plt.plot(depths, test_accuracy, marker='o', label="Test Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
