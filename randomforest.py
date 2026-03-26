"""compare with decision tree and random forest
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\Iris.csv")

print(df.head())

# splite dataset with using the features
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# add the depth range to reduce the complex
depths = range(1, 16)
test_accuracy = []
train_accuracy = []

# get the best depth using the decisiontreeclassifier
for depth in depths:
    # each time start the new model
    dtc = DecisionTreeClassifier(max_depth=depth)
    dtc.fit(x_train, y_train)
    test_accuracy.append(dtc.score(x_test, y_test))
    train_accuracy.append(dtc.score(x_train, y_train))

# get the best depth
best_depth = depths[test_accuracy.index(max(test_accuracy))]

# second time model train using the best depth.
dtc = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
print(classification_report(y_test, y_pred))

# display the tree graph
plt.figure(figsize=(12, 8))
plot_tree(dtc, filled=True)
plt.show()

# display the bar chart
importances = dtc.feature_importances_
plot = plt.figure(figsize=(12, 8))
plt.barh(range(len(importances)), importances)
plt.show()

# use randomforest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)
print(classification_report(y_test, y_pred_rf))


# display the plot
plt.figure(figsize=(12, 8))
plot_tree(rf_classifier.estimators_[0],
          feature_names=x.columns,
          class_names=rf_classifier.classes_,
          filled=True)
plt.show()

# display barh
importances_rf = rf_classifier.feature_importances_
plot = plt.figure(figsize=(12, 8))
plt.barh(range(len(importances_rf)), importances_rf)
plt.show()
