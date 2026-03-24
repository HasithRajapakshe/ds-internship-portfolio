
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np


data = pd.read_csv(
    r'C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\ds-internship\Task3\Housing.csv')
print(data)


plt.scatter(data['area'], data['price'], color='red')
plt.xlabel('area')
plt.ylabel('price')
plt.show()

df = pd.DataFrame(data)
x = df['area'].values.reshape(-1, 1)
y = df['price'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

joblib.dump(model, 'housing_model.pkl')

plt.scatter(data['area'], data['price'], color='red')
plt.plot(x, model.predict(x), color='blue')
plt.xlabel('area')
plt.ylabel('price')
plt.show()

loaded_model = joblib.load('housing_model.pkl')
y_pred = loaded_model.predict(x_test)
rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse1)

print("intercept:", model.intercept_)
print("coef:", model.coef_)
