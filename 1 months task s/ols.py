import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

np.random.seed(42)
x = 2.5 * np.random.randn(100) + 1.5
y = 0.5 * x + np.random.randn(100) * 0.8

df = pd.DataFrame({'x': x, 'y': y})
print(df)

x = sm.add_constant(x)
ols_model = sm.OLS(y, x).fit()
print(ols_model.summary())


plt.figure(figsize=(12, 6))
plt.scatter(x[:, 1], y, color='blue', label='Actual Data')
plt.plot(x[:, 1], ols_model.predict(x),
         color='red', label='OLS Regression line')
plt.xlabel('Independent Variable(x)')
plt.ylabel('Dependent Variable (Y)')
plt.title('Ordinary ls regression')
plt.legend()
plt.grid()
plt.show()
