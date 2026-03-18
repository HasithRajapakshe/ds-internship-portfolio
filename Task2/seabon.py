import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataets\Income.csv')
print(df)

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(df)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxenplot(data=df)
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df)
plt.show()
