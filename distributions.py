import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
N = 1000

normal_sample = np.random.normal(loc=0, scale=1, size=N)
uniform_sample = np.random.uniform(low=0, high=1, size=N)
binomial_sample = np.random.binomial(n=20, p=0.5, size=N)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(normal_sample, kde=True, color="blue")
plt.title("Normal Distribution")
plt.xlabel("Value")


plt.subplot(1, 3, 2)
sns.histplot(uniform_sample, kde=True, color="orange")
plt.title("Uniform Distribution")
plt.xlabel("Value")


plt.subplot(1, 3, 3)
sns.histplot(binomial_sample, kde=False, color="green", discrete=True)
plt.title("Binomial Distribution")
plt.xlabel("Number of Successes")

plt.tight_layout()
plt.show()
