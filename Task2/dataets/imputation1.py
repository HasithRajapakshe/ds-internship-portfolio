import numpy as np

data = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])

print("Original Data:")
print(data, "\n")

mean_value = np.nanmean(data)

data = np.where(np.isnan(data), mean_value, data)

print(data)
