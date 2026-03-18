import numpy as np
values = [1, 2, 3, 4, 4, 5]
mean = np.mean(values)
print("Mean:", mean)
median = np.median(values)
print("Median:", median)
mode = np.bincount(values).argmax()
print("Mode:", mode)
std_dev = np.std(values)
print("Standard Deviation:", std_dev)
variance = np.var(values)
print("Variance:", variance)
