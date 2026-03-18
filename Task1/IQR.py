import numpy as np

scores = [7, 10, 15, 18, 20, 22, 25, 28, 30]

# Calculate Q1 and Q3
q1 = np.percentile(scores, 25)
q3 = np.percentile(scores, 75)

iqr = q3 - q1

print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
