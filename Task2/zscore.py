import numpy as np


def calculate_z_score(data):
    mean = np.mean(data)
    std_dev = np.std(data)

    z_scores = (data - mean) / std_dev

    return z_scores


dataset = [3, 9, 23, 43, 53, 4, 5, 30, 35, 50, 70, 150, 6, 7, 8, 9, 10]
z_score = calculate_z_score(dataset)
print("Z-scores:", z_score)
