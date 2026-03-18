import numpy as np


data = [6, 2, 3, 4, 5, 1, 50]
sort_data = np.sort(data)
print(sort_data)

q1 = np.percentile(data, 25, interpolation='midpoint')
q2 = np.percentile(data, 50, interpolation='midpoint')
q3 = np.percentile(data, 75, interpolation='midpoint')
print(q1, q2, q3)

iqr = q3-q1
print(iqr)

low_lim = q1-1.5*iqr
up_lim = q3+1.5*iqr
print(low_lim, up_lim)

outliers = []
for i in data:
    if i < low_lim or i > up_lim:
        outliers.append(i)
print(outliers)
