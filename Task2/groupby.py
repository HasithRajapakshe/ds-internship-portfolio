import pandas as pd
import numpy as np

data = {
    'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
    'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
    'Sales': [200, 120, 340, 124, 243, 350]

}

df = pd.DataFrame(data)
print(df, "\n")

sales_avg = df.groupby('Company')['Sales'].mean()
print(sales_avg, "\n")

sales_sum = df.groupby('Company')['Sales'].sum()
print(sales_sum, "\n")

multiple_agg = df.groupby('Company').agg({'Sales': ['sum', 'mean', 'std']})
print(multiple_agg, "\n")


grouped = df.groupby('Company')
print(grouped.describe(), "\n")

# transform


def z_score(x):
    return (x - x.mean()) / x.std()


transformed = grouped['Sales'].transform(z_score)
print(transformed, "\n")
