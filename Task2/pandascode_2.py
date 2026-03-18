import pandas as pd
import numpy as np

data = {
    'date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09', '2026-01-10'],
    'sales': [100, 150, 200, 180, 220, 250, 300, 280, 320, 350]
}
df = pd.DataFrame(data)
print(df, "\n")

df['date'] = pd.to_datetime(df['date'])
print(df, "\n")

df.set_index('date', inplace=True)
print(df, "\n")


resampled = df[['sales']].resample('D')
print(resampled.sum(), "\n ")

resampled = df[['sales']].resample('2D')
print(resampled.sum(), "\n ")

resampled = df[['sales']].resample('3D')
print(resampled.mean(), "\n ")

resampled = df[['sales']].resample('5D')
print(resampled.sum(), "\n ")


df['rolling_mean'] = df['sales'].rolling(window=3).sum()
print(df, "\n ")


df['lag_1'] = df['sales'].shift(1)
print(df, "\n ")
