import numpy as np
import pandas as pd
import os

os.chdir("C:\\Users\\hasit\\OneDrive\\Desktop\\ds_internship_portfolio\\Task2\\dataets")

df1 = pd.read_csv("price_final.csv")
df2 = pd.read_csv("insurance_data.csv")
df3 = pd.read_csv("AutoInsurance.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)
print(df, "\n")


if __name__ == "__main__":
    from function2 import DataCleaner
    data_cleaner = DataCleaner(df)
    data_cleaner.detect_missing()
    data_cleaner.fill_missing()
    data_cleaner.remove_outliers()
    data_cleaner.encode_categoricals()
    print("\n")
    print(data_cleaner.data)
