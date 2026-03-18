import numpy as np
import pandas as pd
import os

df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), "train.csv"))

df = pd.concat([df1], ignore_index=True)
print(df, "\n")


if __name__ == "__main__":
    from functions import DataCleaner
    data_cleaner = DataCleaner(df)
    data_cleaner.detect_missing()
    data_cleaner.fill_missing()
    data_cleaner.remove_outliers()
    data_cleaner.encode_categoricals()
    print("\n")
    print(data_cleaner.data)
