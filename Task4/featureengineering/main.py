import pandas as pd
from sklearn.model_selection import train_test_split
import feature
import train
import evaluation

data = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\House Price Prediction Dataset.csv")

selected_columns = ['Area', 'Bedrooms', 'Bathrooms', 'Floors',
                    'YearBuilt', 'Location', 'Condition', 'Garage', 'Price']

df = data[selected_columns].copy()

print("Original Data:")
print(df.head())


df = feature.apply_all(df)

x = df.drop(columns=["Price"])
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

model = train.model_train(x_train, y_train)

evaluation.evaluate_model(model, x_test, y_test)
