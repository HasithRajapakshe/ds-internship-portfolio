import pandas as pd
from sklearn.model_selection import train_test_split
import encode
import scale
import polynomial
import train
import evaluation


data = pd.read_csv(
    r"C:\Users\hasit\OneDrive\Desktop\ds_internship_portfolio\House Price Prediction Dataset.csv")

selected_columns = ['Area', 'Bedrooms', 'Bathrooms', 'Floors',
                    'YearBuilt', 'Location', 'Condition', 'Garage', 'Price']

df = data[selected_columns]

print(df.head())

x = df.drop(columns=["Price"])
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

x_train, x_test = encode.binary_encoder(x_train, x_test)
x_train, x_test = encode.frequency_encoder(x_train, x_test)
x_train, x_test = encode.target_encoder(x_train, x_test, y_train)
print(x_train.head())

x_train, x_test = scale.scale(x_train, x_test)
print(x_train.head())

x_train, x_test = polynomial.polynomial_features(x_train, x_test)
print(x_train.head())

model = train.model_train(x_train, y_train)

evaluation.evaluate_model(model, x_test, y_test)
